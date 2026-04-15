from __future__ import annotations

import argparse
import contextlib
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import torch
import torch.distributed as dist
import timm
import torchvision
from timm.data import create_loader, create_transform
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_utils import SelfSupervisedDataset, build_supervised_dataset
from freq_mae_model_v2 import (
    FreqMAE_ImgShape,
    _extract_state_dict,
    extract_timm_mae_decoder_spec,
    extract_timm_vit_backbone_spec,
    load_partial_vit_mae_weights,
)
from gdrive_fallback import init_gdrive_uploader, upload_artifacts_to_gdrive
from utils import calculate_mse, calculate_psnr, setup_seed


@dataclass
class DistributedContext:
    distributed: bool
    rank: int
    world_size: int
    local_rank: int
    host_sync_group: Any | None = None


class FrequencyMAESelfSupervisedTask:
    def __init__(self, model, mask_ratio: float):
        self.model = model
        self.mask_ratio = mask_ratio

    def train_step(self, images: torch.Tensor) -> torch.Tensor:
        _, loss = self.model(images, training=True, custom_mask_ratio=self.mask_ratio)
        if isinstance(loss, torch.Tensor) and loss.ndim > 0:
            loss = loss.mean()
        return loss

    @torch.no_grad()
    def eval_step(self, images: torch.Tensor):
        reconstructed, _, _ = self.model(images, training=False, custom_mask_ratio=self.mask_ratio)
        batch_mse = calculate_mse(reconstructed, images).item()
        batch_psnr = 0.0
        for i in range(images.size(0)):
            batch_psnr += calculate_psnr(reconstructed[i], images[i], data_range=2.0)
        return reconstructed, batch_mse, batch_psnr


def init_distributed_mode() -> DistributedContext:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return DistributedContext(distributed=False, rank=0, world_size=1, local_rank=0)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timedelta(minutes=30),
    )
    host_sync_group = None
    if backend == "nccl":
        host_sync_group = dist.new_group(backend="gloo", timeout=timedelta(minutes=30))
        dist.barrier(group=host_sync_group)
    else:
        dist.barrier()
    return DistributedContext(True, rank, world_size, local_rank, host_sync_group)


def is_main_process(dist_ctx: DistributedContext) -> bool:
    return dist_ctx.rank == 0


def log_event(dist_ctx: DistributedContext, message: str, *, all_ranks: bool = False) -> None:
    if all_ranks or is_main_process(dist_ctx):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [rank {dist_ctx.rank}] {message}", flush=True)


def maybe_barrier(dist_ctx: DistributedContext) -> None:
    if dist_ctx.distributed:
        if dist_ctx.host_sync_group is not None:
            dist.barrier(group=dist_ctx.host_sync_group)
        elif torch.cuda.is_available():
            dist.barrier(device_ids=[dist_ctx.local_rank])
        else:
            dist.barrier()


def cleanup_distributed(dist_ctx: DistributedContext) -> None:
    if dist_ctx.distributed and dist.is_initialized():
        if dist_ctx.host_sync_group is not None:
            dist.barrier(group=dist_ctx.host_sync_group)
        elif torch.cuda.is_available():
            dist.barrier(device_ids=[dist_ctx.local_rank])
        else:
            dist.barrier()
        dist.destroy_process_group()


def get_device(dist_ctx: DistributedContext) -> torch.device:
    if dist_ctx.distributed:
        return torch.device("cuda", dist_ctx.local_rank) if torch.cuda.is_available() else torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_core_model(model):
    return model.module if hasattr(model, "module") else model


def reduce_sum(value: float, device: torch.device, dist_ctx: DistributedContext) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if dist_ctx.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def build_transforms(args: argparse.Namespace):
    input_size = (3, args.image_size, args.image_size)
    train_transform = create_transform(
        input_size=input_size,
        is_training=True,
        no_aug=args.no_aug,
        scale=(args.train_scale_min, args.train_scale_max),
        ratio=(args.train_ratio_min, args.train_ratio_max),
        hflip=args.hflip,
        color_jitter=args.color_jitter,
        interpolation=args.interpolation,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        re_prob=0.0,
        normalize=True,
        use_prefetcher=False,
    )
    eval_transform = create_transform(
        input_size=input_size,
        is_training=False,
        interpolation=args.interpolation,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
        normalize=True,
        use_prefetcher=False,
    )
    return train_transform, eval_transform


def build_datasets(args: argparse.Namespace, dist_ctx: DistributedContext):
    train_transform, eval_transform = build_transforms(args)

    if args.dataset == "cifar10" and is_main_process(dist_ctx):
        build_supervised_dataset(
            dataset=args.dataset,
            data_root=args.data_root,
            split="train",
            transform=train_transform,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            download=True,
        )
        build_supervised_dataset(
            dataset=args.dataset,
            data_root=args.data_root,
            split="val",
            transform=eval_transform,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            download=True,
        )
    maybe_barrier(dist_ctx)

    train_dataset = SelfSupervisedDataset(
        build_supervised_dataset(
            dataset=args.dataset,
            data_root=args.data_root,
            split="train",
            transform=train_transform,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            download=False,
        )
    )
    val_dataset = SelfSupervisedDataset(
        build_supervised_dataset(
            dataset=args.dataset,
            data_root=args.data_root,
            split="val",
            transform=eval_transform,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            download=False,
        )
    )
    return train_dataset, val_dataset


def build_loaders(args: argparse.Namespace, dist_ctx: DistributedContext):
    train_dataset, val_dataset = build_datasets(args, dist_ctx)
    input_size = (3, args.image_size, args.image_size)
    train_loader = create_loader(
        train_dataset,
        input_size=input_size,
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=False,
        no_aug=args.no_aug,
        distributed=dist_ctx.distributed,
        num_workers=args.num_workers,
        pin_memory=(args.pin_mem and torch.cuda.is_available()),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        interpolation=args.interpolation,
        worker_seeding="all",
    )
    val_loader = create_loader(
        val_dataset,
        input_size=input_size,
        batch_size=args.val_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=False,
        distributed=False,
        num_workers=args.num_workers,
        pin_memory=(args.pin_mem and torch.cuda.is_available()),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        interpolation=args.interpolation,
        crop_pct=1.0,
        worker_seeding="all",
    )
    return train_loader, val_loader


def build_timm_mae_template(args: argparse.Namespace):
    if not args.timm_model_name:
        return None

    use_timm_weights = not args.pretrained_ckpt
    checkpoint_path = args.timm_checkpoint_path if use_timm_weights else ""
    timm_model = timm.create_model(
        args.timm_model_name,
        pretrained=(args.timm_pretrained and use_timm_weights),
        checkpoint_path=checkpoint_path or "",
    )

    patch_embed = getattr(timm_model, "patch_embed", None)
    if patch_embed is not None:
        patch_size = patch_embed.patch_size[0] if isinstance(patch_embed.patch_size, tuple) else patch_embed.patch_size
        num_patches = getattr(patch_embed, "num_patches", None)
        expected_num_patches = (args.image_size // args.patch_size) ** 2
        if int(patch_size) != args.patch_size:
            raise ValueError(
                f"timm patch size mismatch: model {args.timm_model_name} uses {patch_size}, "
                f"but args.patch_size={args.patch_size}"
            )
        if num_patches is not None and int(num_patches) != expected_num_patches:
            raise ValueError(
                f"timm patch count mismatch: model {args.timm_model_name} uses {num_patches}, "
                f"but image_size={args.image_size}, patch_size={args.patch_size} gives {expected_num_patches}"
            )

    spec = extract_timm_vit_backbone_spec(timm_model)
    args.emb_dim = spec["emb_dim"]
    args.encoder_layer = spec["encoder_layers"]
    args.encoder_head = spec["num_heads"]
    decoder_spec = extract_timm_mae_decoder_spec(timm_model)
    if decoder_spec is not None:
        expected_patch_dim = 3 * (args.patch_size ** 2)
        if decoder_spec["decoder_patch_dim"] != expected_patch_dim:
            raise ValueError(
                f"timm decoder patch dim mismatch: model {args.timm_model_name} predicts "
                f"{decoder_spec['decoder_patch_dim']}, expected {expected_patch_dim}"
            )
        args.decoder_emb_dim = decoder_spec["decoder_emb_dim"]
        args.decoder_layer = decoder_spec["decoder_layers"]
        args.decoder_head = decoder_spec["decoder_num_heads"]
    return timm_model


def build_model(
    args: argparse.Namespace,
    device: torch.device,
    dist_ctx: DistributedContext,
    *,
    encoder_template=None,
):
    model = FreqMAE_ImgShape(
        image_size=args.image_size,
        block_size=args.block_size,
        channels=3,
        patch_size=args.patch_size,
        patch_dim=args.patch_size * args.patch_size * 3,
        emb_dim=args.emb_dim,
        encoder_layers=args.encoder_layer,
        decoder_emb_dim=args.decoder_emb_dim,
        decoder_layers=args.decoder_layer,
        num_heads=args.encoder_head,
        decoder_num_heads=args.decoder_head,
        mask_ratio=args.mask_ratio,
        norm_pix_loss=args.norm_pix_loss,
        encoder_template=encoder_template,
    ).to(device)
    if dist_ctx.distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    return model


def verify_timm_template_transfer(model, mae_template, dist_ctx: DistributedContext) -> None:
    if mae_template is None or not is_main_process(dist_ctx):
        return

    core_model = get_core_model(model)

    checks = [
        ("cls_token", core_model.cls_token, mae_template.cls_token),
        ("encoder.blocks.0.attn.qkv.weight", core_model.encoder_blocks[0].attn.qkv.weight, mae_template.blocks[0].attn.qkv.weight),
        ("encoder.norm.weight", core_model.encoder_norm.weight, mae_template.norm.weight),
    ]

    if hasattr(mae_template, "decoder_embed"):
        checks.extend(
            [
                ("decoder_embed.weight", core_model.decoder_embed.weight, mae_template.decoder_embed.weight),
                ("decoder.blocks.0.attn.qkv.weight", core_model.decoder_blocks[0].attn.qkv.weight, mae_template.decoder_blocks[0].attn.qkv.weight),
                ("decoder_pred.weight", core_model.output_proj.weight, mae_template.decoder_pred.weight),
            ]
        )

    for name, target, source in checks:
        max_abs_diff = (target.detach() - source.detach().to(device=target.device, dtype=target.dtype)).abs().max().item()
        print(f"[pretrained] transfer check {name}: max_abs_diff={max_abs_diff:.6e}")


def maybe_load_external_pretrained_weights(model, args: argparse.Namespace, device: torch.device, mae_template=None) -> None:
    core_model = get_core_model(model)

    if args.pretrained_ckpt:
        payload = torch.load(args.pretrained_ckpt, map_location=device)
        state_dict = _extract_state_dict(payload)
        print(f"[pretrained] loading partial weights from checkpoint: {args.pretrained_ckpt}")
        load_partial_vit_mae_weights(core_model, state_dict, verbose=True)
        return

    if mae_template is not None and args.timm_model_name:
        print(
            f"[pretrained] loading partial weights from timm model: {args.timm_model_name} "
            f"(pretrained={args.timm_pretrained}, checkpoint_path={args.timm_checkpoint_path or 'none'})"
        )
        load_partial_vit_mae_weights(core_model, mae_template.state_dict(), verbose=True)


def save_checkpoint(
    path: str,
    *,
    epoch: int,
    model,
    optimizer,
    scheduler,
    scaler,
    args: argparse.Namespace,
    best_val_psnr: float,
    best_val_mse: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": get_core_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "args": vars(args),
            "best_val_psnr": best_val_psnr,
            "best_val_mse": best_val_mse,
        },
        path,
    )


def load_checkpoint(path: str, *, model, optimizer, scheduler, scaler, device: torch.device):
    payload = torch.load(path, map_location=device)
    get_core_model(model).load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    return (
        int(payload.get("epoch", 0)),
        float(payload.get("best_val_psnr", float("-inf"))),
        float(payload.get("best_val_mse", float("inf"))),
    )


def train_one_epoch(
    task: FrequencyMAESelfSupervisedTask,
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    dist_ctx: DistributedContext,
    epoch: int,
    grad_accum_steps: int,
) -> float:
    model.train()
    if dist_ctx.distributed and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    local_loss_sum = 0.0
    local_num_steps = 0.0
    local_num_updates = 0.0
    num_update_steps_per_epoch = math.ceil(len(loader) / grad_accum_steps)
    progress = tqdm(
        total=len(loader),
        desc=f"Train {epoch + 1}",
        disable=not is_main_process(dist_ctx),
        mininterval=0.3,
    )

    optimizer.zero_grad(set_to_none=True)
    window_size = grad_accum_steps
    for step_idx, (images, _) in enumerate(loader):
        if step_idx % grad_accum_steps == 0:
            window_size = min(grad_accum_steps, len(loader) - step_idx)

        images = images.to(device, non_blocking=True)
        is_update_step = ((step_idx + 1) % grad_accum_steps == 0) or (step_idx + 1 == len(loader))

        sync_context = contextlib.nullcontext()
        if dist_ctx.distributed and not is_update_step:
            sync_context = model.no_sync()

        with sync_context:
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss = task.train_step(images)
                loss_for_backward = loss / float(window_size)

            if scaler is None:
                loss_for_backward.backward()
            else:
                scaler.scale(loss_for_backward).backward()

        if is_update_step:
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad(set_to_none=True)
            local_num_updates += 1.0

            if scheduler is not None:
                scheduler.step_update(
                    num_updates=epoch * num_update_steps_per_epoch + int(local_num_updates)
                )

        local_loss_sum += loss.item()
        local_num_steps += 1.0
        if is_main_process(dist_ctx):
            progress.set_postfix(loss=local_loss_sum / max(local_num_steps, 1.0))
            progress.update(1)

    if is_main_process(dist_ctx):
        progress.close()

    global_loss_sum = reduce_sum(local_loss_sum, device, dist_ctx)
    global_num_steps = reduce_sum(local_num_steps, device, dist_ctx)
    return global_loss_sum / max(global_num_steps, 1.0)


@torch.no_grad()
def validate(
    task: FrequencyMAESelfSupervisedTask,
    model,
    loader,
    device: torch.device,
    dist_ctx: DistributedContext,
) -> tuple[float, float]:
    model.eval()
    if not is_main_process(dist_ctx):
        log_event(dist_ctx, "validate skipped on non-main rank", all_ranks=True)
        return 0.0, 0.0

    log_event(dist_ctx, "validate start")
    total_psnr = 0.0
    total_mse = 0.0
    total_images = 0.0

    progress = tqdm(total=len(loader), desc="Validate", mininterval=0.3)
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        _, batch_mse, batch_psnr = task.eval_step(images)
        batch_size = float(images.size(0))
        total_mse += batch_mse * batch_size
        total_psnr += batch_psnr
        total_images += batch_size
        progress.update(1)
    progress.close()

    avg_psnr = total_psnr / max(total_images, 1.0)
    avg_mse = total_mse / max(total_images, 1.0)
    log_event(dist_ctx, "validate done")
    return avg_psnr, avg_mse


@torch.no_grad()
def save_epoch_visualization(
    task: FrequencyMAESelfSupervisedTask,
    model,
    loader,
    device: torch.device,
    dist_ctx: DistributedContext,
    output_dir: str,
    epoch: int,
    count: int,
    writer: SummaryWriter | None,
) -> None:
    if not is_main_process(dist_ctx):
        return

    log_event(dist_ctx, f"epoch {epoch + 1}: visualization start")
    import matplotlib.pyplot as plt

    images, _ = next(iter(loader))
    images = images[:count].to(device, non_blocking=True)
    reconstructed, _, _ = task.model(images, training=False, custom_mask_ratio=task.mask_ratio)
    _, freq_spatial, _ = get_core_model(model).patchify(images)

    inputs = ((images.detach().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
    recons = ((reconstructed.detach().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
    freqs = freq_spatial.detach().cpu()
    count = min(count, inputs.size(0))

    fig, axes = plt.subplots(4, count, figsize=(4 * count, 12), dpi=150)
    if count == 1:
        axes = axes.reshape(4, 1)

    for idx in range(count):
        inp = inputs[idx]
        rec = recons[idx]
        freq = freqs[idx]
        freq_min = freq.amin()
        freq_max = freq.amax()
        freq_vis = (freq - freq_min) / (freq_max - freq_min + 1.0e-6)
        err = (rec - inp).abs().mean(dim=0)

        axes[0, idx].imshow(inp.permute(1, 2, 0).numpy())
        axes[0, idx].set_title("Input" if idx == 0 else "")
        axes[1, idx].imshow(freq_vis.permute(1, 2, 0).numpy())
        axes[1, idx].set_title("Frequency" if idx == 0 else "")
        axes[2, idx].imshow(rec.permute(1, 2, 0).numpy())
        axes[2, idx].set_title("Recon" if idx == 0 else "")
        axes[3, idx].imshow(err.numpy(), cmap="magma")
        axes[3, idx].set_title("Abs Err" if idx == 0 else "")
        for row in range(4):
            axes[row, idx].axis("off")

    fig.tight_layout()
    visuals_dir = os.path.join(output_dir, "visuals")
    os.makedirs(visuals_dir, exist_ok=True)
    save_path = os.path.join(visuals_dir, f"epoch_{epoch + 1:04d}.png")
    fig.savefig(save_path, bbox_inches="tight")
    if writer is not None:
        writer.add_figure("visuals/reconstruction_grid", fig, global_step=epoch + 1)
    plt.close(fig)
    log_event(dist_ctx, f"epoch {epoch + 1}: visualization saved to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="self-supervised training for frequency MAE on ImageNet or CIFAR-10")
    parser.add_argument("--dataset", type=str, choices=["imagenet", "cifar10"], default="imagenet")
    parser.add_argument("--data_root", type=str, default="data/imagenet")
    parser.add_argument("--train_dir", type=str, default="")
    parser.add_argument("--val_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="logs/imagenet/ssl_freq_mae")
    parser.add_argument("--experiment", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--pretrained_ckpt", type=str, default="")
    parser.add_argument("--timm_model_name", type=str, default="vit_base_patch16_224.mae")
    parser.add_argument("--timm_pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timm_checkpoint_path", type=str, default="")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pin_mem", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--encoder_layer", type=int, default=12)
    parser.add_argument("--encoder_head", type=int, default=12)
    parser.add_argument("--decoder_emb_dim", type=int, default=512)
    parser.add_argument("--decoder_layer", type=int, default=8)
    parser.add_argument("--decoder_head", type=int, default=16)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--norm_pix_loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visualize_every", type=int, default=10)
    parser.add_argument("--visualize_count", type=int, default=4)

    parser.add_argument("--opt", type=str, default="adamw")
    parser.add_argument("--base_lr", type=float, default=1.5e-4)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--sched", type=str, default="cosine")
    parser.add_argument("--warmup_epochs", type=int, default=40)
    parser.add_argument("--warmup_lr", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--opt_betas", type=float, nargs=2, default=(0.9, 0.95))

    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--train_scale_min", type=float, default=0.2)
    parser.add_argument("--train_scale_max", type=float, default=1.0)
    parser.add_argument("--train_ratio_min", type=float, default=0.75)
    parser.add_argument("--train_ratio_max", type=float, default=1.3333333333333333)
    parser.add_argument("--hflip", type=float, default=0.5)
    parser.add_argument("--color_jitter", type=float, default=0.0)
    parser.add_argument("--interpolation", type=str, default="bicubic")

    parser.add_argument(
        "--gdrive-upload",
        action=argparse.BooleanOptionalAction,
        default=bool(
            os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON")
            and os.environ.get("GDRIVE_FOLDER_ID")
        ),
    )
    parser.add_argument("--gdrive-service-account-json", type=str, default=os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON", ""))
    parser.add_argument("--gdrive-folder-id", type=str, default=os.environ.get("GDRIVE_FOLDER_ID", ""))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad_accum_steps must be positive")
    dist_ctx = init_distributed_mode()
    device = get_device(dist_ctx)
    setup_seed(args.seed)

    experiment_name = args.experiment or f"mr{str(args.mask_ratio).replace('.', '')}"
    output_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "freq-mae-ssl.pth")
    best_model_path = os.path.join(output_dir, "best_freq-mae-ssl.pth")
    last_checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    last_checkpoint_norm_path = os.path.join(checkpoint_dir, "last_checkpoint_norm_pix.pth")

    writer = SummaryWriter(output_dir) if is_main_process(dist_ctx) else None
    gdrive_uploader = (
        init_gdrive_uploader(
            enabled=args.gdrive_upload,
            service_account_json=args.gdrive_service_account_json,
            folder_id=args.gdrive_folder_id,
        )
        if is_main_process(dist_ctx)
        else None
    )

    train_loader, val_loader = build_loaders(args, dist_ctx)
    mae_template = build_timm_mae_template(args) if args.timm_model_name else None
    model = build_model(args, device, dist_ctx, encoder_template=mae_template)
    maybe_load_external_pretrained_weights(model, args, device, mae_template=mae_template)
    verify_timm_template_transfer(model, mae_template, dist_ctx)
    task = FrequencyMAESelfSupervisedTask(model, mask_ratio=args.mask_ratio)
    effective_batch_size = args.batch_size * dist_ctx.world_size * args.grad_accum_steps
    actual_lr = args.lr if args.lr > 0.0 else args.base_lr * effective_batch_size / 256.0

    optimizer = create_optimizer_v2(
        get_core_model(model),
        opt=args.opt,
        lr=actual_lr,
        weight_decay=args.weight_decay,
        filter_bias_and_bn=False,
        betas=tuple(args.opt_betas),
    )
    scheduler, _ = create_scheduler_v2(
        optimizer,
        sched=args.sched,
        num_epochs=args.epochs,
        min_lr=args.min_lr,
        warmup_lr=args.warmup_lr,
        warmup_epochs=args.warmup_epochs,
        step_on_epochs=False,
        updates_per_epoch=len(train_loader),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    start_epoch = 0
    best_val_psnr = float("-inf")
    best_val_mse = float("inf")
    if args.resume:
        start_epoch, best_val_psnr, best_val_mse = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if scaler.is_enabled() else None,
            device=device,
        )

    if is_main_process(dist_ctx):
        print(f"Using device: {device}")
        print(f"Distributed: {dist_ctx.distributed}")
        print(f"Mask ratio: {args.mask_ratio}")
        print(f"Pixel-normalized loss: {args.norm_pix_loss}")
        print(f"Gradient accumulation steps: {args.grad_accum_steps}")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Base learning rate: {args.base_lr}")
        print(f"Actual learning rate: {actual_lr}")
        print(f"Output dir: {output_dir}")
        print(f"Timm template enabled: {bool(args.timm_model_name)}")
        print(f"Timm model name: {args.timm_model_name or 'none'}")
        print(f"Timm pretrained requested: {bool(args.timm_pretrained)}")
        print(f"External pretrained checkpoint: {args.pretrained_ckpt or 'none'}")
        print(f"Timm template class: {type(mae_template).__name__ if mae_template is not None else 'none'}")
        print(f"Timm template has decoder modules: {bool(mae_template is not None and extract_timm_mae_decoder_spec(mae_template) is not None)}")
        print(f"Decoder initialized from timm template: {bool(mae_template is not None and extract_timm_mae_decoder_spec(mae_template) is not None)}")
        print(f"Model decoder structure source: {'timm-template' if get_core_model(model)._decoder_initialized_from_template else 'local-fallback'}")
        print("Input projection initialization: custom (not patch_embed-transferred)")

    try:
        for epoch in range(start_epoch, args.epochs):
            train_loss = train_one_epoch(
                task=task,
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler if scaler.is_enabled() else None,
                device=device,
                dist_ctx=dist_ctx,
                epoch=epoch,
                grad_accum_steps=args.grad_accum_steps,
            )
            val_psnr, val_mse = validate(
                task=task,
                model=model,
                loader=val_loader,
                device=device,
                dist_ctx=dist_ctx,
            )
            val_loss = val_mse

            if is_main_process(dist_ctx):
                if writer is not None:
                    writer.add_scalar("loss/train", train_loss, epoch)
                    writer.add_scalar("loss/val_mse", val_mse, epoch)
                    writer.add_scalar("psnr/val", val_psnr, epoch)

                print(
                    f"Epoch {epoch + 1:03d}/{args.epochs} | "
                    f"train loss {train_loss:.6f} | "
                    f"val mse {val_mse:.6f} | "
                    f"val psnr {val_psnr:.2f}"
                )

                should_visualize = (
                    args.visualize_every > 0
                    and ((epoch + 1) == 1 or ((epoch + 1) % args.visualize_every == 0) or ((epoch + 1) == args.epochs))
                )
                if should_visualize:
                    save_epoch_visualization(
                        task=task,
                        model=model,
                        loader=val_loader,
                        device=device,
                        dist_ctx=dist_ctx,
                        output_dir=output_dir,
                        epoch=epoch,
                        count=args.visualize_count,
                        writer=writer,
                    )

                save_model = get_core_model(model)
                is_best = val_psnr > best_val_psnr or (
                    math.isclose(val_psnr, best_val_psnr) and val_mse < best_val_mse
                )
                if is_best:
                    best_val_psnr = val_psnr
                    best_val_mse = val_mse
                    log_event(dist_ctx, f"epoch {epoch + 1}: best model save start")
                    torch.save(save_model.state_dict(), best_model_path)
                    log_event(dist_ctx, f"epoch {epoch + 1}: best model save done")

                should_save_latest = ((epoch + 1) % 10 == 0) or (epoch + 1 == args.epochs)
                if should_save_latest:
                    log_event(dist_ctx, f"epoch {epoch + 1}: latest checkpoint save start")
                    save_checkpoint(
                        last_checkpoint_norm_path,
                        epoch=epoch + 1,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler if scaler.is_enabled() else None,
                        args=args,
                        best_val_psnr=best_val_psnr,
                        best_val_mse=best_val_mse,
                    )
                    log_event(dist_ctx, f"epoch {epoch + 1}: latest checkpoint save done")
            log_event(dist_ctx, f"epoch {epoch + 1}: barrier start", all_ranks=True)
            maybe_barrier(dist_ctx)
            log_event(dist_ctx, f"epoch {epoch + 1}: barrier done", all_ranks=True)
    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed(dist_ctx)
        if is_main_process(dist_ctx):
            save_model = get_core_model(model)
            log_event(dist_ctx, "final model save start")
            torch.save(save_model.state_dict(), model_path)
            log_event(dist_ctx, "final model save done")
            log_event(dist_ctx, "artifact upload start")
            upload_artifacts_to_gdrive(
                gdrive_uploader,
                [
                    model_path,
                    best_model_path,
                    last_checkpoint_norm_path,
                    last_checkpoint_path,
                ],
            )
            log_event(dist_ctx, "artifact upload done")


if __name__ == "__main__":
    main()
