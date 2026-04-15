from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset_utils import build_eval_transform, build_supervised_dataset, build_train_transform
from direct_spatial_decoder_model import DirectSpatialDecoderMAE
from freq_mae_model_v2 import FreqMAE_ImgShape, _extract_state_dict, build_fixed_visible_idx
from gdrive_fallback import init_gdrive_uploader, upload_artifacts_to_gdrive
from utils import calculate_psnr, setup_seed


@dataclass
class EpochStats:
    loss: float
    psnr: float


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_checkpoint_payload(path: str, device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    payload = torch.load(path, map_location=device)
    state_dict = strip_module_prefix(_extract_state_dict(payload))
    saved_args = payload.get("args", {}) if isinstance(payload, dict) else {}
    return state_dict, saved_args if isinstance(saved_args, dict) else {}


def load_resume_checkpoint(
    path: str,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
) -> Tuple[int, float]:
    payload = torch.load(path, map_location=device)
    state_dict = strip_module_prefix(payload["model_state_dict"])
    get_core_model(model).load_state_dict(state_dict, strict=False)
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    return int(payload.get("epoch", 0)), float(payload.get("best_val_loss", math.inf))


def build_freq_mae_from_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[FreqMAE_ImgShape, Dict[str, object]]:
    state_dict, saved_args = load_checkpoint_payload(checkpoint_path, device)
    if not saved_args:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain saved args")

    image_size = int(saved_args["image_size"])
    block_size = int(saved_args["block_size"])
    patch_size = int(saved_args["patch_size"])
    emb_dim = int(saved_args["emb_dim"])
    encoder_layers = int(saved_args.get("encoder_layers", saved_args.get("encoder_layer", 12)))
    decoder_emb_dim = int(saved_args.get("decoder_emb_dim", 512))
    decoder_layers = int(saved_args.get("decoder_layers", saved_args.get("decoder_layer", 8)))
    num_heads = int(saved_args.get("num_heads", saved_args.get("encoder_head", 12)))
    decoder_num_heads = int(saved_args.get("decoder_num_heads", saved_args.get("decoder_head", 16)))
    mask_ratio = float(saved_args["mask_ratio"])
    timm_model_name = str(saved_args.get("timm_model_name") or "")

    encoder_template = timm.create_model(timm_model_name, pretrained=False) if timm_model_name else None
    model = FreqMAE_ImgShape(
        image_size=image_size,
        block_size=block_size,
        channels=3,
        patch_size=patch_size,
        patch_dim=patch_size * patch_size * 3,
        emb_dim=emb_dim,
        encoder_layers=encoder_layers,
        decoder_emb_dim=decoder_emb_dim,
        decoder_layers=decoder_layers,
        num_heads=num_heads,
        decoder_num_heads=decoder_num_heads,
        mask_ratio=mask_ratio,
        encoder_template=encoder_template,
    ).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys while loading freq MAE: {missing}")
    if unexpected:
        print(f"[warn] Unexpected keys while loading freq MAE: {unexpected}")
    return model, saved_args


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    train_transform = build_train_transform(dataset=args.dataset, image_size=args.image_size)
    eval_transform = build_eval_transform(dataset=args.dataset, image_size=args.image_size)
    train_dataset = build_supervised_dataset(
        dataset=args.dataset,
        data_root=args.data_root,
        split="train",
        transform=train_transform,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        download=False,
    )
    val_dataset = build_supervised_dataset(
        dataset=args.dataset,
        data_root=args.data_root,
        split="val",
        transform=eval_transform,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        download=False,
    )

    if args.val_size > 0:
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(train_dataset), generator=generator).tolist()
        val_indices = indices[: args.val_size]
        train_indices = indices[args.val_size :]
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(
            build_supervised_dataset(
                dataset=args.dataset,
                data_root=args.data_root,
                split="train",
                transform=eval_transform,
                train_dir=args.train_dir,
                val_dir=args.val_dir,
                download=False,
            ),
            val_indices,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def get_core_model(model: torch.nn.Module) -> DirectSpatialDecoderMAE:
    return model.module if hasattr(model, "module") else model


def train_one_epoch(
    model: DirectSpatialDecoderMAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
) -> EpochStats:
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    num_batches = 0

    progress = tqdm(loader, desc="Train", leave=False)
    for images, _ in progress:
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is None:
            reconstructed, original = model(images)
            loss = F.l1_loss(reconstructed, original)
            loss.backward()
            optimizer.step()
        else:
            with torch.autocast(device_type="cuda", enabled=True):
                reconstructed, original = model(images)
                loss = F.l1_loss(reconstructed, original)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        batch_psnr = 0.0
        for i in range(images.size(0)):
            batch_psnr += calculate_psnr(reconstructed[i], original[i], data_range=2.0)
        batch_psnr /= max(images.size(0), 1)
        total_psnr += batch_psnr
        num_batches += 1

        progress.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{batch_psnr:.2f}")

    return EpochStats(
        loss=total_loss / max(num_batches, 1),
        psnr=total_psnr / max(num_batches, 1),
    )


@torch.no_grad()
def evaluate(model: DirectSpatialDecoderMAE, loader: DataLoader, device: torch.device) -> EpochStats:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    num_batches = 0

    progress = tqdm(loader, desc="Validate", leave=False)
    for images, _ in progress:
        images = images.to(device, non_blocking=True)
        reconstructed, original = model(images)
        loss = F.mse_loss(reconstructed, original)

        total_loss += loss.item()
        batch_psnr = 0.0
        for i in range(images.size(0)):
            batch_psnr += calculate_psnr(reconstructed[i], original[i], data_range=2.0)
        batch_psnr /= max(images.size(0), 1)
        total_psnr += batch_psnr
        num_batches += 1

        progress.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{batch_psnr:.2f}")

    return EpochStats(
        loss=total_loss / max(num_batches, 1),
        psnr=total_psnr / max(num_batches, 1),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train direct spatial decoder on frozen frequency encoder")
    parser.add_argument("--dataset", type=str, choices=["imagenet", "cifar10"], default="imagenet")
    parser.add_argument("--data_root", type=str, default="data/imagenet")
    parser.add_argument("--train_dir", type=str, default="")
    parser.add_argument("--val_dir", type=str, default="")
    parser.add_argument("--freq_mae_ckpt", type=str, default="best_freq-mae-ssl.pth")
    parser.add_argument("--output_dir", type=str, default="logs/imagenet/direct_spatial_decoder")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask_seed", type=int, default=42)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--decoder_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--amp", action="store_true")
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


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    use_data_parallel = device.type == "cuda" and gpu_count > 1
    print(f"Using device: {device}")
    print(f"Visible GPUs: {gpu_count}")

    freq_mae, saved_args = build_freq_mae_from_checkpoint(args.freq_mae_ckpt, device)
    args.image_size = int(saved_args["image_size"])
    args.block_size = int(saved_args["block_size"])
    args.patch_size = int(saved_args["patch_size"])

    num_patches = (args.image_size // args.patch_size) ** 2
    fixed_visible_idx = build_fixed_visible_idx(
        num_patches=num_patches,
        mask_ratio=args.mask_ratio,
        seed=args.mask_seed,
    )

    model = DirectSpatialDecoderMAE(freq_mae, fixed_visible_idx=fixed_visible_idx).to(device)
    if use_data_parallel:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.decoder_lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    gdrive_uploader = init_gdrive_uploader(
        enabled=args.gdrive_upload,
        service_account_json=args.gdrive_service_account_json,
        folder_id=args.gdrive_folder_id,
    )

    train_loader, val_loader = build_dataloaders(args)
    last_path = os.path.join(args.output_dir, "last_direct_spatial_decoder.pth")
    best_val_loss = math.inf
    best_path = os.path.join(args.output_dir, "best_direct_spatial_decoder.pth")
    start_epoch = 0

    if args.resume:
        start_epoch, best_val_loss = load_resume_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        print(f"Resumed direct spatial decoder from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_stats = evaluate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train loss {train_stats.loss:.5f} | "
            f"train psnr {train_stats.psnr:.2f} | "
            f"val loss {val_stats.loss:.5f} | "
            f"val psnr {val_stats.psnr:.2f}"
        )

        save_model = get_core_model(model)
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": save_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "args": vars(args),
                "best_val_loss": best_val_loss,
            },
            last_path,
        )

        if val_stats.loss < best_val_loss:
            best_val_loss = val_stats.loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": save_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                    "args": vars(args),
                    "best_val_loss": best_val_loss,
                },
                best_path,
            )
            print(f"New best checkpoint saved to {best_path}")

    print(f"Final checkpoint saved to {last_path}")

    upload_artifacts_to_gdrive(gdrive_uploader, [best_path, last_path])


if __name__ == "__main__":
    main()
