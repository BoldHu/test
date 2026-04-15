"""Evaluate a trained FreqMAE checkpoint on a fixed mask.

This script loads a compatible FreqMAE checkpoint,
runs reconstruction on ImageNet or CIFAR-10 images with a deterministic visible patch
mask, and reports image-level and self-supervised reconstruction metrics.

Metrics reported:
- PSNR on the final reconstructed image
- SSIM on the final reconstructed image
- MSE / MAE / NMSE on the final reconstructed image
- Masked-patch MSE in the frequency-patch reconstruction space
- Visible-patch MSE in the frequency-patch reconstruction space
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Tuple

import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset_utils import build_eval_transform, build_supervised_dataset
from freq_mae_model_v2 import FreqMAE_ImgShape
from utils import setup_seed


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_checkpoint_payload(checkpoint_path: str, device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        saved_args = payload.get("args", {})
    elif isinstance(payload, dict):
        state_dict = payload
        saved_args = payload.get("args", {})
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(payload)!r}")

    state_dict = strip_module_prefix(state_dict)
    return state_dict, saved_args if isinstance(saved_args, dict) else {}


def load_checkpoint(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys: {missing}")
    if unexpected:
        print(f"[warn] Unexpected keys: {unexpected}")


def build_fixed_visible_indices(
    num_patches: int,
    num_visible: int,
    strategy: str,
    seed: int,
) -> torch.Tensor:
    if num_visible <= 0:
        raise ValueError("num_visible must be positive")
    if num_visible > num_patches:
        raise ValueError("num_visible cannot exceed num_patches")

    if strategy == "random":
        generator = torch.Generator().manual_seed(seed)
        visible = torch.randperm(num_patches, generator=generator)[:num_visible]
        return visible.sort().values

    side = int(math.sqrt(num_patches))
    if side * side != num_patches:
        raise ValueError(f"num_patches must be a perfect square for strategy={strategy!r}")

    if strategy == "top_left":
        return torch.arange(num_visible)

    if strategy == "center":
        coords = torch.stack(torch.meshgrid(
            torch.arange(side, dtype=torch.float32),
            torch.arange(side, dtype=torch.float32),
            indexing="ij",
        ), dim=-1).reshape(-1, 2)
        center = (side - 1) / 2.0
        distances = torch.sum((coords - center) ** 2, dim=-1)
        return torch.argsort(distances)[:num_visible].sort().values

    raise ValueError(f"Unsupported fixed mask strategy: {strategy}")


def psnr_from_mse(mse: torch.Tensor, max_val: float = 2.0) -> torch.Tensor:
    return 20.0 * torch.log10(torch.tensor(max_val, device=mse.device, dtype=mse.dtype) / torch.sqrt(mse.clamp_min(1e-12)))


def batch_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 2.0) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    return psnr_from_mse(mse, max_val=max_val).mean()


def gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gauss_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = gauss_1d[:, None] @ gauss_1d[None, :]
    return gauss_2d.expand(channels, 1, window_size, window_size).contiguous()


def batch_ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    channels = pred.shape[1]
    window = gaussian_window(window_size, sigma, channels, pred.device, pred.dtype)
    padding = window_size // 2

    mu_x = F.conv2d(pred, window, padding=padding, groups=channels)
    mu_y = F.conv2d(target, window, padding=padding, groups=channels)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_x_sq
    sigma_y = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / denominator.clamp_min(1e-12)
    return ssim_map.mean(dim=(1, 2, 3)).mean()


def run_fixed_mask_forward(
    model: FreqMAE_ImgShape,
    x: torch.Tensor,
    visible_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out = model(
        x,
        training=False,
        fixed_visible_idx=visible_idx,
        return_patches=True,
        return_eval_dict=True,
    )
    pred_img = out["pred_img"]
    fused_img = out["fused_img"]
    mask_bool = out["mask_bool"]
    pred_patches_pixel = out["pred_patches_pixel"]
    target_patches = out["target_patches"]
    masked_mse_pixel = out["masked_mse_pixel"]

    mse_per_patch = F.mse_loss(pred_patches_pixel, target_patches, reduction="none").mean(dim=-1)
    masked_weight = mask_bool.float()
    visible_weight = (~mask_bool).float()
    masked_patch_mse = (mse_per_patch * masked_weight).sum(dim=1) / masked_weight.sum(dim=1).clamp_min(1.0)
    visible_patch_mse = (mse_per_patch * visible_weight).sum(dim=1) / visible_weight.sum(dim=1).clamp_min(1.0)

    return pred_img, fused_img, masked_patch_mse, visible_patch_mse, mask_bool


def save_visualization(
    save_path: str,
    inputs: torch.Tensor,
    reconstructions: torch.Tensor,
    count: int,
) -> None:
    import matplotlib.pyplot as plt

    count = min(count, inputs.size(0))
    fig, axes = plt.subplots(3, count, figsize=(4 * count, 9), dpi=150)
    for idx in range(count):
        inp = inputs[idx].detach().cpu().permute(1, 2, 0)
        rec = reconstructions[idx].detach().cpu().permute(1, 2, 0)
        err = (rec - inp).abs().mean(dim=-1)

        axes[0, idx].imshow(((inp + 1.0) / 2.0).clamp(0.0, 1.0))
        axes[0, idx].set_title(f"Input {idx}")
        axes[1, idx].imshow(((rec + 1.0) / 2.0).clamp(0.0, 1.0))
        axes[1, idx].set_title(f"Recon {idx}")
        axes[2, idx].imshow(err, cmap="magma")
        axes[2, idx].set_title(f"Abs Err {idx}")
        for row in range(3):
            axes[row, idx].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FreqMAE checkpoint on a fixed patch mask")
    parser.add_argument("--model_path", type=str, default="freq-mae-ssl.pth")
    parser.add_argument("--timm_model_name", type=str, default="")
    parser.add_argument("--dataset", type=str, choices=["imagenet", "cifar10"], default="imagenet")
    parser.add_argument("--data_root", type=str, default="data/imagenet")
    parser.add_argument("--train_dir", type=str, default="")
    parser.add_argument("--val_dir", type=str, default="")
    parser.add_argument("--eval_split", type=str, choices=["train", "val"], default="val")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", type=int, default=0, help="Optional limit on the number of samples to evaluate")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--encoder_layers", type=int, default=12)
    parser.add_argument("--decoder_emb_dim", type=int, default=512)
    parser.add_argument("--decoder_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--decoder_num_heads", type=int, default=16)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--fixed_mask_strategy", type=str, default="random", choices=["random", "top_left", "center"])
    parser.add_argument("--fixed_mask_seed", type=int, default=1234)
    parser.add_argument("--summary_path", type=str, default="freq_mae_fixed_mask_metrics.json")
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--vis_path", type=str, default="freq_mae_fixed_mask_reconstruction.png")
    parser.add_argument("--vis_count", type=int, default=8)
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    state_dict, saved_args = load_checkpoint_payload(args.model_path, device)
    if saved_args:
        args.image_size = int(saved_args.get("image_size", args.image_size))
        args.block_size = int(saved_args.get("block_size", args.block_size))
        args.patch_size = int(saved_args.get("patch_size", args.patch_size))
        args.emb_dim = int(saved_args.get("emb_dim", args.emb_dim))
        args.encoder_layers = int(saved_args.get("encoder_layers", saved_args.get("encoder_layer", args.encoder_layers)))
        args.decoder_emb_dim = int(saved_args.get("decoder_emb_dim", args.decoder_emb_dim))
        args.decoder_layers = int(saved_args.get("decoder_layers", saved_args.get("decoder_layer", args.decoder_layers)))
        args.num_heads = int(saved_args.get("num_heads", saved_args.get("encoder_head", args.num_heads)))
        args.decoder_num_heads = int(saved_args.get("decoder_num_heads", saved_args.get("decoder_head", args.decoder_num_heads)))
        args.mask_ratio = float(saved_args.get("mask_ratio", args.mask_ratio))
        args.timm_model_name = str(saved_args.get("timm_model_name") or args.timm_model_name or "")

    eval_transform = build_eval_transform(dataset=args.dataset, image_size=args.image_size)
    test_dataset = build_supervised_dataset(
        dataset=args.dataset,
        data_root=args.data_root,
        split=args.eval_split,
        transform=eval_transform,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        download=(args.dataset == "cifar10"),
    )
    if args.subset and args.subset > 0:
        indices = list(range(min(args.subset, len(test_dataset))))
        test_dataset = Subset(test_dataset, indices)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    encoder_template = None
    if args.timm_model_name:
        encoder_template = timm.create_model(args.timm_model_name, pretrained=False)
        patch_embed = getattr(encoder_template, "patch_embed", None)
        if patch_embed is not None:
            patch_size = patch_embed.patch_size[0] if isinstance(patch_embed.patch_size, tuple) else patch_embed.patch_size
            num_patches = getattr(patch_embed, "num_patches", None)
            expected_num_patches = (args.image_size // args.patch_size) ** 2
            if int(patch_size) != args.patch_size:
                raise ValueError(
                    f"timm patch size mismatch: model {args.timm_model_name} uses {patch_size}, "
                    f"but checkpoint expects patch_size={args.patch_size}"
                )
            if num_patches is not None and int(num_patches) != expected_num_patches:
                raise ValueError(
                    f"timm patch count mismatch: model {args.timm_model_name} uses {num_patches}, "
                    f"but checkpoint expects {expected_num_patches}"
                )
        decoder_pred = getattr(encoder_template, "decoder_pred", None)
        if decoder_pred is not None:
            expected_patch_dim = 3 * (args.patch_size ** 2)
            if int(decoder_pred.out_features) != expected_patch_dim:
                raise ValueError(
                    f"timm decoder patch dim mismatch: model {args.timm_model_name} predicts "
                    f"{decoder_pred.out_features}, expected {expected_patch_dim}"
                )

    model = FreqMAE_ImgShape(
        image_size=args.image_size,
        block_size=args.block_size,
        channels=3,
        patch_size=args.patch_size,
        patch_dim=args.patch_size * args.patch_size * 3,
        emb_dim=args.emb_dim,
        encoder_layers=args.encoder_layers,
        decoder_emb_dim=args.decoder_emb_dim,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        decoder_num_heads=args.decoder_num_heads,
        mask_ratio=args.mask_ratio,
        encoder_template=encoder_template,
    ).to(device)
    load_checkpoint(model, state_dict)
    model.eval()

    num_patches = (args.image_size // args.patch_size) ** 2
    num_visible = int(num_patches * (1.0 - args.mask_ratio))
    visible_idx = build_fixed_visible_indices(
        num_patches=num_patches,
        num_visible=num_visible,
        strategy=args.fixed_mask_strategy,
        seed=args.fixed_mask_seed,
    ).to(device)

    total_images = 0
    sum_psnr = 0.0
    sum_ssim = 0.0
    sum_mse = 0.0
    sum_mae = 0.0
    sum_nmse = 0.0
    sum_masked_patch_mse = 0.0
    sum_visible_patch_mse = 0.0

    first_inputs = None
    first_recons = None

    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Evaluating", mininterval=0.3):
            inputs = inputs.to(device, non_blocking=True)
            recon_image, _, masked_patch_mse, visible_patch_mse, _ = run_fixed_mask_forward(
                model=model,
                x=inputs,
                visible_idx=visible_idx,
            )

            batch_size = inputs.size(0)
            batch_mse = F.mse_loss(recon_image, inputs, reduction="none").mean(dim=(1, 2, 3))
            batch_mae = F.l1_loss(recon_image, inputs, reduction="none").mean(dim=(1, 2, 3))
            batch_nmse = batch_mse / inputs.pow(2).mean(dim=(1, 2, 3)).clamp_min(1e-12)

            sum_psnr += batch_psnr(recon_image, inputs).item() * batch_size
            sum_ssim += batch_ssim(recon_image, inputs).item() * batch_size
            sum_mse += batch_mse.sum().item()
            sum_mae += batch_mae.sum().item()
            sum_nmse += batch_nmse.sum().item()
            sum_masked_patch_mse += masked_patch_mse.sum().item()
            sum_visible_patch_mse += visible_patch_mse.sum().item()
            total_images += batch_size

            if args.save_vis and first_inputs is None:
                first_inputs = inputs[: args.vis_count].detach().cpu()
                first_recons = recon_image[: args.vis_count].detach().cpu()

    metrics = {
        "checkpoint": args.model_path,
        "dataset": args.dataset,
        "eval_split": args.eval_split,
        "num_images": total_images,
        "mask_ratio": args.mask_ratio,
        "fixed_mask_strategy": args.fixed_mask_strategy,
        "fixed_mask_seed": args.fixed_mask_seed,
        "psnr_db": sum_psnr / total_images,
        "ssim": sum_ssim / total_images,
        "mse": sum_mse / total_images,
        "mae": sum_mae / total_images,
        "nmse": sum_nmse / total_images,
        "masked_patch_mse": sum_masked_patch_mse / total_images,
        "visible_patch_mse": sum_visible_patch_mse / total_images,
        "num_patches": num_patches,
        "num_visible": num_visible,
    }

    print("\nEvaluation summary")
    for key in [
        "psnr_db",
        "ssim",
        "mse",
        "mae",
        "nmse",
        "masked_patch_mse",
        "visible_patch_mse",
    ]:
        print(f"{key}: {metrics[key]:.6f}")

    os.makedirs(os.path.dirname(args.summary_path) or ".", exist_ok=True)
    with open(args.summary_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {args.summary_path}")

    if args.save_vis and first_inputs is not None and first_recons is not None:
        os.makedirs(os.path.dirname(args.vis_path) or ".", exist_ok=True)
        save_visualization(args.vis_path, first_inputs, first_recons, args.vis_count)
        print(f"Saved visualization to {args.vis_path}")


if __name__ == "__main__":
    main()
