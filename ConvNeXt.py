import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from freq_mae_model_v2 import FreqMAE_ImgShape, build_fixed_visible_idx


# ==========================================
# 1. 核心组件：现代化的 ConvNeXt Block
# ==========================================
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


# ==========================================
# 2. 图像域精修 CNN (Image Refiner)
# ==========================================
class ImageRefinerCNN(nn.Module):
    def __init__(self, in_channels: int = 3, dim: int = 96, num_blocks: int = 5):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[ConvNeXtBlock(dim=dim) for _ in range(num_blocks)])
        self.head = nn.Conv2d(dim, in_channels, kernel_size=3, padding=1)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return residual + x


# ==========================================
# 3. 工具函数：checkpoint 读取与冻结策略
# ==========================================
def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if not any(key.startswith('module.') for key in state_dict):
        return state_dict
    return {key.removeprefix('module.'): value for key, value in state_dict.items()}


def load_freq_mae_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and 'state_dict' in payload:
        state_dict = payload['state_dict']
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise TypeError(f'Unsupported checkpoint type: {type(payload)!r}')

    state_dict = strip_module_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'[warn] Missing keys: {missing}')
    if unexpected:
        print(f'[warn] Unexpected keys: {unexpected}')


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def freeze_parameter(parameter: torch.nn.Parameter) -> None:
    parameter.requires_grad = False


def freeze_freq_mae_encoder(freq_mae: FreqMAE_ImgShape) -> None:
    freeze_module(freq_mae.input_proj)
    freeze_module(freq_mae.encoder_blocks)
    freeze_module(freq_mae.encoder_norm)
    freeze_parameter(freq_mae.enc_pos_embedding)


def freeze_freq_mae_mask_branch(freq_mae: FreqMAE_ImgShape) -> None:
    if hasattr(freq_mae, 'mask_logits_2d'):
        freeze_parameter(freq_mae.mask_logits_2d)


def set_freq_mae_decoder_trainable(freq_mae: FreqMAE_ImgShape, trainable: bool = True) -> None:
    for parameter in [freq_mae.mask_token, freq_mae.dec_pos_embedding]:
        parameter.requires_grad = trainable

    for module in [freq_mae.decoder_embed, freq_mae.decoder_blocks, freq_mae.decoder_norm, freq_mae.output_proj]:
        for param in module.parameters():
            param.requires_grad = trainable


# ==========================================
# 4. 终极组装：双域级联网络
# ==========================================
class DualDomainCascadedModel(nn.Module):
    def __init__(
        self,
        freq_mae_model: Optional[FreqMAE_ImgShape] = None,
        freq_mae_ckpt_path: Optional[str] = None,
        freq_mae_kwargs: Optional[dict] = None,
        cnn_dim: int = 96,
        cnn_blocks: int = 5,
        fixed_visible_idx: Optional[torch.Tensor] = None,
        freeze_encoder: bool = True,
        fine_tune_decoder: bool = True,
        freeze_mask_branch: bool = True,
    ):
        super().__init__()

        if freq_mae_model is None:
            if freq_mae_kwargs is None:
                raise ValueError('freq_mae_kwargs is required when freq_mae_model is not provided')
            freq_mae_model = FreqMAE_ImgShape(**freq_mae_kwargs)

        self.freq_mae = freq_mae_model
        self.fixed_visible_idx = fixed_visible_idx

        if freq_mae_ckpt_path is not None:
            load_freq_mae_checkpoint(self.freq_mae, freq_mae_ckpt_path, device=torch.device('cpu'))

        if freeze_encoder:
            freeze_freq_mae_encoder(self.freq_mae)
        if freeze_mask_branch:
            freeze_freq_mae_mask_branch(self.freq_mae)
        set_freq_mae_decoder_trainable(self.freq_mae, trainable=fine_tune_decoder)

        self.image_cnn = ImageRefinerCNN(
            in_channels=self.freq_mae.channels,
            dim=cnn_dim,
            num_blocks=cnn_blocks,
        )

    def set_fixed_visible_idx(self, fixed_visible_idx: torch.Tensor) -> None:
        self.fixed_visible_idx = fixed_visible_idx

    def forward(
        self,
        x: torch.Tensor,
        fixed_visible_idx: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        visible_idx = fixed_visible_idx if fixed_visible_idx is not None else self.fixed_visible_idx

        if return_aux:
            coarse_img, visible_idx_out, mask_bool, freq_recon_patches, orig_patches = self.freq_mae(
                x,
                training=False,
                input_is_freq=False,
                fixed_visible_idx=visible_idx,
                return_patches=True,
            )
            refined_img = self.image_cnn(coarse_img)
            return refined_img, coarse_img, freq_recon_patches, orig_patches, visible_idx_out, mask_bool

        coarse_img, _, _ = self.freq_mae(
            x,
            training=False,
            input_is_freq=False,
            fixed_visible_idx=visible_idx,
        )
        refined_img = self.image_cnn(coarse_img)
        return refined_img


if __name__ == '__main__':
    print('=' * 70)
    print('Dual-domain ConvNeXt wrapper')
    print('=' * 70)
    model = DualDomainCascadedModel(
        freq_mae_kwargs=dict(
            image_size=32,
            block_size=16,
            channels=3,
            patch_size=2,
            patch_dim=24,
            emb_dim=768,
            encoder_layers=12,
            decoder_emb_dim=512,
            decoder_layers=8,
            num_heads=12,
            decoder_num_heads=16,
            mask_ratio=0.75,
        ),
        cnn_dim=96,
        cnn_blocks=5,
    )
    print(model)
