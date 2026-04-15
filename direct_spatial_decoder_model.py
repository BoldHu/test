from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from freq_mae_model_v2 import FreqMAE_ImgShape, _build_ids_restore


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def freeze_parameter(parameter: torch.nn.Parameter) -> None:
    parameter.requires_grad = False


def set_parameter_trainable(parameter: torch.nn.Parameter, trainable: bool) -> None:
    parameter.requires_grad = trainable


class DirectSpatialDecoderMAE(nn.Module):
    def __init__(
        self,
        freq_mae: FreqMAE_ImgShape,
        *,
        fixed_visible_idx: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.freq_mae = freq_mae
        self.fixed_visible_idx = fixed_visible_idx

        self.image_size = self.freq_mae.image_size
        self.patch_size = self.freq_mae.patch_size
        self.num_patches = self.freq_mae.num_patches
        self.channels = self.freq_mae.channels
        self.num_prefix_tokens = self.freq_mae.num_prefix_tokens

        self._freeze_encoder()
        self._set_decoder_trainable(True)

    def _freeze_encoder(self) -> None:
        freeze_module(self.freq_mae.input_proj)
        freeze_module(self.freq_mae.encoder_blocks)
        freeze_module(self.freq_mae.encoder_norm)
        freeze_parameter(self.freq_mae.cls_token)
        freeze_parameter(self.freq_mae.enc_pos_embedding)
        if hasattr(self.freq_mae, "mask_logits_2d"):
            freeze_parameter(self.freq_mae.mask_logits_2d)

    def _set_decoder_trainable(self, trainable: bool) -> None:
        set_module_trainable(self.freq_mae.decoder_embed, trainable)
        set_module_trainable(self.freq_mae.decoder_blocks, trainable)
        set_module_trainable(self.freq_mae.decoder_norm, trainable)
        set_module_trainable(self.freq_mae.output_proj, trainable)
        set_parameter_trainable(self.freq_mae.mask_token, trainable)
        set_parameter_trainable(self.freq_mae.dec_pos_embedding, trainable)

    def set_fixed_visible_idx(self, fixed_visible_idx: torch.Tensor) -> None:
        self.fixed_visible_idx = fixed_visible_idx

    def _normalize_visible_idx(self, x: torch.Tensor, visible_idx: Optional[torch.Tensor]) -> torch.Tensor:
        resolved = visible_idx if visible_idx is not None else self.fixed_visible_idx
        if resolved is None:
            raise ValueError("visible_idx is required for DirectSpatialDecoderMAE")

        resolved = resolved.to(device=x.device, dtype=torch.long)
        if resolved.dim() == 1:
            resolved = resolved.unsqueeze(0).expand(x.shape[0], -1)
        elif resolved.dim() == 2:
            if resolved.shape[0] != x.shape[0]:
                raise ValueError(
                    f"visible_idx batch size mismatch: got {resolved.shape[0]}, expected {x.shape[0]}"
                )
        else:
            raise ValueError("visible_idx must be 1D or 2D")
        return resolved

    def forward(
        self,
        x: torch.Tensor,
        *,
        visible_idx: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        visible_idx = self._normalize_visible_idx(x, visible_idx)
        batch_size = x.shape[0]

        with torch.no_grad():
            patches, _, _ = self.freq_mae.patchify(x)
            patches_flat = rearrange(patches, "b d n -> b n d")
            target_patches = rearrange(
                F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size),
                "b d n -> b n d",
            )

            embedded = self.freq_mae.input_proj(patches_flat)
            embedded = embedded + self.freq_mae.enc_pos_embedding[:, self.num_prefix_tokens :, :]

            expanded_visible_idx = visible_idx.unsqueeze(-1).expand(-1, -1, embedded.size(-1))
            visible_patches = torch.gather(embedded, 1, expanded_visible_idx)

            cls_tokens = (
                self.freq_mae.cls_token + self.freq_mae.enc_pos_embedding[:, : self.num_prefix_tokens, :]
            ).expand(batch_size, -1, -1)
            encoder_input = torch.cat([cls_tokens, visible_patches], dim=1)
            encoded = self.freq_mae.encoder_blocks(encoder_input)
            encoded = self.freq_mae.encoder_norm(encoded)

        ids_restore, mask_bool = _build_ids_restore(visible_idx, self.num_patches)
        decoded = self.freq_mae.decoder_embed(encoded)
        num_masked = self.num_patches - visible_idx.shape[1]
        mask_tokens = repeat(self.freq_mae.mask_token, "1 1 d -> b n d", b=batch_size, n=num_masked)
        decoded_patches = torch.cat([decoded[:, self.num_prefix_tokens :, :], mask_tokens], dim=1)
        gather_index = ids_restore.unsqueeze(-1).expand(-1, -1, decoded.size(-1))
        decoded_patches = torch.gather(decoded_patches, 1, gather_index)
        full_seq = torch.cat([decoded[:, : self.num_prefix_tokens, :], decoded_patches], dim=1)
        full_seq = full_seq + self.freq_mae.dec_pos_embedding

        reconstructed = self.freq_mae.decoder_blocks(full_seq)
        reconstructed = self.freq_mae.decoder_norm(reconstructed)
        spatial_recon_patches = self.freq_mae.output_proj(reconstructed[:, self.num_prefix_tokens :, :])
        spatial_recon_unfold = rearrange(spatial_recon_patches, "b n d -> b d n")
        spatial_recon_img = self.freq_mae.unpatchify(spatial_recon_unfold)

        if return_aux:
            return spatial_recon_img, x, spatial_recon_patches, target_patches, visible_idx, mask_bool
        return spatial_recon_img, x
