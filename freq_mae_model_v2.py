import copy
import math
from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block


def build_fixed_visible_idx(
    num_patches: int,
    mask_ratio: float,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    num_visible = int(num_patches * (1.0 - mask_ratio))
    if num_visible <= 0:
        raise ValueError("mask_ratio is too large: no visible patches remain")

    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    visible_idx = torch.randperm(num_patches, generator=generator)[:num_visible].sort().values
    if device is not None:
        visible_idx = visible_idx.to(device)
    return visible_idx


def _extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(k, str) for k in payload.keys()):
            return payload
    raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def _adapt_pos_embed(pos_embed: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if pos_embed.ndim != 3:
        raise ValueError(f"Expected 3D pos embed tensor, got {pos_embed.shape}")

    if pos_embed.shape[-1] != target_shape[-1]:
        raise ValueError(
            f"Position embedding dim mismatch: source {pos_embed.shape[-1]}, target {target_shape[-1]}"
        )

    source_tokens = pos_embed.shape[1]
    target_tokens = target_shape[1]
    if source_tokens == target_tokens:
        return pos_embed
    if source_tokens == target_tokens + 1:
        return pos_embed[:, 1:, :]
    if source_tokens + 1 == target_tokens:
        prefix = pos_embed.new_zeros((pos_embed.shape[0], 1, pos_embed.shape[-1]))
        return torch.cat([prefix, pos_embed], dim=1)
    raise ValueError(
        f"Unsupported token count for position embedding: source {pos_embed.shape}, target {tuple(target_shape)}"
    )


def _get_1d_sincos_pos_embed(embed_dim: int, positions: torch.Tensor) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError(f"1D sin-cos embedding dim must be even, got {embed_dim}")
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))
    out = positions.reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, *, cls_token: bool) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError(f"2D sin-cos embedding dim must be even, got {embed_dim}")

    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    emb_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid[0].reshape(-1))
    emb_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid[1].reshape(-1))
    pos_embed = torch.cat([emb_h, emb_w], dim=1)
    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed], dim=0)
    return pos_embed.unsqueeze(0)


def _dht2(x: torch.Tensor) -> torch.Tensor:
    spectrum = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    return spectrum.real - spectrum.imag


def _build_ids_restore(visible_idx: torch.Tensor, num_patches: int) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_visible = visible_idx.shape
    device = visible_idx.device

    visible_mask = torch.zeros(batch_size, num_patches, device=device, dtype=torch.bool)
    visible_mask.scatter_(1, visible_idx, True)

    all_idx = torch.arange(num_patches, device=device).unsqueeze(0).expand(batch_size, -1)
    masked_idx = all_idx.masked_select(~visible_mask).view(batch_size, num_patches - num_visible)
    ids_shuffle = torch.cat([visible_idx, masked_idx], dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    return ids_restore, ~visible_mask


def extract_timm_vit_backbone_spec(model: nn.Module) -> Dict[str, int]:
    if not hasattr(model, "blocks") or len(model.blocks) == 0:
        raise ValueError("Expected a timm VisionTransformer-like model with encoder blocks")
    if not hasattr(model, "embed_dim"):
        raise ValueError("Expected timm model to expose embed_dim")

    first_block = model.blocks[0]
    attn = getattr(first_block, "attn", None)
    num_heads = getattr(attn, "num_heads", None)
    if num_heads is None:
        raise ValueError("Unable to infer attention head count from timm encoder block")

    return {
        "emb_dim": int(model.embed_dim),
        "encoder_layers": int(len(model.blocks)),
        "num_heads": int(num_heads),
    }


def extract_timm_mae_decoder_spec(model: nn.Module) -> Optional[Dict[str, int]]:
    decoder_blocks = getattr(model, "decoder_blocks", None)
    if decoder_blocks is None or len(decoder_blocks) == 0:
        return None

    decoder_embed = getattr(model, "decoder_embed", None)
    decoder_pred = getattr(model, "decoder_pred", None)
    decoder_attn = getattr(decoder_blocks[0], "attn", None)
    decoder_heads = getattr(decoder_attn, "num_heads", None)
    if decoder_embed is None or decoder_pred is None or decoder_heads is None:
        return None

    return {
        "decoder_emb_dim": int(decoder_embed.out_features),
        "decoder_layers": int(len(decoder_blocks)),
        "decoder_num_heads": int(decoder_heads),
        "decoder_patch_dim": int(decoder_pred.out_features),
    }


def load_partial_vit_mae_weights(
    model: "FreqMAE_ImgShape",
    state_dict: Dict[str, torch.Tensor],
    *,
    verbose: bool = True,
) -> Dict[str, list[str]]:
    state_dict = _strip_module_prefix(state_dict)

    mapped: Dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for key, value in state_dict.items():
        if key.startswith("patch_embed.") or key.startswith("head."):
            skipped.append(key)
            continue
        if key == "cls_token":
            mapped[key] = value
            continue
        if key.startswith("blocks."):
            mapped[f"encoder_blocks.{key[len('blocks.') :]}"] = value
            continue
        if key.startswith("norm."):
            mapped[f"encoder_norm.{key[len('norm.') :]}"] = value
            continue
        if key == "pos_embed":
            mapped["enc_pos_embedding"] = _adapt_pos_embed(value, model.enc_pos_embedding.shape)
            continue
        if key.startswith("decoder_embed."):
            mapped[key] = value
            continue
        if key == "mask_token":
            mapped[key] = value
            continue
        if key == "decoder_pos_embed":
            mapped["dec_pos_embedding"] = _adapt_pos_embed(value, model.dec_pos_embedding.shape)
            continue
        if key.startswith("decoder_blocks."):
            mapped[key] = value
            continue
        if key.startswith("decoder_norm."):
            mapped[key] = value
            continue
        if key.startswith("decoder_pred."):
            mapped[f"output_proj.{key[len('decoder_pred.') :]}"] = value
            continue
        if key.startswith("output_proj."):
            mapped[key] = value
            continue
        skipped.append(key)

    current_state = model.state_dict()
    loadable: Dict[str, torch.Tensor] = {}
    shape_skipped: list[str] = []
    for key, value in mapped.items():
        if key not in current_state:
            skipped.append(key)
            continue
        if current_state[key].shape != value.shape:
            shape_skipped.append(f"{key}: source {tuple(value.shape)} target {tuple(current_state[key].shape)}")
            continue
        loadable[key] = value

    missing, unexpected = model.load_state_dict(loadable, strict=False)

    if verbose:
        print(f"[pretrained] loaded {len(loadable)} tensors into FreqMAE")
        if shape_skipped:
            print("[pretrained] skipped shape-mismatched tensors:")
            for item in shape_skipped[:20]:
                print(f"  - {item}")
            if len(shape_skipped) > 20:
                print(f"  ... and {len(shape_skipped) - 20} more")
        if unexpected:
            print(f"[pretrained] unexpected keys after load: {unexpected}")
        if missing:
            preview = missing[:20]
            print(f"[pretrained] missing model keys after partial load: {preview}")
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more")

    return {
        "loaded": sorted(loadable.keys()),
        "skipped": sorted(skipped),
        "shape_skipped": shape_skipped,
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


# ==========================================
# 辅助模块：衔接张量维度
# ==========================================
class PatchifyFreq_ImgShape(nn.Module):
    def __init__(self, image_size=32, block_size=16, channels=3, patch_size=16):
        super().__init__()
        self.image_size = image_size
        self.block_size = block_size
        self.channels = channels
        self.patch_size = patch_size

        # 固定随机符号翻转 mask；使用局部生成器避免污染全局随机状态
        mask_generator = torch.Generator()
        mask_generator.manual_seed(42)
        fixed_mask = torch.randint(
            0,
            2,
            (1, channels, image_size, image_size),
            generator=mask_generator,
        ) * 2 - 1
        
        # 3. 注册为 buffer：它不计入梯度更新，但会随 model.state_dict() 保存
        self.register_buffer('fixed_sign_mask', fixed_mask.float())

    def butterfly_forward(self, blocks: torch.Tensor) -> torch.Tensor:
        B, C_sq, N = blocks.shape
        dim = self.block_size ** 2
        C = C_sq // dim

        # 恢复 [B, N, C, H, W]，在每个 block 上做 2D DHT 并中心化频谱布局
        x_in = blocks.view(B, C, self.block_size, self.block_size, N).permute(0, 4, 1, 2, 3)
        freq_2d = _dht2(x_in)
        freq_2d = torch.fft.fftshift(freq_2d, dim=(-2, -1))

        freq_blocks = freq_2d.permute(0, 2, 3, 4, 1).reshape(B, C * dim, N)
        return freq_blocks
        

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 使用预生成的固定 mask，它会自动移动到 x 所在的设备 (CPU/GPU)
        x_flipped = x * self.fixed_sign_mask
        
        # 后续逻辑保持不变
        unfolded_16 = F.unfold(x_flipped, kernel_size=self.block_size, stride=self.block_size)
        freq_blocks = self.butterfly_forward(unfolded_16)
        freq_spatial = F.fold(freq_blocks, output_size=(self.image_size, self.image_size),
                              kernel_size=self.block_size, stride=self.block_size) 
        unfolded_2 = F.unfold(freq_spatial, kernel_size=self.patch_size, stride=self.patch_size) 
        
        # 注意：这里返回 self.fixed_sign_mask 供逆变换使用
        return unfolded_2, freq_spatial, self.fixed_sign_mask

class Unpatchify_ImgShape(nn.Module):
    def __init__(self, image_size=32, patch_size=2, channels=3):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return F.fold(patches, output_size=(self.image_size, self.image_size),
                      kernel_size=self.patch_size, stride=self.patch_size)

class FreqToSpatial_ImgShape(nn.Module):
    def __init__(self, image_size=32, block_size=16, channels=3):
        super().__init__()
        self.image_size = image_size
        self.block_size = block_size
        self.channels = channels
        
    def butterfly_inverse(self, freq_blocks: torch.Tensor) -> torch.Tensor:
        B, C_sq, N = freq_blocks.shape
        dim = self.block_size ** 2
        C = C_sq // dim

        freq_2d = freq_blocks.view(B, C, self.block_size, self.block_size, N).permute(0, 4, 1, 2, 3)
        freq_2d = torch.fft.ifftshift(freq_2d, dim=(-2, -1))
        spatial = _dht2(freq_2d)
        spatial_blocks = spatial.permute(0, 2, 3, 4, 1).reshape(B, C * dim, N)
        return spatial_blocks
        
    def forward(self, freq_spatial: torch.Tensor) -> torch.Tensor:
        unfolded_16 = F.unfold(freq_spatial, kernel_size=self.block_size, stride=self.block_size)
        spatial_blocks = self.butterfly_inverse(unfolded_16)
        spatial_img = F.fold(spatial_blocks, output_size=(self.image_size, self.image_size),
                             kernel_size=self.block_size, stride=self.block_size)
        return spatial_img


class FreqMAE_ImgShape(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        block_size: int = 16,       
        channels: int = 3,
        patch_size: int = 16,
        patch_dim: Optional[int] = None,
        emb_dim: int = 768,
        encoder_layers: int = 12,
        decoder_emb_dim: int = 512,
        decoder_layers: int = 8,
        num_heads: int = 12,
        decoder_num_heads: int = 16,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
        acs_num: int = 0,
        acs_radius: Optional[float] = None,
        radial_decay: float = 0.8,
        encoder_template: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.image_size = image_size
        self.block_size = block_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.num_patches = (image_size // patch_size) ** 2
        self.num_prefix_tokens = 1
        self.freq_channels = channels
        expected_patch_dim = self.freq_channels * (patch_size ** 2)
        expected_output_patch_dim = channels * (patch_size ** 2)
        grid_size = image_size // patch_size
        if grid_size * grid_size != self.num_patches:
            raise ValueError(f"Expected square patch grid, got image_size={image_size}, patch_size={patch_size}")
        self.channels = channels
        decoder_spec = None
        mae_norm_layer = partial(nn.LayerNorm, eps=1e-6)
        mae_mlp_ratio = 4.0
        if encoder_template is not None:
            spec = extract_timm_vit_backbone_spec(encoder_template)
            emb_dim = spec["emb_dim"]
            encoder_layers = spec["encoder_layers"]
            num_heads = spec["num_heads"]
            decoder_spec = extract_timm_mae_decoder_spec(encoder_template)
            if decoder_spec is not None:
                decoder_emb_dim = decoder_spec["decoder_emb_dim"]
                decoder_layers = decoder_spec["decoder_layers"]
                decoder_num_heads = decoder_spec["decoder_num_heads"]

        self.emb_dim = emb_dim
        self.decoder_emb_dim = decoder_emb_dim

        self.patch_dim = expected_patch_dim if patch_dim is None else patch_dim
        if self.patch_dim != expected_patch_dim:
            raise ValueError(
                f"patch_dim must be {expected_patch_dim} for DHT frequency patches, got {self.patch_dim}"
            )
        
        # 频率-空间转换模块
        self.patchify = PatchifyFreq_ImgShape(image_size, block_size, channels, patch_size)
        self.unpatchify = Unpatchify_ImgShape(image_size, patch_size, channels)
        self.freq_to_spatial = FreqToSpatial_ImgShape(image_size, block_size, channels)
        
        # 标准 MAE 的 encoder / decoder 组件，encoder 可直接从 timm ViT 模板克隆。
        self.input_proj = nn.Linear(self.patch_dim, self.emb_dim)

        if encoder_template is None:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.emb_dim))
            self.enc_pos_embedding = nn.Parameter(
                _get_2d_sincos_pos_embed(self.emb_dim, grid_size, cls_token=True),
                requires_grad=False,
            )
            self.encoder_blocks = nn.Sequential(
                *[
                    Block(
                        self.emb_dim,
                        num_heads,
                        mlp_ratio=mae_mlp_ratio,
                        qkv_bias=True,
                        norm_layer=mae_norm_layer,
                    )
                    for _ in range(encoder_layers)
                ]
            )
            self.encoder_norm = mae_norm_layer(self.emb_dim)
            self._init_token_parameter(self.cls_token)
            self._reset_module(self.encoder_blocks)
            self._reset_module(self.encoder_norm)
        else:
            self.cls_token = nn.Parameter(self._clone_cls_token(encoder_template))
            self.enc_pos_embedding = nn.Parameter(self._clone_encoder_pos_embedding(encoder_template))
            self.encoder_blocks = copy.deepcopy(encoder_template.blocks)
            self.encoder_norm = copy.deepcopy(encoder_template.norm)

        self._decoder_initialized_from_template = decoder_spec is not None
        if self._decoder_initialized_from_template:
            self.decoder_embed = copy.deepcopy(encoder_template.decoder_embed)
            self.mask_token = nn.Parameter(encoder_template.mask_token.detach().clone())
            self.dec_pos_embedding = nn.Parameter(
                _adapt_pos_embed(
                    encoder_template.decoder_pos_embed.detach(),
                    torch.Size((1, self.num_patches + self.num_prefix_tokens, self.decoder_emb_dim)),
                )
            )
            self.decoder_blocks = copy.deepcopy(encoder_template.decoder_blocks)
            self.decoder_norm = copy.deepcopy(encoder_template.decoder_norm)
            self.output_proj = copy.deepcopy(encoder_template.decoder_pred)
            if self.output_proj.out_features != expected_output_patch_dim:
                raise ValueError(
                    f"timm decoder_pred out_features={self.output_proj.out_features} does not match "
                    f"expected spatial patch dim {expected_output_patch_dim}"
                )
        else:
            self.decoder_embed = nn.Linear(self.emb_dim, decoder_emb_dim)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_emb_dim))
            self.dec_pos_embedding = nn.Parameter(
                _get_2d_sincos_pos_embed(decoder_emb_dim, grid_size, cls_token=True),
                requires_grad=False,
            )
            self.decoder_blocks = nn.Sequential(
                *[
                    Block(
                        decoder_emb_dim,
                        decoder_num_heads,
                        mlp_ratio=mae_mlp_ratio,
                        qkv_bias=True,
                        norm_layer=mae_norm_layer,
                    )
                    for _ in range(decoder_layers)
                ]
            )
            self.decoder_norm = mae_norm_layer(decoder_emb_dim)
            self.output_proj = nn.Linear(decoder_emb_dim, expected_output_patch_dim)

        self.init_decoder_weights()
        self._setup_adaptive_mask(acs_num, acs_radius, radial_decay)

    def _clone_cls_token(self, encoder_template: nn.Module) -> torch.Tensor:
        cls_token = getattr(encoder_template, "cls_token", None)
        if cls_token is None:
            cls_token = torch.zeros(1, 1, self.emb_dim)
            trunc_normal_(cls_token, std=0.02)
        return cls_token.detach().clone()

    def _clone_encoder_pos_embedding(self, encoder_template: nn.Module) -> torch.Tensor:
        pos_embed = getattr(encoder_template, "pos_embed", None)
        if pos_embed is None:
            grid_size = self.image_size // self.patch_size
            return _get_2d_sincos_pos_embed(self.emb_dim, grid_size, cls_token=True)
        return _adapt_pos_embed(pos_embed.detach(), torch.Size((1, self.num_patches + self.num_prefix_tokens, self.emb_dim)))

    def init_decoder_weights(self) -> None:
        self._reset_module(self.input_proj)
        if self._decoder_initialized_from_template:
            return
        self._init_token_parameter(self.mask_token)
        self._reset_module(self.decoder_embed)
        self._reset_module(self.decoder_blocks)
        self._reset_module(self.decoder_norm)
        self._reset_module(self.output_proj)

    def _init_token_parameter(self, param: torch.Tensor) -> None:
        trunc_normal_(param, std=0.02)

    def _reset_module(self, module: nn.Module) -> None:
        for submodule in module.modules():
            if isinstance(submodule, nn.Linear):
                nn.init.xavier_uniform_(submodule.weight)
                if submodule.bias is not None:
                    nn.init.constant_(submodule.bias, 0)
            elif isinstance(submodule, nn.LayerNorm):
                nn.init.constant_(submodule.bias, 0)
                nn.init.constant_(submodule.weight, 1.0)
        
    def _setup_adaptive_mask(
        self,
        acs_num: int,
        acs_radius: Optional[float],
        radial_decay: float,
        init_mode: str = 'random',
        random_std: float = 0.1       # ✨ 新增：控制随机初始化的波动幅度
    ):
        """初始化 block-wise 2D 掩码 logits 与 ACS 保护区。"""
        num_blocks_per_axis = self.image_size // self.block_size
        patches_per_block_axis = self.block_size // self.patch_size

        assert self.image_size % self.block_size == 0, "image_size 必须能被 block_size 整除"
        assert self.block_size % self.patch_size == 0, "block_size 必须能被 patch_size 整除"

        # 在每个 block 的 patch 网格上构建坐标
        local_y, local_x = torch.meshgrid(
            torch.arange(patches_per_block_axis, dtype=torch.float32),
            torch.arange(patches_per_block_axis, dtype=torch.float32),
            indexing='ij',
        )
        center = (patches_per_block_axis - 1) / 2.0
        radius = torch.sqrt((local_y - center) ** 2 + (local_x - center) ** 2)

        # 兼容旧参数 acs_num：按圆面积近似映射为半径
        if acs_radius is None:
            acs_radius = math.sqrt(max(float(acs_num), 1.0) / math.pi)
        acs_radius_patch = float(acs_radius) / float(self.patch_size)
        acs_local = radius <= acs_radius_patch

        # ==========================================
        # ✨ 核心消融逻辑：初始化掩码 Logits
        # ==========================================
        if init_mode == 'vd':
            # 山峰先验：外围概率按半径指数衰减
            local_logits = -float(radial_decay) * radius
        elif init_mode == 'uniform':
            # 平坦先验：所有未保护区域概率绝对相等
            local_logits = torch.zeros_like(radius)
        elif init_mode == 'random':
            # 随机先验：高斯噪声分布。
            # 乘以 random_std 避免 exp() 后某些离群值吞噬所有概率配额
            local_logits = torch.randn_like(radius) * random_std
        elif init_mode == 'vd_poly':
            # 长尾多项式分布：对数下降，完美解决边缘死区
            # 常数 2.0 可以视作控制拖尾厚度的超参数 beta
            local_logits = -2.0 * torch.log(1.0 + float(radial_decay) * radius)
        else:
            raise ValueError(f"不支持的 init_mode: {init_mode}. 可选值为 'vd', 'uniform', 'random'")

        # 拼装成全局 2D 网格；mask logits 以 2D 形式保留
        logits_global = local_logits.repeat(num_blocks_per_axis, num_blocks_per_axis)
        acs_global = acs_local.repeat(num_blocks_per_axis, num_blocks_per_axis)

        self.mask_logits_2d = nn.Parameter(logits_global)
        self.register_buffer('acs_mask_2d', acs_global)

    def _normalize_target_patches(self, target_patches: torch.Tensor) -> torch.Tensor:
        if not self.norm_pix_loss:
            return target_patches
        mean = target_patches.mean(dim=-1, keepdim=True)
        var = target_patches.var(dim=-1, keepdim=True)
        return (target_patches - mean) / torch.sqrt(var + 1.0e-6)

    def _denormalize_pred_patches(
        self,
        pred_patches: torch.Tensor,
        target_patches: torch.Tensor,
    ) -> torch.Tensor:
        if not self.norm_pix_loss:
            return pred_patches
        mean = target_patches.mean(dim=-1, keepdim=True)
        var = target_patches.var(dim=-1, keepdim=True)
        std = torch.sqrt(var + 1.0e-6)
        return pred_patches * std + mean

    def _masked_patch_mse(
        self,
        pred_patches: torch.Tensor,
        target_patches: torch.Tensor,
        mask_bool: torch.Tensor,
        *,
        normalize_target: bool,
    ) -> torch.Tensor:
        target = (
            self._normalize_target_patches(target_patches)
            if normalize_target
            else target_patches
        )
        mse = F.mse_loss(pred_patches, target, reduction="none").mean(dim=-1)
        mask = mask_bool.float()
        denom = mask.sum().clamp_min(1e-6)
        return (mse * mask).sum() / denom

    def _get_exact_probabilities(self, num_visible: int) -> torch.Tensor:
        """
        核心数学模块：利用二分查找计算出精确的概率分布 P_i
        保证所有 0 <= P_i <= 1，且其总和精准等于 num_visible
        """
        logits_1d = self.mask_logits_2d.flatten()
        acs_mask_1d = self.acs_mask_2d.flatten()

        # 转为非负权重
        W = torch.exp(logits_1d)
        # 强制 ACS 区域概率无限大
        W = torch.where(acs_mask_1d, torch.full_like(W, 1e6), W)
        
        low = torch.tensor(0.0, device=W.device)
        high = torch.tensor(1e4, device=W.device)
        
        # 二分查找寻找最佳 Scaling Factor
        for _ in range(20):
            mid = (low + high) / 2.0
            P = torch.clamp(W * mid, 0.0, 1.0)
            if P.sum() > num_visible:
                high = mid
            else:
                low = mid
                
        # 返回最终缩放并截断后的真实概率
        return torch.clamp(W * high, 0.0, 1.0)

    def forward(
        self, 
        x: torch.Tensor, 
        training: bool = True,
        custom_mask_ratio: Optional[float] = None,
        input_is_freq: bool = False,
        fixed_visible_idx: Optional[torch.Tensor] = None,
        return_patches: bool = False,
        return_eval_dict: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        batch_size = x.shape[0]
        device = x.device
        target_patches = None
        if input_is_freq:
            freq_spatial = x
            patches = F.unfold(freq_spatial, kernel_size=self.patch_size, stride=self.patch_size)
        else:
            patches, freq_spatial, _ = self.patchify(x)
            target_patches = rearrange(
                F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size),
                "b d n -> b n d",
            )
        patches_flat = rearrange(patches, "b d n -> b n d")
        embedded = self.input_proj(patches_flat)
        embedded = embedded + self.enc_pos_embedding[:, self.num_prefix_tokens :, :]

        ratio = custom_mask_ratio if custom_mask_ratio is not None else self.mask_ratio
        num_visible = int(self.num_patches * (1 - ratio))
        if fixed_visible_idx is not None:
            fixed_visible_idx = fixed_visible_idx.to(device=device, dtype=torch.long)
            if fixed_visible_idx.dim() == 1:
                visible_idx = fixed_visible_idx.unsqueeze(0).expand(batch_size, -1)
            elif fixed_visible_idx.dim() == 2:
                if fixed_visible_idx.shape[0] != batch_size:
                    raise ValueError(
                        f"fixed_visible_idx batch size mismatch: got {fixed_visible_idx.shape[0]}, expected {batch_size}"
                    )
                visible_idx = fixed_visible_idx
            else:
                raise ValueError("fixed_visible_idx must be a 1D or 2D tensor")

            mask_binary = torch.zeros(batch_size, self.num_patches, device=device, dtype=torch.float32)
            mask_binary.scatter_(1, visible_idx, 1.0)
            mask_ste = mask_binary
        else:
            P = self._get_exact_probabilities(num_visible)
            P_batch = P.unsqueeze(0).expand(batch_size, -1)
            
            visible_idx = torch.multinomial(P_batch, num_samples=num_visible, replacement=False)

            mask_binary = torch.zeros_like(P_batch)
            mask_binary.scatter_(1, visible_idx, 1.0)
            mask_ste = mask_binary.detach() - P_batch.detach() + P_batch

        ids_restore, mask_bool = _build_ids_restore(visible_idx, self.num_patches)

        expanded_visible_idx = visible_idx.unsqueeze(-1).expand(-1, -1, embedded.size(-1))
        visible_patches = torch.gather(embedded, 1, expanded_visible_idx)

        cls_tokens = (self.cls_token + self.enc_pos_embedding[:, : self.num_prefix_tokens, :]).expand(batch_size, -1, -1)
        encoder_input = torch.cat([cls_tokens, visible_patches], dim=1)
        encoded = self.encoder_blocks(encoder_input)
        encoded = self.encoder_norm(encoded)

        decoded = self.decoder_embed(encoded)
        num_masked = self.num_patches - visible_idx.shape[1]
        mask_tokens = repeat(self.mask_token, "1 1 d -> b n d", b=batch_size, n=num_masked)
        decoded_patches = torch.cat([decoded[:, self.num_prefix_tokens :, :], mask_tokens], dim=1)
        gather_index = ids_restore.unsqueeze(-1).expand(-1, -1, decoded.size(-1))
        decoded_patches = torch.gather(decoded_patches, 1, gather_index)
        full_seq = torch.cat([decoded[:, : self.num_prefix_tokens, :], decoded_patches], dim=1)
        full_seq = full_seq + self.dec_pos_embedding

        reconstructed = self.decoder_blocks(full_seq)
        reconstructed = self.decoder_norm(reconstructed)
        spatial_recon_patches = self.output_proj(reconstructed[:, self.num_prefix_tokens :, :])
        spatial_recon_unfold = rearrange(spatial_recon_patches, "b n d -> b d n")
        spatial_recon_img = self.unpatchify(spatial_recon_unfold)

        if training:
            if target_patches is None:
                raise ValueError("training=True requires image input, not precomputed frequency input")
            target_for_loss = self._normalize_target_patches(target_patches)
            mse_full = F.mse_loss(spatial_recon_patches, target_for_loss, reduction="none").mean(dim=-1)
            masked_weight = 1.0 - mask_ste
            denom = masked_weight.sum().detach().clamp_min(1e-6)
            loss = (mse_full * masked_weight).sum() / denom
            if return_patches:
                return spatial_recon_img, loss, spatial_recon_patches, target_patches, mask_bool, visible_idx
            return spatial_recon_img, loss
            
        else:
            if target_patches is None:
                if self.norm_pix_loss:
                    raise ValueError(
                        "Eval with norm_pix_loss=True requires image input so target patch statistics are available."
                    )

                if return_patches:
                    if return_eval_dict:
                        return {
                            "pred_img": spatial_recon_img,
                            "pred_patches_raw": spatial_recon_patches,
                            "pred_patches_pixel": spatial_recon_patches,
                            "target_patches": None,
                            "fused_img": None,
                            "visible_idx": visible_idx,
                            "mask_bool": mask_bool,
                            "eval_loss": None,
                            "masked_mse_pixel": None,
                        }
                    return spatial_recon_img, visible_idx, mask_bool, spatial_recon_patches, None
                return spatial_recon_img, visible_idx, mask_bool

            pred_patches_raw = spatial_recon_patches
            pred_patches_pixel = self._denormalize_pred_patches(
                pred_patches_raw,
                target_patches,
            )
            pred_unfold = rearrange(pred_patches_pixel, "b n d -> b d n")
            pred_img = self.unpatchify(pred_unfold)

            fused_patches = torch.where(
                mask_bool.unsqueeze(-1),
                pred_patches_pixel,
                target_patches,
            )
            fused_unfold = rearrange(fused_patches, "b n d -> b d n")
            fused_img = self.unpatchify(fused_unfold)

            eval_loss = self._masked_patch_mse(
                pred_patches=pred_patches_raw,
                target_patches=target_patches,
                mask_bool=mask_bool,
                normalize_target=True,
            )
            masked_mse_pixel = self._masked_patch_mse(
                pred_patches=pred_patches_pixel,
                target_patches=target_patches,
                mask_bool=mask_bool,
                normalize_target=False,
            )

            if return_patches:
                if return_eval_dict:
                    return {
                        "pred_img": pred_img,
                        "pred_patches_raw": pred_patches_raw,
                        "pred_patches_pixel": pred_patches_pixel,
                        "target_patches": target_patches,
                        "fused_img": fused_img,
                        "visible_idx": visible_idx,
                        "mask_bool": mask_bool,
                        "eval_loss": eval_loss,
                        "masked_mse_pixel": masked_mse_pixel,
                    }
                return fused_img, visible_idx, mask_bool, pred_patches_raw, target_patches
            return pred_img, visible_idx, mask_bool

if __name__ == "__main__":
    print("=" * 70)
    print("Frequency Domain MAE with Image-Shaped Processing")
    print("=" * 70)
    print("Toy smoke test only. Runtime defaults are configured by the training scripts.")
    
    # Create a small toy model for quick structure verification.
    model = FreqMAE_ImgShape(
        image_size=32,
        block_size=16,
        channels=3,
        patch_size=2,
        patch_dim=24,
        emb_dim=768,
        encoder_layers=12,
        decoder_layers=8,
        num_heads=12,
        mask_ratio=0.75,
        acs_radius=2.0,
    )
    
    # Dummy input for the toy smoke test
    x = torch.randn(2, 3, 32, 32)
    print(f"\nInput image: {x.shape}")
    
    # Training forward
    print("\n【Training Mode】")
    recon_train, loss = model(x, training=True)
    print(f"Reconstructed: {recon_train.shape}")
    print(f"Loss: {loss.item():.6f}")
    print("✓ Loss computed on masked spatial patches")
    
    # Eval forward
    print("\n【Eval Mode】")
    recon_eval, visible_idx, masked_idx = model(x, training=False)
    print(f"Reconstructed: {recon_eval.shape}")
    print(f"Visible idx: {visible_idx.shape}, Masked idx: {masked_idx.shape}")
    print(f"✓ Ready for PSNR computation on [B, 3, 32, 32]")
    
    print("\n" + "=" * 70)
    print("✅ Model structure verified")
    print("=" * 70)
    
    print("\n【Processing Pipeline】:")
    print("1️⃣  Input: [B, 3, 32, 32]")
    print("2️⃣  unfold(16×16): [B, 3, 32, 32] → [B, 768, 4]")
    print("3️⃣  DHT2 + fftshift per block: → [B, 768, 4]")
    print("4️⃣  fold: [B, 768, 4] → [B, 3, 32, 32] (freq image)")
    print("5️⃣  unfold(2×2): [B, 3, 32, 32] → [B, 12, 256]")
    print("6️⃣  reshape: [B, 12, 256] → [B, 256, 12]")
    print("7️⃣  input_proj: [B, 256, 12] → [B, 256, 768]")
    print("8️⃣  MAE decoder predicts spatial patches directly")
    print("9️⃣  ⭐️ LOSS: masked spatial-patch MSE")
    print("🔟 fold: [B, 12, 256] → [B, 3, 32, 32] (spatial reconstruction)")
    print("1️⃣1️⃣  eval can fuse visible ground-truth patches in image space")
    print("1️⃣2️⃣  PSNR: on [B, 3, 32, 32] (in eval caller)")
    print("=" * 70)
