# Refactor-Based ImageNet Root

[Weights](https://drive.google.com/drive/folders/1Y2WLw7aao23ZG1YYu47rXXGk71_7v0K5?usp=sharing)

The root of this workspace now follows the `refactor` code path, adapted to run on ImageNet with the original `224x224` image resolution.

The active root workflow is:

1. self-supervised pretraining with `train_ssl_freq_mae.py`
2. direct spatial decoder training with `train_direct_spatial_decoder.py`
3. fixed-mask evaluation with `eval_freq_mae_fixed_mask.py`

Key runtime assumptions:

- dataset source: ImageNet directory with `train/` and `val/`
- default input size: `224`
- default frequency block size: `16`
- default MAE patch size: `16`
- resulting token count: `(224 / 16)^2 = 196`
- default patch dimension: `16 x 16 x 3 = 768` (blockwise DHT, real-valued)
- default ViT scale: MAE-B-like (`encoder_dim=768`, `encoder_heads=12`, `encoder_depth=12`, `decoder_dim=512`, `decoder_heads=16`, `decoder_depth=8`)
- optimizer: `AdamW`
- weight decay: `0.05`
- optimizer betas: `(0.9, 0.95)`
- base learning rate: `1.5e-4`
- learning-rate rule: `actual_lr = base_lr * total_batch_size / 256`
- warmup epochs: `40`
- total pretraining epochs: `800`
- default total batch size: `5120` via `8 GPUs x batch 160 x grad_accum_steps 4`

That keeps the MAE token count in a tractable range for single-node multi-GPU training.

Double-checked runtime geometry:

- global patch grid: `14 x 14`
- global adaptive-mask PDF shape: `14 x 14`
- per-block local PDF shape: `1 x 1`

The implementation now matches the documented defaults end to end: the model internals, the training scripts, and the run scripts all use `block_size=16` and `patch_size=16`.

Default pretraining augmentation:

- `RandomResizedCrop` scale: `0.2` to `1.0`
- horizontal flip: `0.5`
- color jitter: disabled
- mixup / cutmix: not used in this workflow
- drop path: disabled in the current model code

## Model Structure

The first-stage model is not a vanilla MAE end to end. The split is:

1. `PatchifyFreq_ImgShape`
   - Applies the fixed sign mask.
   - Converts each block to DHT frequency coefficients.
   - Reassembles them into frequency-space patches.

2. `input_proj`
   - This is the patch embedding layer for the frequency-domain input.
   - It is a learned linear map from DHT patch vectors to the ViT embedding space.
   - It must remain custom because MAE's original `patch_embed.proj` was trained on spatial RGB patches, while this model feeds it DHT frequency patches.
   - Even when both have the same flat dimension (`3 * patch_size^2`), the basis is different:
     - original MAE patch embed sees pixel values in spatial coordinates
     - this model sees Hartley coefficients in frequency coordinates
   - Because of that basis mismatch, `input_proj` cannot safely reuse `patch_embed.proj` weights directly unless an explicit basis transform is derived and applied first.
   - The current implementation therefore keeps `input_proj` task-specific and initializes it separately from the timm MAE weights.

3. `encoder`
   - The encoder is MAE-style ViT.
   - When the timm template is available, the encoder structure and most encoder weights come from `vit_base_patch16_224.mae`.
   - This includes:
     - `cls_token`
     - encoder position embedding shape contract
     - encoder transformer blocks
     - encoder normalization

4. `decoder`
   - The decoder is MAE-style as well.
   - It restores masked tokens with `mask_token`, uses `ids_restore`, adds decoder positional embeddings, runs MAE decoder blocks, and predicts patch outputs.
   - The prediction target in this project is spatial patch content, not frequency patch content.
   - In the current default setup, the active decoder path is the local MAE-style fallback.
   - For the default ImageNet run (`image_size=224`, `patch_size=16`, `emb_dim=768`, `decoder_emb_dim=512`, `decoder_layers=8`, `decoder_heads=16`), the fallback decoder is:
     - `decoder_embed`: `Linear(768 -> 512)`
     - `mask_token`: learnable parameter with shape `[1, 1, 512]`
     - `dec_pos_embedding`: fixed 2D sin-cos positional embedding with shape `[1, 197, 512]`
     - `decoder_blocks`: 8 transformer blocks with:
       - hidden size `512`
       - 16 attention heads
       - head dimension `32`
       - `mlp_ratio=4.0`
       - `qkv_bias=True`
       - `LayerNorm(eps=1e-6)`
     - `decoder_norm`: `LayerNorm(512, eps=1e-6)`
     - `output_proj`: `Linear(512 -> 768)`
   - So the decoder predicts `196` spatial patches of size `16 x 16 x 3`, then `unpatchify` stitches them back into a `224 x 224` RGB image.

5. Loss
   - The pretraining loss is masked patch MSE on spatial patches.
   - By default `norm_pix_loss` is enabled, matching the MAE paper's patch-wise target normalization.

6. Evaluation semantics
   - `pred_img` is the honest decoder prediction in pixel space and is the correct input for PSNR / SSIM / MSE.
   - `fused_img` is only a visualization artifact where visible patches are copied from ground truth and masked patches come from the model.
   - `fused_img` should not be used as the primary reconstruction metric input.

## Layout

Main entrypoints:

- `train_ssl_freq_mae.py`: self-supervised pretraining
- `train_direct_spatial_decoder.py`: frozen-encoder direct spatial decoder training
- `eval_freq_mae_fixed_mask.py`: fixed-mask evaluation
- `direct_spatial_decoder_model.py`: second-stage direct spatial decoder wrapper
- `dataset_utils.py`: dataset helper utilities used by the ImageNet setup

## Data

By default the scripts expect:

```text
data/imagenet/
  train/
  val/
```

You can also point at any other ImageNet location with:

- `--data_root`
- `--train_dir`
- `--val_dir`

If you want a scripted download helper, use:

```bash
cd others/refactor
bash download_imagenet.sh
```

Default download location:

```text
others/refactor/data/imagenet/
```

Downloaded shard archives:

```text
others/refactor/data/imagenet/valid.zip
others/refactor/data/imagenet/train_part0.zip
others/refactor/data/imagenet/train_part1.zip
others/refactor/data/imagenet/train_part2.zip
others/refactor/data/imagenet/train_part3.zip
```

Assembled dataset layout after download:

```text
others/refactor/data/imagenet/train/<class_name>/*
others/refactor/data/imagenet/val/<class_name>/*
```

You can override that path with:

```bash
DATA_ROOT=/path/to/imagenet bash download_imagenet.sh
```

Authentication is handled through the Kaggle CLI environment, typically either:

```bash
KAGGLE_KEY=... bash download_imagenet.sh
```

or:

```bash
~/.kaggle/kaggle.json
```

## Local Examples

Pretraining:

```bash
python train_ssl_freq_mae.py \
  --dataset imagenet \
  --data_root /path/to/imagenet \
  --train_dir /path/to/imagenet/train \
  --val_dir /path/to/imagenet/val \
  --image_size 224 \
  --block_size 16 \
  --patch_size 16 \
  --mask_ratio 0.75 \
  --batch_size 160 \
  --val_batch_size 160 \
  --num_workers 2 \
  --grad_accum_steps 4 \
  --amp
```

To disable MAE-style patch normalization, add:

```bash
--no-norm_pix_loss
```

Direct Spatial Decoder:

```bash
python train_direct_spatial_decoder.py \
  --dataset imagenet \
  --data_root /path/to/imagenet \
  --train_dir /path/to/imagenet/train \
  --val_dir /path/to/imagenet/val \
  --freq_mae_ckpt /path/to/best_freq-mae-ssl.pth \
  --batch_size 16 \
  --amp
```

Evaluation:

```bash
python eval_freq_mae_fixed_mask.py \
  --dataset imagenet \
  --data_root /path/to/imagenet \
  --train_dir /path/to/imagenet/train \
  --val_dir /path/to/imagenet/val \
  --model_path logs/imagenet/ssl_freq_mae/mr075/best_freq-mae-ssl.pth \
  --image_size 224 \
  --block_size 16 \
  --patch_size 16
```

## CLI Reference

### `train_ssl_freq_mae.py`

- Data and checkpoints:
  - `--dataset`
  - `--data_root`
  - `--train_dir`
  - `--val_dir`
  - `--output_dir`
  - `--experiment`
  - `--resume`
  - `--pretrained_ckpt`
  - `--timm_model_name`
  - `--timm_pretrained` / `--no-timm_pretrained`
  - `--timm_checkpoint_path`

- Optimization and runtime:
  - `--epochs`
  - `--batch_size`
  - `--val_batch_size`
  - `--num_workers`
  - `--grad_accum_steps`
  - `--seed`
  - `--amp`
  - `--pin_mem` / `--no-pin_mem`

- Model:
  - `--image_size`
  - `--block_size`
  - `--patch_size`
  - `--emb_dim`
  - `--encoder_layer`
  - `--encoder_head`
  - `--decoder_emb_dim`
  - `--decoder_layer`
  - `--decoder_head`
  - `--mask_ratio`
  - `--norm_pix_loss` / `--no-norm_pix_loss`
  - `--visualize_every`
  - `--visualize_count`

- Optimizer / scheduler:
  - `--opt`
  - `--base_lr`
  - `--lr`
  - `--weight_decay`
  - `--sched`
  - `--warmup_epochs`
  - `--warmup_lr`
  - `--min_lr`
  - `--opt_betas`

- Augmentation:
  - `--no_aug`
  - `--train_scale_min`
  - `--train_scale_max`
  - `--train_ratio_min`
  - `--train_ratio_max`
  - `--hflip`
  - `--color_jitter`
  - `--interpolation`

- Optional upload:
  - `--gdrive-upload` / `--no-gdrive-upload`
  - `--gdrive-service-account-json`
  - `--gdrive-folder-id`

### `train_direct_spatial_decoder.py`

- Data and checkpoints:
  - `--dataset`
  - `--data_root`
  - `--train_dir`
  - `--val_dir`
  - `--freq_mae_ckpt`
  - `--output_dir`
  - `--resume`

- Runtime:
  - `--epochs`
  - `--batch_size`
  - `--num_workers`
  - `--val_size`
  - `--seed`
  - `--mask_seed`
  - `--mask_ratio`
  - `--decoder_lr`
  - `--weight_decay`
  - `--amp`

- Optional upload:
  - `--gdrive-upload` / `--no-gdrive-upload`
  - `--gdrive-service-account-json`
  - `--gdrive-folder-id`

### `eval_freq_mae_fixed_mask.py`

- Checkpoint and data:
  - `--model_path`
  - `--timm_model_name`
  - `--dataset`
  - `--data_root`
  - `--train_dir`
  - `--val_dir`
  - `--eval_split`

- Runtime:
  - `--batch_size`
  - `--num_workers`
  - `--seed`
  - `--subset`

- Model shape / config recovery overrides:
  - `--image_size`
  - `--block_size`
  - `--patch_size`
  - `--emb_dim`
  - `--encoder_layers`
  - `--decoder_emb_dim`
  - `--decoder_layers`
  - `--num_heads`
  - `--decoder_num_heads`
  - `--mask_ratio`

- Fixed-mask evaluation:
  - `--fixed_mask_strategy`
  - `--fixed_mask_seed`

- Output artifacts:
  - `--summary_path`
  - `--save_vis`
  - `--vis_path`
  - `--vis_count`

## Direct Invocation

Shared environment setup:

```bash
cd /users/$USER/scratch/grind_project
python3 -m venv others/venv
source others/venv/bin/activate
python3 -m pip install --upgrade pip wheel setuptools
python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
python3 -m pip install numpy scipy tqdm tensorboard einops timm matplotlib diffusers google-api-python-client google-auth huggingface_hub kaggle
```

Single-node multi-GPU pretraining:

```bash
cd /users/$USER/scratch/grind_project
source others/venv/bin/activate
torchrun --standalone --nnodes=1 --nproc_per_node=8 train_ssl_freq_mae.py \
  --dataset imagenet \
  --data_root /path/to/imagenet \
  --train_dir /path/to/imagenet/train \
  --val_dir /path/to/imagenet/val \
  --output_dir logs/imagenet/ssl_freq_mae \
  --experiment mr075 \
  --epochs 800 \
  --batch_size 160 \
  --val_batch_size 160 \
  --num_workers 2 \
  --grad_accum_steps 4 \
  --base_lr 1.5e-4 \
  --image_size 224 \
  --block_size 16 \
  --patch_size 16 \
  --mask_ratio 0.75 \
  --timm_model_name vit_base_patch16_224.mae \
  --timm_pretrained \
  --amp
```

Direct spatial decoder stage:

```bash
cd /users/$USER/scratch/grind_project
source others/venv/bin/activate
python3 train_direct_spatial_decoder.py \
  --dataset imagenet \
  --data_root /path/to/imagenet \
  --train_dir /path/to/imagenet/train \
  --val_dir /path/to/imagenet/val \
  --freq_mae_ckpt logs/imagenet/ssl_freq_mae/mr075/best_freq-mae-ssl.pth \
  --output_dir logs/imagenet/direct_spatial_decoder \
  --epochs 50 \
  --batch_size 16 \
  --num_workers 8 \
  --mask_ratio 0.75 \
  --decoder_lr 1e-5 \
  --weight_decay 0.05 \
  --amp
```

Simple mask-ratio sweep without wrapper scripts:

```bash
cd /users/$USER/scratch/grind_project
source others/venv/bin/activate
for ratio in 0.75; do
  tag="${ratio/./}"
  torchrun --standalone --nnodes=1 --nproc_per_node=8 train_ssl_freq_mae.py \
    --dataset imagenet \
    --data_root /path/to/imagenet \
    --output_dir logs/imagenet/ssl_freq_mae_sweep \
    --experiment "mr${tag}" \
    --epochs 800 \
    --batch_size 160 \
    --val_batch_size 160 \
    --num_workers 2 \
    --grad_accum_steps 4 \
    --base_lr 1.5e-4 \
    --image_size 224 \
    --block_size 16 \
    --patch_size 16 \
    --mask_ratio "${ratio}" \
    --timm_model_name vit_base_patch16_224.mae \
    --timm_pretrained \
    --amp
done
```

Google Drive / Google Cloud is optional. Nothing in the root `refactor` workflow requires it. Upload support is only used when you explicitly enable `GDRIVE_UPLOAD=1` or pass the matching CLI flags.

Default paths remain in the root project layout:

- data: `data/imagenet`
- pretrain outputs: `logs/imagenet/ssl_freq_mae/<experiment>/`
- direct spatial decoder outputs: `logs/imagenet/direct_spatial_decoder/`
- default decoder checkpoint input:
  `logs/imagenet/ssl_freq_mae/<experiment>/best_freq-mae-ssl.pth`
- default evaluation input:
  `logs/imagenet/ssl_freq_mae/<experiment>/best_freq-mae-ssl.pth`

### Checkpoint Usage

- `best_freq-mae-ssl.pth`
  - Use this for evaluation.
  - Use this as the input checkpoint for `train_direct_spatial_decoder.py`.
  - This is the best first-stage checkpoint according to validation metrics.

- `checkpoints/last_checkpoint_norm_pix.pth`
  - Use this to resume the current normalization-enabled training run.
  - This is the preferred latest checkpoint for auto-resume.

- `checkpoints/last_checkpoint.pth`
  - This is the legacy latest checkpoint from the non-normalized version.
  - It is only used as a fallback when `last_checkpoint_norm_pix.pth` is missing.

- `best_direct_spatial_decoder.pth`
  - Use this as the best second-stage decoder checkpoint.

- `last_direct_spatial_decoder.pth`
  - Use this to resume second-stage direct spatial decoder training.

## Default Runtime Settings

- shared virtualenv: `others/venv`
- default multi-GPU pretraining launcher: `torchrun --nproc_per_node=8`
- default direct decoder launcher: `python`

Pretraining defaults:

- base learning rate: `1.5e-4`
- actual learning rate: scaled from total batch size
- batch size: `160`
- val batch size: `160`
- gradient accumulation steps: `4`
- dataloader workers per rank: `2`
- epochs: `800`
- warmup epochs: `40`
- mask ratio: `0.75`

Direct spatial decoder defaults:

- batch size: `16`
- epochs: `50`

Sweep defaults:

- ratios: `0.75`
- output root: `logs/imagenet/ssl_freq_mae_sweep`
- default pretraining GPU count: `8`

## Parameter Overrides

Common knobs to change directly on the command line:

- `--epochs`
- `--batch_size`
- `--val_batch_size`
- `--num_workers`
- `--grad_accum_steps`
- `--base_lr`
- `--data_root`
- `--train_dir`
- `--val_dir`
- `--output_dir`
- `--mask_ratio`
- `--timm_model_name`
- `--timm_checkpoint_path`

Pretrained initialization options for the ViT core:

- default automatic path: `TIMM_MODEL_NAME=vit_base_patch16_224.mae` with `TIMM_PRETRAINED=1`
- `PRETRAINED_CKPT=/path/to/mae_or_vit_checkpoint.pth`
- `TIMM_MODEL_NAME=vit_base_patch16_224 TIMM_PRETRAINED=1`
- `TIMM_MODEL_NAME=vit_base_patch16_224 TIMM_CHECKPOINT_PATH=/path/to/timm_checkpoint.pth`

These paths only partially load compatible weights:

- encoder blocks and norm
- decoder middle layers when MAE-style weights are available
- learned position embeddings when shapes match

The custom frequency input projection and reconstruction head remain task-specific.

## Verification

Quick syntax check:

```bash
python -m py_compile \
  dataset_utils.py \
  direct_spatial_decoder_model.py \
  eval_freq_mae_fixed_mask.py \
  freq_mae_model_v2.py \
  gdrive_fallback.py \
  train_direct_spatial_decoder.py \
  train_ssl_freq_mae.py \
  utils.py
```

## Notes

- The migrated branch snapshots remain under `others/`.
- The root directory intentionally avoids wrapper `.sh` scripts; invoke Python, torchrun, and sbatch directly.
- The `if __name__ == "__main__"` block at the bottom of `freq_mae_model_v2.py` is only a toy smoke test and is not the runtime default configuration.
