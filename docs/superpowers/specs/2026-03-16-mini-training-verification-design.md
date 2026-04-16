# Mini Dataset Training Verification for TRELLIS.2 Stage 1

## Goal

Verify that the TRELLIS.2 training pipeline works end-to-end by training the **Sparse Structure Flow Model (Stage 1)** on a minimal synthetic dataset of 20 samples, running for 1000 steps, and confirming loss decreases.

## Scope

- **In scope**: mini dataset generation, training config, training launch, loss verification
- **Out of scope**: data_toolkit pipeline fixes, multi-GPU verification, real data training, other training stages

## Data Design

### Target directory structure

```
datasets/mini_test/
├── metadata.csv
├── ss_latents/
│   └── ss_enc_conv3d_16l8_fp16_64/
│       ├── metadata.csv
│       └── {sha256}.npz          # z: float32 [8, 16, 16, 16]
└── renders_cond/
    ├── metadata.csv
    └── {sha256}/
        ├── transforms.json
        └── 000.png ~ 015.png     # 1024×1024 RGBA
```

### Data generation approach

1. **ss_latent**: Use pretrained SS encoder (`microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16`) to encode 20 random binary occupancy grids into latent tensors of shape `[8, 16, 16, 16]`. The encoder expects a float tensor of shape `[1, 1, 64, 64, 64]` (batch, channel, D, H, W) with binary values (0/1). After encoding, log statistics (mean, std, min, max) of generated latents to verify they are in a reasonable range.
2. **render_cond**: Use images from `assets/example_image/`, resize to 1024×1024 RGBA (add full-white alpha channel if source is RGB). Construct `transforms.json` with `{"frames": [{"file_path": "000.png"}, ...]}` — only `file_path` is consumed by the Stage 1 training code; camera parameters (yaw, pitch, etc.) are not used.
3. **metadata.csv**: Each of the three directories (`base`, `ss_latent`, `render_cond`) must contain its own `metadata.csv`, as `StandardDatasetBase` reads and merges metadata from all sub-paths via `combine_first`. The simplest approach: create one master CSV with columns `sha256`, `aesthetic_score` (5.0), `ss_latent_encoded` (True), `cond_rendered` (True), and copy it to all three locations.

### Sample count: 20

Enough to see loss trends, small enough to generate quickly.

## Training Config

New file: `configs/gen/ss_flow_img_dit_1_3B_64_bf16_mini.json`

Based on `ss_flow_img_dit_1_3B_64_bf16.json` with these changes:

| Parameter | Original | Mini | Reason |
|-----------|----------|------|--------|
| `max_steps` | 1,000,000 | 1,000 | Only need loss trend |
| `batch_size_per_gpu` | 8 | 2 | 20 samples |
| `batch_split` | 4 | 1 | No gradient accumulation needed |
| `i_log` | 500 | 50 | More frequent logging |
| `i_sample` | 10,000 | 500 | Earlier sampling |
| `i_save` | 10,000 | 500 | More frequent checkpoints |
| `min_aesthetic_score` | 4.5 | 0.0 | Include all samples |

Model architecture (1.3B params), optimizer, EMA, gradient clipping, mixed precision, DinoV3 conditioning — all unchanged. The `image_size: 512` in the dataset args must be preserved (the `ImageConditionedMixin` default is 518, which would mismatch the DinoV3 extractor's expected input).

## Implementation Plan

### Files to create

| File | Purpose |
|------|---------|
| `scripts/create_mini_dataset.py` | Generate synthetic mini dataset |
| `configs/gen/ss_flow_img_dit_1_3B_64_bf16_mini.json` | Training config |

### No existing files modified.

### Training command

```bash
python train.py \
  --config configs/gen/ss_flow_img_dit_1_3B_64_bf16_mini.json \
  --output_dir results/ss_flow_mini_test \
  --data_dir '{"mini_test": {"base": "datasets/mini_test", "ss_latent": "datasets/mini_test/ss_latents/ss_enc_conv3d_16l8_fp16_64", "render_cond": "datasets/mini_test/renders_cond"}}' \
  --num_gpus 1
```

## Success Criteria

1. Training starts without errors
2. Loss shows clear downward trend within 1000 steps (visible in TensorBoard)
3. Checkpoints save and load correctly

## Risks

- **Random occupancy grids may produce degenerate latents**: Mitigated by using the pretrained encoder which normalizes output distribution.
- **20 samples will overfit rapidly**: Expected and acceptable — we only need loss to decrease, not generalization.
- **DinoV3 model loading may fail**: It's already downloaded in `pretrained/dinov3-vitl16-pretrain-lvd1689m/`, so low risk. The data generation script should include a pre-flight check to verify this.
- **Latent distribution out of range**: Random occupancy grids may produce latents with unusual statistics. The data generation script logs latent stats to catch this early.
