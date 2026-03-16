# Mini Training Verification Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify the TRELLIS.2 Stage 1 (Sparse Structure Flow Model) training pipeline works end-to-end on a mini synthetic dataset of 20 samples, running 1000 steps with visible loss decrease.

**Architecture:** Generate synthetic training data (ss_latents via pretrained encoder + condition images from example assets), create a mini training config, and launch single-GPU training. No existing code is modified.

**Tech Stack:** PyTorch 2.6.0, CUDA 12.4, TRELLIS.2 codebase (trellis2 package), pretrained SS encoder from HuggingFace.

**Spec:** `docs/superpowers/specs/2026-03-16-mini-training-verification-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `scripts/create_mini_dataset.py` | Create | Generate 20 synthetic training samples (ss_latents + condition images + metadata) |
| `configs/gen/ss_flow_img_dit_1_3B_64_bf16_mini.json` | Create | Mini training config (1000 steps, batch 2, frequent logging) |

No existing files are modified.

---

## Chunk 1: Data Generation Script + Training Config

### Task 1: Create the mini training config

**Files:**
- Create: `configs/gen/ss_flow_img_dit_1_3B_64_bf16_mini.json`
- Reference: `configs/gen/ss_flow_img_dit_1_3B_64_bf16.json`

- [ ] **Step 1: Create the mini config file**

Copy the original config and apply the parameter changes from the spec. Key differences from original:
- `max_steps`: 1000
- `batch_size_per_gpu`: 2
- `batch_split`: 1
- `i_log`: 50
- `i_sample`: 500
- `i_save`: 500
- `min_aesthetic_score`: 0.0
- `image_size`: 512 (must be preserved from original)

```json
{
    "models": {
        "denoiser": {
            "name": "SparseStructureFlowModel",
            "args": {
                "resolution": 16,
                "in_channels": 8,
                "out_channels": 8,
                "model_channels": 1536,
                "cond_channels": 1024,
                "num_blocks": 30,
                "num_heads": 12,
                "mlp_ratio": 5.3334,
                "pe_mode": "rope",
                "share_mod": true,
                "initialization": "scaled",
                "qk_rms_norm": true,
                "qk_rms_norm_cross": true
            }
        }
    },
    "dataset": {
        "name": "ImageConditionedSparseStructureLatent",
        "args": {
            "min_aesthetic_score": 0.0,
            "image_size": 512,
            "pretrained_ss_dec": "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
        }
    },
    "trainer": {
        "name": "ImageConditionedFlowMatchingCFGTrainer",
        "args": {
            "max_steps": 1000,
            "batch_size_per_gpu": 2,
            "batch_split": 1,
            "optimizer": {
                "name": "AdamW",
                "args": {
                    "lr": 1e-4,
                    "weight_decay": 0.01,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8
                }
            },
            "ema_rate": [0.9999],
            "mix_precision_mode": "amp",
            "mix_precision_dtype": "bfloat16",
            "grad_clip": {
                "name": "AdaptiveGradClipper",
                "args": {
                    "max_norm": 1.0,
                    "clip_percentile": 95
                }
            },
            "i_log": 50,
            "i_sample": 500,
            "i_save": 500,
            "p_uncond": 0.1,
            "t_schedule": {
                "name": "logitNormal",
                "args": {
                    "mean": 1.0,
                    "std": 1.0
                }
            },
            "sigma_min": 1e-5,
            "image_cond_model": {
                "name": "DinoV3FeatureExtractor",
                "args": {
                    "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                    "image_size": 512
                }
            }
        }
    }
}
```

- [ ] **Step 2: Verify config is valid JSON**

Run: `python -c "import json; json.load(open('configs/gen/ss_flow_img_dit_1_3B_64_bf16_mini.json')); print('OK')"`
Expected: `OK`

---

### Task 2: Create the mini dataset generation script

**Files:**
- Create: `scripts/create_mini_dataset.py`

- [ ] **Step 1: Create the data generation script**

The script must:
1. Load the pretrained SS encoder
2. Generate 20 random binary occupancy grids (64^3)
3. Encode each through the SS encoder → latent [8, 16, 16, 16]
4. Save as `.npz` with key `z`
5. Copy/resize example images as condition images (RGBA, 1024x1024)
6. Create `transforms.json` per sample (16 views, only `file_path` needed)
7. Create `metadata.csv` in all three required directories
8. Log latent statistics for sanity check

```python
"""
Generate a minimal synthetic dataset for TRELLIS.2 Stage 1 training verification.

Creates 20 samples with:
- ss_latent: encoded from random occupancy grids via pretrained SS encoder
- render_cond: condition images from assets/example_image/
- metadata.csv: in all required directories

Usage:
    python scripts/create_mini_dataset.py
"""

import os
import sys
import json
import hashlib
import shutil
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

NUM_SAMPLES = 20
DATASET_DIR = PROJECT_ROOT / "datasets" / "mini_test"
EXAMPLE_IMAGE_DIR = PROJECT_ROOT / "assets" / "example_image"
SS_LATENT_SUBDIR = "ss_latents/ss_enc_conv3d_16l8_fp16_64"
RENDER_COND_SUBDIR = "renders_cond"
NUM_VIEWS = 16
IMAGE_SIZE = 1024


def generate_sha256_ids(n):
    """Generate n deterministic sha256-like IDs."""
    ids = []
    for i in range(n):
        h = hashlib.sha256(f"mini_test_sample_{i}".encode()).hexdigest()
        ids.append(h)
    return ids


def generate_ss_latents(sha256_ids, output_dir):
    """Encode random occupancy grids through pretrained SS encoder."""
    from trellis2 import models

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pretrained SS encoder...")
    encoder = models.from_pretrained(
        "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"
    )
    encoder = encoder.eval().cuda()

    all_latents = []
    print(f"Generating {len(sha256_ids)} ss_latents...")
    with torch.no_grad():
        for i, sha256 in enumerate(sha256_ids):
            # Random binary occupancy grid with ~10-30% fill rate
            fill_rate = 0.1 + 0.2 * (i / len(sha256_ids))
            x = (torch.rand(1, 1, 64, 64, 64) < fill_rate).float().cuda()
            z = encoder(x, sample_posterior=False)
            z_np = z[0].cpu().numpy().astype(np.float32)
            all_latents.append(z_np)

            save_path = output_dir / f"{sha256}.npz"
            np.savez_compressed(str(save_path), z=z_np)

            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{len(sha256_ids)}] done")

    # Log statistics
    all_z = np.stack(all_latents)
    print(f"\nLatent statistics:")
    print(f"  Shape: {all_z.shape}")
    print(f"  Mean:  {all_z.mean():.4f}")
    print(f"  Std:   {all_z.std():.4f}")
    print(f"  Min:   {all_z.min():.4f}")
    print(f"  Max:   {all_z.max():.4f}")

    del encoder
    torch.cuda.empty_cache()


def generate_render_cond(sha256_ids, output_dir):
    """Create condition images from example images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect source images
    source_images = sorted(EXAMPLE_IMAGE_DIR.glob("*.webp")) + sorted(
        EXAMPLE_IMAGE_DIR.glob("*.png")
    )
    if not source_images:
        raise RuntimeError(f"No images found in {EXAMPLE_IMAGE_DIR}")
    print(f"Found {len(source_images)} source images")

    for i, sha256 in enumerate(sha256_ids):
        sample_dir = output_dir / sha256
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Pick a source image (cycle through available ones)
        src_path = source_images[i % len(source_images)]
        img = Image.open(src_path)

        # Ensure RGBA
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        # Resize to target size
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)

        # Save NUM_VIEWS copies (same image, different "views" for simplicity)
        frames = []
        for v in range(NUM_VIEWS):
            filename = f"{v:03d}.png"
            img.save(sample_dir / filename)
            frames.append({"file_path": filename})

        # Write transforms.json
        transforms = {"frames": frames}
        with open(sample_dir / "transforms.json", "w") as f:
            json.dump(transforms, f, indent=2)

    print(f"Generated condition images for {len(sha256_ids)} samples")


def create_metadata(sha256_ids, dataset_dir, ss_latent_dir, render_cond_dir):
    """Create metadata.csv in all three required directories."""
    records = []
    for sha256 in sha256_ids:
        records.append(
            {
                "sha256": sha256,
                "aesthetic_score": 5.0,
                "ss_latent_encoded": True,
                "cond_rendered": True,
            }
        )
    df = pd.DataFrame(records)

    # Save to all three locations
    for d in [dataset_dir, ss_latent_dir, render_cond_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
        csv_path = Path(d) / "metadata.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved metadata.csv to {csv_path}")


def main():
    print("=" * 60)
    print("TRELLIS.2 Mini Dataset Generator")
    print("=" * 60)

    # Pre-flight checks
    assert EXAMPLE_IMAGE_DIR.exists(), f"Example images not found: {EXAMPLE_IMAGE_DIR}"

    ss_latent_dir = DATASET_DIR / SS_LATENT_SUBDIR
    render_cond_dir = DATASET_DIR / RENDER_COND_SUBDIR

    # Generate deterministic IDs
    sha256_ids = generate_sha256_ids(NUM_SAMPLES)
    print(f"\nGenerating {NUM_SAMPLES} samples...")
    print(f"Dataset dir: {DATASET_DIR}")

    # Step 1: Generate ss_latents
    print("\n--- Step 1: Generating SS Latents ---")
    generate_ss_latents(sha256_ids, ss_latent_dir)

    # Step 2: Generate condition images
    print("\n--- Step 2: Generating Condition Images ---")
    generate_render_cond(sha256_ids, render_cond_dir)

    # Step 3: Create metadata
    print("\n--- Step 3: Creating Metadata ---")
    create_metadata(sha256_ids, DATASET_DIR, ss_latent_dir, render_cond_dir)

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print(f"Output: {DATASET_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the data generation script**

Run:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
.venv/bin/python scripts/create_mini_dataset.py
```

Expected output:
- 20 `.npz` files in `datasets/mini_test/ss_latents/ss_enc_conv3d_16l8_fp16_64/`
- 20 directories in `datasets/mini_test/renders_cond/`, each with 16 PNG images + transforms.json
- `metadata.csv` in 3 locations
- Latent statistics printed (mean ~0, std ~2, finite min/max)

- [ ] **Step 3: Verify dataset structure**

Run:
```bash
# Check file counts
echo "=== ss_latents ===" && ls datasets/mini_test/ss_latents/ss_enc_conv3d_16l8_fp16_64/*.npz | wc -l
echo "=== render_cond dirs ===" && ls -d datasets/mini_test/renders_cond/*/ | wc -l
echo "=== metadata files ===" && find datasets/mini_test -name "metadata.csv" | sort
echo "=== sample transforms ===" && cat datasets/mini_test/renders_cond/$(ls datasets/mini_test/renders_cond/ | head -1)/transforms.json | head -5
```

Expected:
- 20 `.npz` files
- 20 render_cond directories
- 3 metadata.csv files
- transforms.json with `frames` array

---

### Task 3: Launch training and verify

**Files:**
- Reference: `train.py`
- Reference: `configs/gen/ss_flow_img_dit_1_3B_64_bf16_mini.json`

- [ ] **Step 1: Launch training (single GPU)**

Run:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
.venv/bin/python train.py \
  --config configs/gen/ss_flow_img_dit_1_3B_64_bf16_mini.json \
  --output_dir results/ss_flow_mini_test \
  --data_dir '{"mini_test": {"base": "datasets/mini_test", "ss_latent": "datasets/mini_test/ss_latents/ss_enc_conv3d_16l8_fp16_64", "render_cond": "datasets/mini_test/renders_cond"}}' \
  --num_gpus 1
```

Expected:
- Dataset loads successfully, prints sample count (~20)
- Model summary prints (~1.3B parameters)
- Training loop starts, logging loss every 50 steps
- Checkpoint saves at step 500
- Training completes at step 1000

- [ ] **Step 2: Verify loss trend**

Run:
```bash
# Check training log for loss values
grep "loss" results/ss_flow_mini_test/log.txt | head -20
```

Or launch TensorBoard:
```bash
tensorboard --logdir results/ss_flow_mini_test --port 6006
```

Expected: Loss decreases over the 1000 steps (early loss ~1.0+, later loss should be noticeably lower).

- [ ] **Step 3: Verify checkpoint exists**

Run:
```bash
ls -la results/ss_flow_mini_test/ckpts/
```

Expected: Checkpoint files at step 500 and 1000 (or `latest`).

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `metadata.csv not found` | metadata.csv missing from one of the 3 dirs | Re-run `create_mini_dataset.py` |
| `No module named 'trellis2'` | Wrong Python interpreter | Use `.venv/bin/python` |
| `CUDA out of memory` | 1.3B model + DinoV3 too large | Reduce `batch_size_per_gpu` to 1 in config |
| `DinoV3 model not found` | Model not downloaded | Run inference example first to trigger download |
| Loss is NaN | Degenerate latents | Check latent stats from data generation; regenerate with different random seed |
| `ss_latent_encoded` filter removes all samples | metadata.csv column type issue | Ensure `ss_latent_encoded` is boolean `True`, not string |
