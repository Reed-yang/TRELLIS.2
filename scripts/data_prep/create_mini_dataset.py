"""
Generate a minimal synthetic dataset for TRELLIS.2 Stage 1 training verification.

Creates 20 samples with:
- ss_latent: encoded from random occupancy grids via pretrained SS encoder
- render_cond: condition images from assets/example_image/
- metadata.csv: in all required directories

Usage:
    python scripts/data_prep/create_mini_dataset.py
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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
