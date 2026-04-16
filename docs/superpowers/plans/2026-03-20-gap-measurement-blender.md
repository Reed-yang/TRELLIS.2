# Gap Measurement with Training-Matched Conditioning Images

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run the Gap Measurement experiment using Blender CYCLES-rendered conditioning images (identical distribution to DiT training data) instead of normal map placeholders, to get a scientifically valid comparison.

**Architecture:** Install Blender 3.0, write a lightweight wrapper that calls the existing `data_toolkit/blender_script/render_cond.py` for each pilot mesh, then re-run `scripts/gap_measurement.py` with the new images. Output to `experiments/gap_measurement_blender/` to preserve the original results for comparison.

**Tech Stack:** Blender 3.0.1 (CYCLES engine, CUDA GPU), existing `data_toolkit/` rendering scripts, `scripts/gap_measurement.py`

**Prior results:** `experiments/gap_measurement/results/summary.md` — used normal-map images, DiT CD/VAE CD ratio = 176x

---

## File Structure

```
scripts/
└── render_blender_cond.py     # NEW: Wrapper to render Blender conditioning images for pilot meshes

experiments/gap_measurement_blender/   # NEW: Output directory (gitignored)
├── renders_cond/                      # Blender-rendered RGBA images per model
│   └── {uid}/
│       ├── 000.png ... 015.png        # 16 multi-view RGBA renders
│       └── transforms.json            # Camera parameters
├── pilot_data/
│   └── manifest.json                  # Updated manifest with Blender image paths
└── results/
    ├── per_sample.csv
    └── summary.md
```

---

## Key Technical Context

### How training conditioning images are produced

`data_toolkit/render_cond.py` calls Blender in batch mode:
```
blender -b -P data_toolkit/blender_script/render_cond.py -- \
    --object model.glb \
    --cond_views '[{"yaw":..., "pitch":..., "radius":..., "fov":...}]' \
    --cond_resolution 1024 \
    --cond_output_folder renders_cond/{sha256}/ \
    --engine CYCLES
```

Blender script does: load object → normalize scene (fit to unit cube) → random CYCLES lighting → render RGBA PNGs at 1024×1024 → save camera params in `transforms.json`.

Camera distribution: `sphere_hammersley_sequence` for yaw/pitch, random FOV in [10°, 70°], radius computed from FOV. 16 views per object.

### How pipeline consumes conditioning images

`pipeline.run(image)` calls `pipeline.preprocess_image(image)`:
1. Remove background (if not already RGBA with alpha)
2. Crop to object bounding box
3. Resize, premultiply `rgb * alpha`

Since Blender renders have transparent backgrounds, the pipeline will correctly handle them.

### What the wrapper needs to do

For each pilot mesh:
1. Call Blender subprocess with the same parameters as `_render_cond()`
2. Pick one rendered view (e.g., view 0) as the DiT conditioning image
3. Update manifest.json to point to the new Blender image paths

### Important: Mesh file paths

The pilot meshes are symlinks from `experiments/gap_measurement/pilot_data/meshes/{uid}.glb` → Objaverse cache. Blender needs **the actual `.glb` file** (follows symlinks automatically).

---

### Task 1: Install Blender 3.0

- [ ] **Step 1: Check if Blender 3.0 is already installed**

Run:
```bash
ls /tmp/blender-3.0.1-linux-x64/blender 2>/dev/null && echo "EXISTS" || echo "NOT FOUND"
```

- [ ] **Step 2: Install system dependencies and download Blender**

Run:
```bash
sudo apt-get update && sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6 libxfixes3 libgl1
wget -q https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz -P /tmp
tar -xf /tmp/blender-3.0.1-linux-x64.tar.xz -C /tmp
```
Expected: Blender binary at `/tmp/blender-3.0.1-linux-x64/blender`

- [ ] **Step 3: Verify Blender runs headless**

Run:
```bash
/tmp/blender-3.0.1-linux-x64/blender -b --version 2>&1 | head -3
```
Expected: `Blender 3.0.1`

---

### Task 2: Write Blender Rendering Wrapper

**Files:**
- Create: `scripts/render_blender_cond.py`

- [ ] **Step 1: Create the rendering wrapper**

```python
# scripts/render_blender_cond.py
"""
Render Blender CYCLES conditioning images for pilot meshes.
Calls the existing data_toolkit/blender_script/render_cond.py via Blender subprocess.
Produces images identical to the DiT training distribution.
"""

import os
import sys
import json
import argparse
import numpy as np
from subprocess import call, DEVNULL
from tqdm import tqdm
from PIL import Image

# Blender path (same as data_toolkit/render_cond.py)
BLENDER_PATH = '/tmp/blender-3.0.1-linux-x64/blender'
BLENDER_SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'data_toolkit', 'blender_script', 'render_cond.py')

# Camera distribution functions (copied from data_toolkit/utils.py to avoid import issues)
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val

def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + [radical_inverse(PRIMES[i], n) for i in range(dim - 1)]

def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]


def build_cond_views(num_views=16):
    """Build camera view parameters matching training distribution."""
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)

    fov_min, fov_max = 10, 70
    radius_min = np.sqrt(3) / 2 / np.sin(fov_max / 360 * np.pi)
    radius_max = np.sqrt(3) / 2 / np.sin(fov_min / 360 * np.pi)
    k_min = 1 / radius_max**2
    k_max = 1 / radius_min**2
    ks = np.random.uniform(k_min, k_max, (1000000,))
    radius = [1 / np.sqrt(k) for k in ks]
    fov = [2 * np.arcsin(np.sqrt(3) / 2 / r) for r in radius]

    return [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f}
            for y, p, r, f in zip(yaws, pitchs, radius, fov)]


def render_single_mesh(mesh_path, uid, output_root, num_views=16, resolution=1024):
    """
    Render conditioning images for a single mesh using Blender CYCLES.

    Args:
        mesh_path: path to .glb/.obj mesh file
        uid: unique identifier for this mesh
        output_root: root directory for renders_cond/
        num_views: number of views to render (default 16, matches training)
        resolution: image resolution (default 1024, matches training)

    Returns:
        True if successful, False otherwise
    """
    output_folder = os.path.join(output_root, 'renders_cond', uid)
    os.makedirs(output_folder, exist_ok=True)

    # Skip if already rendered
    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        return True

    # Resolve symlinks
    mesh_path = os.path.realpath(mesh_path)

    # Build camera views
    cond_views = build_cond_views(num_views)

    args = [
        BLENDER_PATH, '-b', '-P', os.path.realpath(BLENDER_SCRIPT),
        '--',
        '--object', mesh_path,
        '--cond_views', json.dumps(cond_views),
        '--cond_resolution', str(resolution),
        '--cond_output_folder', output_folder,
        '--engine', 'CYCLES',
    ]

    ret = call(args, stdout=DEVNULL, stderr=DEVNULL)
    return os.path.exists(os.path.join(output_folder, 'transforms.json'))


def pick_best_view(renders_dir, uid):
    """
    Pick the first rendered view as conditioning image.
    Returns path to the selected PNG, or None if not found.
    """
    view_path = os.path.join(renders_dir, uid, '000.png')
    if os.path.exists(view_path):
        return view_path
    return None


def main():
    parser = argparse.ArgumentParser(description="Render Blender conditioning images for pilot meshes")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to pilot data manifest.json")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/gap_measurement_blender",
                        help="Output directory for experiment")
    parser.add_argument("--num_views", type=int, default=16,
                        help="Number of views per model (default: 16, matches training)")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Render resolution (default: 1024, matches training)")
    args = parser.parse_args()

    # Verify Blender is installed
    if not os.path.exists(BLENDER_PATH):
        print(f"ERROR: Blender not found at {BLENDER_PATH}")
        print("Install with: data_toolkit/render_cond.py's _install_blender() or manually")
        sys.exit(1)

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)

    renders_dir = os.path.join(args.output_dir, 'renders_cond')
    os.makedirs(renders_dir, exist_ok=True)

    # Render each mesh
    successes = 0
    failures = []
    for item in tqdm(manifest, desc="Rendering with Blender"):
        uid = item['uid']
        mesh_path = item['mesh_path']

        ok = render_single_mesh(mesh_path, uid, args.output_dir,
                                num_views=args.num_views,
                                resolution=args.resolution)
        if ok:
            successes += 1
        else:
            failures.append(uid)
            print(f"  FAILED: {uid}")

    print(f"\nRendering complete: {successes}/{len(manifest)} succeeded")
    if failures:
        print(f"Failed UIDs: {failures}")

    # Create updated manifest with Blender image paths
    updated_manifest = []
    for item in manifest:
        uid = item['uid']
        blender_image = pick_best_view(renders_dir, uid)
        if blender_image:
            updated_item = dict(item)
            updated_item['image_path'] = os.path.abspath(blender_image)
            updated_item['image_source'] = 'blender_cycles'
            updated_manifest.append(updated_item)
        else:
            print(f"  Skipping {uid}: no rendered image")

    # Save updated manifest
    manifest_dir = os.path.join(args.output_dir, 'pilot_data')
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(updated_manifest, f, indent=2)

    print(f"Updated manifest: {manifest_path} ({len(updated_manifest)} models)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

Run: `.venv/bin/python -c "import ast; ast.parse(open('scripts/render_blender_cond.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/render_blender_cond.py
git commit -m "feat: add Blender CYCLES conditioning image renderer for gap measurement"
```

---

### Task 3: Render Conditioning Images

- [ ] **Step 1: Run Blender rendering for all 30 pilot meshes**

Run:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/render_blender_cond.py \
    --manifest experiments/gap_measurement/pilot_data/manifest.json \
    --output_dir experiments/gap_measurement_blender \
    --num_views 16 --resolution 1024
```
Expected: ~30 models × ~30s each ≈ ~15 minutes. Each model produces 16 RGBA PNGs at 1024×1024 in `experiments/gap_measurement_blender/renders_cond/{uid}/`.

Note: If Blender CYCLES GPU rendering fails (CUDA unavailable to Blender), it will fall back to CPU rendering (slower but still works). Check output for "CUDA" errors.

- [ ] **Step 2: Verify rendered images**

Run:
```bash
# Check one rendered image
ls experiments/gap_measurement_blender/renders_cond/ | head -3
ls experiments/gap_measurement_blender/renders_cond/$(ls experiments/gap_measurement_blender/renders_cond/ | head -1)/
# Verify it's a proper RGBA image
.venv/bin/python -c "
from PIL import Image
import os, glob
imgs = glob.glob('experiments/gap_measurement_blender/renders_cond/*/000.png')
img = Image.open(imgs[0])
print(f'Image: {imgs[0]}')
print(f'Size: {img.size}, Mode: {img.mode}')
print(f'Has alpha: {\"A\" in img.mode}')
"
```
Expected: 1024×1024 RGBA PNG images

- [ ] **Step 3: Verify updated manifest**

Run:
```bash
.venv/bin/python -c "
import json
m = json.load(open('experiments/gap_measurement_blender/pilot_data/manifest.json'))
print(f'Models: {len(m)}')
print(f'First image: {m[0][\"image_path\"]}')
print(f'Image source: {m[0].get(\"image_source\", \"unknown\")}')
"
```
Expected: `blender_cycles` source, paths pointing to rendered PNGs

---

### Task 4: Re-run Gap Measurement with Blender Images

- [ ] **Step 1: Run full gap measurement with Blender-rendered images**

Run:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/gap_measurement.py \
    --manifest experiments/gap_measurement_blender/pilot_data/manifest.json \
    --output_dir experiments/gap_measurement_blender/results \
    --grid_size 512
```
Expected: ~30 models, Path A + Path B. ~30-45 minutes total.

Note: The `gap_measurement.py` script checks for placeholder images (`pixels.std() < 5`) before Path B. Blender images will NOT be detected as placeholders (they have proper content), so they will be used directly.

- [ ] **Step 2: Review results**

Run:
```bash
cat experiments/gap_measurement_blender/results/summary.md
```

- [ ] **Step 3: Compare with normal-map experiment**

Run:
```bash
echo "=== ORIGINAL (normal map placeholders) ==="
cat experiments/gap_measurement/results/summary.md
echo ""
echo "=== BLENDER CYCLES (training-matched) ==="
cat experiments/gap_measurement_blender/results/summary.md
```

---

### Task 5: Final Summary and Commit

- [ ] **Step 1: Commit all scripts**

```bash
git add scripts/render_blender_cond.py
git commit -m "feat: complete gap measurement with Blender CYCLES conditioning images"
```

- [ ] **Step 2: Review the decision**

Compare DiT CD / VAE CD ratios between experiments:
- If Blender experiment shows **much smaller gap** than normal-map experiment → confirms the normal-map experiment was unfair, gap is smaller than initially measured
- If Blender experiment shows **similar gap** → confirms DiT is genuinely the bottleneck regardless of input image quality
- If Blender experiment shows **gap < 2x** → VAE is the ceiling per roadmap-v2.md decision rule, pivot to SC-VAE improvements

---

## Troubleshooting

**Blender download fails:**
```bash
# Alternative mirror
wget -q https://mirrors.ocf.berkeley.edu/blender/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz -P /tmp
```

**Blender CYCLES crashes on GPU:**
- Check `CUDA_VISIBLE_DEVICES` is set
- Blender CYCLES uses its own CUDA context, separate from PyTorch
- If GPU fails, remove `--engine CYCLES` or set `bpy.context.scene.cycles.device = 'CPU'` (slower but works)

**Blender fails to import .glb:**
- Some Objaverse .glb files have incompatible features
- Failures are logged and skipped; remaining models still get evaluated

**Pipeline `preprocess_image` strips alpha:**
- Blender renders have transparent background (RGBA)
- `preprocess_image` uses alpha channel directly when present, no background removal needed
- This matches the training pipeline exactly
