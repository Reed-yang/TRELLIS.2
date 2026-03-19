# Gap Measurement Pipeline Design

> Phase 0 of TRELLIS.2 improvement roadmap: quantify the quality gap between SC-VAE reconstruction upper bound and DiT generation quality, to determine whether to prioritize VAE or DiT improvements.

## Context

TRELLIS.2's core bet is that SC-VAE reconstruction fidelity is the system ceiling. Before investing in any SC-VAE improvement (Phase 1+), we must verify this by measuring:

1. **VAE Upper Bound**: GT mesh → SC-VAE encode → decode → reconstructed mesh (best possible quality)
2. **DiT Generation**: test image → full pipeline → generated mesh (actual generation quality)
3. **Gap** = (2) - (1), quantifying how much DiT falls short of VAE's potential

**Decision rule** (from roadmap-v2.md):
- Gap small → prioritize SC-VAE improvements
- Gap large (DiT < VAE by >50%) → prioritize DiT optimization
- Gap varies by region → region-specific strategy

## Architecture

### Two Parallel Tracks

**Track 1: Pilot Evaluation Pipeline** (immediate, ~1 week)

Build the full evaluation pipeline using a small set of accessible 3D models, bypassing the missing `data_toolkit/datasets/` module entirely.

```
Pilot mesh (.glb/.obj via trimesh)
    │
    ├── Path A: VAE Reconstruction Upper Bound
    │   trimesh → vertices + faces (torch tensors)
    │     → o_voxel.convert.mesh_to_flexible_dual_grid()
    │     → SC-VAE encoder (from HF Hub: microsoft/TRELLIS.2-4B/ckpts/shape_enc_*)
    │     → SC-VAE decoder (from HF Hub: microsoft/TRELLIS.2-4B/ckpts/shape_dec_*)
    │     → decoder.set_resolution(512) → flexible_dual_grid_to_mesh()
    │     → reconstructed mesh
    │
    ├── Path B: DiT Generation Quality
    │   Render or provide test image of the same object
    │     → Trellis2ImageTo3DPipeline.from_pretrained()
    │     → pipeline.run(image, pipeline_type='512') → List[MeshWithVoxel]
    │     → generated mesh
    │
    └── Evaluation: both meshes vs GT mesh
        → Geometric: CD, F-score (τ=0.01), Normal Consistency
        → Rendering: PSNR, SSIM on rendered normal maps
        → Output: per-sample CSV + aggregate statistics
```

**Track 2: Standard Test Set Acquisition** (async, best-effort)

Research and obtain the paper's standard test sets (Toys4K-PBR 473 instances + Sketchfab Featured 90 instances) for reproducible comparison.

### Evaluation Metrics Implementation

**Approach: trimesh + torch (zero new dependencies)**

All metrics computed from point clouds sampled from mesh surfaces.

| Metric | Method | Reference |
|--------|--------|-----------|
| **Chamfer Distance (CD)** | trimesh.sample.sample_surface() → 10K points per mesh, `torch.cdist` on GPU with chunking | Standard |
| **F-score** (τ=0.01) | From CD computation: fraction of points within threshold τ in normalized [-0.5, 0.5] space | Standard |
| **Normal Consistency (NC)** | Sample points + normals from both meshes, match nearest points, compute mean cosine similarity | Standard |
| **PSNR** | Render normal maps from multiple views via NVDiffRast, compute pixel-level PSNR | loss_utils.py |
| **SSIM** | Same rendered normal maps, compute SSIM (expects [B,C,H,W] tensor) | loss_utils.py |

**Not included in pilot** (defer to later):
- Sharp Normal Error (SNE) — requires Dora-Bench implementation, add after pipeline is validated
- CLIP/ULIP-2/Uni3D multimodal scores — not needed for gap measurement

### Pilot Data

**Source**: Objaverse via `objaverse` Python package.

**Selection criteria**:
- 20-50 models spanning different complexity levels (simple geometric, medium organic, complex detailed)
- Must have clean manifold geometry (trimesh.is_watertight or reasonable quality)
- Need corresponding reference images (render from GT mesh, or use existing Objaverse thumbnails)

**Alternative**: If `objaverse` package is not installed or download is slow, use publicly available .glb/.obj files from any source.

### Resolution

Both paths use **512³** O-Voxel resolution for a fair comparison:
- Path A: `decoder.set_resolution(512)` before decoding
- Path B: `pipeline.run(image, pipeline_type='512')` to use the 512-resolution DiT

This ensures the gap measurement is not confounded by resolution differences. 512 also balances quality and speed for the pilot.

## File Structure

```
scripts/
├── eval_metrics.py           # Evaluation metrics module (CD, F-score, NC)
├── gap_measurement.py        # Main gap measurement script
└── prepare_pilot_data.py     # Download and prepare pilot test data
```

**Output directory**: `experiments/gap_measurement/`
```
experiments/gap_measurement/
├── pilot_data/               # Downloaded test meshes + rendered images
│   ├── meshes/               # GT meshes (.glb/.obj)
│   └── images/               # Test images for DiT path
├── results/
│   ├── vae_reconstructions/  # Path A output meshes
│   ├── dit_generations/      # Path B output meshes
│   ├── per_sample.csv        # Per-sample metrics
│   └── summary.md            # Aggregate gap analysis
```

## Key Technical Details

### Prerequisites: Download Encoder Weights

The encoder is NOT included in the local pretrained cache (only decoder is cached). It must be downloaded from HuggingFace Hub before running Path A:

```python
from trellis2.models import from_pretrained
# This will auto-download from HF Hub on first call:
encoder = from_pretrained("microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16")
```

Alternatively, pre-download:
```bash
huggingface-cli download microsoft/TRELLIS.2-4B \
    ckpts/shape_enc_next_dc_f16c32_fp16.json \
    ckpts/shape_enc_next_dc_f16c32_fp16.safetensors
```

### Bypassing data_toolkit

The pilot pipeline does NOT use `data_toolkit/dual_grid.py` main entry (which depends on missing `datasets` module). Instead, it directly calls:

```python
import trimesh
import torch
import o_voxel

# Load mesh
mesh = trimesh.load("model.glb", force="mesh")
vertices = torch.from_numpy(mesh.vertices).float()
faces = torch.from_numpy(mesh.faces).long()

# Normalize to [-0.5, 0.5] (same logic as dual_grid.py:50-55)
center = (vertices.min(0)[0] + vertices.max(0)[0]) / 2
scale = 0.99999 / (vertices.max(0)[0] - vertices.min(0)[0]).max()
vertices = (vertices - center) * scale

# Convert to O-Voxel
voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
    vertices=vertices, faces=faces,
    grid_size=512,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2,
)
```

### SC-VAE Encode-Decode (Path A)

```python
from trellis2.models import from_pretrained
from trellis2.modules.sparse import SparseTensor

# Load pretrained encoder + decoder (encoder downloads from HF Hub if not cached)
encoder = from_pretrained("microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16").eval().cuda()
decoder = from_pretrained("microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16").eval().cuda()

# IMPORTANT: set decoder resolution to match O-Voxel grid_size
decoder.set_resolution(512)

# Prepare O-Voxel output for encoder input
# dual_vertices: quantize to uint8 then normalize (same as dual_grid.py:67-70 + encode_shape_latent.py:127-130)
dual_vertices_quantized = torch.clamp(dual_vertices * 512 - voxel_indices, 0, 1)  # normalize to [0, 1]
dual_vertices_quantized = (dual_vertices_quantized * 255).to(torch.uint8)

# Build SparseTensor for encoder
# coords need batch dim prepended: [batch_idx, x, y, z]
coords_with_batch = torch.cat([torch.zeros(voxel_indices.shape[0], 1, dtype=torch.int), voxel_indices], dim=-1)
vertices_st = SparseTensor(
    feats=(dual_vertices_quantized / 255.0).float(),
    coords=coords_with_batch,
)

# intersected: o_voxel.convert returns [N, 3] bool tensor directly
# (no pack/unpack needed since we skip .vxz serialization)
intersected_st = vertices_st.replace(intersected.bool().float())

# Encode → latent → decode → mesh
with torch.no_grad():
    z = encoder(vertices_st.cuda(), intersected_st.cuda())
    reconstructed_meshes = decoder(z)  # returns List[Mesh] in eval mode
    # Each Mesh has .vertices (CUDA tensor) and .faces (CUDA tensor)
    recon_mesh = reconstructed_meshes[0]  # first (and only) mesh in batch
```

### DiT Generation (Path B)

```python
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from PIL import Image

pipeline = Trellis2ImageTo3DPipeline.from_pretrained("pretrained/TRELLIS.2-4B")
pipeline.to("cuda")

image = Image.open("test_image.png")

# run() returns List[MeshWithVoxel], NOT a dict
# pipeline_type='512' ensures 512³ resolution to match Path A
meshes = pipeline.run(image, pipeline_type='512')
generated_mesh = meshes[0]  # MeshWithVoxel inherits from Mesh
```

### Mesh to Point Cloud for Evaluation

Both paths produce `Mesh` objects with CUDA tensors. To evaluate:

```python
import trimesh as tm
import numpy as np

def mesh_to_trimesh(trellis_mesh):
    """Convert trellis2 Mesh to trimesh for point sampling."""
    return tm.Trimesh(
        vertices=trellis_mesh.vertices.cpu().numpy(),
        faces=trellis_mesh.faces.cpu().numpy(),
    )

def sample_points_and_normals(trimesh_mesh, num_points=10000):
    """Sample points and face normals from mesh surface."""
    points, face_indices = tm.sample.sample_surface(trimesh_mesh, num_points)
    normals = trimesh_mesh.face_normals[face_indices]
    return torch.from_numpy(points).float(), torch.from_numpy(normals).float()
```

### Chamfer Distance Implementation

```python
def chamfer_distance(points1, points2, chunk_size=2048):
    """
    Compute bidirectional Chamfer Distance using torch.cdist with chunking.
    points1, points2: [N, 3] and [M, 3] torch tensors on GPU.
    Returns: mean bidirectional CD (scalar).
    """
    # Forward: min dist from points1 to points2
    min_dists_1 = []
    for i in range(0, len(points1), chunk_size):
        chunk = points1[i:i+chunk_size]
        dists = torch.cdist(chunk, points2)  # [chunk, M]
        min_dists_1.append(dists.min(dim=1)[0])
    min_dists_1 = torch.cat(min_dists_1)

    # Backward: min dist from points2 to points1
    min_dists_2 = []
    for i in range(0, len(points2), chunk_size):
        chunk = points2[i:i+chunk_size]
        dists = torch.cdist(chunk, points1)  # [chunk, N]
        min_dists_2.append(dists.min(dim=1)[0])
    min_dists_2 = torch.cat(min_dists_2)

    return (min_dists_1.mean() + min_dists_2.mean()) / 2

def f_score(points1, points2, threshold=0.01, chunk_size=2048):
    """
    Compute F-score: harmonic mean of precision and recall at distance threshold.
    threshold=0.01 in [-0.5, 0.5] normalized space = 1% of model extent.
    """
    # Precision: fraction of points1 within threshold of points2
    min_dists_1 = []
    for i in range(0, len(points1), chunk_size):
        chunk = points1[i:i+chunk_size]
        dists = torch.cdist(chunk, points2)
        min_dists_1.append(dists.min(dim=1)[0])
    min_dists_1 = torch.cat(min_dists_1)
    precision = (min_dists_1 < threshold).float().mean()

    # Recall: fraction of points2 within threshold of points1
    min_dists_2 = []
    for i in range(0, len(points2), chunk_size):
        chunk = points2[i:i+chunk_size]
        dists = torch.cdist(chunk, points1)
        min_dists_2.append(dists.min(dim=1)[0])
    min_dists_2 = torch.cat(min_dists_2)
    recall = (min_dists_2 < threshold).float().mean()

    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall / (precision + recall)).item()

def normal_consistency(points1, normals1, points2, normals2, chunk_size=2048):
    """
    Compute Normal Consistency: mean |cos(angle)| between matched point normals.
    Matches each point in points1 to nearest point in points2.
    """
    # Find nearest neighbor indices
    nn_indices = []
    for i in range(0, len(points1), chunk_size):
        chunk = points1[i:i+chunk_size]
        dists = torch.cdist(chunk, points2)
        nn_indices.append(dists.argmin(dim=1))
    nn_indices = torch.cat(nn_indices)

    matched_normals = normals2[nn_indices]
    cos_sim = torch.abs((normals1 * matched_normals).sum(dim=1))
    return cos_sim.mean().item()
```

## Success Criteria

- Pipeline runs end-to-end on ≥20 test models without errors
- Produces per-sample CSV with all metrics (CD, F-score, NC, PSNR, SSIM)
- Produces summary with mean ± std for each metric, separately for Path A and Path B
- Gap analysis clearly indicates whether VAE or DiT is the bottleneck

## Dependencies

All already installed:
- `trimesh` (mesh I/O and point sampling)
- `torch` (GPU-accelerated metric computation)
- `o_voxel` (mesh ↔ O-Voxel conversion)
- `nvdiffrast` (normal map rendering for PSNR/SSIM)
- `trellis2` (SC-VAE and pipeline)

**Not installed locally but available on HF Hub:**
- Encoder weights (`shape_enc_next_dc_f16c32_fp16`) — auto-downloads on first use

No new Python packages required.

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Encoder weights download fails (HF auth) | Pre-download via `huggingface-cli download`; verify access |
| Pilot data not representative | Select diverse complexity levels; note limitations in summary |
| SC-VAE decoder OOM at 512³ | Fall back to 256³; `decoder.set_resolution(256)` |
| DiT generation fails on some inputs | Log failures, report success rate, analyze failure modes |
| Mesh alignment issues (coordinate systems) | Both paths normalize to [-0.5, 0.5] consistently |
| CD computation OOM for large point clouds | Chunked `torch.cdist` with chunk_size=2048 |
| Path A vs Path B resolution mismatch | Both explicitly set to 512 (decoder.set_resolution + pipeline_type) |
