# O-Voxel Representation Fidelity Test on Sketchfab-Hard Samples

> Date: 2026-03-26
> Status: Approved
> Motivation: Toys4k models are too simple to challenge the O-Voxel representation. Use manually curated hard Sketchfab models to probe the upper-bound fidelity of the O-Voxel format and SC-VAE compression at multiple resolutions, **without involving DiT generation**.

---

## Goal

Quantify how much geometric information is lost at two distinct stages:
1. **Layer A (O-Voxel roundtrip):** Mesh → `mesh_to_flexible_dual_grid()` → `flexible_dual_grid_to_mesh()` — pure discretization loss, no neural network
2. **Layer B (SC-VAE roundtrip):** Mesh → O-Voxel → SC-VAE Encoder → Latent → SC-VAE Decoder → Mesh — adds 16× compression loss

Sweep resolutions from 512 to 2048+ to find the resolution-fidelity ceiling.

## Test Samples

| ID | Model | Source File | Vertices | Faces | Sub-meshes | Key Challenge |
|----|-------|------------|----------|-------|------------|---------------|
| helmet | Early Medieval Nasal Helmet | `early_medieval_nasal_helmet.glb` | 362K | 324K | 17 | Chainmail ring topology (275K faces in 6 sub-meshes) |
| bugatti | Bugatti EB110 Super Sport 1992 | `.zip → .7z → .obj` | 148K | 169K | 61 | Extreme aspect ratio (10.6×1.2×10.6), thin car shell, fine spokes |
| spacesuit | Franz Viehbock's Sokol Space Suit | `.zip → .zip → .obj` | 109K | 200K | 1 | Fabric wrinkles, single non-watertight mesh |

All three are **non-watertight**, which is a core strength of O-Voxel vs SDF representations.

## Data Preprocessing

For each model:
1. Load raw file (GLB/OBJ), merge all sub-meshes into a single `trimesh.Trimesh`
2. Normalize: center at origin, scale by `1 / max(extents)` to fit in [-0.5, 0.5]³ (preserves aspect ratio)
3. Save normalized GT mesh as OBJ to `experiments/ovoxel_repr_test/data/{model_id}.obj`
4. Record: original vertex/face count, bounding box extents before/after normalization, watertight status

**Note on aspect ratio:** Bugatti's Y extent is ~11% of max extent. After normalization, most of the 512³ voxel budget is "wasted" on empty space. This is an inherent representation challenge worth documenting.

## Layer A: O-Voxel Pure Representation Test

### Resolution Range

512, 1024, 1536, 2048

256 is skipped — pilot testing showed IndexError (integer overflow) on the helmet mesh at this resolution.

### Per (model, resolution) procedure

1. **Encode:** `mesh_to_flexible_dual_grid(vertices, faces, grid_size=res, aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]], face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2)`
   - Record: voxel count, encode time
2. **Decode:** `flexible_dual_grid_to_mesh(coords, dual_vertices, intersected, split_weight=None, grid_size=res, aabb=...)`
   - Record: output vertex/face count, decode time
3. **Geometric metrics** (100K sampled points from each mesh):
   - Chamfer Distance (CD)
   - F-score at τ = {0.005, 0.01, 0.05}
   - Normal Consistency (NC)
4. **Export** reconstructed mesh as OBJ
5. **Render previews** (see Preview Rendering section)

### Output

- `experiments/ovoxel_repr_test/layer_a/{model_id}_{res}/recon.obj`
- `experiments/ovoxel_repr_test/layer_a/results.csv` — columns: `model_id, resolution, n_voxels, encode_time_s, decode_time_s, out_verts, out_faces, cd, nc, fscore_0.005, fscore_0.01, fscore_0.05`

## Layer B: SC-VAE Compression Test

### Procedure

For each (model, resolution) that succeeded in Layer A:
1. Construct SparseTensor from O-Voxel output (same as `gap_measurement.py` / `component_eval.py`)
2. `encoder(vertices_st, intersected_st)` → latent
3. `decoder(latent)` → reconstructed mesh
4. Same metrics as Layer A
5. Record latent token count and latent shape

### Constraints

- Pretrained SC-VAE: trained at 256, fine-tuned to 512. Resolutions 1024+ are out-of-training-distribution — quality may degrade or OOM may occur.
- `decoder.set_resolution(res)` must be called before decode.
- If OOM at a given resolution: record failure, do not retry. The failure itself is informative (marks the practical upper bound of SC-VAE).

### Output

- `experiments/ovoxel_repr_test/layer_b/{model_id}_{res}/recon.obj`
- `experiments/ovoxel_repr_test/layer_b/results.csv` — same columns as Layer A plus: `latent_tokens, latent_shape, error`

## Preview Rendering

### Per (model, resolution, layer) rendering

Use NVDiffRast to render **normal maps** (headless-compatible, consistent with existing eval pipeline).

- **Camera setup:** 8 fixed viewpoints evenly distributed around the object (azimuth 0°, 45°, 90°, ..., 315°; elevation 15°; fixed distance)
- **Resolution:** 512×512 per view
- **GT and reconstruction** rendered with identical camera parameters

### Output format

1. **Side-by-side images:** `previews/{layer}/{model_id}_{res}_view{N}.png` — GT on left, reconstruction on right, 1024×512 per image
2. **Per-model grid:** `previews/{layer}/{model_id}_grid.png` — rows = resolutions (512/1024/1536/2048), columns = viewpoints (8), plus GT row at top. Enables instant visual comparison across all resolutions.
3. **Cross-layer grid (optional):** `previews/{model_id}_ab_grid.png` — rows = Layer A then Layer B at each resolution, columns = viewpoints. Shows compression impact visually.

## Qualitative Topology Analysis

For each model, focus on its signature difficult structure:

- **Helmet:** Do chainmail rings remain as separate loops, or merge into solid blobs? At which resolution?
- **Bugatti:** Do thin panels (hood, doors), glass surfaces, and wheel spokes survive? How does the extreme aspect ratio affect voxel utilization?
- **Space Suit:** Are fabric wrinkles preserved, or smoothed into a featureless surface? How does NC change across resolutions?

Record observations in `report.md` with references to specific preview images.

## Metrics Reference

Reuse functions from `scripts/eval_metrics.py`:
- `sample_points_and_normals(mesh, n_points)` — uniform surface sampling
- `chamfer_distance(pts_a, pts_b)` — symmetric CD
- `f_score_multi(pts_a, pts_b, thresholds)` — multi-threshold F-score
- `normal_consistency(pts_a, nrm_a, pts_b, nrm_b)` — mean abs cosine similarity

For rendering, reuse `render_normal_maps_paper_config()` or implement a simpler fixed-viewpoint variant using NVDiffRast directly.

## Implementation Plan

### Script: `scripts/ovoxel_repr_test.py`

Single script with two modes:
- `--layer a` — O-Voxel roundtrip
- `--layer b` — SC-VAE roundtrip

Steps:
1. Preprocess all models (merge, normalize, save GT)
2. For each (model, resolution): encode → decode → metrics → export mesh → render previews
3. Generate results CSV
4. Generate grid preview images
5. Print summary table

### Dependencies

- `o_voxel` (mesh_to_flexible_dual_grid, flexible_dual_grid_to_mesh)
- `trimesh` (mesh loading, merging, point sampling)
- `scripts/eval_metrics.py` (CD, NC, F-score)
- `trellis2.models.from_pretrained` (SC-VAE encoder/decoder, Layer B only)
- `nvdiffrast` or existing render utils (normal map rendering)
- `PIL` / `matplotlib` (preview grid assembly)

### Estimated Runtime

| Step | Time |
|------|------|
| Preprocessing | < 1 min |
| Layer A (3 models × 4 resolutions) | ~5 min (dominated by 2048 encode ~48s/model) |
| Layer B (3 models × 2-4 resolutions) | ~10-30 min (SC-VAE inference, may OOM at high res) |
| Preview rendering (12+ mesh pairs × 8 views) | ~5-10 min |
| Grid image generation | < 1 min |

Total: ~30-45 minutes on single GPU.
