# TRELLIS.2 Improvement — Findings & Issues Log

## Phase 0: Gap Measurement

### Finding 1: data_toolkit/datasets/ Module Missing
**Date:** 2026-03-19
**Impact:** Blocks standard 7-step data preparation pipeline
**Description:** `data_toolkit/build_metadata.py`, `download.py`, `dual_grid.py` all depend on `importlib.import_module(f'datasets.{sys.argv[1]}')`, but the `datasets/` directory does not exist in the repo. This prevents using `Toys4k` or `SketchfabPicked` test sets through the standard pipeline.
**Workaround:** Bypassed entirely by directly calling core functions (`o_voxel.convert.mesh_to_flexible_dual_grid()`, `trimesh.load()`) without going through the data_toolkit CLI wrappers. Pilot data obtained from Objaverse instead.
**Status:** Workaround in place. Standard test sets still not available.

---

### Finding 2: DINOv3 and RMBG-2.0 Gated Model Access
**Date:** 2026-03-19
**Impact:** DiT pipeline (Path B) fails without monkey-patching
**Description:** `Trellis2ImageTo3DPipeline.from_pretrained()` tries to download `facebook/dinov3-vitl16-pretrain-lvd1689m` and `ZhengPeng7/BiRefNet` from HuggingFace Hub, but these are gated repos requiring authentication. Both models are already available locally at `pretrained/dinov3/` and `pretrained/rmbg2/`.
**Solution:** Added `_patch_gated_models()` in `gap_measurement.py` (same approach as `example_local.py`) to redirect HF Hub paths to local directories.
**Status:** Resolved. Patch applied in gap_measurement.py.

---

### Finding 3: PbrMeshRenderer vs MeshRenderer API Mismatch
**Date:** 2026-03-19
**Impact:** Rendering metrics fail for DiT-generated meshes
**Description:** DiT pipeline returns `MeshWithVoxel` objects. `render_utils.get_renderer()` routes these to `PbrMeshRenderer`, which does NOT accept `return_types` parameter (unlike `MeshRenderer`). This causes `render_normal_maps()` to fail.
**Solution:** In `render_normal_maps()`, detect `MeshWithVoxel` instances and convert to plain `Mesh` before rendering, so `MeshRenderer` is used instead.
**Status:** Resolved.

---

### Finding 4: Headless Server — No pyrender/pyglet for trimesh Rendering
**Date:** 2026-03-19
**Impact:** `trimesh.scene.save_image()` fails, all reference images are gray placeholders
**Description:** Server has no display server (X11/Wayland), so `pyglet` (used by trimesh's built-in renderer) cannot initialize a GL context. All 30 pilot reference images were saved as uniform gray (200,200,200) placeholders.
**Solution:** Two-pronged:
1. `gap_measurement.py` auto-detects placeholder images (`pixels.std() < 5`) and re-renders using NVDiffRast (GPU, headless-capable)
2. Installed Blender 3.0 for CYCLES rendering (training-matched conditioning images)
**Status:** Resolved. Blender rendering produces proper 1024×1024 RGBA images.

---

### Finding 5: SC-VAE Encoder Not in Local Pretrained Cache
**Date:** 2026-03-19
**Impact:** Path A (VAE reconstruction) cannot run
**Description:** `pretrained/TRELLIS.2-4B/ckpts/` contains only decoders and DiT flow models. The encoder (`shape_enc_next_dc_f16c32_fp16`) is available on HuggingFace Hub but was not downloaded during initial model setup.
**Solution:** Downloaded via `huggingface-cli download microsoft/TRELLIS.2-4B ckpts/shape_enc_next_dc_f16c32_fp16.json ckpts/shape_enc_next_dc_f16c32_fp16.safetensors`
**Status:** Resolved. Encoder loads successfully (FlexiDualGridVaeEncoder, 354M params).

---

### Finding 6: Coordinate Axis Misalignment Between GT and DiT Output
**Date:** 2026-03-20
**Impact:** **Critical** — All geometric metrics (CD, F-score, NC) severely overestimated the gap
**Description:** DiT generates meshes in its own canonical coordinate system. GT meshes from Objaverse have their own coordinate convention. The axes are swapped (e.g., GT longest axis = Y, DiT longest axis = Z for the same object). Direct point cloud comparison without alignment gives meaningless results.
**Evidence:**
- Band_Aid: GT extent [1.0, 0.76, 0.76], DiT extent [0.75, 1.0, 0.75] — X↔Y swap
- Christmas_tree: GT extent [0.59, 1.0, 0.61], DiT extent [0.68, 0.69, 1.0] — Y↔Z swap
**Solution:** Implemented `find_best_rotation_24()` — enumerate all 24 axis-aligned rotations (proper rotations of cube), compute CD for each, pick minimum. Uses 2048-point subset for speed (~10ms overhead per model).
**Impact on results:**
- CD ratio: 406x → **49x** (8x correction)
- F-score: 0.054 → **0.321** (6x correction)
- NC: 0.500 → **0.741** (significant correction)
**Status:** Resolved. Alignment is now standard in the evaluation pipeline.
**Lesson:** Any evaluation comparing meshes from different generation pipelines MUST include pose alignment. This is well-known in the 3D generation literature but easy to overlook in practice.

---

### Finding 7: Normal-Map vs Blender CYCLES Conditioning — Unexpected Result
**Date:** 2026-03-20
**Impact:** Informational — helps interpret results
**Description:** Expected Blender CYCLES images (training-matched) to reduce the DiT gap compared to normal-map placeholders. The opposite happened: DiT performed worse with Blender images (CD ratio 49x) vs normal maps (after alignment: ~20x estimated). Possible explanations:
1. Random Blender viewpoints (extreme angles, narrow FOV) are harder for DiT than the consistent front-facing normal map views
2. Normal maps provide cleaner geometric cues that help DiT despite being out-of-distribution
3. Objaverse pilot models may include unusual categories not well-represented in DiT training data
**Status:** Noted. Does not change the conclusion (DiT is bottleneck regardless).

---

## Methodology Notes

### Evaluation Protocol
- **Test set:** 30 Objaverse models from diverse LVIS categories (pilot set, not standard benchmark)
- **Resolution:** 512³ O-Voxel grid
- **Point sampling:** 10,000 points per mesh (trimesh.sample.sample_surface)
- **Alignment:** 24 axis-aligned rotation search before DiT metrics
- **F-score threshold:** τ = 0.01 in [-0.5, 0.5] normalized space
- **Rendering metrics:** 8-view normal maps via NVDiffRast, PSNR/SSIM

### Caveats
1. Pilot set (30 models) is small and not the paper's standard test set (Toys4K + Sketchfab Featured)
2. PSNR/SSIM rendering metrics do not apply rotation alignment (only geometric metrics do)
3. Blender conditioning images use random viewpoints — a fixed canonical viewpoint may give different results
