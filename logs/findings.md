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

---

## Component Evaluation

### Finding 8: OBJ Files Lack PBR Material Information
**Date:** 2026-03-21
**Impact:** PBR filtering cannot be done on OBJ files alone
**Description:** All 4000 Toys4k OBJ files pass any OBJ-based material check because OBJ format does not encode PBR properties (metallic, roughness textures). The paper's Toys4k-PBR subset (~473) is defined by Blender material node trees.
**Solution:** Created `filter_pbr_blender.py` and `pbr_strict_filter.py` to parse `.blend` files. Strict filter (all 3 PBR inputs texture-linked in Principled BSDF) yields 590 assets. Tested 3 strictness levels: strict=590, loose=790, partial=597.
**Status:** Resolved. Using strict=590 as test set.

---

### Finding 9: GPU Memory Contention Causes False OOM and Evaluation Errors
**Date:** 2026-03-21 ~ 2026-03-22
**Impact:** **Critical** — Multiple processes per GPU cause 5x slowdown and false "all 16 views failed" errors
**Description:** Running >1 component_eval process on the same 80GB GPU causes: (1) sampling speed drops from 20 it/s to 4 it/s, (2) CUDA OOM errors even for small meshes, (3) false errors where all 16 DiT generations fail. All 109 Phase A "OOM" samples and all 23 Phase B "error" samples completed successfully on dedicated single-process GPUs.
**Lesson:** Never share GPU between evaluation processes. Use 1 process per GPU, balance load via LPT scheduling.
**Status:** Resolved.

---

### Finding 10: fill_holes Post-processing Has Negligible Impact
**Date:** 2026-03-22
**Impact:** Answers component eval question #3 — fill_holes is NOT a meaningful contributor
**Description:** Phase B simultaneously evaluated with/without fill_holes. Results: CD identical, NC delta -0.0004, only 50.5% of samples improved by fill_holes. F-score also essentially unchanged across all thresholds.
**Implication:** Post-processing optimization (P1d) is low priority; gap is entirely in DiT generation quality.
**Status:** Confirmed.

---

### Finding 11: DiT Quality Degrades Systematically with Object Complexity
**Date:** 2026-03-22
**Impact:** Identifies where DiT improvement efforts should focus
**Description:** By-tier analysis on 590 PBR assets shows clear degradation: NC 0.913→0.870→0.819 (Tier 1→2→3), LPIPS 0.077→0.108→0.115. Worst cases are consistently complex shapes (balls with texture, trees with fine branches).
**Implication:** DiT struggles most with complex geometry — structure prediction may be the bottleneck stage.

---

### Finding 12: Blender CPU Rendering Outperforms GPU for Simple Scenes
**Date:** 2026-03-21
**Impact:** Informational — guides future rendering strategy
**Description:** Blender CYCLES GPU rendering (7.2s/view) is slower than CPU (5.4s/view) for Toys4k scenes due to ~1.5s CUDA kernel initialization overhead per Blender subprocess. With 112 CPU cores available, distributed CPU rendering across 6 Slurm nodes (24 concurrent processes) achieved ~26 assets/min, completing 4000 assets in ~2.5 hours.

---

## O-Voxel Representation Fidelity Test (Sketchfab-Hard)

### Finding 13: flexible_dual_grid_to_mesh Requires CUDA Tensors
**Date:** 2026-03-26
**Impact:** **Critical** — O-Voxel decode returns all-zero face indices on CPU
**Description:** `o_voxel.convert.flexible_dual_grid_to_mesh()` internally uses `_C.hashmap_insert_3d_idx_as_val_cuda()` for face index computation. When called with CPU tensors, the hashmap doesn't populate correctly, resulting in all face indices being 0 (degenerate mesh). All O-Voxel examples in the codebase pass `.cuda()` tensors.
**Solution:** Always pass CUDA tensors to `flexible_dual_grid_to_mesh()`.
**Status:** Resolved.

---

### Finding 14: O-Voxel Resolution Sweet Spot at 512-1536
**Date:** 2026-03-26
**Impact:** Guides resolution selection for evaluation benchmarks
**Description:** Testing 3 hard Sketchfab models (helmet 324K faces, bugatti 169K, spacesuit 200K) across 512-2048:
- **CD** is near-identical at 512-1536 (0.000005-0.000013), with catastrophic degradation at 2048
- **2048 failure**: helmet CD 160x worse, spacesuit CD 58x worse than 1536. Bugatti unaffected.
- **NC is uniformly low for helmet** (~0.62 across all resolutions) — chainmail topology defeats normal consistency
- **SC-VAE (Layer B) improves NC** over raw O-Voxel: helmet 0.62→0.78 at 512, spacesuit 0.66→0.97
**Implication:** 512-1024 is the practical operating range. Higher resolutions provide diminishing returns and can degrade quality (2048 overflow behavior).

---

### Finding 15: O-Voxel Causes Severe Mesh Fragmentation
**Date:** 2026-03-26
**Impact:** **Serious** — topology is fundamentally altered by O-Voxel representation
**Description:** Topology analysis reveals massive fragmentation:
- **Helmet**: GT 42K components → 97K-545K recon components (2.3-12.8x increase)
- **Spacesuit**: GT 285 → 2.5K-56K components (9-198x increase)
- **Bugatti**: GT 3K → 3.2K-8.4M components (similar at low res, explodes at 2048)
- **Area ratio**: helmet recon has 1.37-5.89x GT surface area; bugatti has 0.86-1.88x
- **Boundary edges**: recon has fewer boundary edges than GT for bugatti (8K vs 123K at 512), suggesting open surfaces are being sealed
**Implication:** O-Voxel fundamentally changes mesh topology. The representation creates many disconnected fragments while also sealing some open surfaces. This is intrinsic to the voxel-based approach and cannot be fixed by resolution alone.

---

### Finding 16: SC-VAE Compression Has Minimal Geometric Impact but Improves Normals
**Date:** 2026-03-26
**Impact:** Positive — SC-VAE is not a bottleneck for geometric quality
**Description:** Comparing Layer A (O-Voxel roundtrip) vs Layer B (SC-VAE roundtrip):
- **CD**: Nearly identical between layers (e.g., helmet@1024: 0.000013 both)
- **NC**: SC-VAE significantly IMPROVES normals — helmet 0.62→0.81, spacesuit 0.66→0.97
- **Topology**: SC-VAE produces more components and boundary edges than raw O-Voxel
**Interpretation:** The SC-VAE decoder's learned mesh extraction produces more consistent surface normals than the deterministic `flexible_dual_grid_to_mesh`, despite creating more topological fragments. The 16x compression adds negligible geometric error.

---

## Methodology Notes (Updated)

### Component Evaluation Protocol
- **Test set:** 590 Toys4k-PBR assets (strict PBR filter: all 3 Principled BSDF inputs texture-linked)
- **Complexity stratification:** 3 tiers (face count + edge complexity), 197/240/153 per tier
- **Resolution:** 512³ O-Voxel grid
- **Point sampling:** 100K points (Phase B/C), 1M points (Phase A CD)
- **Alignment:** 24 axis-aligned rotation + ICP refinement
- **F-score thresholds:** τ = {0.005, 0.01, 0.05, 0.1, 0.2}
- **Rendering metrics:** 4-view paper config + 8-view coverage, PSNR/SSIM/LPIPS
- **Conditioning:** 16 Blender CYCLES views per asset (Hammersley distribution, FoV 10-70°)
- **DiT evaluation:** Best-of-16 views (lowest CD) as upper bound
