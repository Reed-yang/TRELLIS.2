# TRELLIS.2 Component-Level Evaluation Design

## Goal

Systematically measure the upper-bound capability of each TRELLIS.2 pipeline component (SC-VAE, Sparse Structure DiT, Shape DiT, Material DiT) and post-processing, using paper-aligned test sets and metrics. The results guide which component to prioritize for optimization in subsequent research phases.

## Motivation

Phase 0 gap measurement (30 Objaverse models, random views) revealed a ~19x CD gap between VAE reconstruction and DiT generation. However, several confounds prevented actionable conclusions:

1. **Input viewpoint quality**: Worst-case DiT results correlated with poor conditioning views, not model limitations
2. **Post-processing asymmetry**: DiT output includes `fill_holes()`, VAE output does not
3. **Small sample size**: 30 models insufficient for statistical significance
4. **Non-standard test set**: Objaverse pilot cannot be compared with paper baselines
5. **No stage-level diagnosis**: Unknown which of the 3 DiT stages contributes most error

This evaluation addresses all five issues.

## Architecture

Three-phase progressive evaluation on Toys4k-PBR (paper-standard test set):

- **Phase A**: VAE reconstruction baseline on full test set — establishes upper bound and validates pipeline
- **Phase B**: DiT generation with best-of-16 views — measures generation quality ceiling with/without post-processing
- **Phase C**: GT injection experiments — isolates error contribution of each DiT stage

Each phase produces independent, actionable results. Later phases build on earlier phase outputs.

---

## 1. Test Set Construction

### Data Source
- **Toys4k** public dataset (~3,229 3D assets)
- Not in TRELLIS.2 training set (confirmed by paper Table 6)

### PBR Filtering
- Retain only assets containing all three PBR maps: base color, metallic, roughness
- Expected yield: ~473 instances (matching paper's Toys4k-PBR)
- Implementation: parse material node tree per asset, check for required texture maps

### Complexity Stratification
Three tiers based on dual criteria:
- **Primary**: GT mesh face count
- **Secondary**: Canny edge pixel ratio on rendered normal maps (captures visual complexity)
- Thresholds set at tercile boundaries of the filtered dataset, ensuring ~equal sample count per tier
  - Tier 1 (Simple): low face count, simple silhouette
  - Tier 2 (Medium): moderate complexity
  - Tier 3 (Complex): high face count, detailed geometry

Note: Dora (CVPR 2025) uses salient edge count (N_Gamma) with 4 levels. Our approach differs intentionally — we use tercile-based 3-tier split for balanced sample sizes. Dora's absolute thresholds are dataset-specific and may not transfer well to Toys4k. The per-sample face count and edge ratio are recorded in CSV for post-hoc re-stratification if needed.

### Data Preparation Implementation
`prepare_toys4k.py` bypasses the missing `data_toolkit/datasets/` module (see findings.md Finding 1) by directly calling:
- `trimesh.load()` for mesh loading
- Material node tree parsing for PBR filtering
- `o_voxel.convert.mesh_to_flexible_dual_grid()` for O-Voxel conversion
This follows the same workaround validated in Phase 0.

---

## 2. Conditioning Image Generation

### Rendering Protocol
- **Engine**: Blender CYCLES (training-matched)
- **Views per asset**: 16
- **Camera distribution**: Hammersley sequence on sphere, FoV random 10-70 deg (matching training)
- **Lighting**: Randomized environment lighting
- **Output**: 1024x1024 RGBA images
- **Script**: Extend existing `scripts/render_blender_cond.py`

### Best-of-N Strategy
- Run DiT inference for all 16 views per object
- Select the view producing the lowest CD (after 24-rotation alignment + ICP) as the object's DiT upper bound
- Record all 16 CDs for optional viewpoint sensitivity analysis

---

## 3. Evaluation Conditions

### Phase A — VAE Reconstruction Baseline (473 objects)

| Condition | Input | Pipeline | Post-processing |
|-----------|-------|----------|-----------------|
| VAE recon | GT mesh | encode -> decode | None (raw) |

### Phase B — DiT Generation (473 objects x 16 views)

| Condition | Input | Pipeline | Post-processing |
|-----------|-------|----------|-----------------|
| DiT + fill_holes | Rendered image | Full pipeline | fill_holes(max_hole_perimeter=0.03), matching pipeline default in `trellis2/pipelines/trellis2_image_to_3d.py:474` |
| DiT raw | Rendered image | Full pipeline | None (monkey-patch no-op) |

Best view selected by min CD of "DiT + fill_holes" condition. Same best view used for "DiT raw" comparison.

### Phase C — GT Injection Stage Breakdown (100 objects, sampled by tier)

| Experiment | Structure | Shape | Material | Measures |
|------------|-----------|-------|----------|----------|
| Baseline | DiT | DiT | DiT | Full pipeline (= Phase B best) |
| C1 | **GT** | DiT | DiT | Structure prediction error contribution |
| C2 | **GT** | **GT** | DiT | Material DiT isolated capability |
| C3 | DiT | DiT | **GT** | Geometry pipeline isolated (excludes material) |
| C4 | **GT** | DiT | **GT** | Shape DiT isolated (excludes structure + material) |

GT sources:
- GT sparse structure: `mesh_to_flexible_dual_grid()` on GT mesh → extract active voxel positions
- GT shape latent: GT mesh → SC-VAE shape encoder → shape latent $z_{shape}$
- GT material latent: GT mesh → SC-VAE material encoder → material latent $z_{mat}$

Analysis logic:
- Baseline vs C1: quantifies structure prediction error (structure DiT contribution)
- C1 vs C4: isolates material DiT error (C4 fixes material, C1 doesn't)
- C4 vs VAE recon: shape DiT's remaining gap with perfect structure and material
- C1 vs C2: shape SLat generation error + its cascade on material (note: not purely shape error)
- C2 vs VAE recon: remaining gap = material DiT error + decode path differences

Note: VAE recon is encode-decode of ALL components together, so "C1 ~ VAE recon" does NOT directly imply shape DiT is near upper bound. The C4 experiment is needed to isolate shape DiT cleanly.

### Phase C Sampling Strategy
- **100 objects** sampled from Phase B results, stratified by complexity tier: ~33 per tier
- **Sampling method**: Within each tier, select objects spanning the Phase B CD ratio distribution — 11 from bottom quartile (best DiT), 11 from top quartile (worst DiT), 11 from median region. This ensures the stage attribution covers both easy and hard cases.
- Each object uses its Phase B best view as conditioning image
- 5 experiments × 100 objects = 500 inferences

---

## 4. Metrics

### Geometric Metrics
- **Chamfer Distance (CD)**: Bidirectional mean squared L2
  - Phase A: **both 1M and 100K** sampled points (1M for paper-aligned comparison with Table 1; 100K for cross-phase comparability with B/C)
  - Phase B/C: 100K sampled points (efficiency trade-off for 7,568+ evaluations)
  - Cross-phase comparison always uses the 100K version
- **F-score**: Multi-threshold — tau = 0.005, 0.01, 0.05, 0.1, 0.2 (covers TRELLIS.2 paper, Dora, and TripoSR/SF3D/InstantMesh standards)
- **Normal Consistency (NC)**: Mean |cos(angle)| of matched point normals

### Rendering Metrics
- **Config A (paper-aligned)**: 4 fixed views matching TRELLIS.2 paper — pitch 30deg, FoV 6deg, yaw 30/120/210/300 deg, radius 10
- **Config B (broader coverage)**: 8 views at 512x512 (matching Phase 0 setup)
- Metrics: PSNR, SSIM, **LPIPS** (new, more perceptually meaningful)
- **Rotation alignment for rendering**: Normal maps are rendered from the **aligned** mesh (after applying the best 24-rotation + ICP transform to the predicted mesh). This ensures rendering metrics are meaningful despite coordinate system differences (addresses findings.md Caveat 2).

### Alignment
- **Stage 1**: 24-rotation enumeration (2048-point subset for speed)
- **Stage 2** (new): ICP refinement on best rotation — **rigid transform only** (rotation + translation, no scale). Implementation: Open3D `registration_icp()` with `TransformationEstimationPointToPoint`, max iterations=50, convergence threshold=1e-8. Scale is excluded because all meshes are pre-normalized to unit cube.
- Full metric computation uses aligned 100K/1M points

### Normalization
- All meshes normalized to fit within unit cube ([-0.5, 0.5]^3), matching paper protocol

---

## 5. Reporting

### Per-Phase Outputs
Each phase produces:
- `per_sample.csv` — Full per-object metrics
- `summary.md` — Aggregate statistics (mean +/- std)
- `by_tier.md` — Breakdown by complexity tier

### Phase B Additional Outputs
- `postprocess_comparison.md` — with vs without fill_holes analysis
- Best/worst case tables with conditioning image previews
- Per-object: best_view_idx, all 16 view CDs

### Phase C Additional Outputs
- `stage_attribution.md` — Error contribution per DiT stage
- Comparison table: Baseline/C1/C2/C3/C4 across tiers

### Final Summary
- `my-docs/component-eval-summary.md` — Chinese summary with key findings and optimization direction recommendation

---

## 6. Engineering

### Directory Structure
```
experiments/component_eval/
├── phase_a/
│   ├── results/ (per_sample.csv, summary.md, by_tier.md)
│   └── previews/
├── phase_b/
│   ├── results/ (per_sample.csv, summary.md, by_tier.md, postprocess_comparison.md)
│   ├── meshes/ (best OBJ with/without fill_holes)
│   └── previews/
└── phase_c/
    ├── results/ (per_sample.csv, stage_attribution.md, by_tier.md)
    └── meshes/
```

### Scripts (new, under `scripts/`)
- `prepare_toys4k.py` — Download, PBR filter, complexity stratification
- `component_eval.py` — Main pipeline, `--phase a|b|c`, `--rank`, `--world_size`
- Extend `eval_metrics.py` — Add ICP refinement, LPIPS, multi-threshold F-score

### Constraints
- **No modification** to `trellis2/` core code
- Post-processing toggle via monkey-patch on `Mesh.fill_holes` (set to no-op)

### GT Injection Implementation (Phase C)

The pipeline's `run()` method in `trellis2/pipelines/trellis2_image_to_3d.py` executes three stages sequentially. GT injection is implemented by monkey-patching specific stage outputs:

1. **GT sparse structure injection**: After the sparse structure flow samples (`run()` line ~430-440), replace the sampled `z_ss` with the GT sparse structure derived from `mesh_to_flexible_dual_grid()`. The GT structure needs to be encoded through the sparse structure VAE encoder to get the latent representation, then decoded to match the expected format.

2. **GT shape latent injection**: After the shape SLat flow samples (`run()` line ~450-460), replace the sampled `z_shape` with the GT shape latent from SC-VAE shape encoder output. The encoder is already available (`shape_enc_next_dc_f16c32_fp16`).

3. **GT material latent injection**: After the material SLat flow samples (`run()` line ~470), replace `z_mat` with the GT material latent from SC-VAE material encoder output.

Implementation approach: Wrap `pipeline.run()` with a custom function that intercepts stage outputs via hooks on the flow model's `sample()` method, replacing the denoised output with pre-computed GT latents. This avoids modifying `trellis2/` source code.

### Compute Estimate
- Phase A: ~473 VAE encode-decode, ~30 min on 1 GPU
- Phase B: 7,568 DiT inferences × 2 (with/without fill_holes), ~4 hours on 8x H100 (both versions extracted per inference, no doubling needed)
- Phase C: 500 inferences (5 conditions × 100 objects), ~40 min on 8x H100
- Conditioning image rendering: 473 × 16 = 7,568 Blender renders. At ~30-60s per CYCLES render, this is 63-126 hours sequential. With 6 parallel Blender processes (one per GPU): **~10-20 hours**. Can be reduced by lowering CYCLES samples (e.g., 64 samples instead of default 128) or using EEVEE for non-photorealistic renders. This is the longest single step and should be started first.

---

## 7. Success Criteria

The evaluation is successful when it can answer:

1. **What is the VAE reconstruction upper bound on Toys4k-PBR?** — Directly comparable to paper Table 1
2. **What is the DiT generation ceiling under ideal conditions (best view)?** — The true VAE-DiT gap
3. **How much does post-processing contribute?** — fill_holes impact quantified
4. **Which DiT stage is the primary bottleneck?** — Actionable for Phase 1 optimization direction
5. **Does the gap vary with object complexity?** — Identifies where DiT struggles most

---

## References

- TRELLIS.2 paper: evaluation protocol (Section 4, Appendix D.1)
- TRELLIS v1 paper: reconstruction/generation eval split (Section 4, Appendix C.1)
- Dora (CVPR 2025): complexity-stratified VAE benchmark
- TripoSR/SF3D: rotation search + ICP alignment standard
- Cue3D (NeurIPS 2025): visible vs full surface evaluation
- Survey: `logs/eval_methodology_survey.md`
