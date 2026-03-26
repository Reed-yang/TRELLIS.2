# TRELLIS.2 Improvement — Progress Log

## Phase 0: Gap Measurement (2026-03-19 ~ 2026-03-20)

### Goal
Quantify the quality gap between SC-VAE reconstruction upper bound and DiT generation quality, to determine Phase 1 priority (improve VAE or DiT).

### Timeline

| Date | Milestone |
|------|-----------|
| 03-19 | Evaluation metrics module (CD, F-score, NC, PSNR, SSIM) implemented and tested |
| 03-19 | Pilot data: 30 Objaverse models downloaded (diverse LVIS categories) |
| 03-19 | Encoder weights downloaded from HF Hub (shape_enc_next_dc_f16c32_fp16, 354M params) |
| 03-19 | gap_measurement.py: Path A (VAE encode-decode) + Path B (DiT generation) complete |
| 03-19 | First run with normal-map placeholder images — DiT/VAE CD ratio = 176x |
| 03-20 | Blender 3.0 installed, CYCLES conditioning images rendered (6-GPU parallel) |
| 03-20 | Second run with Blender images — DiT/VAE CD ratio = 406x (unaligned) |
| 03-20 | Discovered coordinate axis misalignment issue between GT and DiT output |
| 03-20 | Added 24-rotation alignment — DiT/VAE CD ratio = **49x** (aligned, final) |

### Current Status: **Phase 0 COMPLETE**

### Key Result

| Metric | VAE Recon | DiT Gen (aligned) | Ratio |
|--------|-----------|-------------------|-------|
| CD | 0.000071 | 0.00348 | 49x |
| F-score | 0.775 | 0.321 | 2.4x worse |
| NC | 0.932 | 0.741 | 1.26x worse |

**Decision: DiT is the bottleneck.** Per roadmap-v2.md decision rule, Phase 1 should prioritize DiT optimization (P2b promoted to highest priority).

### Deliverables
- `scripts/eval_metrics.py` — Reusable evaluation metrics with 24-rotation alignment
- `scripts/gap_measurement.py` — Full pipeline with multi-GPU support
- `scripts/prepare_pilot_data.py` — Objaverse pilot data downloader
- `scripts/render_blender_cond.py` — Blender CYCLES conditioning renderer
- `experiments/gap_measurement/` — Normal-map experiment results
- `experiments/gap_measurement_blender/` — Blender experiment results (final)

---

## Component Evaluation: Toys4k-PBR (2026-03-20 ~ 2026-03-22)

### Goal
Systematic measurement of each TRELLIS.2 component's upper-bound capability on the paper-standard Toys4k-PBR test set (590 strict PBR assets).

### Timeline

| Date | Milestone |
|------|-----------|
| 03-20 | Component eval plan designed (3-phase: VAE baseline / DiT best-of-16 / GT injection) |
| 03-20 | eval_metrics.py extended: ICP refinement, LPIPS, multi-threshold F-score |
| 03-20 | component_eval.py + report_gen.py implemented |
| 03-21 | Toys4k dataset downloaded (4000 OBJ + 4000 Blender files) |
| 03-21 | prepare_toys4k.py: UID collision fix (parent dir name for unique UIDs) |
| 03-21 | Blender CYCLES 16-view rendering: 6-node distributed CPU (Slurm), 3999/4000 complete |
| 03-21 | Phase A first run: 3434/4000 processed, 614 OOM errors (GPU memory contention) |
| 03-21 | PBR material filtering via Blender node tree parsing: strict=590 (all 3 PBR inputs texture-linked) |
| 03-22 | Phase B: 590/590 PBR complete on 19 GPUs (local 8 + N118×8 + N120×3), LPT load balancing |
| 03-22 | Phase A PBR retry: 109 remaining items completed on N118 (OOM was from GPU sharing, not model) |
| 03-22 | CSV merge + report generation for PBR subset (Phase A & B) |

### Current Status: **Phase A & B COMPLETE, Phase C PENDING**

### Key Results (590 PBR assets)

#### Phase A: VAE Reconstruction Upper Bound

| Metric | Mean | Median |
|--------|------|--------|
| CD | 0.0000 | 0.0000 |
| NC | 0.9742 | 0.9813 |
| PSNR | 48.46 dB | 49.26 dB |
| F@0.005 | 0.9611 | 0.9912 |

#### Phase B: DiT Best-of-16 Generation

| Metric | Mean | Median |
|--------|------|--------|
| CD | 0.0001 | 0.0001 |
| NC | 0.8714 | 0.8916 |
| PSNR | 25.12 dB | 24.56 dB |
| F@0.005 | 0.5633 | 0.5533 |

#### Gap Ratio (DiT / VAE)

| Metric | Ratio | Interpretation |
|--------|-------|----------------|
| CD | **24.68x** | DiT geometry 25x worse |
| LPIPS | **73.39x** | Perceptual quality much worse |
| NC | 0.89x | Normal consistency 11% worse |
| PSNR | 0.52x | Rendering PSNR halved |

#### fill_holes Post-processing Impact
Negligible: CD identical, NC -0.0004, only 50.5% of samples improved. **fill_holes is NOT a meaningful contributor.**

#### By-Tier Degradation
DiT quality degrades with complexity: NC 0.913→0.870→0.819 (Tier 1→2→3), LPIPS 0.077→0.108→0.115.

### Deliverables
- `experiments/component_eval/phase_a/results_pbr/` — Phase A PBR reports
- `experiments/component_eval/phase_b/results_pbr/` — Phase B PBR reports
- `scripts/component_eval.py` — Main pipeline (--phase a|b|c)
- `scripts/phase_b_scheduler.py` — LPT load-balanced GPU scheduler with backfill
- `scripts/prepare_toys4k.py` — Toys4k download + PBR filter + complexity stratification
- `scripts/filter_pbr_blender.py` — Blender node-tree PBR material filter
- `scripts/pbr_strict_filter.py` — Strict PBR UID list generator

---

---

## O-Voxel Representation Fidelity Test (2026-03-26)

### Goal
Test O-Voxel representation and SC-VAE compression upper bound on hard Sketchfab models, without DiT involvement. Replaced Toys4k (too simple) with manually curated difficult cases.

### Timeline

| Date | Milestone |
|------|-----------|
| 03-26 | 3 Sketchfab-Hard samples curated: helmet (chainmail, 324K faces), bugatti (extreme 9:1 aspect ratio, 169K), spacesuit (fabric wrinkles, 200K) |
| 03-26 | Design spec + implementation plan written |
| 03-26 | Critical fix: `flexible_dual_grid_to_mesh` requires CUDA tensors (returns all-zero face indices on CPU) |
| 03-26 | Layer A (O-Voxel roundtrip) + Layer B (SC-VAE roundtrip) complete: 24 combinations (3 models × 4 resolutions × 2 layers) |
| 03-26 | Topology metrics added: connected components, Euler number, boundary edges, area ratio |

### Current Status: **Layer A & B COMPLETE**

### Key Results

#### Geometric (Layer A — O-Voxel roundtrip, 512-1536)

| Model | CD Range | NC Range | F@0.01 |
|-------|----------|----------|--------|
| Helmet | 0.000013 | 0.62 | 0.999 |
| Bugatti | 0.000005 | 0.89-0.91 | 1.000 |
| Spacesuit | 0.000007-8 | 0.66 | 1.000 |

#### 2048 Degradation
Helmet and spacesuit show catastrophic quality loss at 2048: CD 160x and 58x worse than 1536, likely int32 overflow in O-Voxel library.

#### SC-VAE vs O-Voxel
SC-VAE (Layer B) significantly improves NC over raw O-Voxel: helmet 0.62→0.81, spacesuit 0.66→0.97. CD unchanged. The 16× compression adds negligible geometric error.

#### Topology (Fragmentation)
O-Voxel causes severe mesh fragmentation:
- Helmet: 42K → 98K-545K connected components
- Spacesuit: 285 → 2.5K-56K connected components
- Surface area increases 1.37-5.89x for helmet

### Deliverables
- `scripts/ovoxel_repr_test.py` — Main test script (subprocess parallel, 8-GPU)
- `experiments/ovoxel_repr_test/results_a.csv` — Layer A results
- `experiments/ovoxel_repr_test/results_b.csv` — Layer B results
- `experiments/ovoxel_repr_test/previews/` — Multi-view normal map comparisons
- `docs/superpowers/specs/2026-03-26-ovoxel-repr-test-design.md` — Design spec

---

## Next: Phase C (GT Injection) → Summary

Phase C: 100 sampled objects, 5 GT-injection conditions to locate bottleneck stage.
Then write `my-docs/component-eval-summary.md`.
