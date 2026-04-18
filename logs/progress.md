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

---

## CoReP-Fast: Stage 1 Torch Vectorization (2026-04-15 ~)

### Goal
Rewrite `custom/` CoReP pipeline as `corep_fast/` with Torch vectorization for batch dataset generation.

### Phase 0: Infrastructure (2026-04-15)
- 17 TDD tasks: CubeBatch, MeshTensors, interop bridges, profiling harness, topology equivalence, A/B rig
- 88 tests, all passing. Commits `64dd3f6` → `8be3489`.

### Phase 1a: s8_collapse Torch Rewrite (2026-04-15)
- s8_collapse is #1 bottleneck at 48% of pipeline runtime
- 9 TDD tasks: hybrid pipeline, edge enumeration, geometry processing, vertex welding, PLY writer, A/B validation

**Key Performance Milestones (resolution=128, icosphere subdiv=3, 68K cubes):**

| Version | s8 Time | vs Custom | Commit |
|---------|---------|-----------|--------|
| Custom baseline | 16.0s | 1.0x | — |
| Phase 1a initial (Python loops) | 23.5s | 0.7x | `9b3ee9e` |
| Vectorized _weld_and_dedup | 7.3s | 2.2x | `cffc719` |
| Optimized edge iteration | ~6.7s | ~2.4x | `2020b1f` |

**Optimization Details:**
- `_weld_and_dedup` 17.7s → 2.65s (6.7x): numpy array pipeline + `scatter_reduce_('amin')` replaces Python for-loops
- Edge iteration: frozenset lookup, cached sentinels, dict reuse
- PLY writer: `np.savetxt` bulk writes
- Multiprocessing for geometry processing: 263K edge tasks distributed across CPU cores
- GPU `torch.unique` for vertex welding: 97x faster than CPU (1.75s → 0.018s at 800K verts)

**Performance Progression (res=256, H100):**

| Optimization | s8 Time | vs Custom | Commit |
|-------------|---------|-----------|--------|
| Phase 1a initial | ~95s | 0.7x | `9b3ee9e` |
| Vectorized welding | ~33s | 2.1x | `cffc719` |
| + edge iter optimize | ~33s | 2.1x | `2020b1f` |
| + multiprocessing | 23.2s | 3.0x | `2ec9bc3` |
| + GPU welding | 13.9s | **5.0x** | `1511525` |

**Correctness:** A/B tests confirm exact vertex/face count match with custom/ implementation at all resolutions tested.

### Status: **5.0x speedup achieved on s8_collapse at res=256 (H100). Stage 2 Triton spec written, branch `triton-s8` created.**

---

## CoReP Deep Profiling (2026-04-16 → 2026-04-17)

### Goal
Pre-Triton kernel-level bottleneck investigation to inform next-stage choice
among Triton K1 / mesh-cleanup-port / s1-sat-hardening.

### Output
- Spec: `docs/superpowers/specs/2026-04-16-corep-deep-profiling-design.md`
- Plan: `docs/superpowers/plans/2026-04-16-corep-deep-profiling-implementation.md`
- Results: `my-docs/20260417-corep-deep-profiling-results.md`
- DoD: `tmp/profile_deep/DONE.md`
- Raw data: `tmp/profile_deep/results/`

### Method
4-layer profile on 116 GPU 0:
- Layer 0 — Nsight Systems macro timeline (res=256)
- Layer 1 — torch.profiler Chrome trace, 3-run median @ res=256 (run2)
- Layer 2 — Post-process Layer 1 trace → top-20 GPU kernels + heuristic CMB/MMB/LNB/CPU/UNK
- Layer 3 — Repeat Layer 1 @ res=128 (run1 median), compare per-stage scaling ratios

### Three headline findings
1. Pipeline is ~96% GPU-idle at res=256 (nsys: kernel ~260ms + memcpy ~435ms over warmup+measured = 347ms/run; e2e 10s).
2. Zero compute-bound kernels in top-20 (17 MMB, 3 LNB, 0 CMB). Biggest single-call kernel is an 11 ms `tensor_kernel_scan_innermost_dim_with_indices<long, greater_equal>` in s6, introduced by the Task 3 s7 bugfix's cummax.
3. s4 and s7 scale at 1.68x and 1.11x for R doubling (128→256), vs expected R²=4x — both are host-bound.

### Recommended next step
Shelve Triton K1 (originally #1, now #4 in the ROI list — s4 device time is only ~37 ms at res=256; even zero-GPU would save <40 ms). Primary target: eliminate the 1.0 s `cudaDeviceSynchronize` + 4022 `.item()`-style D2H leaks in s4/s6/s7 Python loops.

### Notable incidents during the investigation
- s7 W2 GPU path bug (`83d230d`) landed mid-investigation. Re-established baseline at res=256 e2e ~10.2 s (vs pre-fix 9.13 s).
- Sub-stage CUDA event patches added 41% overhead at res=128 (`3af0d99` smoke check) → fell back to stage-level NVTX only (`36a1e9d`).
- NVTX monkey-patches from Tasks 2-3 don't appear in torch.profiler Chrome trace (only nsys). Layers 1-3 fell back to matching `python_function` events by stage entry name.
- torch.profiler Chrome traces at res=256 are 3.8 GB each (above GitHub's 2 GB file limit). Kept local-only; 11.4 GB across 3 runs. For future runs, reduce with `with_stack=False`.

## 2026-04-17 — sync-spike findings complete

- Spec: `docs/superpowers/specs/2026-04-17-sync-spike-design.md` (commit 687b02f)
- Plan: `docs/superpowers/plans/2026-04-17-sync-spike-implementation.md` (commit b9048bf)
- Driver tooling: `tmp/profile_deep/driver.py` with_stack default flipped to False, --with-stack CLI flag added (ROI #6, commit 99c06b7)
- Findings: `logs/findings_sync_sources.md` — 1.0s DeviceSync root-caused (Bucket C @ s1_voxelize.py:56); top-20 blocking D2H classified into 5 buckets (A=2, B=6, C=7, D=4, E=0); recommended next spec = Option Y "A + §1 DeviceSync fix" with expected Δ -1.0 to -1.1s @ res=256, ~3-5 d effort
- Branch: `post-profile-sync-elim`
- corep_fast changes: zero (investigation-only spec; fixes deferred to next sync-elimination spec)
- Key architectural insight: Bucket B = 98% of top-20 ms = 6 bulk `.cpu().numpy()` dispatches at GPU→CPU-MP boundary in s6/s7/s4/s8. Architectural rewrite belongs with Triton K1/K2 track, NOT bundled into next sync spec.

## 2026-04-17 — T0 decisive experiment: MP vs serial e2e wall-time

- Two subagents ran corep_pipeline @ res=256 on host-10-240-99-116, 3 trials each.
- **MP default (GPU 3): median 8.631 s** (V=551079, F=1102152, variance 4.3%)
- **Serial nw=1 (GPU 4): median 75.312 s** (same V/F, variance 0.15%, bit-deterministic)
- **Serial is 8.7x slower than MP.** MP is net-positive; plan direction (W2 persistent pool) is correct.
- Full writeup: `logs/findings_t0_mp_vs_serial.md`
- Artifacts: `tmp/cpu_profile/t0_driver.py`, `t0_{mp_default,serial}.md`, `t0_{default,serial}.json`
- Plan update: T1 uses nw=1 for bit-exact golden; T3 (W2) adds PYTHONHASHSEED=0 in pool initializer to fix MP topology nondeterminism (pre-existing ~0.7% vertex-set drift).
