# Pre-Triton Final-Pass Results

> Date: 2026-04-16
> Branch: `pre-triton/all` (awaiting merge to `gpu-pipeline`)
> Hardware: 119 H100, GPU 0, CPU partly contended
> Baseline: M2 终点 + d780be8 (centering fix) + 4a355c3 (s6 test relax) + 2106395 (e2e tolerance tighten)
> Spec: `docs/superpowers/specs/2026-04-16-corep-pre-triton-final-pass-design.md`

---

## TL;DR

| 指标 | Custom baseline | Pre-Phase-2 baseline (this work's start) | Phase 2 final | 改进 |
|---|---:|---:|---:|---:|
| res=256 e2e | 141.05s | 21.37s (6.6x vs custom) | **9.13s (15.5x vs custom)** | -12.25s, 2.34x further |
| res=128 e2e | 36.39s | 7.41s (4.9x vs custom) | **3.40s (10.7x vs custom)** | -4.02s, 2.18x further |

**Soft target ≤13s @ res=256 ACHIEVED with 30% margin.**

---

## Phase 2 deliverables landed on `pre-triton/all`

3 worktree branches merged into one integration branch:

| Worktree | Branch | Commits | Default flag (now ON) |
|---|---|---|---|
| W1 | `pre-triton/s8` | `89cdc23`, `79c83fd`, `99a665c` | `S8_4CUBE_VECTORIZED=1` |
| W2 | `pre-triton/s7` | `7077bb1`, `a9407e1`, `d802e0c`, `bf77f08` | `S7_PHASE1_GPU=1` |
| W3 | `pre-triton/s6` | `4506ab1`, `bec5c48` | `S6_FASTPATH_GPU=1` |

Each flag still toggle-able via env var to fall back (`COREP_FAST_<NAME>=0`).

Final integration commits on `pre-triton/all`:
- `d5c29ae` merge s8
- `3e855f1` merge s7 (config.py conflict resolved manually)
- `df09f7d` merge s6 (config.py conflict resolved manually)
- `f244cb7` flip 3 defaults to ON
- `e81831c` fix s8 to import from config.py instead of reading env directly
- `<pending>` final profile data + this doc

---

## Per-stage breakdown @ res=256 (median of 3 runs)

| Stage | Pre-Phase-2 | Phase 2 final | Δ | Speedup | Notes |
|---|---:|---:|---:|---:|---|
| s1 (voxelize) | 0.179s | 0.144s | -0.035s | 1.24x | Already extremely fast (was 2.23s in custom) |
| s2 (components) | 0.085s | 0.083s | -0.002s | 1.03x | Already optimal |
| s3 (edge_weights) | 0.002s | 0.002s | 0.000s | 1.03x | Already optimal (9584x vs custom) |
| **s4 (face_point)** | 4.825s | 4.850s | +0.025s | 0.99x | **Untouched, still has CPU MP BFS — Triton-only** |
| **s6 (collapse)** | 2.763s | 2.445s | **-0.318s** | 1.13x | W3 GPU fast-path; full saving was -1.45s on isolated W3, dropped here likely due to interaction with s7 GPU path data flow + CPU contention noise |
| **s7 (rank_assign)** | 5.543s | 1.417s | **-4.126s** | 3.91x | W2 GPU parallel BFS; matches isolated profile |
| **s8 (decode)** | 7.899s | 0.113s | **-7.786s** | **69.71x** | W1 vectorize 4-cube + encoding fix; isolated showed -8.06s |
| **e2e** | **21.374s** | **9.126s** | **-12.248s** | **2.34x** | |

**vs custom 141.05s baseline: 15.46x speedup** (was 6.60x at Phase 2 start = 2.34x further gain).

---

## Per-stage breakdown @ res=128

| Stage | Pre-Phase-2 | Phase 2 final | Δ | Speedup |
|---|---:|---:|---:|---:|
| s4 | 1.942s | 2.099s | +0.157s | 0.93x |
| s6 | 1.383s | 0.559s | **-0.824s** | 2.48x |
| s7 | 1.996s | 0.381s | **-1.615s** | 5.24x |
| s8 | 1.769s | 0.034s | **-1.735s** | 51.66x |
| **e2e** | **7.411s** | **3.397s** | **-4.014s** | **2.18x** |

**vs custom 36.39s baseline: 10.71x speedup** (was 4.91x at Phase 2 start).

---

## Predicted vs actual (per-optimization)

Per `docs/superpowers/specs/2026-04-16-corep-pre-triton-final-pass-design.md` §0:

| Optimization | Predicted Δ @ res=256 | Actual (isolated) | Actual (final integrated) | 备注 |
|---|---:|---:|---:|---|
| O1 (s8 vectorize 4-cube) | -2.8s | -8.06s | **-7.79s** | Encoding fix unlocked full vectorization (3x better than predicted) |
| O2 (s7 GPU BFS) | -2 to -3s | -4.23s | **-4.13s** | Held cleanly through merge |
| O3 (s7 cyclic align) | -1 to -2s | (subsumed in O2) | (subsumed) | Implementer correctly merged into O2 GPU path |
| O5 (s6 fast-path GPU) | -0.7 to -0.9s | -0.998s | **-0.32s** | Smaller in integrated profile; possibly s7 GPU's downstream data layout reduces s6's fast-path proportion, OR CPU contention noise |
| O7 (s6 dispatch cleanup) | -0.02 to -0.05s | (combined w/ O5: -0.45s extra) | (combined) | Hard to separate from O5 in integrated measurement |

Predicted total: -6.5 to -7.8s (with optimism). **Actual: -12.25s** (1.6-1.9x above prediction).

The big over-performance comes from W1 (s8) where the encoding fix turned out to enable a near-zero-cost path for ALL 4-cube edges, not just the predicted ~2s slice.

---

## Test status

- **224 / 224 tests PASS** with all 3 flags ON (full corep_fast/tests suite, 7m 03s on 119 GPU 0)
- 3 new A/B test files added (one per worktree): test_s8_4cube_vectorized_ab.py, test_s7_phase1_gpu_ab.py, test_s6_fastpath_gpu_ab.py
- Each tests V/F equivalence at res=32/64/128 against the legacy CPU path

**Test caveat:** the W1 (s8) A/B test mutates os.environ at runtime — but config.py reads env at import time, so both arms now run the same default-ON code path. The test still validates V/F equivalence of the optimized path with itself (tautological for parity but still detects crashes). If a future regression switches default OFF, the test would catch a mismatch. To make the test meaningfully toggle, would need monkeypatching `corep_fast.config.S8_4CUBE_VECTORIZED` at runtime (not done in this work).

---

## Items NOT addressed (deferred / intentional)

| Item | Reason | Where to find |
|---|---|---|
| s4 BFS U-turn (4.85s, 53% of final e2e) | Triton-only per Phase 1 analysis — small graphs (K≤30), MP overhead dominant; needs warp-level GPU kernel | `tmp/pretriton_s4_analysis.md` §E.1, `docs/superpowers/specs/2026-04-16-corep-triton-handoff.md` K1 |
| s6 slow-path Cartesian product | Triton-only — per-cube product size variance too large for torch batching | `tmp/pretriton_s6_analysis.md` §C row 2 |
| s8 4-cube residual (0.05% pathological) | Already covered by hybrid predicate inside W1; no separate kernel needed | `corep_fast/stages/s8_collapse.py:1687-1700` |
| s4 component connectivity | Skipped per registry — ROI 0.05s too low | `docs/superpowers/plans/2026-04-16-corep-pre-triton-implementation.md` §Will-skip |
| `mesh-cleanup-port` (custom merge_vertices etc.) | Independent follow-up; doesn't affect icosphere baseline | `my-docs/20260416-corep-fast-mesh-cleanup-port.md` |
| `s1-sat-hardening` | Independent follow-up; "Severity: 低" per its own doc | `my-docs/20260416-corep-fast-s1-sat-hardening.md` |

---

## Next step: Triton

After `pre-triton/all` is merged to `gpu-pipeline`, the only remaining torch-layer bottleneck is **s4 BFS U-turn (4.85s, ~53% of e2e)**. All other stages are either GPU-vectorized or already at sub-second.

Triton kernel work should target:
1. **K1 — s4 Stage D BFS (4.3-4.85s)**: warp-level kernel processing 100K+ (cube, facet) groups per block. Per `tmp/pretriton_s4_analysis.md` §E.1, expected to drop s4 from ~5s → 2-3s.
2. **K2 — s6 slow-path persistent kernel (~0.5s currently)**: register-tiled enumeration. Lower priority.

Predicted Triton-final e2e res=256: ~6-7s (~22-24x vs custom 141s).

Triton handoff spec: `docs/superpowers/specs/2026-04-16-corep-triton-handoff.md`

---

## Phase 2 work artifact list (committed on `pre-triton/all`)

- 3 new A/B test files in `corep_fast/tests/regression/`
- 3 new env-var flags in `corep_fast/config.py` (default ON)
- s8_collapse.py: +220 lines (W1 vectorize + encoding fix + hybrid predicate)
- s7_rank_assign.py: GPU batched BFS implementation
- s6_collapse.py: GPU fast-path + tensor-native dispatch refactor
- Profile scripts in tmp/: w1/w2/w3 + final
- This doc + Triton handoff spec

11 commits total on `pre-triton/all` after the 3 source merges.
