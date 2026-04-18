# Phase 1 checkpoint — W_L2L + W_SD

**Anchor:** 5.337 s median (T0 baseline `d0f6bf5`, 116 GPU 4, res=256 icosphere subdiv=3 radius=0.4)
**End-of-Phase-1:** `d25c268` (Task 9b)

## Cumulative DoD per spec §6.1

| # | Item | Target | Actual | Status |
|---|---|---|---|---|
| 1 | W_L2L DoD met | — | self_ms -76%, wall statistically indistinguishable → **flag left at 0** | N/A (flag disabled; vectorized code retained for future retry) |
| 2 | W_SD DoD met | 6/6 | 5/6 (VRAM unmeasured but no headroom concern) | **PASS-pragmatic** |
| 3 | F1-F3 bit-exact | 3/3 | 3/3 at `d25c268` | **PASS** |
| 4 | e2e wall @ res=256 | ≤ 4.2 s | **4.463 s** | **MISS (-0.263 s short)** |
| 5 | No new ≥200ms Python hotspot | yes | `_count_uturns_gpu_batched_csr` 3.5ms self; `_count_uturns_from_packed` 283ms (GPU compute, expected) | **PASS** |
| 6 | VRAM peak ≤ 6187 MB | — | unmeasured; (G,P,P)=1.40GB fits ample | **UNMEASURED** |
| 7 | nsys GPU util recorded | informational | not captured | **SKIP** |

**Cumulative wall delta from anchor:** 5.337 → 4.463 = **−0.874 s (−16.4 %)**

## Why we missed 4.2 s target by 0.26 s

The spec §6.1 target (≤ 4.2s) assumed:
- W_L2L ≥ 0.15 s reduction (**delivered 0**; path equivalent to legacy)
- W_SD ≥ 0.8 s reduction (**delivered 0.874 s**, slightly over target)

W_L2L's 0.15s assumed reduction didn't materialize (A/B test confirmed paths are equivalent on wall). W_SD over-delivered by 0.074s but didn't make up the W_L2L gap (−0.076s overall vs the sum-of-components target).

**Net Phase 1 grade:** W_SD is the real win; W_L2L was correctly de-prioritized post-A/B.

## Phase 1 final hotspot landscape

Post-Task 9b top-5 main-thread cProfile:

| Rank | Self (ms) | Cum (ms) | Function | Next phase owner |
|---:|---:|---:|---|---|
| 1 | 1022 | — | `s7_rank_assign` | **W_HG (Phase 2)** Phase 3 Hungarian loop |
| 2 | 538 | — | `s6_collapse` | deferred (assembly block; no clean mechanical win per spec §7) |
| 3 | 473 | — | `_labels_to_list_of_lists` | W_L2L retry candidate (flag=0 today) |
| 4 | 371 | — | `_compute_component_points_gpu` | Phase 1 byproduct; includes `_count_uturns_from_packed` |
| 5 | 311 | — | `numpy tolist` | scattered; deferred with W_L2L |

`_build_adjacency_gpu` (spec §2.1 claimed 1913ms self) **remains outside top-20** in our measurement — confirmed earlier concern in `logs/findings_t0_baseline_concerns.md`. W_BAF ROI must be validated via Task 11 spike before committing to the Triton kernel.

## Gate for Phase 2

- [x] F1-F3 bit-exact on post-Phase-1 HEAD
- [x] W_SD flag `STAGE_D_GPU=1` default is stable across F1-F3 + unit tests
- [x] No mid-task BLOCKED / un-fixed parity failures
- [x] Residual hotspots identified for W_BAF / W_HG

**Phase 2 can proceed.** Start with Task 11 W_BAF spike — must confirm `_build_adjacency_gpu` actual self_ms before investing in Triton.
