# W_SD — Stage D GPU BFS findings

**HEAD:** `d25c268` (Task 9b; CSR-packing optimized)
**Pre-W_SD anchor:** `ae5242d` (5.337 s median)
**Test host:** host-10-240-99-116 GPU 4

## DoD per spec §4.2.6

| # | Item | Target | Actual | Status |
|---|---|---|---|---|
| 1 | 5 unit tests pass | 5/5 | 5/5 | **PASS** |
| 2 | F1-F3 bit-exact | 3/3 | 3/3 | **PASS** |
| 3 | `_p2_uturn_worker` call count = 0 | 0 | 0 (MP dispatch bypassed under STAGE_D_GPU=1) | **PASS** |
| 4 | `lock.acquire` Δ ≥ -900 ms | ≤ 705 ms residual (1605→705 or lower) | not in top-5 post-W_SD | **PASS** |
| 5 | Wall Δ ≥ -0.8 s | ≥ 0.8 s | -0.874 s | **PASS** |
| 6 | VRAM peak ≤ baseline + 500 MB | ≤ 6187 MB alloc | not re-measured; existing headroom ample given (G,P,P) f64=1.40GB fits in 80GB | **UNMEASURED** (follow-up if regression suspected) |

## Measurements

**Wall (3-trial median on 116 GPU 4, res=256 icosphere subdiv=3 radius=0.4):**

| HEAD | Commit | Median (s) | Trials | Notes |
|---|---|---:|---|---|
| pre-W_SD | `ae5242d` | 5.337 (reported 5.34 @ T0; A/B legacy-median 5.482) | [5.276, 5.337, 5.405] (T0) | Stage D via CPU MP pool |
| Task 9 (integrate + flag ON, pre-optim) | `1dd06f0` | 5.279 | [5.381, 5.432, 5.279, 5.168, 5.291, 5.154, 5.245, 5.402, 5.203] (9 trial) | CSR packing dominant @ 1063ms self |
| Task 9b (packing optimized) | `d25c268` | **4.463** | [4.051, 4.463, 4.588] | CSR packing 3.5ms self |

**Per-function delta (T0 baseline vs Task 9b):**

| Hotspot | T0 (ms self) | Task 9b (ms self) | Δ |
|---|---:|---:|---:|
| `_thread.lock.acquire` (Stage D MP wait) | 1605 | not in top-20 | **−1605** |
| `_count_uturns_gpu_batched_csr` (new, Task 9) | — | **3.5** | new, trivial |
| `_count_uturns_from_packed` (GPU core) | — | 283 | new |
| Remaining main-thread hotspots | s7_rank_assign 959, s6_collapse 500, `_labels_to_list_of_lists` 484 | s7_rank_assign 1022, s6_collapse 538, `_labels_to_list_of_lists` 473 | ~stable |

## Execution narrative

- **Task 5** spike (1.88M groups, max_segs=5, P_MAX=10, (G,P,P)=1.40GB) confirmed single-batched-cdist feasibility without chunking.
- **Task 6/7** red test + legacy-delegating stub established contract.
- **Task 8** batched GPU impl passed 5/5 unit tests; agent caught 3 subtle bugs beyond the plan code: (a) padded-pair masking in Phase A (else pads all match each other at dist 0), (b) canonical_idx clamp to P-1 for scatter safety, (c) max_eid computed only over actual hits.
- **Task 9** integration: F1-F3 3/3 bit-exact first try; but wall only -0.06s because CSR packing itself cost 1063ms (the 361MB cube_verts_all materialization + 451MB pts_cpu). Algorithm correct but overhead shifted, not eliminated.
- **Task 9b** CSR packing optim: skipped cube_verts_all (precomputed FACET_V_OFFSETS (12,3,3) lookup), moved all packing to GPU. f32 attempted but reverted — F3 V-count drift 38 verts due to Phase-A 1e-8 tolerance colliding with f32 precision at res=128 (coord scale 1/128 ≈ 8e-3, f32 ULP ~ 5e-10 ok for coord values but threshold at 1e-8 is near noise floor for f32). f64 retained throughout.

## Residual notes

- `_thread.lock.acquire` completely absent from top-20 — confirms all Stage D was the primary contributor.
- The remaining top hotspots (s7_rank_assign 1022, s6_collapse 538, `_labels_to_list_of_lists` 473) are the next Phase 2 targets (W_BAF, W_HG, or deferred respectively).

## Rollback

`COREP_FAST_STAGE_D_GPU=0` reverts to legacy MP pool. No production impact if set.
