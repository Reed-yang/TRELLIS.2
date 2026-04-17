# Task 0 baseline — concerns to address before Phase 2

**Baseline captured:** commit `d0f6bf5` on 116 GPU 4 (2026-04-17).

## Data vs spec §2.1 — 4/5 hotspots matched, 1 missing

| Hotspot | Spec expected self_ms | T9 artifact self_ms | T0 baseline self_ms | Status |
|---|---:|---:|---:|---|
| `_thread.lock.acquire` | 1648.5 | 1648.5 | 1605.4 | ✓ |
| `s7_rank_assign` | 999.0 | 999.0 | 958.7 | ✓ |
| `s6_collapse` | 513.7 | 513.7 | 499.8 | ✓ |
| `_labels_to_list_of_lists` | 464.1 | 464.5 | 484.0 | ✓ |
| `_build_adjacency_gpu` | 1913.7 | **NOT in top-20** | **NOT in top-20** | ❌ |

**Finding:** `_build_adjacency_gpu` self_ms is NOT in main-thread cProfile top-20, either in
current T0 baseline OR in the original T9 artifact. The spec §2.1's "1913 ms self"
figure is incorrect as a main-thread self_ms measurement.

## Consequence for W_BAF (Task 11-15)

**ROI assumption is unverified.** The spec estimated W_BAF delivers -0.5~1.0s by
reducing `_build_adjacency_gpu` self 1913 → 200 ms (-90 %). If the Python self_ms
is already small (e.g., < 100 ms), the actual wall reduction is upper-bounded by
GPU kernel time, which was not measured directly.

## Mitigation — fold into Task 11 spike

Task 11's spike already measures legacy vs Triton wall time on a 1000-cube synthetic
input. Before writing the full kernel (Task 13), the spike's wall delta must show
≥ 300 ms improvement extrapolated to N=275k; otherwise W_BAF should be **descoped**
or **replaced with Strategy C** (pure PyTorch vectorize of the 576-iteration loop).

If Task 11 spike shows < 100 ms extrapolated gain, escalate to human before Task 13.

## No impact on Phase 1

Phase 1 targets (W_L2L, W_SD) are data-confirmed:
- W_L2L: `_labels_to_list_of_lists` 484 ms ✓ (spec said 464)
- W_SD: `lock.acquire` 1605 ms ✓ (Stage D share is primary contributor)

Phase 1 execution continues unblocked.

## Secondary observation — VRAM ~6 % below spec window

- peak_alloc: 5304 MB (spec said 5687 ± 100) — 7 % low
- peak_reserved: 12406 MB (spec said 13220 ± 400) — 6 % low

Both within ±20 % tolerance. Possibly correlates with `_build_adjacency_gpu` no
longer materializing the large `(N, 12, 3, W)` intermediates (a recent refactor
may have fused / streamed them). Non-blocking.
