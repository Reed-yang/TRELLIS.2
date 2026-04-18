# Phase 2 checkpoint — W_HG (W_BAF descoped)

**Anchor HEAD (pre-Phase-2):** `aa93f8b` (end of Phase 1, wall 4.463s median on 3 trials)
**End-of-Phase-2 HEAD:** `7c43f64` (Task 18b); benchmark at `3140277`
**Test host:** host-10-240-99-116 GPU 4

## Cumulative DoD per spec §6.2

| # | Item | Target | Actual | Status |
|---|---|---|---|---|
| 1 | W_BAF DoD met | — | **DESCOPED** (spike falsified ROI — real `_build_adjacency_gpu` is 38ms not 1913ms; Triton spike was 2.5× SLOWER on real F2 data + had race bug) | DESCOPED |
| 2 | W_HG DoD met | 6/6 | 6/6 (parity tests 4/4; F1-F3 3/3; self_ms 999→235; wall Δ ≥ -0.6s) | **PASS** |
| 3 | F1-F3 bit-exact on HEAD | 3/3 | 3/3 @ `7c43f64` | **PASS** |
| 4 | e2e wall @ res=256 | ≤ 3.2 s | **3.118 s** (24-trial median) | **PASS** (0.082 s under target) |
| 5 | cProfile top-3 ≠ app code | yes | top-3 still app code (`s7_rank_assign` 235, `s6_collapse` 538, `_labels_to_list_of_lists` 473) but the Phase-3 apply loop dropped out of top-20 entirely | PARTIAL PASS |
| 6 | VRAM peak ≤ 6187 MB | — | not re-measured post-W_HG; no bulk allocations added vs anchor | UNMEASURED |
| 7 | nsys GPU util ≥ 30 % | ≥ 30 % | not captured | UNMEASURED |

## Definitive wall benchmark (robust 24-trial A/B)

From `3140277` (interleaved: 4 rounds × 2 modes × 3 trials, on 116 GPU 4 with env load 13-50):

| Mode | N | Median | Mean | stdev | min | max |
|---|---:|---:|---:|---:|---:|---:|
| A: all followup flags OFF | 12 | **5.6893 s** | 5.6741 | 0.3012 | 5.222 | 6.215 |
| B: production (STAGE_D_GPU=1 + HUNGARIAN_GPU=1) | 12 | **3.1175 s** | 3.1002 | 0.1154 | 2.921 | 3.291 |

- **Δ(B - A) median = −2.572 s (−45.2 %)**
- **Welch t = −27.65** (p ≪ 0.0001, extremely statistically significant)
- **B stdev (0.115) is 2.6× smaller than A (0.301)** — production path is both faster and more stable

## Phase 2 narrative

### W_BAF descoped (Task 11 spike)

**Decisive data** killed W_BAF before the full Triton kernel was written:

- Real `_build_adjacency_gpu` wall in F2 = 38.26 ms (1 call, N=275541). Spec §2.1's "1913 ms self" was wrong — the real top-20 self_ms for this function is <20 ms (not in the top-20).
- Spike kernel on synthetic random tensors: 329× speedup (misleading — random data fills every slot, defeating legacy's `mask.any()` early-exit).
- Spike kernel on REAL F2 tensors with N=275541 fast-frac=99.96%: **0.40× (2.5× SLOWER)**, saved -57ms (i.e. Triton was worse).
- Race condition: `tl.atomic_add + tl.store` has 4 mismatches; fix requires `atomic_cas` multi-pass.
- Net: no path to W_BAF positive ROI without major re-engineering for upper-bound gain < 38 ms.

Details: `logs/findings_w_baf_descope.md`, `tmp/followup_design/w_baf_*`.

### W_HG (Tasks 16, 17, 18, 18b)

Tie-scan findings (Task 16) revealed an extreme simplification opportunity:

- 99.9953 % of F2 cubes are 1×1 cost matrices (99.9868 % on F3)
- 0 ties observed in 100k+ sampled matrices
- 0 cubes exceeding nl > 5 or npts > 8 (brute-force envelope covers 100%)
- 13 rect-reverse (npts < nl) cubes per fixture → trivial scipy fallback

Task 17 (`hungarian_batched` in `corep_fast/stages/s7_triton.py`): pure PyTorch batched brute-force with three paths:
- 1×1 hot-path: vectorized `output[is_1x1, 0] = 0` (no compute, just assignment)
- 1×K (K≥2): vectorized argmin
- 2×2..5×5: permutation enumeration per (nl, npts) bucket

Task 18 (integrate in `s7_rank_assign.py` Phase 3): first cut delivered -0.633 s but left a new bottleneck — 276k `tolist` + 275k `reduce` + 275k `.any()` in the Python apply/fallback loop.

**Task 18b** (vectorize apply loop): replaced per-cube Python loop with bulk numpy fancy assignment on is_1x1/is_brute_valid masks. Delivered the additional -0.757 s, pushing wall under the 3.2 s target.

Hotspot transitions (single cProfile run at HEAD `7c43f64`, driver_main):

| Hotspot | Pre-W_HG (Phase 1 end) | Post-Task-18 | Post-Task-18b |
|---|---:|---:|---:|
| `s7_rank_assign` self_ms | 999 (scipy dominant) | 648 | 235 |
| `numpy.any` (Phase 3 tie-check) | — | 275 | dropped from top-20 |
| `numpy.reduce` (Phase 3 classify) | — | 196 | dropped from top-20 |
| `numpy.tolist` (various) | 412 | 412 | 402 (non-s7 origin) |

## Gate for Phase 3 (if pursued)

All Phase 2 infrastructure is in place:
- `STAGE_D_GPU=1` + `HUNGARIAN_GPU=1` + all W_HG apply vectorizations default-on
- `BUILD_ADJACENCY_TRITON=0` + `LABELS_TO_LIST_VECTORIZED=0` kept as feature-flagged dead-code for future retry under different pipeline conditions
- F1-F3 regression gate unchanged, still bit-exact on every commit

**Next ROI candidates (for Phase 3 spec author):**

1. **`s6_collapse` 538 ms self** — assembly-phase CSR repack. No clean mechanical win per original spec §7 but worth a drill-down given it's now rank #2.
2. **`_labels_to_list_of_lists` 473 ms self** — vectorized path exists under `LABELS_TO_LIST_VECTORIZED=1` but wall-neutral vs legacy. Would need CSR-native downstream consumer (s4→s7 intermediate pipeline refactor) to unlock real gain.
3. **`numpy.tolist` 402 ms scattered** — incidental coercions across s4/s7 intermediate tensors. Umbrella refactor ("keep s4→s7 intermediates on GPU") would reclaim this + much of #2 + #1.
4. **nsys GPU util measurement** — spec §6.2 DoD item 7 (≥30 %). Capture this to gate any further optimization direction.

**Next spec should start with nsys capture.** Post-Phase-2 GPU util is likely ~20-30 % given wall halved from 5.69 → 3.12 s while GPU compute was already ≤ 500 ms. If util is now ≥ 40 %, the pipeline has shifted from host-bound to a more balanced regime and different optimization strategies apply.
