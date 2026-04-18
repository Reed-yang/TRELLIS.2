# corep_fast cpu-worker-optim-followup — V3 Handoff

**Date:** 2026-04-18
**Branch:** `post-profile-sync-elim`
**HEAD at close:** `3140277` (robust 24-trial A/B benchmark commit); last code change at `7c43f64` (Task 18b)
**Spec:** `docs/superpowers/specs/2026-04-17-cpu-worker-optim-followup-design.md`
**Plan:** `docs/superpowers/plans/2026-04-17-cpu-worker-optim-followup-implementation.md`
**Prior handoff (V2 close-out):** `docs/superpowers/specs/2026-04-17-cpu-worker-optim-handoff.md`

---

## 1. What was delivered (3-line summary)

1. **W_SD (Stage D batched GPU BFS + U-turn count):** replaced MP-pool-per-cube Python BFS (1605 ms `lock.acquire` wait) with a single batched GPU pass using vectorized numpy → GPU packing. F1-F3 bit-exact. Delivered Phase 1 single-handedly.

2. **W_BAF (Triton kernel fusion for `_build_adjacency_gpu`): DESCOPED** after spike. Real wall was 38 ms (spec assumed 1913 ms); Triton on real data was 2.5× SLOWER than legacy; design had atomic 2-slot race. Net negative ROI.

3. **W_HG (batched Hungarian on GPU) + W_HG apply vectorization (Task 18b):** replaced 275k scipy.linear_sum_assignment calls + per-cube Python apply loop with bulk numpy fancy-assignment hot-path (99.99 % of cubes are 1×1). F1-F3 bit-exact. Delivered Phase 2 single-handedly after W_BAF descope.

**W_L2L (numpy-vectorize `_labels_to_list_of_lists`): wall-neutral** post A/B re-test — flag retained but defaulted OFF for cProfile observability.

---

## 2. Measured impact

### Robust 24-trial A/B benchmark (definitive) — `3140277`

4 rounds × (A: all followup flags OFF, B: production flags ON) × 3 trials, interleaved on 116 GPU 4 at env load 13-50:

| Mode | N | Median wall | Mean | stdev | min | max |
|---|---:|---:|---:|---:|---:|---:|
| A (pre-followup equivalent) | 12 | **5.6893 s** | 5.6741 | 0.3012 | 5.222 | 6.215 |
| B (production, STAGE_D_GPU=1 + HUNGARIAN_GPU=1) | 12 | **3.1175 s** | 3.1002 | 0.1154 | 2.921 | 3.291 |

**Δ median = −2.572 s (−45.2 %)** · Welch t = −27.65 (p ≪ 0.0001) · B variance 2.6× lower than A.

### DoD check against spec

**Spec §6.1 (Phase 1):**

| Item | Target | Actual | Status |
|---|---|---|---|
| W_L2L | deliver | flag=0 (legacy kept, paths equivalent) | NEUTRAL |
| W_SD | deliver | -0.874 s alone | **PASS** |
| F1-F3 bit-exact | 3/3 | 3/3 | PASS |
| Phase-1 wall ≤ 4.2 s | yes | 4.463 s (first-batch 3-trial) | MISS -0.26 s |

**Spec §6.2 (Phase 2):**

| Item | Target | Actual | Status |
|---|---|---|---|
| W_BAF | deliver | DESCOPED (real-data spike falsified spec §2.1 ROI) | DESCOPED |
| W_HG | deliver | -2.572 s vs A baseline (robust 24-trial) | **PASS** |
| F1-F3 bit-exact | 3/3 | 3/3 | PASS |
| Phase-2 wall ≤ 3.2 s | yes | **3.118 s** | **PASS** (0.082 s under target) |
| cProfile top-3 ≠ app code | yes | top-3 app code, but Phase 3 apply loop eliminated | PARTIAL |
| VRAM ≤ +500 MB (alloc) | yes | **peak_alloc=5304.5 MB (Δ=+0.2 MB vs T0 5304.3)**; peak_reserved=31044.0 MB (+18638 MB arena-only, informational) — see `tmp/followup_baseline/vram_peak_head.log` | **PASS** |
| nsys GPU util ≥ 30 % | ≥ 30 % | not captured | UNMEASURED |

---

## 3. What was tried and rejected

### W_BAF (Task 11 spike `045996b`) — descoped `c1283de`

Two killing facts:

1. **Wrong ROI assumption.** Spec §2.1's "_build_adjacency_gpu self_ms = 1913" does NOT match our baseline cProfile. Real wall per call on F2 (N=275541) is 38.26 ms. The 1913 figure may have come from a cumulative-time view or pre-W2 state; regardless, it's false in our measurement frame.

2. **Triton loses to legacy on real data.** Random synthetic tensors masquerade as a 329× speedup because they fill every `(t_idx, pi, jj)` slot. Real F2 data is sparse (99.96 % fast-path cubes → most `valid_arc[:, t, pi, jj]` entries are False), so legacy's `mask.any()` early-exit skips the vast majority of its 576 iterations. Triton runs all 576 programs unconditionally; on real data, it's 2.5× SLOWER (95.78 ms vs 38.26 ms).

Plus a correctness bug: atomic 2-slot assignment via `tl.atomic_add + tl.store` has a race. Fix requires `atomic_cas` loops or multi-pass reduction — major additional work for an upper-bound saving of < 38 ms.

Conclusion: skip Tasks 12-15 entirely. `corep_fast/stages/s7_triton.py` retained with only the spike kernel (not wired to any call site); `BUILD_ADJACENCY_TRITON` flag defaults to `'0'` and is unused. Artifacts at `tmp/followup_design/w_baf_*`.

### W_L2L (Tasks 1-4) — wall-neutral

First measurement showed vectorized wall +0.362 s regression. A/B interleaved re-test (`ae5242d`) showed the regression was environmental noise (CPU contention on 116 between baseline and first W_L2L measurement). Interleaved data: N=9 per mode, Δ(B−A) median = +0.0003 s (essentially tied), Welch t = −1.075.

Decision: keep flag at `'0'` (legacy) for cProfile observability — the hidden cost of vectorized path (np.split + per-component tolist) is scattered across multiple top-20 rows instead of owned by one function. Code + tests retained for future retry if s4→s7 becomes CSR-native downstream.

### Closed paths (from prior V2 handoff, unchanged here)

- Delete MP: dead (T0 proved serial is 8.7× slower)
- W6 Angle 3 ThreadPool: dead (T7a GIL-holding 57.9 %)
- W7 s7 orchestration cleanup: zero mechanical ≥200 ms candidate (T8a)

---

## 4. Branch + commit chain

Since Phase 1+2 anchor `c0a2956`:

```
3140277 followup(p2): robust 24-trial A/B benchmark — definitive wall data
7c43f64 w_hg(p2): Task 18b — vectorize Phase 3 apply loop
1a38761 w_hg(p2): Tasks 17+18 — batched Hungarian (PyTorch) + Phase 3 integrate
c1283de followup(p2): descope W_BAF — spike falsified ROI on real F2 data
045996b w_baf(p2): Task 11 spike — fast-path Triton kernel + ROI measurement
777618e w_hg(p2): Task 16 — Phase 3 cost-matrix tie scan on F2+F3
6413043 followup(p2): add BUILD_ADJACENCY_TRITON + HUNGARIAN_GPU flags (default off)
aa93f8b w_sd(p1): Task 10 — W_SD findings + Phase 1 checkpoint
d25c268 w_sd(p1): Task 9b — eliminate CSR packing overhead
1dd06f0 w_sd(p1): integrate batched GPU U-turn at Stage D driver (flag ON)
f0a64de w_sd(p1): batched GPU BFS + U-turn count impl (Task 8)
4e528d5 w_sd(p1): _count_uturns_gpu_batched stub (legacy-delegating)
ebeac76 w_sd(p1): red test for _count_uturns_gpu_batched (ImportError phase)
cb812dc w_sd(p1): add STAGE_D_GPU flag (default off) + spike group-size histogram
ae5242d w_l2l(p1): interleaved A/B re-test — no statistically significant wall diff
c621580 w_l2l(p1): default flag to 0 — vectorized path regresses wall +0.36s
9ffb76f w_l2l(p1): post-change findings + wall/hotspot artifacts
cb69001 w_l2l(p1): vectorize _labels_to_list_of_lists bucket loop
7ac331e w_l2l(p1): add parity tests for _labels_to_list_of_lists (green on current HEAD)
bc3ef2c w_l2l(p1): add LABELS_TO_LIST_VECTORIZED feature flag
a837eda followup(t0): note _build_adjacency_gpu absent from main-thread cProfile top-20
d0f6bf5 followup(t0): capture Phase-0 baseline on 116 GPU 4 (F1-F3 green, wall 5.34s)
dfbd3c1 plan(cpu-worker-optim-followup): Phase 1+2 implementation, 20 TDD tasks
9eed5fb spec(cpu-worker-optim-followup): Phase 1+2 combined design
```

---

## 5. Files touched

**Production code (3 files):**
- `corep_fast/stages/s4_face_point.py` — W_L2L vectorized path (flagged off); W_SD stub, batched GPU core, CSR entry, module constant `FACET_V_OFFSETS`; Stage D dispatcher branches on `STAGE_D_GPU`.
- `corep_fast/stages/s7_rank_assign.py` — Phase 3 loop refactored; vectorized apply block behind `HUNGARIAN_GPU=1` feature flag; scipy fallback for rect-reverse + tie cases.
- `corep_fast/stages/s7_triton.py` (NEW) — `hungarian_batched` PyTorch brute-force + `build_adjacency_triton_fast_only` spike kernel (retained but unused in production).

**Config (1 file):**
- `corep_fast/config.py` — 4 new flags: `LABELS_TO_LIST_VECTORIZED` (default 0), `STAGE_D_GPU` (default 1), `BUILD_ADJACENCY_TRITON` (default 0, unused), `HUNGARIAN_GPU` (default 1).

**Tests (3 files NEW, all under `corep_fast/tests/unit/`):**
- `test_labels_to_list_vectorized.py` — 4 parity tests (green against legacy on HEAD regardless of flag)
- `test_stage_d_gpu_bfs.py` — 5 parity tests (green against legacy `_count_uturns`)
- `test_hungarian_batched.py` — 4 parity tests (green vs scipy)

**F1-F3 regression gate unchanged:** `corep_fast/tests/regression/test_cpu_worker_optim.py` + `_cpu_worker_optim_runner.py` reused verbatim. 3/3 bit-exact on every production-affecting commit.

**Findings + design docs (under `logs/` and `tmp/`):**
- `logs/findings_t0_baseline_concerns.md` — flagged `_build_adjacency_gpu` spec-vs-reality gap early
- `logs/findings_w_l2l_vectorized.md` + `logs/findings_w_l2l_ab_rerun.md`
- `logs/findings_w_sd_gpu_bfs.md`
- `logs/findings_w_baf_descope.md`
- `logs/findings_phase1_checkpoint.md`
- `logs/findings_phase2_checkpoint.md`
- `tmp/followup_design/w_sd_stage_d_spike.md`
- `tmp/followup_design/w_baf_triton_kernel_sketch.md`
- `tmp/followup_design/w_hg_tie_scan.md`

**Measurement artifacts (under `tmp/followup_baseline/`, `tmp/cpu_profile/`):**
- T0 baseline + W_L2L A/B (9+9 trials)
- Robust Phase 2 A/B (12+12 trials)
- Per-task wall JSONs and cProfile hotspot dumps

Total: 21 source/test/doc files changed, ~80 files under tmp/ (profile artefacts). 28 new commits on the branch.

---

## 6. What's next — proposed future spec

Post-Phase-2 hotspot landscape (single cProfile, HEAD `7c43f64` with all production flags on):

| Rank | Self (ms) | Function | Next action |
|---:|---:|---|---|
| 1 | 235 | `s7_rank_assign` | largely done; tail is hungarian_batched GPU time itself (~150 ms) + scipy fallback for 15 cubes |
| 2 | 538 | `s6_collapse` | Drill-down candidate (prior spec said "no clean mechanical win" but with new rank, worth a 0.5-day look) |
| 3 | 473 | `_labels_to_list_of_lists` | Vectorized path exists; needs CSR-native downstream to unlock real gain |
| 4 | 371 | `_compute_component_points_gpu` | Includes our `_count_uturns_from_packed` GPU work — genuine GPU compute, Triton-candidate |
| 5 | 402 | `numpy.tolist` (scattered) | Umbrella refactor target |

### Recommended Phase 3 scope

**"nsys-first GPU-util spec"**:

1. **Capture nsys GPU utilization** at HEAD (blocking first step — not yet measured).
2. If util ≥ 40 %: pipeline now in balanced regime; target GPU kernel speedups (Triton is now viable for the remaining compute hotspots).
3. If util still 15-30 %: continue host-side work. s6_collapse drill-down + labels_to_list CSR refactor are the top candidates.
4. W_L2L retry under a different data shape (`list[np.ndarray]` instead of `list[list[int]]`) may become positive if combined with point 3.

### Secondary / deferred

- s6 assembly block (rank 2, 538 ms) — 0.5-day drill-down worth doing before any bigger refactor.
- Umbrella s4→s7 GPU-resident intermediate refactor — high impact but multi-week scope; only justified if GPU util stays < 30 %.

### Closed paths (do NOT revisit without new data)

- W_BAF — quantitatively disproven in this spec run (see §3).
- W_L2L under `list[list[int]]` return contract — A/B proven wall-neutral.
- MP removal, ThreadPool W_SD, W7 orchestration — carried over from V2 close-out, still closed.

---

## 7. Measurement discipline carry-over

Preserve these conditions or the F1-F3 gate will break:

1. **Determinism layers on every regression run:** `PYTHONHASHSEED=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic=True`, SerialPool monkeypatch, subprocess-per-fixture isolation. Pattern lives in `corep_fast/tests/regression/_cpu_worker_optim_runner.py`.

2. **Clean wall-time is the authoritative metric.** cProfile inflates e2e 15-25 % in this pipeline. Use `tmp/cpu_profile/t0_driver.py` for DoD numbers; cProfile only for hotspot ordering.

3. **cProfile self_time is Python-only.** `_build_adjacency_gpu` in the spec was mis-profiled as 1913 ms self (it's 38 ms wall per call). Cross-check wall with direct timing (perf_counter + `torch.cuda.synchronize()`) before investing in any optimization that targets a top-20 hotspot figure.

4. **Interleave A/B on the same invocation cycle** when measuring wall deltas. Non-interleaved captures (the original W_L2L +0.362 s "regression") are contaminated by environmental drift. Our robust 24-trial harness (`tmp/robust_ab_bench_116.sh`) is the reusable template.

5. **Test on 116 GPU 4** per user policy (memory file `feedback_profiling_on_119.md` updated). When 116 is loaded, expect wall variance 0.3-0.8 s; mitigate by interleaving or waiting for quiet windows.

---

## 8. Pickup checklist

If you are the next spec author:

1. Read `logs/findings_phase2_checkpoint.md` (§2 hotspot transitions, §4 nsys gap).
2. Run F1-F3 on HEAD to confirm baseline is still green:
   ```
   ssh host-10-240-99-116 "bash" < tmp/baseline_f123_116.sh
   ```
   All three must PASS before any modifications.
3. Run robust A/B benchmark to confirm the 3.12 s median still holds:
   ```
   ssh host-10-240-99-116 "bash" < tmp/robust_ab_bench_116.sh
   ```
4. **Capture nsys GPU util** (blocking first step of Phase 3 spec — closes the unmeasured DoD item and gates optimization direction).
5. Read `logs/findings_w_baf_descope.md` before anyone proposes another Triton direction on `_build_adjacency_gpu` or similar — the descope decision is data-driven, not preference-driven.
6. Choose scope per §6 recommended and write V3-style data-driven spec.

End of handoff.
