# CPU Worker Optimization — Handoff (V2 close-out)

**Date:** 2026-04-17
**Branch:** `post-profile-sync-elim`
**HEAD at close:** `c0a29566bf2f48f5e707f3321d516e39cc9d835e` (T7a W6 GIL spike)
**DoD status:** **MET** (all 6 DoD items green; see §Measured impact)
**Spec:** `docs/superpowers/specs/2026-04-17-cpu-worker-optim-design.md`
**Plan:** `docs/superpowers/plans/2026-04-17-cpu-worker-optim-implementation.md`

Standalone handoff for the next CPU/GPU-perf spec author. Cross-references every
raw artifact so pickup does not require rebuilding context.

---

## 1. What was delivered (3-line summary)

1. Persistent MP pool across stages (W2) eliminated 124 per-dispatch `posix.fork` calls; 19 `lock.acquire` waits collapsed to 4.
2. GPU-batched padded-walk loop tracer (W4) replaced 275 426 numpy `_fastpath_trace_loops_numpy` calls on main thread with 1 GPU call (~7 ms).
3. GPU-batched label-prop union-find (W5) replaced 275 541 numpy `_get_local_components_np` calls with 1 GPU call (~0.1 ms).

Net effect: CPU-main-thread hotspot table collapses three top-5 rows (1822 ms,
1014 ms, 816 ms) to ~7 ms; residual top hotspot is now `lock.acquire` (Stage D
pool.map wait) — see §5 for why that path was closed out and deferred.

---

## 2. Measured impact

### End-to-end wall (res=256, icosphere subdiv=3, host-10-240-99-116 GPU 4, 3-trial median, clean — no cProfile)

| Mode | Wall (s) | Δ vs baseline |
|---|---:|---:|
| Baseline MP default (T0 `b6bb12c`) | **8.631** | — |
| Serial nw=1 (T0 decisive expt) | 75.312 | **+66.68 s** (confirms MP is net-positive 8.7×) |
| Post-W2+W4+W5 (T9 `1ced857`) | **5.354** | **-3.277 s (-37.97 %)** |

Raw: `tmp/cpu_profile/t9_clean_wall.json` (samples: 5.421, 5.224, 5.354).
VRAM peak post-W5 = 5687.8 MB allocated / 13220.4 MB reserved
(`tmp/cpu_profile/t9_vram.log`). No VRAM regression (spec ceiling: +500 MB).

### Per-function elimination (pre-W2 vs post-W5, main-thread cProfile)

| Function | pre-W2 calls / self_ms | post calls / self_ms | Δ |
|---|---:|---:|---|
| `posix.fork` | 124 / 816.3 | 0 / 0.0 | -100 % (W2) |
| `_thread.lock.acquire` | 19 / 2710.8 | 4 / 1648.5 | -1062 ms |
| `_fastpath_trace_loops_numpy` (s6) | 275 426 / 1822.5 | 0 / 0.0 | -100 % (W4) |
| `_get_local_components_np` (s4) | 275 541 / 1013.5 | 0 / 0.0 | -100 % (W5) |
| `_fastpath_trace_loops_gpu` (NEW W4) | — | 1 / 7.0 | new |
| `_get_local_components_gpu_batched` (NEW W5) | — | 1 / 0.1 | new |

Full top-20 at `tmp/cpu_profile/t9_final_hotspots.txt`.

### DoD check

| # | Item | Status |
|---|---|---|
| 1 | No Triton, no new deps | PASS — pure PyTorch vectorization + stdlib MP |
| 2 | W2 fork reduction ≥90 % | PASS — 124 → 0 |
| 3 | W4 `_fastpath_trace_loops_numpy` elimination ≥80 % | PASS — 275 426 → 0 |
| 4 | W5 `_get_local_components_np` elimination ≥80 % | PASS — 275 541 → 0 |
| 5 | e2e wall reduction ≥3 s | PASS — -3.277 s (9.2 % margin over bar) |
| 6 | F1-F3 bit-exact vs nw=1 goldens | PASS — 3/3 every commit from T5c onward |

---

## 3. What was tried and rejected

### T0 decisive experiment (`b6bb12c`) — "Delete MP" hypothesis DEAD

Pre-V2 main-thread cProfile reported 2.7 s `lock.acquire`; worker cProfile reported
~60 ms worker Python self-time. The naive reading said "MP is pure overhead,
serial would reclaim 2.7 s."

Two-subagent decisive wall-time run (`tmp/cpu_profile/t0_mp_default.md` vs
`t0_serial.md`): **serial was 8.7× slower (75.3 s vs 8.6 s)**. cProfile
`self`-time is Python-only and misses C-extension time — the 60 ms worker
Python self hid seconds of torch/numpy kernel compute that was genuinely
parallelizing. Keeping MP was unambiguously correct.

Cost: ~15 min of subagent wall. Saved: days of misdirected W2/W6 work.
Artifact: `logs/findings_t0_mp_vs_serial.md`.

### W1 Bucket A audit (T2 `09f003e`) — zero safe deletions

Target functions `_process_shared_edges_from_tensors` (s8:1629-1800) and
`process_geometry_vectorized` (s8:920-1247) scanned for `.item()` / `.cpu()` /
`.tolist()`. Found 2 sites, both **load-bearing** (Bucket C alloc-size and
Bucket B bulk transfer). 0 deletions shipped; purpose served was to validate
the W1-W7 commit-flow mechanics before tackling larger work.

Artifact: `tmp/cpu_worker_optim_audit/w1_s8_audit.md`.

### W6 Angle 3 (ThreadPool) DEAD — GIL-holding 57.9 % (T7a `c0a2956`)

After W2 collapsed Stage D wait from 2.7 s to 1.67 s, the decision tree in
`logs/findings_w6_angle_decision.md` prescribed either Angle 2 (GPU
BFS+UTurn, 3-5 d) or Angle 3 (ThreadPool, contingent on GIL-holding < 30 %).

T7a spike cProfile-instrumented `_p2_uturn_worker` via SerialPool monkeypatch:
- 125 273 ms worker wall
- 71 657 ms Python self (GIL-holding)
- 52 080 ms C-ext self (GIL-releasing)
- **GIL-holding lower bound: 57.9 %** — nearly 2× the 30 % viability threshold
- `_count_uturns` alone: 49 679 ms (40 % of total self_tt)

ThreadPool replacement would be serialized by the GIL — dead on arrival. With
DoD already MET by W4+W5 and Angle 2 estimated at 3-5 d for a ~1-1.5 s gain,
recommendation was skip W6 entirely. Artifact: `logs/findings_w6_gil_spike.md`.

### W7 s7 orchestration drill-down (T8a `ee7a9c1`) — no mechanical win

Drill into `s7_rank_assign.py` looking for ≥200 ms mechanical cleanup:
- `_build_adjacency_gpu` (1913 ms self) — genuine GPU compute, Triton
  territory, NOT orchestration.
- `s7_rank_assign` Phase 3 loop (984 ms self) — 275 539-iteration Hungarian
  per cube; vectorizing requires padded cost-matrix + custom GPU Hungarian
  (major refactor, not cleanup).
- Top-of-function redundant `.cpu().numpy()` × 4 (lines 1206/1207/1215/1218):
  two dead vars (`ci_np`, `fw_np`), two legacy-path-only (`ew_np`, `uturn_np`).
  Total attributable ≈ 29 ms — below threshold.

Verdict: skip W7. Minor hygiene (15-25 ms) is worth folding into any future
s7 edit, not a standalone ticket. Artifact:
`tmp/cpu_worker_optim_audit/w7_s7_drilldown.md`.

### W4 Option A residual

Per `tmp/cpu_worker_optim_design/t5a_s6_fastpath_gpu_sketch.md` §4, T5d shipped
Option B (CSR → list-of-lists adapter, ~100-300 ms absorbed into the numpy
repack post-tracer). Option A (emit CSR directly from GPU + teach downstream
consumers to accept CSR) would reclaim the last 10-20 % of the W4 win.
Deferred — low ROI vs s7/Stage D work.

---

## 4. Branch + commit chain

Linear, 17 commits on `post-profile-sync-elim` since sync-spike progress
(parent `0b1ef14`):

```
c0a2956 cpu-worker-optim(t7a/w6): GIL-holding spike — Angle 3 viability check
ee7a9c1 cpu-worker-optim(t8a/w7): s7 orchestration drill-down — ≥200ms win check
1ced857 cpu-worker-optim(t9):    clean post-W2+W4+W5 re-profile + final findings
ff461a3 cpu-worker-optim(t5d/w4): _fastpath_trace_loops_gpu vectorized + integrated
d16c73c cpu-worker-optim(t6d/w5): batched GPU label-prop UF + integrated
2b72a98 cpu-worker-optim(t4):    W6 angle decision post-W2
c923e85 cpu-worker-optim(t3/w2): persistent MP pool across stages
276b715 cpu-worker-optim(t6c/w5): _get_local_components_gpu numpy-delegating stub
2cd8167 cpu-worker-optim(t5c/w4): _fastpath_trace_loops_gpu numpy-delegating stub
09f003e cpu-worker-optim(t2/w1):  s8_collapse.py Bucket A audit + safe removals
6d3dada cpu-worker-optim(t6b/w5): s4 local UF TDD scaffold (red phase)
243b349 plan(cpu-worker-optim):   fix stale T6c note — face_adj is GLOBAL
244ad96 cpu-worker-optim(t5b/w4): s6 fast-path tracer TDD scaffold (red phase)
9061c7f cpu-worker-optim(t5a,t6a):spike design sketches (s6 tracer, s4 UF)
a3daaf5 plan(cpu-worker-optim):   note T1 shipped design diverged from Step 1
949aed9 cpu-worker-optim(t1):     golden-snapshot regression gate F1-F3
b6bb12c cpu-worker-optim(t0):     decisive MP vs serial wall-time experiment
```

This handoff commit appends on top.

---

## 5. Files touched

**Production code** (5 files):
- `corep_fast/stages/s4_face_point.py` (+311 lines): W5 integration, GPU UF call-site, `_snap_centroids_to_components` CSR-adaptation.
- `corep_fast/stages/s6_collapse.py` (+242 lines): W4 integration, GPU tracer call-site, list-of-lists adapter.
- `corep_fast/stages/s7_rank_assign.py` (+8 lines): persistent-pool integration (W2).
- `corep_fast/stages/s8_collapse.py` (+24 lines): persistent-pool integration (W2) + W1 annotations.
- `corep_fast/utils/persistent_pool.py` (+84 lines, NEW): `get_global_pool()` shared-pool factory.

**Tests** (6 files, all NEW):
- `corep_fast/tests/regression/test_cpu_worker_optim.py` — F1/F2/F3 fixture harness.
- `corep_fast/tests/regression/_cpu_worker_optim_runner.py` — subprocess-isolated runner.
- `corep_fast/tests/regression/cpu_worker_optim_goldens/F{1,2,3}_*.pkl` — three bit-exact goldens (icosphere_s3 r128/r256, triple_icosphere r128).
- `corep_fast/tests/unit/test_persistent_pool.py` — W2 unit tests.
- `corep_fast/tests/unit/test_s4_uf.py` — W5 unit tests.
- `corep_fast/tests/unit/test_s6_fastpath_tracer.py` — W4 unit tests.

**Documentation + findings** (7 files, all NEW or append):
- `docs/superpowers/specs/2026-04-17-cpu-worker-optim-design.md` (V2 reframe spec).
- `docs/superpowers/plans/2026-04-17-cpu-worker-optim-implementation.md` (10-task plan).
- `docs/superpowers/specs/2026-04-17-cpu-worker-optim-handoff.md` (this file).
- `logs/findings_cpu_worker_post_fix.md` (post-W5 summary + this handoff's Section 4).
- `logs/findings_t0_mp_vs_serial.md` (T0 decisive expt).
- `logs/findings_w6_angle_decision.md` (post-W2 angle decision).
- `logs/findings_w6_gil_spike.md` (T7a GIL-fraction spike).

**Design sketches + audits** (4 files, all NEW):
- `tmp/cpu_worker_optim_design/t5a_s6_fastpath_gpu_sketch.md` — s6 tracer GPU design.
- `tmp/cpu_worker_optim_design/t6a_s4_uf_gpu_sketch.md` — s4 UF GPU design.
- `tmp/cpu_worker_optim_audit/w1_s8_audit.md` — W1 Bucket A audit.
- `tmp/cpu_worker_optim_audit/w7_s7_drilldown.md` — W7 s7 drilldown.

**Profile artifacts** (14 files under `tmp/cpu_profile/`): pre/post `.prof`
captures, wall-time JSONs, T0/T7a spike scripts + logs, W2 diff parse. Full
list in `git diff b6bb12c^..HEAD --stat`. Total: 52 files, 3801 insertions.

---

## 6. What's next — proposed spec

### Recommendation: **"Stage D + s7 Phase 3 Triton port"**

Scope (three coupled workstreams in one spec):

1. **s7 Phase 3 batched Hungarian on GPU** (replaces 275 539 × scipy
   `linear_sum_assignment` calls, ~1536 ms addressable):
   - Pad per-cube cost matrices to max `(n_loops, n_points)`; run custom
     Triton Hungarian kernel (or warp-parallel assignment for tiny n).
   - Consumer: `s7_rank_assign.py:1175` Phase 3 loop, lines 1485-1512.
   - Dependency: batched GPU input already available via `_build_adjacency_gpu`.
2. **s7 `_build_adjacency_gpu` fusion** (1913 ms self, line 634):
   - Fuse 12×3×W scatter iterations + U-turn loop into one Triton kernel.
   - Lower priority than Phase 3 but in the same file; ship as a single spec.
3. **s4 Stage D GPU BFS + UTurn** (W6 Angle 2):
   - Port `_count_uturns` + its BFS driver onto the on-device CSR segments
     already built in Stage B of `_compute_face_weights_gpu`.
   - Cleans the 1648 ms `lock.acquire` residual.

**Estimated reward:** 1.5–2.5 s e2e at res=256.
**Estimated effort:** 7–12 days (Triton-first spec, careful determinism via
F1-F3 regression gate on every commit).
**Gate DoD:** reuse F1-F3 fixture subprocess pattern + ≥1.5 s e2e reduction +
cProfile post-top-20 must not re-introduce pure-Python hotspots ≥200 ms.

### Secondary / deferred (do not bundle with above)

- W4 Option A (CSR-native consumer downstream of `_fastpath_trace_loops_gpu`)
  — 10-20 % of the current W4 win. Small ROI unless s6 assembly block at
  lines 935-1001 is refactored for other reasons.
- s4 `_labels_to_list_of_lists` vectorization (464 ms self @ `s4:966`) —
  pure Python label-bucketing, candidate for `np.argsort` + `np.split` or
  sparse-COO on GPU. ~200-400 ms reward at ~1 d effort. Standalone micro-spec
  if a future change touches s4.
- Miscellaneous numpy coercion cluster (~759 ms across 276k+ asarray/tolist/
  astype calls) — would be reclaimed incidentally by any "keep intermediates
  on GPU through s4→s7" pass; not a standalone target.

### Closed / rejected paths (do not revisit without new data)

- W6 Angle 3 ThreadPool — dead per T7a GIL 57.9 %.
- W7 s7 orchestration cleanup — no ≥200 ms mechanical candidate per T8a.
- Delete MP entirely — dead per T0 (serial 8.7× slower).

---

## 7. Measurement discipline carry-over

The next spec author should preserve these conditions or the F1-F3 gate will break:

1. **Determinism layers on every regression run:**
   `PYTHONHASHSEED=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
   `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic=True`.
   Plus SerialPool monkeypatch (nw=1) + subprocess-per-fixture isolation.
   Pattern lives in `corep_fast/tests/regression/_cpu_worker_optim_runner.py`.
2. **Clean wall-time is the authoritative metric.** cProfile overhead
   inflates e2e by 15-25 % in this pipeline. Use `tmp/cpu_profile/t0_driver.py
   --mode default` for authoritative numbers and cProfile only for hotspot
   ordering.
3. **cProfile self-time is Python-only.** Cross-check worker timing with
   wall-clock, not cProfile alone. The T0 false-delete-MP pivot cost nothing
   because of the 15-min decisive experiment — but could have cost days.
4. **Test on 119 (GPU 0-3 idle) per user policy** — all profiling already
   used 116/119 GPUs per `feedback_profiling_on_119.md`.

---

## 8. Pickup checklist

If you are the next spec author:

1. Read `logs/findings_cpu_worker_post_fix.md` §4 for the annotated top-20
   with W-coverage status.
2. Run `tmp/cpu_profile/t0_driver.py --mode default --res 256` on 119 to
   confirm the 5.35 s baseline still holds from your HEAD.
3. Run F1/F2/F3:
   `pytest corep_fast/tests/regression/test_cpu_worker_optim.py -v`
   All three must PASS before you start modifying s4/s6/s7.
4. Choose scope (§6 recommendation vs. alternative) and write a V2-style
   data-driven spec — the V2 format in
   `docs/superpowers/specs/2026-04-17-cpu-worker-optim-design.md` is the
   house pattern.

End of handoff.
