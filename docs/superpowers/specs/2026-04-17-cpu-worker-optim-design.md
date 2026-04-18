# CPU MP Worker Optimization — Design Spec (V2, data-driven reframe)

**Date:** 2026-04-17
**Branch:** `post-profile-sync-elim` (continuation; user directed to stay on current branch)
**Data basis:**
- `logs/findings_sync_sources.md` (sync-spike, commit `c64b7cc`)
- `tmp/cpu_profile/findings_cpu_worker.md` (worker-body profile)
- `tmp/cpu_profile/findings_main_thread.md` (main-thread cProfile — **decisive data for this V2**)

**V1 → V2 reframe:** V1 assumed Bucket B `.cpu().numpy()` (198 ms) was the dominant cost and built W1-W4 around MP worker optimization. Main-thread cProfile showed Bucket B is only ~0.2 s of 10 s; the real cost is **275k main-thread Python iterations** across two stages plus **124 per-dispatch MP Pool forks**. V2 rebuilds the scope around this evidence.

---

## 1. Goal

把 corep_fast e2e wall-time 从 ~10 s @ res=256 下拉 3-5 s（30-50 %），**不引入 Triton**。策略：消除主线程 275k 次 Python cube 循环、合并 124 次 MP Pool fork、把已在 GPU 上的数据保留在 device 而不是 round-trip 回 CPU。

## 2. Background — what the profile actually says

e2e measure @ res=256 (with cProfile overhead): 12.37 s; per-stage wall:

| Stage | Wall | % of e2e |
|---|---:|---:|
| s4 | 6.69 s | 54 % |
| s6 | 3.50 s | 28 % |
| s7 | 1.74 s | 14 % |
| s1+s2+s3+s8 | <0.4 s | ~3 % |

Top main-thread hotspots (self-time, from `tmp/cpu_profile/results_main/main_thread_res256.prof`):

| Rank | Self (ms) | Calls | Function | File:line |
|---:|---:|---:|---|---|
| 1 | 2711 | — | `_thread.lock.acquire` | main thread blocked on `Pool.map` in `s4_face_point.py:1186` (Stage D workers) |
| 2 | 1891 | 275 426 | `_fastpath_trace_loops_numpy` | `corep_fast/stages/s6_collapse.py:379` (main thread, **NOT MP-dispatched**) |
| 3 | 1030 | 275 541 | `_get_local_components_np` | `corep_fast/stages/s4_face_point.py:734` (main-thread Python UF) |
| 4 | 816 | 124 forks | `posix.fork` | per-dispatch `Pool(num_workers)` context, 4 sites across s4/s6/s7/s8 |
| 5 | ~1010 | — | s7 orchestration / CSR conversion | `corep_fast/stages/s7_rank_assign.py:1175` (needs further drill-down) |

**Key counter-findings (contradicting V1 assumptions):**

- **`.cpu()` / `.numpy()` D2H is NOT the bottleneck.** Total `.cpu()` self-time = 232 ms / 55 calls; `.to()` = 43 ms / 228 calls. `findings_sync_sources.md`'s "Bucket B 198 ms (98 %)" was true within its own top-20 universe (torch.profiler memcpy events) but only ~2 % of true wall-time.
- **Worker bodies are trivial.** Prior worker-profile showed s6/s7/s8 workers combined = 60 ms. The 9.4 s of "missing time" is NOT in worker bodies.
- **Only 2 stages actually route through the monkey-patched workers.** s6 fast-path runs entirely on main thread (275 k iterations); s4 Part 1 GPU path skips `_fw_worker_indexed`. Workers that DO run (s6 slow-path, s7, s8) handle tiny fractions of cubes.

**Implication:** the architecture is "main-thread-bound Python over N≈275 k work items plus fork-heavy Pool orchestration", not "GPU-bound with D2H leaks". V2 targets the former.

## 3. Scope

### 3.1 In-scope

| # | Work item | Target file:line (primary) | Expected Δ | Effort |
|---|---|---|---:|---|
| **W1** | Bucket A cleanup (inherited from V1) | `s8_collapse.py:1629` + 1 secondary | ~0 s (cleanup) | 2 h |
| **W2** | Persistent MP Pool across stages (124 forks → 1 reused) | 4 sites: `s4:1186`, `s6:925`, `s7:1315`, `s8:1428/1769/1939` | ~0.8 s (reclaims `posix.fork` 816 ms) | 1-2 d |
| **W4** | s6 fast-path tracer GPU-vectorize | `s6_collapse.py:379 _fastpath_trace_loops_numpy` (275 k calls on main thread) | 1.5-2 s | 3-5 d |
| **W5** | s4 Part 2 batched GPU union-find | `s4_face_point.py:734 _get_local_components_np` (275 k calls, 1030 ms self) | 0.8-1.2 s | 3-5 d |
| **W6** | s4 Part 1 Stage D: eliminate `lock.acquire` blocker | `s4_face_point.py:1186 Pool(...) with p:` (2711 ms main-thread wait) | 1.0-1.8 s | 3-7 d (depends on W2 outcome; partially covered by W2) |
| **W7** | s7 main-thread orchestration slim-down | `s7_rank_assign.py:1175` onwards (~1010 ms self in orchestration / CSR / work-item build) | 0.3-0.6 s | 2-3 d |

**Target total wall reduction: 3.4-5.6 s @ res=256.** DoD conservative threshold: **≥3 s** (§8 item 6).

### 3.2 Dropped (was V1 W3 — `.cpu().numpy()` payload trim)

V1 W3 is **deleted** from scope. Data: total `.cpu()` self = 232 ms across 55 calls. Even 100 % removal saves <250 ms — not worth the per-site regression-risk surface when W4-W6 are available at 2-5× the ROI.

### 3.3 Out-of-scope (unchanged from V1)

- §1 1.0 s `.item()` sync at `s1_voxelize.py:56` — deferred to future Triton-introduction spec
- Any new external dependency (Triton / Cython / Numba / Rust)
- s1/s2/s3/s8 optimization (combined <0.4 s; below noise floor)
- Bucket C / D / E sites from `findings_sync_sources.md` §2

### 3.4 VRAM discipline

- W4/W5/W6 keep already-on-GPU tensors on device instead of `.cpu()`. Net VRAM impact is **small positive increase** (the tensors exist either way; we skip the pinned-buffer staging). Each W task must record peak VRAM @ res=256 before/after.
- W2 (persistent pool) affects CPU RSS, not VRAM.
- Hard limit: spec rejects any W variant that raises peak VRAM by >500 MB @ res=256 vs. pre-change baseline without explicit user sign-off.

## 4. Specific per-W design notes

### 4.1 W1 — Bucket A cleanup

Inherited verbatim from V1. Two `.item()` call sites where the returned scalar is only used for Python-side logging/debug. Remove or defer to end-of-pipeline collection. See `logs/findings_sync_sources.md` §2 (rank 8 + secondary A row).

### 4.2 W2 — Persistent MP Pool

**Current reality (from code survey):**

```python
# s4_face_point.py:1186, s6_collapse.py:925, s7_rank_assign.py:1315,
# s8_collapse.py:1428/1769/1939 (3 sites in s8) — all share this pattern:
from multiprocessing import Pool as _Pool
with _Pool(num_workers) as p:
    results = p.map(_worker_fn, chunks)
```

Each `with _Pool(...)` fork()s num_workers children, runs, joins, tears down. `posix.fork` was called 124 times over e2e → many of these context managers execute multiple times per pipeline run.

**Design:**

- New module `corep_fast/utils/persistent_pool.py` exposes a **lazily-initialized module-level pool** (or a `get_or_create_pool(num_workers)` function with LRU cache by `num_workers`).
- Pool is created on first use, reused across stages, shut down at pipeline exit (or kept alive for batch inference).
- Replace each `with _Pool(...) as p:` site with a call that uses the shared pool.
- Thread-safety: if corep_fast is ever called from multiple threads simultaneously, pool access needs a lock. For now, assume single-threaded orchestration (matches current invariant).

**Risk:** inherits the classic multiprocessing pool pitfalls — if a worker crashes the shared pool is poisoned for subsequent stages. Mitigation: detect exceptions from `p.map` and re-create pool on next use.

**Coverage with W6:** if W6 moves Stage D off MP entirely, some of W2's value for s4 evaporates. Keep both tracked; if W6 lands first, re-estimate W2's delta before committing to it.

### 4.3 W4 — s6 fast-path tracer GPU-vectorize

**Current code path (`s6_collapse.py:~320-450`):**

1. `_fastpath_gpu_build_adjacency` produces edge adjacency as GPU tensors.
2. That adjacency is `.cpu().numpy()`'d.
3. `_fastpath_trace_loops_numpy` iterates per-cube in a Python loop (275 k cubes), walking the numpy adjacency to emit loop traces.

**Proposed design:**

Replace the numpy loop with a GPU kernel sequence that traces loops in parallel per cube. Since each cube has ≤12 edges, loop-tracing is a small per-cube graph walk; vectorizable with padded tensors + gather / scatter.

- Write `_fastpath_trace_loops_gpu` producing the same output signatures as `_fastpath_trace_loops_numpy` (loops as GPU tensors).
- Downstream consumer: verify `_collapse_with_uturns_tracked` (worker fn, runs on 113 slow-path cubes) can accept GPU input or needs a `.cpu()` fallback for the slow-path.
- Fast-path: full GPU pipeline, no main-thread iteration.
- Slow-path (113 cubes with pathological topology): unchanged, still routes through `_s6_worker`.

**Risk:** topology correctness is load-bearing for s6 (collapse decisions depend on loop orientation). Every commit under W4 must pass F1-F3 fixtures (§5) with bit-exact tensor comparison.

### 4.4 W5 — s4 Part 2 batched GPU union-find

**Current code path (`s4_face_point.py:~720-780`):**

`_compute_component_points_gpu` calls `_get_local_components_np` per cube (275 k times), each running a small Python union-find over ≤12 faces to compute component labels.

**Proposed design:**

Batched per-cube parallel UF on GPU. Two options:

- **(a) Iterative label-propagation:** pad each cube's adjacency to (12, 12); run iterative `min`/`max` propagation until labels converge (max ~log₂(12) ≈ 4 iterations). Vectorized across all 275 k cubes.
- **(b) Sparse segmented UF:** use `torch_scatter`-style segment-reduce ops (no new dep — stock PyTorch `scatter_reduce_` suffices).

Option (a) is simpler; start there. Option (b) only if (a) shows numerical or bandwidth issues.

**Risk:** UF label convention (which member of a component becomes the "root") differs between numpy code and label-propagation; downstream code must not depend on specific label values, only on equivalence classes. Audit `s4_face_point.py:~780` onward for label-value dependencies; encapsulate if needed.

### 4.5 W6 — s4 Part 1 Stage D unblocking

**Current code path (`s4_face_point.py:~1150-1250`, Stage D):**

`_compute_face_weights_gpu` Stage D dispatches BFS + U-Turn detection to MP workers via `Pool(num_workers).map`. Main thread blocks on `_thread.lock.acquire` for 2.71 s.

**Three feasible angles (pick during plan drafting after W2 lands):**

1. **(low-risk)** After W2 lands, measure residual Stage D wait. If it drops significantly (fork overhead gone), declare W6 done.
2. **(medium)** Move BFS+UTurn to GPU. Stage B already produces CSR adjacency on device; Stage D could consume it on GPU directly with a vectorized BFS (batched per cube, bounded depth).
3. **(high risk)** Parallelize via thread pool instead of process pool — avoids fork but hits GIL. Only viable if BFS+UTurn kernel releases GIL (pure numpy/scatter ops do; Python loops don't). Measure before committing.

Plan task chooses angle after W2's delta is measured.

### 4.6 W7 — s7 orchestration slim-down

**Current code path:** 1.01 s self-time in `s7_rank_assign.py:1175`-ish range, covering CSR conversion + work-item construction for the rank-assign MP dispatch.

**Action:** drill into this block during plan drafting (read the 100-line window, run `cProfile.stats_by_filename(suffix='s7_rank_assign.py')` against the existing `.prof`). Likely candidates:

- Repeated `.cpu().numpy()` of already-transferred tensors (payload partially overlaps with W3-dropped scope but on a different axis)
- Inefficient Python list-of-lists to ragged tensor conversion
- Redundant recomputation of `counts` / `offsets` from GPU tensors already present

Commit as W7 only if the drill-down identifies a concrete ≥200 ms win; otherwise demote to "future work" and recover the effort budget for W5/W6.

## 5. Topology correctness methodology (unchanged from V1)

First post-profile spec that actually modifies `corep_fast/` semantics. Fixture regression gate is load-bearing.

### 5.1 Fixture matrix

| Fixture | Resolution | Mesh | Baseline |
|---|---|---|---|
| F1 | res=128 | icosphere subdiv=3 (1280 faces) | `custom/` + pre-change `corep_fast/` |
| F2 | res=256 | 同上 | 同上 |
| F3 | res=128 | triple-concentric sphere (r=1.00/1.01/1.02) | 同上 (covers s7 multi-loop path) |

### 5.2 Bit-exact comparison targets

- `cube_indices` (shape / dtype / values bit-equal)
- `V` (vertex count scalar)
- `F` (face count scalar)
- `decode_from_cubebatch` final output (V×3 vertices, F×3 faces)

Allowed float delta: **0 bit-exact** except for W5 label-propagation UF which may reorder equivalence-class representatives. In that case, use a canonical-form comparison (sort within each component, compare as sets).

### 5.3 Per-commit gate

Every commit under W1-W7 must run:

```
pytest tests/corep_fast/test_cpu_worker_optim_regression.py::test_F1_F2_F3 -v
```

Commit message must contain a `Fixture: F1 PASS, F2 PASS, F3 PASS` line; failed fixtures block the commit.

## 6. Files changed

**Modified (production code):**

- `corep_fast/stages/s4_face_point.py` — W1 (secondary A site), W5 (Part 2 UF), W6 (Part 1 Stage D), W2 (pool call site at :1186)
- `corep_fast/stages/s6_collapse.py` — W4 (fastpath tracer), W2 (pool at :925)
- `corep_fast/stages/s7_rank_assign.py` — W7 (orchestration), W2 (pool at :1315)
- `corep_fast/stages/s8_collapse.py` — W1 (primary A site at :1629), W2 (pool at :1428/:1769/:1939)

**Created:**

- `corep_fast/utils/persistent_pool.py` — W2 shared pool helper (new module)
- `tests/corep_fast/test_cpu_worker_optim_regression.py` — F1-F3 fixture gate

**Untouched:**

- `corep_fast/stages/s1_voxelize.py` — deferred to future Triton spec
- `corep_fast/stages/s2_components.py`, `s3_edge_weights.py` — below noise floor

## 7. Task breakdown (high-level; plan refines)

| # | Task | Deps | Effort |
|---|---|---|---|
| T1 | Create F1-F3 fixture regression test against current `post-profile-sync-elim` HEAD | — | 1 d |
| T2 | W1 Bucket A cleanup | T1 | 2 h |
| T3 | W2 persistent pool module + replace 4 call sites | T1 | 1-2 d |
| T4 | Measure W2 delta; decide W6 angle (if residual Stage D wait is small, skip W6 or choose low-risk variant) | T3 | 0.5 d |
| T5 | W4 s6 fast-path GPU tracer | T1 | 3-5 d |
| T6 | W5 s4 Part 2 batched GPU UF | T1 | 3-5 d |
| T7 | W6 s4 Part 1 Stage D (angle from T4) | T3, T4 | 2-7 d |
| T8 | W7 s7 orchestration (drill-down first, commit only if ≥200 ms win) | T1 | 1-3 d |
| T9 | Re-profile + update findings doc with post-fix per-stage wall | T2-T8 | 0.5 d |
| T10 | Handoff: write next-spec recommendation (Triton feasibility after this round) | T9 | 0.5 d |

**Estimated total effort:** 12-22 d (W4 + W5 + W6 are the major time sinks; W1/W2/W7 are shorter).

## 8. Definition of Done

| # | Item | Check |
|---|---|---|
| 1 | Every commit passes F1-F3 fixture gate | Commit message `Fixture: F1 PASS, F2 PASS, F3 PASS` line |
| 2 | W1 landed (2 `.item()` sites removed) | `git diff` + findings §2 update |
| 3 | W2 landed; `posix.fork` count drops from 124 to ≤4 (one per worker count) | `.prof` re-measure |
| 4 | W4 landed; `_fastpath_trace_loops_numpy` call count drops ≥95% OR self-time drops ≥80% | `.prof` re-measure |
| 5 | W5 landed; `_get_local_components_np` call count drops ≥95% OR self-time drops ≥80% | `.prof` re-measure |
| 6 | Measured e2e wall @ res=256 drops ≥ **3 s** vs. `post-profile-sync-elim` HEAD | T9 re-profile |
| 7 | `corep_fast/stages/s1_voxelize.py` unchanged | `git diff 52f5a8c HEAD -- corep_fast/stages/s1_voxelize.py` empty |
| 8 | No new external dependency | `pip freeze` diff empty for python deps |
| 9 | Peak VRAM @ res=256 ≤ baseline + 500 MB | `nvidia-smi` capture in T9 re-profile |
| 10 | W6 and W7 explicitly scoped — landed OR documented-skip with justification | Task notes in `logs/progress.md` |

## 9. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| W4 GPU loop-tracer correctness subtly differs from numpy (edge ordering, loop closure) | High | F1-F3 bit-exact gate per-commit; slow-path fallback for cubes where GPU path disagrees; staging commits so each is independently revertable |
| W5 UF label-propagation gives different representative labels | Medium | Equivalence-class comparison in F1-F3 for s4 outputs; audit downstream consumers for label-value deps |
| W2 persistent pool gets poisoned by a worker exception → downstream stages hang | Medium | Wrap pool access; detect `BrokenPipeError`/similar; lazy re-create |
| W6 angle (GPU BFS) blows up scope: BFS on ragged per-cube graphs is not trivial to vectorize | Medium-High | T4 decision gate; prefer angle (1) "W2 fixes it"; if angle (2) picked, require a 1-d spike before committing to 3-7d path |
| Total e2e reduction < 3 s (DoD #6 miss) | Medium | Mid-spec review after T6 (W4+W5 done): if Δ ≥ 2 s, decide whether to push harder on W6/W7 or accept partial win |
| Pool persistence breaks a test that assumes Pool shutdown between stages | Low | All tests run under F1-F3 fixture which covers full pipeline |
| W4/W5 increase VRAM (keeping tensors on device vs. round-trip) beyond 500 MB | Low | Per-task VRAM delta measurement; early exit if exceeded |
| `.item()` at s1:56 turns out to be load-bearing for the 1 s idle bubble and this spec makes GPU idle MORE visible (Amdahl) | Medium | Post-fix profile will re-measure; ROI of Triton spec gets re-evaluated with new denominator |

## 10. Handoff to the next spec

After this lands (T9 re-profile done):

- `logs/findings_cpu_worker_post_fix.md` documents new hotspot distribution
- Residual e2e wall (expected: ~5-7 s) informs whether Triton is finally worth the maintenance cost
- VRAM baseline reported for future pre-alloc-buffer feasibility analyses

## 11. Deltas from V1 (for reviewers)

| V1 | V2 | Reason |
|---|---|---|
| W3 (Bucket B payload trim, 1-2d) | **Dropped** | `.cpu()` is 232 ms total, not worth the per-site regression risk |
| W4 (algorithmic worker-body fix) | **Replaced** by new W4 (s6 fastpath GPU trace), W5 (s4 UF), W6 (s4 Stage D), W7 (s7 orchestration) | Worker bodies are 60 ms total; real cost is main-thread |
| §1 1.0 s sync handled via Triton (Option Y) | Still deferred to separate Triton spec | Unchanged |
| Effort estimate 7-14 d | 12-22 d | V2 scope is larger (4 real dev items vs. 2) but addresses the actual bottleneck |
| Wall-time target ≥1 s | **≥3 s** | Profile data supports a higher target |
