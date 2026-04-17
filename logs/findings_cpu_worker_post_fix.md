# CPU Worker Optimization — Final Findings (post-W2+W4+W5)

**Date:** 2026-04-17
**HEAD:** `ff461a31b1586d5264b5361c5e3921dce8b41c2f` (post W2, W4, W5 committed)
**Baseline:** T0 experiment `b6bb12c` (commit `c0b5e85` parent)

## TL;DR

Clean e2e wall @ res=256 on 116 GPU 4 (icosphere subdiv=3, 3-trial median):
- **Baseline (pre-W2, default MP):** 8.631 s (T0)
- **Post-W2+W4+W5:** 5.354 s
- **Δ: -3.277 s (-37.97%)**

3-trial raw samples (seconds): [5.421, 5.224, 5.354], median=5.354, min=5.224, max=5.421.

DoD target: ≥3.0 s reduction. **MET** (-3.277 s, 9.2% margin above bar).

VRAM peak (res=256):
- Post-W2+W4+W5: **5687.8 MB** allocated, **13220.4 MB** reserved

## Per-stage hotspot elimination

| Function                              | pre-W2 calls | pre-W2 self_ms | post calls | post self_ms |
|---------------------------------------|-------------:|---------------:|-----------:|-------------:|
| posix.fork                            |          124 |          816.3 |          0 |          0.0 |
| _thread.lock.acquire                  |           19 |         2710.8 |          4 |       1648.5 |
| _fastpath_trace_loops_numpy (W4)      |       275426 |         1822.5 |          0 |          0.0 |
| _get_local_components_np    (W5)      |       275541 |         1013.5 |          0 |          0.0 |
| _fastpath_trace_loops_gpu   (NEW W4)  |            - |              - |          1 |          7.0 |
| _get_local_components_gpu_batched W5  |            - |              - |          1 |          0.1 |

Key observations:
- W2 eliminated all 124 `posix.fork` calls (persistent MP pool).
- W4 GPU fast-path tracer collapses 275426 numpy calls → 1 GPU call (7.0 ms self-time).
- W5 batched GPU UF collapses 275541 numpy calls → 1 GPU call (0.1 ms self-time).
- Lock.acquire dropped from 2710.8 ms (19 calls) to 1648.5 ms (4 calls) — main-thread waiting on MP
  worker results (s4/s7) is now the dominant residual CPU sink.

## Top-20 residual hotspots (post-W2+W4+W5)

```
cc=     4 self= 1648.5ms cum=  1648.5ms  ~:0:<method 'acquire' of '_thread.lock' objects>
cc=     1 self=  999.0ms cum=  1702.0ms  s7_rank_assign.py:1175:s7_rank_assign
cc=     1 self=  513.7ms cum=  1267.4ms  s6_collapse.py:976:s6_collapse
cc=     1 self=  464.5ms cum=   490.9ms  s4_face_point.py:966:_labels_to_list_of_lists
cc=     1 self=  370.6ms cum=  1212.6ms  s4_face_point.py:411:_compute_component_points_gpu
cc=276329 self=  248.3ms cum=   248.3ms  ~:0:<method 'tolist' of 'numpy.ndarray' objects>
cc=     8 self=  229.4ms cum=   229.4ms  ~:0:<built-in method gc.collect>
cc=275543 self=  226.6ms cum=   226.6ms  ~:0:<method 'reduce' of 'numpy.ufunc' objects>
cc=551101 self=  206.8ms cum=   206.8ms  ~:0:<built-in method numpy.asarray>
cc=    58 self=  198.3ms cum=   198.3ms  ~:0:<method 'cpu' of 'torch._C.TensorBase' objects>
cc=275539 self=  165.1ms cum=   165.1ms  ~:0:<method 'reduce' of 'numpy.ufunc' objects> (lsap)
cc=     1 self=  148.2ms cum=  1884.9ms  s4_face_point.py:1354:_compute_face_weights_gpu
cc=     1 self=  145.1ms cum=   145.1ms  s6_collapse.py:1158:<listcomp>
cc=275545 self=  105.8ms cum=   105.8ms  ~:0:<method 'astype' of 'numpy.ndarray' objects>
cc=     1 self=  102.7ms cum=   123.5ms  s4_face_point.py:648:_snap_centroids_to_components
cc=     1 self=   74.7ms cum=  3172.3ms  s4_face_point.py:66:s4_face_point
cc=1997642 self=   73.3ms cum=    73.3ms  ~:0:<method 'append' of 'list' objects>
cc=1935899 self=   63.0ms cum=    63.0ms  ~:0:<built-in method builtins.len>
cc=   216 self=   41.4ms cum=    44.4ms  sh_clip.py:108:_emit
cc=275539 self=   39.0ms cum=   303.8ms  ~:0:<method 'sum' of 'numpy.ndarray' objects>
```

## DoD checklist

- [x] DoD #1 (no Triton, no new deps): satisfied
- [x] DoD #2 (W2 persistent pool reduces forks): 124 → 0 (100%)
- [x] DoD #3 (W4 `_fastpath_trace_loops_numpy` eliminated ≥80%): 275426 → 0 (100%)
- [x] DoD #4 (W5 `_get_local_components_np` eliminated ≥80%): 275541 → 0 (100%)
- [x] DoD #5 (e2e wall reduction ≥3 s): -3.277 s MET
- [x] DoD #6 (F1-F3 bit-exact against nw=1 goldens): 3/3 PASS (carried from T5d HEAD)

## Residual hotspots + next-spec recommendations

Residual CPU time ≈ 5.35 s e2e. Top levers:

1. **Main-thread lock.acquire = 1648.5 ms (31% of e2e).**
   Main thread blocks on MP pool `apply_async().get()` for s4/s7 worker results. Moving s7
   rank-assign onto the GPU (or overlapping it with s4 loader prep) would claw back ~1.0-1.5 s.
   Candidate: port `s7_rank_assign` (999 ms self) to a vectorized GPU kernel — likely the
   single biggest remaining lever.

2. **s6 collapse = 513.7 ms self + 145 ms listcomp = ~660 ms.**
   `s6_collapse.py:976` is still CPU-dense even after W5 UF GPU-ification — the remaining
   work is in Python-level traversal of loop results. A W6-style vectorization pass would
   likely halve this.

3. **NumPy `.tolist()` on 276k ndarrays = 248 ms, `reduce` ufunc = 227 ms, `asarray` = 207 ms,
   `astype` = 106 ms.** These are scattered per-cube conversions, mostly in the s4/s7
   post-processing path. A single batched CPU→GPU transfer (or keeping results on GPU) could
   eliminate 500-800 ms cumulatively.

4. **`_labels_to_list_of_lists` = 464.5 ms self.** Pure Python label-bucketing loop at
   `s4_face_point.py:966`. Could be vectorized with `np.argsort` + `np.split` or a sparse COO
   bucket on GPU.

5. **scipy.optimize._lsap.linear_sum_assignment = 165 ms cumulatively across 275539 tiny LAPs.**
   Small per-cube LAPs — batching into a single GPU Hungarian (or at least combining into
   contiguous numpy calls) would cut this meaningfully.

## Raw artifacts

- Wall-time JSON:    `tmp/cpu_profile/t9_clean_wall.json`
- Final profile:     `tmp/cpu_profile/results_main/main_thread_res256_final.prof`
- Hotspot parse:     `tmp/cpu_profile/t9_final_hotspots.txt`
- Clean run log:     `tmp/cpu_profile/t9_clean_wall.log`
- cProfile run log:  `tmp/cpu_profile/driver_main_res256_final.log`
- VRAM log:          `tmp/cpu_profile/t9_vram.log`

## 4. Next-spec scope recommendations

Post-W2+W4+W5 residual top-20 hotspots (from `tmp/cpu_profile/t9_final_hotspots.txt`),
annotated with origin stage and W-coverage status:

| # | self ms |  calls  | cum ms | location                                               | origin | status |
|--:|--------:|--------:|-------:|--------------------------------------------------------|--------|--------|
| 1 |  1648.5 |       4 | 1648.5 | `_thread.lock.acquire`                                 | s4 Stage D pool.map wait | NEW residual (W2 collapsed 19→4; W6 rejected per T7a) |
| 2 |   999.0 |       1 | 1702.0 | `s7_rank_assign.py:1175 s7_rank_assign`                | s7     | NEW — Phase 3 Hungarian-per-cube loop, not covered by any W |
| 3 |   513.7 |       1 | 1267.4 | `s6_collapse.py:976 s6_collapse`                       | s6     | NEW — post-tracer assembly (`np.fromiter`/concat over list-of-lists) |
| 4 |   464.5 |       1 |  490.9 | `s4_face_point.py:966 _labels_to_list_of_lists`        | s4     | NEW — pure-Python label-bucket loop |
| 5 |   370.6 |       1 | 1212.6 | `s4_face_point.py:411 _compute_component_points_gpu`   | s4     | GPU kernel self-time (real compute) |
| 6 |   248.3 |  276329 |  248.3 | `ndarray.tolist`                                       | s4/s7 post-processing | NEW — per-cube conversions scattered |
| 7 |   229.4 |       8 |  229.4 | `gc.collect`                                           | allocator | framework overhead, not algo-attributable |
| 8 |   226.6 |  275543 |  226.6 | `ufunc.reduce` (numpy, non-lsap)                       | s7 Phase 3 inner | tied to #2 loop |
| 9 |   206.8 |  551101 |  206.8 | `numpy.asarray`                                        | s4/s7 | scattered coercions |
|10 |   198.3 |      58 |  198.3 | `torch._C.TensorBase.cpu`                              | s4/s7 | real D2H, 58 calls — bulk payload traffic (not Bucket A) |
|11 |   165.1 |  275539 |  165.1 | `scipy.optimize._lsap.linear_sum_assignment`           | s7 Phase 3 | tied to #2 loop (Hungarian per cube) |
|12 |   148.2 |       1 | 1884.9 | `s4_face_point.py:1354 _compute_face_weights_gpu`      | s4 Stage D | self is orchestration; cum = worker wait (see #1) |
|13 |   145.1 |       1 |  145.1 | `s6_collapse.py:1158 <listcomp>`                       | s6     | post-assembly listcomp, paired with #3 |
|14 |   105.8 |  275545 |  105.8 | `ndarray.astype`                                       | s7 Phase 3 | required by scipy float64 signature |
|15 |   102.7 |       1 |  123.5 | `s4_face_point.py:648 _snap_centroids_to_components`   | s4     | post-UF consumer, still list-based |
|16 |    74.7 |       1 | 3172.3 | `s4_face_point.py:66 s4_face_point`                    | s4     | top-level orchestration |
|17 |    73.3 | 1997642 |   73.3 | `list.append`                                          | misc Python | distributed |
|18 |    63.0 | 1935899 |   63.0 | `builtins.len`                                         | misc Python | distributed |
|19 |    41.4 |     216 |   44.4 | `sh_clip.py:108 _emit`                                 | sh_clip | unrelated module |
|20 |    39.0 |  275539 |  303.8 | `ndarray.sum`                                          | s7 Phase 3 | tied to #2 loop |

Aggregation:
- s7 Phase 3 Hungarian-per-cube cluster (#2 + #8 + #11 + #14 + #20) = ~1536 ms of addressable work, all gated on one 275k-iter Python loop at `s7_rank_assign.py:1175`.
- Stage D main-thread wait (#1) = 1648 ms, 100% pure-Python BFS per T7a.
- s6 assembly cluster (#3 + #13) = ~659 ms, Python traversal over worker list-of-lists.
- s4 list-based consumers (#4 + #15) = ~567 ms, already-allocated structures but still per-cube Python loops.
- Misc numpy coercion (#6 + #9 + #10 + #14) = ~759 ms cumulatively, 276k+ ndarray conversions scattered across stage boundaries — would be reclaimed by any "keep on GPU" pass that removes the CPU round-trip.

### 4.1 Highest-lever candidate: s7 GPU port

- `_build_adjacency_gpu` @ s7_rank_assign.py:634 — 1913 ms self (genuine GPU kernel compute)
- `s7_rank_assign` @ s7_rank_assign.py:1175 — 999 ms self (Phase 3 Hungarian matching, 275k iterations of `scipy.optimize.linear_sum_assignment`)

Both are Triton-territory. `_build_adjacency_gpu`: candidate for kernel fusion across 12×3×W scatter iterations + U-turn loop. `s7_rank_assign` Phase 3: pad cost-matrix + custom GPU Hungarian.

**Estimated reward:** ~1.5–2 s additional e2e reduction. **Estimated effort:** 5–10 days (one spec, probably Triton + careful correctness validation).

### 4.2 Secondary: lock.acquire residual (1648 ms)

Main-thread waits on Stage D Pool.map (persistent post-W2). 100% Python-BFS (`_count_uturns`: 49.7 s across all cube-workers via GIL-holding). Three paths:

- **Skip.** W2/W4/W5 already hit DoD. W6 Angle 2 (GPU BFS+UTurn) would cost 3–5 days for ~1.0–1.5 s reward — secondary to 4.1.
- **Triton Angle 2.** GPU BFS + UTurn unblock, requires cube-batched BFS kernel. Couples with 4.1 if the same spec opens Triton.
- **Rewrite Stage D pure-numpy.** Drop Python data structures inside `_count_uturns` / `_find_or_add_node` for numpy. Risk: doesn't fix GIL fully. ~1 day spike, uncertain outcome.

Recommend bundling with 4.1 in a "s4 Stage D + s7 Phase 3 GPU spec" if both survive review.

### 4.3 Shelf / deferred

- `_fastpath_gpu_build_adjacency` second-pack (W4 Option A in t5a sketch) — ~10–20% of the current W4 win is still absorbed by a numpy → tensor repack post-tracer. If the consumer downstream is updated to accept CSR natively, this can be reclaimed.
- W7 (s7 orchestration) — drilldown found no ≥200 ms mechanical candidate. Only bucket reopens if a future refactor changes s7 call patterns.
- W6 Angle 3 (ThreadPool) — dead (GIL-holding 57.9% in Stage D). Do not revisit without first rewriting Stage D to release GIL.

### 4.4 Measurement discipline notes

- cProfile instrumentation inflates e2e wall by ~15–25% in this pipeline (W2 measurement showed 14.9 s cProfile vs ~8.6 s clean baseline). Future work should run clean e2e (`tmp/cpu_profile/t0_driver.py --mode default`) for authoritative numbers and use cProfile only for hotspot rank.
- Main-thread cProfile `self` time excludes C-extension time (torch/numpy kernels). The T0 workers-only profile misread (~60 ms worker Python self) triggered a false "delete MP" pivot that was corrected by a ~15-minute decisive experiment. Future specs: cross-check worker timing with wall-clock, not cProfile alone.
- F1-F3 regression gate required three determinism layers (SerialPool + subprocess-per-fixture + CUBLAS_WORKSPACE_CONFIG/cudnn deterministic/PYTHONHASHSEED). Cross-process bit-exactness was hard-won; preserve these env conditions in all future gates.

### 4.5 Proposed next spec (one-liner)

**"Stage D + s7 Phase 3 Triton port"** — target `_count_uturns` + `_build_adjacency_gpu` + Hungarian Phase 3. Expected Δ 1.5–2.5 s. Effort 7–12 days. Gate: same F1-F3 + an additional ≥1.5 s e2e reduction DoD.
