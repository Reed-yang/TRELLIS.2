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
