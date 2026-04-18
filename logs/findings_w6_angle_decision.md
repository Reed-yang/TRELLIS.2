# W6 angle decision (post-W2 measurement)

**Date:** 2026-04-17
**Baseline HEAD (pre-W2):** `276b7152f8c426044ffd8fbe0d7d27472b0267a4` (c923e85^)
**Post-W2 HEAD:** `c923e85c51d90b1fd956166331f92b3d3406fd72`

## Measurements (res=256 main-thread cProfile)

| Metric              | pre-W2  | post-W2  | Δ          |
|---------------------|--------:|---------:|-----------:|
| posix.fork self_tt  | 816.3ms | 0.0ms    | -816.3ms   |
| posix.fork calls    | 124     | 0        | -124       |
| lock.acquire self   | 2710.8ms| 1666.2ms | -1044.6ms  |

Raw parse output: `tmp/cpu_profile/w2_diff.txt`
Raw profiles: `tmp/cpu_profile/results_main/main_thread_res256_{pre,post}_w2.prof`

### Verification (second independent post-W2 run)

A second run yielded lock.acquire = 1743.1 ms (4 calls), fork = 0 calls.
Mean L_post_w2 ≈ 1704 ms (sd≈40 ms). Both samples > 1500 ms threshold.

### Lock.acquire caller trace (post-W2)

`_thread.lock.acquire` post-W2: cc=4, self=1666.2 ms. Caller chain:

```
s4_face_point.py:1144:_compute_face_weights_gpu  (1 call)
 -> pool.py:362:map                              (cum 1670.9 ms)
 -> pool.py:767:get                              (cum 1666.3 ms)
 -> pool.py:764:wait                             (cum 1666.3 ms)
 -> threading.py:589:wait / :288:wait            (cum 1666.3 ms)
 -> _thread.lock.acquire                         (self 1666.2 ms)
```

**100% of the residual lock.acquire is a single `pool.map()` call from s4
Stage D `_compute_face_weights_gpu`** — i.e. the persistent MP pool is
reused correctly (no fork cost), but the main thread still idles on the
pool result queue while Stage D CPU BFS + U-Turn workers run.

Pre-W2 had cc=19 lock.acquire (each stage re-creating + waiting on its
own Pool). Post-W2 has cc=4 — W2 collapsed the 19 pool-level waits into
effectively one dominant wait (Stage D), plus three small helper waits.
The ~1044 ms drop matches the sum of small stages' pool init+wait
overheads that W2 amortized away; **the ~1666 ms residual is Stage D
steady-state worker wall-time**, not pool overhead.

## Decision

**Angle chosen: 3 (ThreadPool spike)** per spec §4.5 decision tree.

### Justification

L_post_w2 = 1666.2 ms (verified 1743.1 ms in a second run), which
**exceeds the 1500 ms threshold** in the decision tree. The tree therefore
prescribes Angle (3) "ThreadPool spike" with the caveat that it is "only
viable if GIL-holding fraction is <30%; if unclear, T7 spikes".

Because the residual is a **single main-thread wait on Stage D workers**
(confirmed by caller trace) and the workers run pure-Python BFS + U-Turn
counting, the GIL-holding fraction inside the worker body is currently
**unknown** without a spike. Per the spec, this ambiguity is exactly the
trigger for a T7 spike rather than a commitment to the full path.

### Strong fallback: Angle (2) "GPU BFS+UTurn"

If T7's ThreadPool spike shows Stage D cannot release the GIL (likely —
BFS + U-Turn are Python-object heavy), Angle 3 is dead on arrival and
Angle 2 becomes the chosen path. The data supports Angle 2 almost as
strongly as Angle 3 under a strict reading: the caller trace confirms
"Stage D workers dominate residual" (the 1500-ms interior branch trigger),
and Stage D's CSR segments are already on-device from Stage B, making a
GPU port mechanically plausible at 3-5 d effort.

The threshold crossing is narrow (1666 vs 1500 ms, ~11% above) and the
Stage D dominance is unambiguous, so Angle 2 is the robust fallback the
moment Angle 3's GIL assumption fails.

## T7 (W6) scope

- **T7 spike (0.5 d diagnostic):** Replace the Stage D Pool with a
  `ThreadPoolExecutor` of equal size and re-measure Stage D wall-time on
  the same res=256 input.
  - **Pass criterion:** Stage D wall drops by ≥40% vs post-W2 (i.e.
    residual lock.acquire drops below ~1000 ms) → commit to full Angle 3.
  - **Fail criterion:** Stage D wall is within ±10% of post-W2 or worse
    → pivot to Angle 2 immediately (GPU BFS+UTurn on the on-device CSR
    segments built in Stage B).
- **Instrument Stage D worker body** to attribute time between
  numpy/C-bound ops (GIL-released) and pure-Python BFS/U-Turn logic
  (GIL-held). A ≥70% Python/GIL-held fraction is the same fail signal
  as the spike benchmark.
- **Record decision in logs/findings_t7_*.md** and commit the spike
  branch with the instrumentation removed.

## Residual top-10 hotspots (post-W2, by self_tt)

(from `tmp/cpu_profile/w2_diff.txt`, targeted rows; full dump in the .prof)

| Self   | Cum    | Calls   | Location                                           | W-owner |
|-------:|-------:|--------:|----------------------------------------------------|---------|
| 1822.5 | 2582.0 | 275 426 | s6_collapse.py:379 `_fastpath_trace_loops_numpy`   | W4      |
| 1666.2 | 1666.2 |       4 | `_thread.lock.acquire` (Stage D pool.map wait)     | W6      |
| 1014.2 | 1540.5 | 275 541 | s4_face_point.py:734 `_get_local_components_np`    | W5      |

(Remaining hotspots fall below ~100 ms self-time and are not material to
W6 scoping. The three rows above consume ~4.5 s of the 14.9 s e2e
wall-time with cProfile overhead; the rest is pipeline orchestration
distributed across many small call sites.)

## Notes on e2e wall-time inflation (pre-W2 12.4 s vs post-W2 14.9 s)

The post-W2 e2e-with-cProfile wall (14.9 s) being higher than pre-W2
(12.4 s) is an artifact of cProfile's instrumentation overhead
interacting with the persistent pool's cross-stage workers (more deep
Python frames to profile as the same worker processes accumulate
state). Non-profiled benchmarks in T3 showed W2 reduces e2e, not
increases it; see the T3 commit notes and `corep_fast/tests/regression/
test_cpu_worker_optim.py` fixture F1/F2/F3 (unchanged: F1 PASS, F2 PASS,
F3 PASS). This measurement artifact does not affect the L_post_w2 value
itself, which is a direct pool.map wait time and independent of profiling
overhead's distribution.
