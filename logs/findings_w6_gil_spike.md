# W6 GIL-holding spike — Angle 3 viability (T7a)

**Date:** 2026-04-17
**HEAD at measurement:** `ee7a9c1d06fd027fb32a01a59b0d6454ac7bc8dc`
**Post-W2+W4+W5 reference HEAD:** `ff461a31b1586d5264b5361c5e3921dce8b41c2f`
**s4_face_point.py unchanged between `ff461a3..HEAD`** — measurement applies
to the post-W4+W5 state (`git log ff461a3..HEAD -- corep_fast/stages/
s4_face_point.py` returns empty).
**Machine / GPU:** `host-10-240-99-116`, `CUDA_VISIBLE_DEVICES=3`.

## Approach

cProfile-based classification of every frame emitted *inside*
`_p2_uturn_worker` (the per-cube Stage-D worker at
`corep_fast/stages/s4_face_point.py:1329`, invoked by `p.map` from
`_compute_face_weights_gpu` at line 1424). The Stage-D `Pool` is swapped
for a `SerialPool` monkeypatch so the worker body runs in the main
thread where cProfile can observe every child frame.

Each profiled call record is classified as:

- **C-ext (GIL-releasing):** fname=`~` entries whose `fn_name` matches
  `built-in method torch._C.*`, `built-in method numpy*`, or the common
  ndarray C methods that release the GIL around their numeric kernel
  (`dot`, `matmul`, `sum`, `ravel`, `reshape`, `astype`, `copy`,
  `flatten`, `mean`, `max`, `min`, `argsort`, `cumsum`, `nonzero`); plus
  any Python frame whose filename lies under `torch/_C` or
  `numpy/core/_multiarray_umath` / `numpy/linalg`.
- **Python (GIL-held):** everything else — pure-Python functions, list /
  dict / set ops, `builtins.len / issubclass / min / max`, etc.

The ratio `py_tt / (cext_tt + py_tt)` is a **lower bound** on the
GIL-holding fraction (it assumes all listed ndarray ops release the GIL,
which is optimistic; real GIL-holding is ≥ this number).

Spec rule (from `logs/findings_w6_angle_decision.md`): Angle 3
(ThreadPool) is viable iff GIL-holding fraction **< 30 %**. Boundary
band 25-30 % recommends Angle 2 to avoid risk.

Tool availability: **py-spy is NOT installed** in the project venv on
116 (verified via `.venv/bin/python -c "import py_spy"` →
`ModuleNotFoundError`; no `py-spy` binary on PATH). cProfile fallback
path was therefore taken.

## Measurement

| Metric                              | Value          |
|-------------------------------------|---------------:|
| e2e wall (with cProfile overhead)   | 129 499.3 ms   |
| `_p2_uturn_worker` calls            | 1 881 777      |
| `_p2_uturn_worker` total wall       | 125 273.3 ms   |
| cProfile C-ext self_tt              |  52 079.6 ms   |
| cProfile Python self_tt             |  71 656.8 ms   |
| cProfile total self_tt              | 123 736.4 ms   |
| **GIL-holding fraction (lower bd)** | **57.9 %**     |

Raw outputs:
- `tmp/cpu_profile/t7a_gil_result.pkl` (pickled summary dict)
- `tmp/cpu_profile/t7a_gil.log` (human-readable run log)
- `tmp/cpu_profile/t7a_gil_spike.py` (spike script, READ-ONLY — never
  imported by production code)

### Top 10 Python (GIL-held) self_tt contributors inside the worker

| # | self_tt (ms) | calls          | location                                          |
|--:|-------------:|---------------:|---------------------------------------------------|
|  1 |      49 679.0 |      1 881 777 | `s4_face_point.py:305:_count_uturns`             |
|  2 |       9 265.6 |      1 881 777 | `s4_face_point.py:1329:_p2_uturn_worker`         |
|  3 |       3 963.1 |      3 974 410 | `s4_face_point.py:398:_find_or_add_node`         |
|  4 |       2 926.2 |     49 682 890 | `builtins.issubclass`                            |
|  5 |       1 266.0 |     21 332 775 | `list.append`                                    |
|  6 |         995.4 |      1 881 909 | `s4_face_point.py:359:<listcomp>` (endpoints)    |
|  7 |         793.9 |     11 501 782 | `builtins.len`                                   |
|  8 |         756.0 |      1 881 777 | `s4_face_point.py:1339:<listcomp>` (segs build)  |
|  9 |         502.8 |      1 881 777 | `s4_face_point.py:331:<dictcomp>` (adj init)     |
| 10 |         351.6 |      1 987 205 | `builtins.min`                                   |

Pure-Python `_count_uturns` self_tt alone (row #1) is ~40 % of the
total self_tt — by itself it already puts the GIL-holding fraction
above 30 % with a comfortable margin.

## Decision

**Angle 3 (ThreadPool) VIABLE: NO.**

GIL-holding lower bound 57.9 % is **~2x the 30 % viability threshold**
and is not a boundary case. A ThreadPool replacement of the Stage-D
`Pool.map` would therefore be serialized by the GIL and cannot recover
any of the 1666 ms residual. The empirical breakdown confirms the a-
priori expectation from `findings_w6_angle_decision.md` §"Strong
fallback":

> "BFS + U-Turn are Python-object heavy" → ThreadPool dead on arrival.

## Fallback plan

### Option A (preferred): skip W6 entirely

- W4 + W5 composite gain (per `findings_w6_angle_decision.md` / T9 re-
  profile) is **≈ -3.5 s** stage wall, already **exceeding the DoD ≥3 s
  target** for this cpu-worker-optim iteration.
- W6's upper-bound contribution is ~1-1.5 s (the entire Stage-D residual).
- With Angle 3 off the table, the only path is Angle 2 (GPU BFS +
  UTurn), estimated **3-5 days effort** — large risk for a ~1 s win that
  is not required by DoD.
- **Recommendation: skip W6, declare cpu-worker-optim complete, close
  the budget.** Re-open W6 as a separate phase if future regressions
  make Stage D the new bottleneck.

### Option B (if user wants to push Stage D further): Angle 2 (GPU BFS+UTurn)

- Port `_count_uturns` + its BFS driver to GPU kernels operating on the
  on-device CSR segments already built in Stage B of
  `_compute_face_weights_gpu`.
- Effort: 3-5 days (new custom CUDA/Triton kernels, golden-test sweep
  across res ∈ {32, 64, 128, 256}, numerical parity vs current CPU path).
- Upside: removes the full 1666 ms Stage-D residual; enables future
  batched graph processing for other pipelines.

## Recommendation

**Skip W6. Close cpu-worker-optim. Reassign the W6 budget to the next
phase (triton-ingest or gpu-pipeline merge).**

Rationale: W4 + W5 already exceed DoD, Angle 3 is eliminated by the
spike, and Angle 2's 3-5 d effort is disproportionate to the remaining
~1-1.5 s upside. Re-open W6 in a separate sprint if Stage D becomes a
new regression bottleneck after downstream optimizations land.

## Fixture status (no production code changed)

The spike script `tmp/cpu_profile/t7a_gil_spike.py` is READ-ONLY and
never imported by production code (monkeypatches are applied only at
`__main__` run-time). F1, F2, F3 regression fixtures in
`corep_fast/tests/regression/test_cpu_worker_optim.py` remain unaffected.
