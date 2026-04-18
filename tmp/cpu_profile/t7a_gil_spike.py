"""T7a — GIL-holding spike for s4 Stage D (`_p2_uturn_worker`).

Goal: decide whether W6 Angle 3 (ThreadPool) is viable by measuring the
GIL-holding fraction of `_p2_uturn_worker` (the per-cube Stage D worker
invoked by Pool.map in s4_face_point._compute_face_weights_gpu).

Approach (cProfile classification, GIL-release fraction is a lower bound
on "could release GIL"; GIL-holding is therefore an upper bound here):

  1. Force the Stage-D Pool to run serially (SerialPool monkeypatch), so
     the worker body executes in the main thread where cProfile can see
     every call.
  2. Wrap `_p2_uturn_worker` with a timing + per-call cProfile so we only
     capture frames emitted *inside* the worker body.
  3. Run the pipeline once at res=256 on the same icosphere fixture used
     for prior W-series measurements.
  4. Walk the collected cProfile Stats and classify each (fname, lineno,
     fn_name) entry as:
       * "C-ext" (GIL-released)  — torch C dispatch, numpy ufunc /
         multiarray, built-in functions (method-wrapper / builtin).
       * "Python" (GIL-held)     — user Python frames, pure-Python stdlib.
  5. Report self_time totals + ratio.

Rule from the W6 decision spec: ThreadPool viable iff GIL-holding
fraction < 30%. Flag 28-32% as a "boundary" case and recommend Angle 2
to avoid a speculative ThreadPool commit.

Invocation (from repo root, on host-10-240-99-116):
    CUDA_VISIBLE_DEVICES=3 .venv/bin/python tmp/cpu_profile/t7a_gil_spike.py

Output:
    tmp/cpu_profile/t7a_gil.log          (human-readable run log)
    tmp/cpu_profile/t7a_gil_result.pkl   (pickled summary dict)
"""
from __future__ import annotations

import argparse
import cProfile
import gc
import pickle
import pstats
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# SerialPool monkeypatch (copied from t0_driver.py so the worker runs in the
# main thread where cProfile can observe every child frame).
# ---------------------------------------------------------------------------
class SerialPool:
    def __init__(self, processes=None, *a, **kw):
        self.processes = processes

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def map(self, fn, items, *a, **kw):
        return [fn(x) for x in items]

    def imap(self, fn, items, *a, **kw):
        for x in items:
            yield fn(x)

    def imap_unordered(self, fn, items, *a, **kw):
        for x in items:
            yield fn(x)

    def starmap(self, fn, items, *a, **kw):
        return [fn(*x) for x in items]

    def close(self):
        pass

    def terminate(self):
        pass

    def join(self):
        pass


def _maybe_force_serial() -> None:
    import multiprocessing as mp
    import multiprocessing.pool as mp_pool
    mp.Pool = SerialPool
    mp_pool.Pool = SerialPool


def _patch_stage_pools() -> None:
    """Also replace already-imported aliases in each stage module."""
    for mod_name in (
        "corep_fast.stages.s4_face_point",
        "corep_fast.stages.s6_collapse",
        "corep_fast.stages.s7_rank_assign",
        "corep_fast.stages.s8_collapse",
    ):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for alias in ("_Pool", "Pool"):
            if hasattr(mod, alias):
                setattr(mod, alias, SerialPool)


def _patch_persistent_pool() -> None:
    """Force persistent_pool.get_pool to return a SerialPool, so the Stage-D
    pool.map runs in the current thread."""
    import importlib
    try:
        pp = importlib.import_module("corep_fast.utils.persistent_pool")
    except Exception:
        return

    def _fake_get_pool(num_workers: int = 1, *a, **kw):
        return SerialPool(processes=num_workers)

    pp.get_pool = _fake_get_pool


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------
_TORCH_ROOTS = ("torch/_C", "torch\\_C")  # Linux + Windows slashes
_NUMPY_CORE_ROOTS = (
    "numpy/core/_methods",
    "numpy/core/multiarray",
    "numpy/core/_multiarray_umath",
    "numpy/core/umath",
    "numpy/linalg",  # dotted linalg paths vary
    "numpy/_core",   # new numpy 2.x layout
)


def _is_cext(fname: str, fn_name: str) -> bool:
    """Best-effort classification: True if this profiled frame corresponds
    to a C-extension call that is known to release the GIL.

    cProfile represents C/builtins with pseudo-filename "~" and fn_name like
    "<built-in method numpy.core._multiarray_umath.implement_array_function>"
    or "<method 'dot' of 'numpy.ndarray' objects>" or
    "<built-in method torch._C._*>".
    """
    if fname == "~":
        # Built-in or C-extension function. Examine the fn_name.
        low = fn_name.lower()
        if "built-in method torch" in low or "torch._c" in low:
            return True
        if "method_descriptor" in low and "torch" in low:
            return True
        if "built-in method numpy" in low or "numpy.core" in low or "numpy._core" in low:
            return True
        # ndarray C methods; most release the GIL for their numeric kernel.
        # Conservative: count the big ones as C-ext (GIL-released).
        _NDARRAY_GIL_RELEASING = (
            "method 'dot'",
            "method 'matmul'",
            "method 'sum'",
            "method 'ravel'",
            "method 'reshape'",
            "method 'astype'",
            "method 'copy'",
            "method 'flatten'",
            "method 'mean'",
            "method 'max'",
            "method 'min'",
            "method 'argsort'",
            "method 'cumsum'",
            "method 'nonzero'",
        )
        if any(m in low for m in _NDARRAY_GIL_RELEASING):
            return True
        # Everything else (<built-in method builtins.*>, list.pop, set.add,
        # dict lookup, issubclass, len, etc.) is tiny GIL-held C that *does*
        # hold the GIL across the call. Treat as GIL-held.
        return False
    # Python file paths. Check for torch/numpy C-ext modules (rare — usually
    # they show up under "~" fname).
    for root in _TORCH_ROOTS:
        if root in fname:
            return True
    for root in _NUMPY_CORE_ROOTS:
        if root in fname:
            return True
    return False


def _top_python_functions(ps: pstats.Stats, n: int = 5) -> list[tuple[str, float, int]]:
    rows: list[tuple[str, float, int]] = []
    for (fname, lineno, fn_name), (cc, nc, tt, ct, _callers) in ps.stats.items():
        if _is_cext(fname, fn_name):
            continue
        short = f"{Path(fname).name}:{lineno}:{fn_name}" if fname != "~" else fn_name
        rows.append((short, tt * 1000.0, cc))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows[:n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=256)
    args = ap.parse_args()

    _maybe_force_serial()

    # Import AFTER monkeypatch so any local binding `from multiprocessing
    # import Pool` in stage code resolves to SerialPool.
    import numpy as np
    import torch
    from tmp.profile_deep.build_icosphere import build_icosphere_subdiv3

    _patch_stage_pools()
    _patch_persistent_pool()

    from corep_fast.stages import s4_face_point

    # Wrap the target worker with a per-call cProfile. We keep a single
    # Profile instance and enable/disable it across the full measurement
    # run — cProfile's enable/disable is nest-safe and adds negligible
    # overhead compared to worker body wall (sanity-checked in t0).
    pr = cProfile.Profile()
    call_stats = {"calls": 0, "total_wall_ns": 0}

    original = s4_face_point._p2_uturn_worker

    def wrapped(gi):
        call_stats["calls"] += 1
        t0 = time.perf_counter_ns()
        pr.enable()
        try:
            return original(gi)
        finally:
            pr.disable()
            call_stats["total_wall_ns"] += time.perf_counter_ns() - t0

    s4_face_point._p2_uturn_worker = wrapped

    # Also patch the module-level name that _compute_face_weights_gpu
    # captures at call time. In the source `p.map(_p2_uturn_worker, ...)`
    # the lookup goes through the module, so the module-attribute patch
    # above is sufficient — but defend against the SerialPool fall-through
    # path `results = [_p2_uturn_worker(gi) for gi in range(G)]` which
    # references the local import-time name.
    # Inspecting s4_face_point.py: the list-comprehension path references
    # the *global* _p2_uturn_worker in the module namespace, so the attr
    # patch above covers both cases.

    # Build fixture
    torch.manual_seed(42)
    np.random.seed(42)
    mesh = build_icosphere_subdiv3()
    device = torch.device("cuda:0")

    from corep_fast.containers import MeshTensors
    from corep_fast.stages.s1_voxelize import s1_voxelize
    from corep_fast.stages.s2_components import s2_components
    from corep_fast.stages.s3_edge_weights import s3_edge_weights
    from corep_fast.stages.s4_face_point import s4_face_point as _s4
    from corep_fast.stages.s6_collapse import s6_collapse
    from corep_fast.stages.s7_rank_assign import s7_rank_assign

    def _run_pipeline():
        mt = MeshTensors.from_trimesh(mesh, args.res, device=device)
        batch = s1_voxelize(mt, args.res, device)
        batch = s2_components(batch, mt)
        batch = s3_edge_weights(batch, mt)
        batch = _s4(batch, mt)
        batch = s6_collapse(batch)
        batch = s7_rank_assign(batch)
        return batch

    print(f"[warmup] res={args.res}")
    gc.collect(); torch.cuda.empty_cache()
    # Warmup run — do NOT profile, just warm caches/JIT.
    # Temporarily disable the wrapper's cProfile enable to avoid polluting
    # measurements. Simplest: reset the call_stats and profile after warmup.
    _ = _run_pipeline()
    torch.cuda.synchronize()

    # Reset state for measurement run
    pr = cProfile.Profile()
    call_stats["calls"] = 0
    call_stats["total_wall_ns"] = 0

    def wrapped2(gi):
        call_stats["calls"] += 1
        t0 = time.perf_counter_ns()
        pr.enable()
        try:
            return original(gi)
        finally:
            pr.disable()
            call_stats["total_wall_ns"] += time.perf_counter_ns() - t0

    s4_face_point._p2_uturn_worker = wrapped2

    print(f"[measure] res={args.res}")
    gc.collect(); torch.cuda.empty_cache()
    t0 = time.perf_counter()
    _ = _run_pipeline()
    torch.cuda.synchronize()
    e2e = time.perf_counter() - t0

    if call_stats["calls"] == 0:
        print("ERROR: _p2_uturn_worker was not called — Stage D did not run", file=sys.stderr)
        return 2

    ps = pstats.Stats(pr)

    cext_tt = 0.0
    py_tt = 0.0
    for (fname, lineno, fn_name), (cc, nc, tt, ct, _callers) in ps.stats.items():
        if _is_cext(fname, fn_name):
            cext_tt += tt
        else:
            py_tt += tt

    total_tt = cext_tt + py_tt
    if total_tt <= 0.0:
        print("ERROR: cProfile collected no tt data", file=sys.stderr)
        return 2

    gil_holding_lb = py_tt / total_tt
    total_wall_ms = call_stats["total_wall_ns"] / 1e6
    top_py = _top_python_functions(ps, n=10)

    print()
    print("=" * 72)
    print(f"T7a GIL spike results (res={args.res})")
    print("=" * 72)
    print(f"e2e wall (ms):                     {e2e*1000:.1f}")
    print(f"_p2_uturn_worker calls:            {call_stats['calls']}")
    print(f"_p2_uturn_worker total wall (ms):  {total_wall_ms:.1f}")
    print(f"cProfile C-ext self_tt (ms):       {cext_tt*1000:.1f}")
    print(f"cProfile Python self_tt (ms):      {py_tt*1000:.1f}")
    print(f"cProfile total self_tt (ms):       {total_tt*1000:.1f}")
    print(f"GIL-holding fraction (lower bound):{gil_holding_lb:>6.1%}")
    print()
    print("Top Python (GIL-held) self_tt contributors inside worker:")
    for i, (name, tt_ms, cc) in enumerate(top_py, 1):
        print(f"  {i:>2}. {tt_ms:>8.1f} ms  cc={cc:>9d}  {name}")
    print()

    if gil_holding_lb >= 0.30:
        print(f"VERDICT: Angle 3 (ThreadPool) NOT VIABLE "
              f"(lower bound {gil_holding_lb:.1%} ≥ 30%)")
        verdict = "NO"
    elif 0.25 <= gil_holding_lb < 0.30:
        print(f"VERDICT: Angle 3 BOUNDARY ({gil_holding_lb:.1%}); "
              f"recommend Angle 2 to avoid risk.")
        verdict = "BOUNDARY"
    else:
        print(f"VERDICT: Angle 3 (ThreadPool) VIABLE "
              f"(lower bound {gil_holding_lb:.1%} < 30%)")
        verdict = "YES"

    out = {
        "res": args.res,
        "e2e_wall_ms": e2e * 1000.0,
        "worker_calls": call_stats["calls"],
        "worker_total_wall_ms": total_wall_ms,
        "cext_tt_ms": cext_tt * 1000.0,
        "py_tt_ms": py_tt * 1000.0,
        "total_tt_ms": total_tt * 1000.0,
        "gil_holding_lower_bound": gil_holding_lb,
        "verdict": verdict,
        "top_python": top_py,
    }
    out_path = _REPO_ROOT / "tmp/cpu_profile/t7a_gil_result.pkl"
    out_path.write_bytes(pickle.dumps(out))
    print(f"[write] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
