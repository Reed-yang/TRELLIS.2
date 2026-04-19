"""Single shared multiprocessing.Pool across corep_fast stages.

Rationale: main-thread cProfile measured 124 posix.fork calls and 816 ms
self-time per e2e run. Per-dispatch temp pools pay fork + teardown cost
each time. This module exposes one reused pool.

Determinism: the pool initializer sets PYTHONHASHSEED=0 in each worker.
Default MP produces ~0.7% vertex-set drift across runs (T0 finding) because
stages use set/dict iteration to pick representative elements, and worker
hash seeds are randomized by default. Fixing the seed eliminates that
source of nondeterminism. Any remaining nondeterminism after this fix is
algorithmic and surfaces in T9's MP topology A/B check.

Semantics:
- get_pool(num_workers) returns a shared pool. Re-invocations with the
  same num_workers return the same pool. Different num_workers shuts
  down and replaces the pool (rare — typically num_workers is derived
  once per pipeline run and stays constant).
- shutdown_pool() explicitly terminates the pool (idempotent).
- On interpreter exit, atexit calls shutdown_pool automatically.

Safety:
- Not thread-safe for concurrent get_pool across threads; corep_fast runs
  single-threaded orchestration so this is acceptable. If that changes,
  add a lock.
- If a worker crashes (BrokenPipeError / OSError on map), call
  shutdown_pool() and retry with a fresh pool.
"""
import atexit
import multiprocessing as _mp
import multiprocessing.pool as _mp_pool
import os as _os
from typing import Optional

_pool: Optional[_mp_pool.Pool] = None
_pool_workers: Optional[int] = None


def _worker_initializer():
    """Runs once in each worker process immediately after fork.

    Fixes PYTHONHASHSEED so that set/dict iteration inside stage code is
    deterministic across workers. Without this, stage code that picks a
    representative element from an unordered collection selects a different
    element each run, causing the ~0.7% vertex-set drift documented in T0.
    """
    _os.environ["PYTHONHASHSEED"] = "0"


def get_pool(num_workers: int) -> _mp_pool.Pool:
    """Return a shared multiprocessing.Pool. Reuses if num_workers matches."""
    global _pool, _pool_workers
    if _pool is not None and _pool_workers == num_workers:
        return _pool
    if _pool is not None:
        shutdown_pool()
    _pool = _mp.Pool(processes=num_workers, initializer=_worker_initializer)
    _pool_workers = num_workers
    return _pool


def shutdown_pool() -> None:
    """Terminate the shared pool. Idempotent."""
    global _pool, _pool_workers
    if _pool is None:
        return
    try:
        _pool.close()
        _pool.join()
    except Exception:
        try:
            _pool.terminate()
        except Exception:
            pass
    _pool = None
    _pool_workers = None


atexit.register(shutdown_pool)


def default_num_workers() -> int:
    """Mirror the default used at call sites across s4/s6/s7/s8."""
    return max(1, (_os.cpu_count() or 4) - 4)
