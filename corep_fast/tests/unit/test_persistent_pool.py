"""Unit test for persistent_pool shared MP pool helper."""
import multiprocessing

import pytest

from corep_fast.utils.persistent_pool import get_pool, shutdown_pool


def _square(x):
    return x * x


def test_pool_reuse_returns_same_object():
    shutdown_pool()  # clean state
    p1 = get_pool(num_workers=2)
    p2 = get_pool(num_workers=2)
    assert p1 is p2, "Same num_workers should reuse the same pool"
    shutdown_pool()


def test_pool_map_correct_output():
    shutdown_pool()
    p = get_pool(num_workers=2)
    out = p.map(_square, [1, 2, 3, 4])
    assert out == [1, 4, 9, 16]
    shutdown_pool()


def test_pool_reinit_on_different_worker_count():
    shutdown_pool()
    p1 = get_pool(num_workers=2)
    p2 = get_pool(num_workers=4)
    assert p1 is not p2, "Different num_workers should produce a new pool"
    shutdown_pool()


def test_shutdown_idempotent():
    shutdown_pool()
    shutdown_pool()  # second call no-op


def _get_hashseed(_ignored):
    import os
    return os.environ.get("PYTHONHASHSEED")


def test_workers_get_pythonhashseed_zero():
    """T0 finding: MP default is nondeterministic because worker hash seeds
    are randomized, which shuffles set/dict iteration in tiebreakers. The
    pool initializer sets PYTHONHASHSEED=0 in each worker to fix this.
    """
    shutdown_pool()
    p = get_pool(num_workers=2)
    seeds = p.map(_get_hashseed, range(4))
    assert all(s == "0" for s in seeds), f"expected all '0', got {seeds}"
    shutdown_pool()
