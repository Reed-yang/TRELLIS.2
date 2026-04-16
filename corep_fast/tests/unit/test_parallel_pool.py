"""Unit tests for PersistentWorkerPool."""
import time
import pytest


def _double(x):
    """Module-level helper (picklable for multiprocessing)."""
    return x * 2


def test_pool_map_basic():
    from corep_fast.parallel.worker_pool import PersistentWorkerPool
    with PersistentWorkerPool(num_workers=2) as pool:
        results = pool.map_chunked(_double, list(range(10)), chunk_size=3)
    assert sorted(results) == [x * 2 for x in range(10)]


def test_pool_map_empty():
    from corep_fast.parallel.worker_pool import PersistentWorkerPool
    with PersistentWorkerPool(num_workers=2) as pool:
        results = pool.map_chunked(lambda x: x, [], chunk_size=5)
    assert results == []


def test_pool_map_single_worker():
    from corep_fast.parallel.worker_pool import PersistentWorkerPool
    with PersistentWorkerPool(num_workers=1) as pool:
        results = pool.map_chunked(lambda x: x + 1, [10, 20, 30], chunk_size=2)
    assert sorted(results) == [11, 21, 31]


def test_pool_small_batch_runs_serial():
    """Batches smaller than chunk_size should run without spawning workers."""
    from corep_fast.parallel.worker_pool import PersistentWorkerPool
    with PersistentWorkerPool(num_workers=4) as pool:
        results = pool.map_chunked(lambda x: x, [1, 2, 3], chunk_size=100)
    assert sorted(results) == [1, 2, 3]


def test_pool_context_manager_cleanup():
    """Pool must be usable only within context manager."""
    from corep_fast.parallel.worker_pool import PersistentWorkerPool
    pool = PersistentWorkerPool(num_workers=2)
    pool.close()  # Should not raise
