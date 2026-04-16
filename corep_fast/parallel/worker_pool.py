"""Persistent multiprocessing pool for CPU-bound stage work.

Created once per pipeline invocation, reused across s4/s6/s7 stages.
For batches smaller than chunk_size, runs serially to avoid spawn overhead.
"""
from __future__ import annotations

import os
from multiprocessing import Pool
from typing import Any, Callable, TypeVar

T = TypeVar('T')
R = TypeVar('R')


def _apply_fn_to_chunk(args: tuple[Callable, list]) -> list:
    """Worker target: apply fn to each item in a chunk."""
    fn, items = args
    return [fn(item) for item in items]


class PersistentWorkerPool:
    """Process pool that is created once and reused across stage calls.

    Usage::

        with PersistentWorkerPool(num_workers=8) as pool:
            results_s4 = pool.map_chunked(s4_worker, cubes_s4, chunk_size=500)
            results_s6 = pool.map_chunked(s6_worker, cubes_s6, chunk_size=500)
    """

    def __init__(self, num_workers: int | None = None):
        if num_workers is None:
            num_workers = max(1, (os.cpu_count() or 4) // 2)
        self._num_workers = num_workers
        self._pool: Pool | None = None

    def __enter__(self) -> 'PersistentWorkerPool':
        if self._num_workers > 1:
            self._pool = Pool(self._num_workers)
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def map_chunked(
        self,
        fn: Callable[[Any], Any],
        items: list,
        chunk_size: int = 500,
    ) -> list:
        """Apply fn to each item, chunking for multiprocessing efficiency.

        If len(items) <= chunk_size or num_workers <= 1, runs serially.

        Args:
            fn: Function to apply to each item.
            items: List of work items.
            chunk_size: Items per worker chunk.

        Returns:
            Flat list of results in the same order as input.
        """
        if not items:
            return []

        # Serial path: small batches or single worker
        if len(items) <= chunk_size or self._num_workers <= 1 or self._pool is None:
            return [fn(item) for item in items]

        # Chunk and dispatch
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        chunk_args = [(fn, chunk) for chunk in chunks]
        batch_results = self._pool.map(_apply_fn_to_chunk, chunk_args)
        return [r for batch in batch_results for r in batch]
