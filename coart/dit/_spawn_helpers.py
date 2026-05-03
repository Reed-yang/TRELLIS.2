"""Module-level helpers for torch.multiprocessing.spawn workers.

Functions here must be importable from a real module (not ``__main__``)
so that mp.spawn's pickle-based dispatch can locate them in the spawn child.
The launcher script (``scripts/coart_train_dit_entry.py``) itself runs as
``__main__`` and then hands control to ``train.py`` via ``runpy``, which
replaces ``__main__`` namespace — so anything defined in the launcher is
unpicklable from the child's perspective.
"""
from __future__ import annotations

import os
import pathlib

_REPO = pathlib.Path(__file__).resolve().parents[2]


def per_rank_main(rank, original_fn, *args):
    """Wrapper executed in each mp.spawn child as the entry function.

    Sets a per-rank Triton cache directory under ``<repo>/.cache/triton/``
    (avoids wekafs atomic-rename collisions when 8 workers compute the
    same kernel hash concurrently) and re-imports ``coart.dit`` to
    register dataset/trainer classes inside the spawn child. Then
    delegates to the original main fn that train.py was about to call.
    """
    triton_cache = _REPO / ".cache" / "triton" / f"rank-{rank}"
    triton_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
    import coart.dit  # noqa: F401  -- side effect: register classes
    return original_fn(rank, *args)
