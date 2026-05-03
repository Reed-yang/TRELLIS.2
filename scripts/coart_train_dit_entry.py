#!/usr/bin/env python
"""Wrapper around train.py that registers coart.dit dataset/trainer classes
into the trellis2 namespace BEFORE train.py's mp.spawn workers look them up.

train.py uses torch.multiprocessing.spawn(main, ...) to launch worker
processes. Each spawned worker is a fresh Python interpreter that does not
inherit the parent's imports — so an `import coart.dit` in the parent does
not register the classes inside the workers. We monkey-patch train.main
here to perform the registration as the first thing each worker does.
"""
from __future__ import annotations

import os
import pathlib
import runpy
import sys

# Make repo root importable (coart, trellis2, train.py module).
_REPO = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Register classes in this (parent) process so the initial config dump prints OK.
import coart.dit  # noqa: F401  -- side effect


# Patch torch.multiprocessing.spawn before train.py runs. We can't reliably
# monkey-patch train.main because runpy.run_path creates fresh module globals
# (any patches on sys.modules['train'].main are lost). Instead, intercept at
# the mp.spawn boundary so our wrapper applies regardless of how main is
# resolved inside train.py.
#
# The wrapper fn must be importable from a real module (not __main__) so
# the spawn child can pickle-resolve it after runpy has rewritten __main__
# to train.py — see coart/dit/_spawn_helpers.py for the wrapper.
import torch.multiprocessing as _tmp
from coart.dit._spawn_helpers import per_rank_main as _per_rank_main
_orig_spawn = _tmp.spawn


def _patched_spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method="spawn"):
    return _orig_spawn(
        _per_rank_main,
        args=(fn,) + tuple(args),
        nprocs=nprocs,
        join=join,
        daemon=daemon,
        start_method=start_method,
    )


_tmp.spawn = _patched_spawn

# Now run train.py as __main__. Its mp.spawn(main, ...) calls go through
# our _patched_spawn → _per_rank_main(rank, main, cfg).
_train_path = str(_REPO / "train.py")
sys.argv[0] = _train_path
runpy.run_path(_train_path, run_name="__main__")
