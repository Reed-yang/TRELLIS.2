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

import functools
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

# Load train.py as a module and wrap its main so each spawned worker also
# triggers the registration. We must do this BEFORE __main__ in train.py
# fires (which calls mp.spawn). Strategy: import train.py via runpy with a
# pre-patched module so its `mp.spawn(main, ...)` picks up the wrapped main.
import importlib.util

_train_path = str(_REPO / "train.py")
_spec = importlib.util.spec_from_file_location("train", _train_path)
_train_mod = importlib.util.module_from_spec(_spec)
sys.modules["train"] = _train_mod
_spec.loader.exec_module(_train_mod)

_orig_main = _train_mod.main


@functools.wraps(_orig_main)
def _wrapped_main(rank, cfg):
    # Re-register classes inside the spawned worker. Cheap: side-effect import.
    import coart.dit  # noqa: F401
    return _orig_main(rank, cfg)


_train_mod.main = _wrapped_main

# Now run train.py as __main__. Argv is already correct (caller passed
# args after the script name). Use runpy with the module name so its
# `if __name__ == '__main__'` block fires.
sys.argv[0] = _train_path
runpy.run_path(_train_path, run_name="__main__")
