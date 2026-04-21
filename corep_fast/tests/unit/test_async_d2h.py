"""Tests for the 2026-04-21 async-D2H track (P3): remove the s1:56 deviceSync
and batch .item() leaks in s4/s6/s7."""
import importlib
import re
from pathlib import Path

import pytest


def _set_flag(monkeypatch, on):
    monkeypatch.setenv("COREP_FAST_ASYNC_D2H", "1" if on else "0")
    import corep_fast.config as cfg
    importlib.reload(cfg)


def test_s1_line56_removed_when_flag_on():
    """When ASYNC_D2H is on, the blocking .item() at s1_voxelize.py line 56
    must be either gone or behind a `if not ASYNC_D2H:` guard."""
    src = Path("corep_fast/stages/s1_voxelize.py").read_text()
    # Look for the pattern `total_pairs = int(offsets[-1].item())` without
    # a preceding `if not ASYNC_D2H:` guard on the same or prior line.
    # Simple check: the raw line is gone OR it sits inside a conditional.
    match = re.search(r"total_pairs = int\(offsets\[-1\]\.item\(\)\)", src)
    if match is None:
        return  # removed entirely — good
    # If still present, must be under an ASYNC_D2H guard
    prefix = src[:match.start()].splitlines()[-5:]
    assert any("ASYNC_D2H" in line for line in prefix), (
        "s1 line 56 still contains an unconditional .item(); expected guard")
