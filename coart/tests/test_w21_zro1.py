# coart/tests/test_w21_zro1.py
"""Verify ZRO-1 produces same loss as DDP for first N steps (within 1e-3)."""
import glob
import json
import os
import subprocess

import pytest


def _run_short(mode: str, label: str, host: str = "host-10-240-99-117",
               steps: int = 5) -> dict:
    cmd = ["bash", "scripts/wave_verify.sh", label,
           "--mode", mode, "--steps", str(steps), "--host", host]
    subprocess.check_call(cmd)
    out = sorted(glob.glob(f"logs/wave_verify/{label}_*/result.json"))[-1]
    return json.load(open(out))


@pytest.mark.skipif(not os.path.exists("/dev/nvidia0"), reason="no GPU")
def test_zro1_loss_matches_ddp_within_1e3():
    a = _run_short("ddp", "w21_eq_ddp")
    b = _run_short("zro1", "w21_eq_zro1")
    a_loss = a["summary"]["metrics"]["loss/loss"]["mean"]
    b_loss = b["summary"]["metrics"]["loss/loss"]["mean"]
    diff = abs(a_loss - b_loss)
    # < 1e-3 is loose threshold — first 5 steps see different sample order
    # variance, so we tolerate sample-noise level. ZRO is bit-equivalent in math
    # but data sampling RNG state may diverge slightly between ddp/zro1 launches.
    assert diff < 1e-3, f"DDP vs ZRO-1 loss mismatch: {diff} (5-step mean)"
