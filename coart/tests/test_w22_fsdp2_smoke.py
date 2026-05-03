# coart/tests/test_w22_fsdp2_smoke.py
"""W2.2: FSDP2 smoke — short run with parallel_mode=fsdp2_zero2.

Runs via subprocess (wave_verify.sh) to bypass single-process FSDP2 init
constraints — FSDP2 needs torch.distributed initialized, which only the
profile_dit spawn path sets up.
"""
import glob
import json
import os
import subprocess

import pytest


@pytest.mark.skipif(not os.path.exists("/dev/nvidia0"), reason="no GPU")
def test_fsdp2_smoke():
    label = "w22_pytest_smoke"
    cmd = [
        "bash",
        "scripts/wave_verify.sh",
        label,
        "--mode",
        "fsdp2_zero2",
        "--steps",
        "5",
        "--host",
        "host-10-240-99-117",
    ]
    subprocess.check_call(cmd)
    out = sorted(glob.glob(f"logs/wave_verify/{label}_*/result.json"))[-1]
    d = json.load(open(out))
    step_mean = d["summary"]["metrics"]["time/step"]["mean"]
    assert 0 < step_mean < 10, f"FSDP2 step.mean implausible: {step_mean}"
