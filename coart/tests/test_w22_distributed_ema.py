# coart/tests/test_w22_distributed_ema.py
"""W2.2: distributed EMA save/reload sidecar check.

A short FSDP2 run rarely hits i_save (default 2500), so this test mostly
asserts no-crash and skips when no save was triggered. Production verify
(longer run hitting i_save) is deferred to W2.2.8 manual test.
"""
import glob
import json
import os
import subprocess

import pytest


@pytest.mark.skipif(not os.path.exists("/dev/nvidia0"), reason="no GPU")
def test_distributed_ema_save_reload_consistent():
    label = "w22_pytest_ema"
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
    sidecars = glob.glob("results/**/ckpts/*_ema_check.json", recursive=True)
    if not sidecars:
        pytest.skip("no save triggered — i_save > test steps")
    max_diff = max(json.load(open(s))["ema_max_abs_diff"] for s in sidecars)
    assert max_diff < 1e-4, f"EMA round-trip diff too large: {max_diff}"
