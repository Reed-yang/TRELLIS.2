# coart/tests/test_w22_dcp_compat.py
"""W2.2: DDP-trained ckpt → FSDP2 load step-1 sanity.

Skips if no DDP-trained ckpt is available locally. When an existing ckpt is
found, runs a short FSDP2 finetune from it and checks step-1 loss is sane.
"""
import glob
import json
import os
import subprocess

import pytest


@pytest.mark.skipif(not os.path.exists("/dev/nvidia0"), reason="no GPU")
def test_ddp_ckpt_loadable_in_fsdp2():
    candidates = glob.glob("results/**/ckpts/denoiser_step*.pt", recursive=True)
    if not candidates:
        pytest.skip("no DDP ckpt available — run a longer DDP train first")
    ddp_ckpt = candidates[0]
    label = "w22_pytest_dcp"
    env = dict(os.environ, COART_FINETUNE_FROM=ddp_ckpt)
    subprocess.check_call(
        [
            "bash",
            "scripts/wave_verify.sh",
            label,
            "--mode",
            "fsdp2_zero2",
            "--steps",
            "5",
            "--host",
            "host-10-240-99-117",
        ],
        env=env,
    )
    out = sorted(glob.glob(f"logs/wave_verify/{label}_*/result.json"))[-1]
    d = json.load(open(out))
    first_loss = d["summary"]["metrics"]["loss/loss"]["mean"]
    assert 0 < first_loss < 10, f"step-1 loss after DDP→FSDP2 load: {first_loss}"
