"""Verify new wandb + EMA fields exist in VaeTrainConfig + parse_args."""
import sys
from dataclasses import fields

import pytest

from coart.vae.config import VaeTrainConfig, parse_args


def test_new_fields_on_dataclass():
    names = {f.name for f in fields(VaeTrainConfig)}
    for f in (
        "use_wandb", "wandb_project", "wandb_mode",
        "rolling_ckpts_ema", "log_3d", "n_dump_names",
    ):
        assert f in names, f"missing {f}"


def test_argparse_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--run_tag", "dummy"])
    cfg = parse_args()
    assert cfg.use_wandb is True
    assert cfg.wandb_project == "coart-vae"
    assert cfg.wandb_mode == "online"
    assert cfg.rolling_ckpts_ema == 1
    assert cfg.log_3d is False
    assert cfg.n_dump_names == ["helmet", "val_p95"]


def test_wandb_mode_validation(monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        ["prog", "--run_tag", "dummy", "--wandb_mode", "bad"],
    )
    with pytest.raises(SystemExit):
        parse_args()
