"""Test CoartTBLogger dual-write to wandb + TB, robustness to wandb failure."""
from __future__ import annotations

import tempfile
from unittest import mock

import numpy as np


def test_logger_wandb_disabled_is_noop():
    """use_wandb=False → no wandb.init called, TB still works."""
    with tempfile.TemporaryDirectory() as td:
        from coart.common.logging import CoartTBLogger
        with mock.patch("coart.common.logging.wandb") as w:
            lg = CoartTBLogger(
                td, is_master=True, use_wandb=False,
                wandb_project="x", wandb_mode="online",
                wandb_run_name="r", wandb_tags=[], config={},
            )
            w.init.assert_not_called()
            lg.scalar("a/b", 1.0, step=1)
            lg.flush_if_due(step=100, i_log=100)
            lg.close()


def test_logger_wandb_network_failure_fallback():
    """wandb.init() raising must not crash; logger still writes to TB."""
    with tempfile.TemporaryDirectory() as td:
        from coart.common.logging import CoartTBLogger
        with mock.patch("coart.common.logging.wandb") as w:
            w.init.side_effect = RuntimeError("no network")
            lg = CoartTBLogger(
                td, is_master=True, use_wandb=True,
                wandb_project="x", wandb_mode="online",
                wandb_run_name="r", wandb_tags=[], config={},
            )
            assert lg._wandb is None
            lg.scalar("a/b", 1.0, step=1)
            lg.flush_if_due(step=100, i_log=100)
            lg.close()


def test_logger_image_calls_wandb():
    """image() forwards to wandb.Image when wandb is live."""
    with tempfile.TemporaryDirectory() as td:
        from coart.common.logging import CoartTBLogger
        with mock.patch("coart.common.logging.wandb") as w:
            w.init.return_value = mock.MagicMock()
            lg = CoartTBLogger(
                td, is_master=True, use_wandb=True,
                wandb_project="x", wandb_mode="online",
                wandb_run_name="r", wandb_tags=[], config={},
            )
            arr = np.zeros((64, 64, 3), dtype=np.uint8)
            lg.image("deep_eval/render/helmet", arr, step=1000)
            # log was called on the mock init-returned run
            lg._wandb.log.assert_called()


def test_logger_non_master_is_silent():
    """non-master ranks must not init wandb nor write to TB."""
    with tempfile.TemporaryDirectory() as td:
        from coart.common.logging import CoartTBLogger
        with mock.patch("coart.common.logging.wandb") as w:
            lg = CoartTBLogger(
                td, is_master=False, use_wandb=True,
                wandb_project="x", wandb_mode="online",
                wandb_run_name="r", wandb_tags=[], config={},
            )
            w.init.assert_not_called()
            lg.scalar("a/b", 1.0, step=1)
            lg.close()
