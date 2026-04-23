"""Unit tests for Watchdog conditions + dump file creation."""
from __future__ import annotations

import json

import pytest

from coart.eval.watchdog import Watchdog


class _FakeLogger:
    def __init__(self):
        self.alerts = []

    def alert(self, title, text, level="WARN"):
        self.alerts.append((title, text, level))


@pytest.fixture
def wd(tmp_path):
    return Watchdog(str(tmp_path), _FakeLogger())


def test_grad_spike_fires_after_consecutive(wd):
    for i in range(Watchdog.GRAD_SPIKE_CONSECUTIVE):
        wd.update_train(i, grad_pre=100.0, grad_post=1.0, grad_p95=0.5,
                        loss_ef=0.1, lr=1e-5)
    wd.check_train(Watchdog.GRAD_SPIKE_CONSECUTIVE - 1)
    assert len(wd.logger.alerts) == 1
    assert "grad spike" in wd.logger.alerts[0][0]


def test_grad_spike_resets_on_good_step(wd):
    for i in range(Watchdog.GRAD_SPIKE_CONSECUTIVE - 1):
        wd.update_train(i, grad_pre=100.0, grad_post=1.0, grad_p95=0.5,
                        loss_ef=0.1, lr=1e-5)
    wd.update_train(1000, grad_pre=0.5, grad_post=0.4, grad_p95=0.5,
                    loss_ef=0.1, lr=1e-5)
    wd.check_train(1000)
    assert wd.logger.alerts == []


def test_ef_diverge_fires_on_monotonic_rise(wd):
    for s in range(5000, 10001, 500):
        wd.update_train(s, grad_pre=0.1, grad_post=0.1, grad_p95=0.5,
                        loss_ef=0.1 + 0.01 * (s - 5000) / 500, lr=1e-5)
    wd.check_train(10000)
    assert any("ef diverge" in a[0] for a in wd.logger.alerts)


def test_ef_diverge_doesnt_fire_on_stable(wd):
    for s in range(5000, 10001, 500):
        wd.update_train(s, grad_pre=0.1, grad_post=0.1, grad_p95=0.5,
                        loss_ef=0.1, lr=1e-5)
    wd.check_train(10000)
    assert not any("ef diverge" in a[0] for a in wd.logger.alerts)


def test_helmet_bad_fires_above_min_step(wd):
    wd.update_helmet(step=Watchdog.HELMET_BAD_MIN_STEP, helmet_nc=0.3)
    assert any("helmet bad" in a[0] for a in wd.logger.alerts)


def test_helmet_bad_silent_before_min_step(wd):
    wd.update_helmet(step=Watchdog.HELMET_BAD_MIN_STEP - 1, helmet_nc=0.1)
    assert wd.logger.alerts == []


def test_helmet_bad_none_is_noop(wd):
    wd.update_helmet(step=Watchdog.HELMET_BAD_MIN_STEP, helmet_nc=None)
    assert wd.logger.alerts == []


def test_dump_file_written(wd, tmp_path):
    for i in range(Watchdog.GRAD_SPIKE_CONSECUTIVE):
        wd.update_train(i, grad_pre=100.0, grad_post=1.0, grad_p95=0.5,
                        loss_ef=0.1, lr=1e-5)
    wd.check_train(Watchdog.GRAD_SPIKE_CONSECUTIVE - 1)
    dumps = list(tmp_path.glob("watchdog_grad_spike_step*.json"))
    assert len(dumps) == 1
    payload = json.loads(dumps[0].read_text())
    assert payload["condition"] == "grad_spike"


def test_fires_only_once(wd):
    for i in range(Watchdog.GRAD_SPIKE_CONSECUTIVE * 2):
        wd.update_train(i, grad_pre=100.0, grad_post=1.0, grad_p95=0.5,
                        loss_ef=0.1, lr=1e-5)
        wd.check_train(i)
    assert sum(1 for a in wd.logger.alerts if "grad spike" in a[0]) == 1
