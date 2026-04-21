"""Smoke test that the 2026-04-21 throughput flags load with correct defaults."""
import importlib, os


def test_flags_default_off(monkeypatch):
    for k in ("COREP_FAST_VRAM_RESCUE", "COREP_FAST_S2_SPARSE", "COREP_FAST_ASYNC_D2H"):
        monkeypatch.delenv(k, raising=False)
    import corep_fast.config as cfg
    importlib.reload(cfg)
    assert cfg.VRAM_RESCUE is False
    assert cfg.S2_SPARSE is False
    assert cfg.ASYNC_D2H is False


def test_flags_env_toggle(monkeypatch):
    monkeypatch.setenv("COREP_FAST_VRAM_RESCUE", "1")
    monkeypatch.setenv("COREP_FAST_S2_SPARSE", "1")
    monkeypatch.setenv("COREP_FAST_ASYNC_D2H", "1")
    import corep_fast.config as cfg
    importlib.reload(cfg)
    assert cfg.VRAM_RESCUE is True
    assert cfg.S2_SPARSE is True
    assert cfg.ASYNC_D2H is True
