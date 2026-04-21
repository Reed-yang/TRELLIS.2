"""Smoke test that the 2026-04-21 throughput flags load with correct defaults."""
import importlib, os


def test_flags_default_states(monkeypatch):
    """After the 2026-04-21 flag flip, VRAM_RESCUE defaults on; the other
    two tracks stay default-off pending their own validation (S2_SPARSE
    has a known p99 regression; ASYNC_D2H hasn't been A/B-measured
    end-to-end yet)."""
    for k in ("COREP_FAST_VRAM_RESCUE", "COREP_FAST_S2_SPARSE", "COREP_FAST_ASYNC_D2H"):
        monkeypatch.delenv(k, raising=False)
    import corep_fast.config as cfg
    importlib.reload(cfg)
    assert cfg.VRAM_RESCUE is True, "VRAM_RESCUE flipped to default-on after 2026-04-21 500-mesh validation"
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
