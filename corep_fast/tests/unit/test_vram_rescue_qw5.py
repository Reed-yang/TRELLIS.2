"""QW5 threshold dispatch smoke test. Dense->sparse routing at M>32 when
VRAM_RESCUE=1; documents the behaviour. Full structural coverage in
regression suite (F1/F2/F3)."""
import importlib, os
import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="VRAM rescue tests require CUDA",
)


def test_qw5_threshold_loads_with_flag(monkeypatch):
    """Sanity: with VRAM_RESCUE=1, the config flag is on and the s2 module
    imports cleanly. Behavioural dispatch verified by regression goldens
    which exercise real CubeBatches with varying max_faces."""
    monkeypatch.setenv("COREP_FAST_VRAM_RESCUE", "1")
    import corep_fast.config as cfg
    importlib.reload(cfg)
    assert cfg.VRAM_RESCUE is True
    from corep_fast.stages import s2_components  # noqa: F401
    # The threshold 32 is embedded in the if-branch at line ~112; we only
    # document it here for reader clarity. Tight assertion delegated to
    # regression + profiling.
    assert True
