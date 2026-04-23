"""Smoke deep_eval with mocked encoder/decoder and fake golden manifest."""
from __future__ import annotations

import json
from unittest import mock


def test_run_deep_eval_empty_manifest(tmp_path, monkeypatch):
    """With empty manifest, run_deep_eval returns {} without crash."""
    from coart.eval import deep_eval as de

    empty_json = tmp_path / "golden_assets.json"
    empty_json.write_text("[]")
    monkeypatch.setattr(de, "_GOLDEN_LIST", empty_json)

    class FakeCfg:
        resolution = 512
        n_dump_names = []

    class FakeLogger:
        def __init__(self): self.vals = {}
        def scalar(self, tag, value, step): self.vals[tag] = value
        def image(self, *a, **kw): pass

    import torch
    logger = FakeLogger()
    stats = {
        "mean": torch.zeros(18),
        "std": torch.ones(18),
    }
    out = de.run_deep_eval(
        encoder=mock.MagicMock(),
        decoder=mock.MagicMock(),
        stats=stats,
        step=100,
        logger=logger,
        cfg=FakeCfg(),
    )
    assert out == {}
    assert logger.vals == {}


def test_run_deep_eval_missing_manifest(tmp_path, monkeypatch):
    """If golden manifest file does not exist, skip quietly."""
    from coart.eval import deep_eval as de
    monkeypatch.setattr(de, "_GOLDEN_LIST", tmp_path / "nonexistent.json")

    class FakeCfg:
        resolution = 512
        n_dump_names = []

    class FakeLogger:
        def __init__(self): self.vals = {}
        def scalar(self, *a, **kw): self.vals[a[0]] = a[1]
        def image(self, *a, **kw): pass

    import torch
    out = de.run_deep_eval(
        encoder=mock.MagicMock(),
        decoder=mock.MagicMock(),
        stats={"mean": torch.zeros(18), "std": torch.ones(18)},
        step=100,
        logger=FakeLogger(),
        cfg=FakeCfg(),
    )
    assert out == {}


def test_aggregation_skips_failed_assets(tmp_path, monkeypatch):
    """If _one_asset returns {} for some assets, mean is over successful ones."""
    from coart.eval import deep_eval as de

    assets_json = tmp_path / "golden_assets.json"
    assets_json.write_text(json.dumps([
        {"name": "a1", "npz_path": "x", "local_path_gt": "x"},
        {"name": "a2", "npz_path": "x", "local_path_gt": "x"},
        {"name": "a3", "npz_path": "x", "local_path_gt": "x"},
    ]))
    monkeypatch.setattr(de, "_GOLDEN_LIST", assets_json)

    def fake_one_asset(asset, *a, **kw):
        if asset["name"] == "a2":
            return {}
        return {
            "cd": 0.1, "nc": 0.9, "f005": 0.8, "f01": 0.7, "f05": 0.6,
            "n_components": 1, "euler": 2, "n_boundary_edges": 0,
            "is_watertight": 1.0,
        }

    monkeypatch.setattr(de, "_one_asset", fake_one_asset)

    class FakeCfg:
        resolution = 512
        n_dump_names = []

    class FakeLogger:
        def __init__(self): self.vals = {}
        def scalar(self, tag, value, step): self.vals[tag] = value
        def image(self, *a, **kw): pass

    import torch
    logger = FakeLogger()
    out = de.run_deep_eval(
        encoder=mock.MagicMock(),
        decoder=mock.MagicMock(),
        stats={"mean": torch.zeros(18), "std": torch.ones(18)},
        step=100,
        logger=logger,
        cfg=FakeCfg(),
    )
    # Mean over 2 successful assets (a1, a3) → mean cd = 0.1
    assert abs(logger.vals["deep_eval/online/mean/cd"] - 0.1) < 1e-6
    assert abs(logger.vals["deep_eval/online/mean/watertight_rate"] - 1.0) < 1e-6
    assert out["a2"] == {}
    assert len(out) == 3
