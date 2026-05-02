"""Test the inline atomic_savez helper."""
import importlib.util, os, pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(name):
    p = ROOT / "scripts" / "coart_data_v0" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def test_atomic_savez_no_partial_visible(tmp_path: pathlib.Path):
    cache_dino = _load("coart_cache_dino")
    target = tmp_path / "out.npz"
    arr = np.arange(100, dtype=np.float32)
    cache_dino.atomic_savez(str(target), x=arr)
    assert target.exists()
    assert not list(tmp_path.glob("*.tmp.*"))
    with np.load(target) as z:
        np.testing.assert_array_equal(z["x"], arr)
