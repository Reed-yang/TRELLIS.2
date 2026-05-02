"""Test cache_dino output schema using mocked DinoV3 model."""
import importlib.util, pathlib
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(name):
    p = ROOT / "scripts" / "coart_data_v0" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def _make_dummy_renders(root: pathlib.Path, sha: str, n_views: int = 16):
    d = root / sha; d.mkdir(parents=True, exist_ok=True)
    for v in range(n_views):
        img = Image.new("RGBA", (1024, 1024), (128, 128, 128, 255))
        img.save(d / f"{v:03d}.png")


def test_dino_output_schema_with_mock(tmp_path: pathlib.Path):
    cache_dino = _load("coart_cache_dino")
    sha = "a" * 64
    renders_dir = tmp_path / "renders"
    out_dir = tmp_path / "dino"; out_dir.mkdir()
    _make_dummy_renders(renders_dir, sha)

    import torch
    fake_T = 1029
    fake_D = 1024

    class FakeExtractor:
        def __init__(self, *a, **kw): pass
        def cuda(self): return self
        def __call__(self, image):
            return torch.zeros((image.shape[0], fake_T, fake_D), dtype=torch.float32)

    cache_dino.process_one(
        sha=sha,
        renders_dir=str(renders_dir),
        out_dir=str(out_dir),
        extractor=FakeExtractor(),
        image_size=512,
    )

    out = np.load(out_dir / f"{sha}.npz")
    assert out["features"].shape == (16, fake_T, fake_D)
    assert out["features"].dtype == np.float16
    assert int(out["n_tokens"]) == fake_T
    assert int(out["image_size"]) == 512
    assert str(out["model_id"]) == "facebook/dinov3-vitl16-pretrain-lvd1689m"
