"""Test cache_slat output schema using a mock encoder."""
import importlib.util, pathlib

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(name):
    p = ROOT / "scripts" / "coart_data_v0" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def test_slat_output_schema_with_mock(tmp_path: pathlib.Path):
    slat = _load("coart_cache_slat")
    sha = "a" * 64

    feat18 = tmp_path / "feat18"; feat18.mkdir()
    N = 50
    np.savez(feat18 / f"{sha}.npz",
             cube_indices=np.random.randint(0, 32, (N, 3), dtype=np.int32),
             feats=np.random.randn(N, 18).astype(np.float32),
             num_boundary=np.zeros(N, dtype=np.int32))

    out_dir = tmp_path / "slat" / "vae_test_tag"
    out_dir.mkdir(parents=True)

    class FakeEncoder:
        def __call__(self, x, sample_posterior=False):
            class Z:
                def __init__(self, f, c):
                    self.feats = f; self.coords = c
            # Mock returns a Z whose feats match the input batch dimension
            n = x.feats.shape[0] if hasattr(x.feats, 'shape') else len(x.feats)
            return Z(torch.randn(n, 32), x.coords)
        def eval(self): return self

    slat.process_one(
        sha=sha,
        feat18_dir=str(feat18),
        out_dir=str(out_dir),
        encoder=FakeEncoder(),
        mean=torch.zeros(18), std=torch.ones(18),
        vae_ckpt_rel="results/x.pt",
        vae_io_arch="three_branch",
    )

    z = np.load(out_dir / f"{sha}.npz")
    assert z["coords"].shape == (N, 3)
    assert z["coords"].dtype == np.int16
    assert z["feats"].shape == (N, 32)
    assert z["feats"].dtype == np.float16
    assert int(z["num_voxels"]) == N
    assert str(z["vae_ckpt_rel"]) == "results/x.pt"
    assert str(z["vae_io_arch"]) == "three_branch"
