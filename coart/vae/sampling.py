"""Eval-time mesh dump for feat18 VAE.

Runs encoder/decoder in eval mode on first-N samples of a dataset, denormalises
the decoder output to raw feat18 space, converts to CorepParam via
feature_to_mesh (imported from the existing train_overfit_feat18 module), and
exports .ply meshes to out_dir.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from trellis2.modules import sparse as sp
from train_overfit_feat18 import feature_to_mesh

from ..data.stats import denormalize, normalize


@torch.no_grad()
def dump_samples(
    encoder,
    decoder,
    dataset,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    resolution: int,
    n_dump: int,
    out_dir: str,
    device,
) -> None:
    """Dump up to `n_dump` meshes from `dataset` to `out_dir` as .ply files."""
    os.makedirs(out_dir, exist_ok=True)
    encoder.eval()
    decoder.eval()
    n = min(n_dump, len(dataset))
    for k in range(n):
        item = dataset[k]
        ci = item["cube_indices"].astype(np.int32)
        feats_raw = item["feats"].astype(np.float32)
        sha = item["sha"]

        bi = np.zeros((ci.shape[0], 1), dtype=np.int32)
        coords = (
            torch.from_numpy(np.concatenate([bi, ci], axis=1))
            .int()
            .to(device)
        )
        feats_t = torch.from_numpy(feats_raw).float().to(device)
        x = sp.SparseTensor(feats=normalize(feats_t, mean_t, std_t), coords=coords)

        z = encoder(x, sample_posterior=False)
        h = decoder(z)
        h = h[0] if isinstance(h, tuple) else h
        pred_raw = denormalize(h.feats, mean_t, std_t).cpu().numpy()

        try:
            mesh = feature_to_mesh(pred_raw, ci, resolution=resolution, device=str(device))
            if mesh is not None:
                mesh.export(os.path.join(out_dir, f"{sha}_pred.ply"))
            else:
                tqdm.write(f"  [mesh] {sha} returned empty placeholder")
        except Exception as e:
            tqdm.write(f"  [mesh] {sha} failed: {type(e).__name__}: {e}")
    encoder.train()
    decoder.train()
