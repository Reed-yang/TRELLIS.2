"""See spec §7 in
docs/superpowers/specs/2026-05-02-vae-finetune-effectiveness-analysis-design.md.
"""
from __future__ import annotations

from typing import List

import torch


ABLATION_CONDITIONS: List[str] = [
    "full", "zero_ef", "zero_p2", "oracle_ef", "oracle_p2",
]


def apply_head_substitution(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    condition: str,
) -> torch.Tensor:
    """Return a new tensor with the requested head substitution applied.

    All inputs/outputs live in *normalised* feature space (the same space
    seen at the decoder output). Caller is responsible for `denormalize`
    afterwards. The returned tensor is always a fresh clone — never aliases
    the input.
    """
    if condition not in ABLATION_CONDITIONS:
        raise ValueError(
            f"unknown condition {condition!r}; choose from {ABLATION_CONDITIONS}"
        )
    if pred_norm.shape != target_norm.shape:
        raise ValueError(
            f"pred/target shape mismatch: {pred_norm.shape} vs {target_norm.shape}"
        )

    out = pred_norm.clone()
    if condition == "full":
        return out
    if condition == "zero_ef":
        out[:, 6:18] = 0.0
    elif condition == "zero_p2":
        out[:, 3:6] = 0.0
    elif condition == "oracle_ef":
        out[:, 6:18] = target_norm[:, 6:18]
    elif condition == "oracle_p2":
        out[:, 3:6] = target_norm[:, 3:6]
    return out


import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


_REPO = Path("/mnt/novita2/siyuan/workspace/TRELLIS.2")


def label_head_status(
    delta_zero: float,
    delta_oracle: float,
    *,
    dead_zero_threshold: float = 0.05,
    undertrained_oracle_threshold: float = -0.30,
) -> str:
    """Apply spec §7 verdict rules.

    delta_zero    : (cd_zeroH - cd_full) / cd_full   (positive = zeroing hurts)
    delta_oracle  : (cd_oracleH - cd_full) / cd_full (negative = oracle helps)
    """
    if abs(delta_zero) < dead_zero_threshold:
        return "dead"
    if delta_oracle <= undertrained_oracle_threshold:
        return "undertrained"
    return "alive"


def _feats_to_metrics(
    feats_pred_raw: np.ndarray,
    cube_np: np.ndarray,
    gt_npz: Dict[str, np.ndarray],
    resolution: int,
) -> Dict[str, float]:
    """Mirror coart.eval.deep_eval._one_asset post-decoder portion.

    Builds a mesh from feats_pred_raw + cube_np, transforms vertices into
    EXP-5 [-0.5, 0.5]^3 space, samples 100k surface points, computes
    chamfer / normal-consistency / multi-threshold f-score / topology.
    Returns {} on mesh failure.
    """
    sys.path.insert(0, str(_REPO))
    from train_overfit_feat18 import feature_to_mesh

    from coart.eval.metrics import (
        chamfer_distance, compute_topo_metrics, f_score_multi,
        normal_consistency, sample_surface,
    )
    from coart.eval.normalization import corep_to_exp5_vertices

    mesh = feature_to_mesh(feats_pred_raw, cube_np, resolution)
    if (mesh is None
            or getattr(mesh, "faces", None) is None
            or len(mesh.faces) == 0):
        return {}

    mesh.vertices = corep_to_exp5_vertices(
        np.asarray(mesh.vertices, dtype=np.float32)
    )
    pts, nrms = sample_surface(mesh, num_points=100_000)
    cd = chamfer_distance(pts, gt_npz["gt_points"])
    nc = normal_consistency(pts, nrms, gt_npz["gt_points"], gt_npz["gt_normals"])
    fs = f_score_multi(pts, gt_npz["gt_points"], thresholds=[0.005, 0.01, 0.05])
    topo = compute_topo_metrics(mesh)
    return {
        "cd": float(cd),
        "nc": float(nc),
        "f005": float(fs[0.005]),
        "f01": float(fs[0.01]),
        "f05": float(fs[0.05]),
        "n_components": float(topo["n_components"]),
        "euler": float(topo["euler_number"]),
        "n_boundary_edges": float(topo["n_boundary_edges"]),
        "is_watertight": float(topo["is_watertight"]),
    }


def run_ablation_per_asset(
    encoder,
    decoder,
    asset: Dict[str, object],
    stats_mean,
    stats_std,
    resolution: int,
    device,
) -> List[Dict[str, object]]:
    """Run all 5 conditions on one asset; return one row per condition."""
    import torch

    from trellis2.modules import sparse as sp

    from coart.common.dist_utils import unwrap
    from coart.data.stats import denormalize, normalize

    npz_path = _REPO / asset["npz_path"]
    d = np.load(npz_path, allow_pickle=True)
    meta = d["meta"].item() if "meta" in d.files else {}
    if meta.get("normalization") != "exp5":
        return [{
            "asset": asset["name"],
            "condition": cond,
            "skipped": True,
            "reason": f"npz_normalization={meta.get('normalization')!r}",
        } for cond in ABLATION_CONDITIONS]

    cube_indices = torch.from_numpy(d["cube_indices"].astype(np.int32)).to(device)
    feats_raw = torch.from_numpy(d["feats"].astype(np.float32)).to(device)
    feats_n = normalize(feats_raw, stats_mean, stats_std)

    enc = unwrap(encoder).eval()
    dec = unwrap(decoder).eval()
    N = cube_indices.shape[0]
    batch_col = torch.zeros((N, 1), dtype=torch.int32, device=device)
    coords_bn = torch.cat([batch_col, cube_indices], dim=1)
    x = sp.SparseTensor(feats=feats_n, coords=coords_bn)

    with torch.no_grad():
        z = enc(x, sample_posterior=False)
        pred = dec(z)
        pred = pred[0] if isinstance(pred, tuple) else pred
    pred_norm = pred.feats.detach().float().cpu()
    target_norm = feats_n.detach().float().cpu()

    cube_mismatch = pred_norm.shape[0] != target_norm.shape[0]
    cube_np = cube_indices.cpu().numpy().astype(np.int32)

    rows: List[Dict[str, object]] = []
    for cond in ABLATION_CONDITIONS:
        if cube_mismatch and cond not in ("full",):
            rows.append({
                "asset": asset["name"], "condition": cond,
                "skipped": True, "reason": "cube_mismatch",
                "cube_mismatch": True,
            })
            continue

        sub_norm = apply_head_substitution(pred_norm, target_norm, cond)
        feats_pred_raw = denormalize(
            sub_norm, stats_mean.cpu(), stats_std.cpu()
        ).numpy()
        metrics = _feats_to_metrics(
            feats_pred_raw, cube_np,
            {"gt_points": d["gt_points"], "gt_normals": d["gt_normals"]},
            resolution,
        )
        row: Dict[str, object] = {
            "asset": asset["name"], "condition": cond,
            "skipped": not bool(metrics),
            "reason": "" if metrics else "mesh_empty",
            "cube_mismatch": cube_mismatch,
            **metrics,
        }
        rows.append(row)
    return rows


def analyze_head_ablation(
    encoder,
    decoder,
    stats_mean,
    stats_std,
    asset_list: List[Dict[str, object]],
    resolution: int,
    device,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for asset in asset_list:
        rows.extend(run_ablation_per_asset(
            encoder, decoder, asset, stats_mean, stats_std, resolution, device,
        ))
    return pd.DataFrame(rows)


def plot_ablation_delta_per_asset(
    ablation_df: pd.DataFrame,
    metric: str,
    out_path: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = ablation_df[~ablation_df["skipped"].astype(bool)].copy()
    if df.empty:
        # Empty placeholder figure so report.md doesn't break.
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "no ablation data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return

    base = df[df["condition"] == "full"][["asset", metric]].rename(
        columns={metric: "metric_full"}
    )
    merged = df.merge(base, on="asset", how="left")
    merged = merged[merged["condition"] != "full"]
    merged["delta"] = (merged[metric] - merged["metric_full"]) / merged["metric_full"]

    pivot = merged.pivot(index="asset", columns="condition", values="delta")
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot.bar(ax=ax)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel(f"Δ{metric} / {metric}_full")
    ax.set_title(f"head ablation Δ{metric} per asset")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
