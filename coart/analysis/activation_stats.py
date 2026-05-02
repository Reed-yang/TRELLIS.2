"""See spec §6 in
docs/superpowers/specs/2026-05-02-vae-finetune-effectiveness-analysis-design.md.
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


def compute_per_channel_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
    zero_eps: float = 0.05,
) -> pd.DataFrame:
    """Per-channel pred/target stats over voxels.

    Args:
        pred, target: (N, C) tensors in raw (denormalised) space.
        zero_eps: |x| < zero_eps counts as "zero" for `pred_zero_rate` and
            `target_zero_rate`. Default 0.05 matches the spec.

    Returns: DataFrame with one row per channel.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    p = pred.detach().to(torch.float64).cpu()
    t = target.detach().to(torch.float64).cpu()
    N, C = p.shape

    rows: List[Dict[str, object]] = []
    for c in range(C):
        pc = p[:, c]
        tc = t[:, c]
        nz_mask = tc.abs() > zero_eps
        n_nz = int(nz_mask.sum().item())

        if n_nz >= 2:
            mse_cond = ((pc[nz_mask] - tc[nz_mask]) ** 2).mean().item()
        else:
            mse_cond = float("nan")

        pc_centred = pc - pc.mean()
        tc_centred = tc - tc.mean()
        denom = pc_centred.norm() * tc_centred.norm()
        pearson = (
            (pc_centred * tc_centred).sum() / denom
        ).item() if denom.item() > 1e-12 else 0.0

        rows.append({
            "channel": c,
            "pred_mean": pc.mean().item(),
            "pred_std": pc.std(unbiased=False).item(),
            "target_mean": tc.mean().item(),
            "target_std": tc.std(unbiased=False).item(),
            "pearson_r": pearson,
            "mse_overall": ((pc - tc) ** 2).mean().item(),
            "mse_conditional": mse_cond,
            "pred_zero_rate": (pc.abs() < zero_eps).float().mean().item(),
            "target_zero_rate": (tc.abs() < zero_eps).float().mean().item(),
            "n_target_nonzero": n_nz,
        })
    return pd.DataFrame(rows)


def compute_branch_norms(
    contributions: torch.Tensor,
    signal_mask: torch.Tensor,
) -> Dict[str, float]:
    """Per-voxel L2 norms of a branch's contribution, partitioned by signal.

    Args:
        contributions: (N, c_model) tensor of one branch's output before sum.
        signal_mask: (N,) bool tensor; True = voxel is "signal-bearing"
            (e.g. p2 input non-zero, or any ef channel non-zero in the GT).
    """
    norms = contributions.detach().to(torch.float64).norm(dim=1).cpu()
    sig = signal_mask.bool().cpu()
    n_sig = int(sig.sum().item())
    n_zero = int((~sig).sum().item())

    return {
        "norm_all_mean": norms.mean().item(),
        "norm_all_std": norms.std(unbiased=False).item(),
        "norm_signal_mean": (
            norms[sig].mean().item() if n_sig > 0 else float("nan")
        ),
        "norm_signal_std": (
            norms[sig].std(unbiased=False).item() if n_sig > 1 else float("nan")
        ),
        "norm_zero_mean": (
            norms[~sig].mean().item() if n_zero > 0 else float("nan")
        ),
        "norm_zero_std": (
            norms[~sig].std(unbiased=False).item() if n_zero > 1 else float("nan")
        ),
        "n_signal": n_sig,
        "n_zero": n_zero,
    }


class _EncIOTap:
    """Forward-pre hook capturing per-branch contributions of Feat18EncIO.

    Attaches to the Feat18EncIO module. Replays its three branches without
    summing so we can record each contribution separately. Values are stored
    on CPU to keep VRAM steady across many val items.
    """

    def __init__(self, enc_io_module):
        self.module = enc_io_module
        self._handle = enc_io_module.register_forward_hook(self._hook)
        self.records: List[Dict[str, torch.Tensor]] = []

    def _hook(self, module, inputs, output):
        x = inputs[0]
        f = x.feats.detach()
        with torch.no_grad():
            p1 = module.p1_branch(f[:, 0:3]).detach().cpu()
            p2 = module.p2_branch(f[:, 3:6]).detach().cpu()
            ef = module.ef_branch(f[:, 6:18]).detach().cpu()
        self.records.append({
            "p1_contrib": p1,
            "p2_contrib": p2,
            "ef_contrib": ef,
            "raw_feats": f.detach().cpu(),
        })

    def close(self):
        self._handle.remove()


class _DecIOTap:
    """Forward hook capturing per-head outputs of Feat18DecIO BEFORE concat."""

    def __init__(self, dec_io_module):
        self.module = dec_io_module
        self._handle = dec_io_module.register_forward_hook(self._hook)
        self.records: List[Dict[str, torch.Tensor]] = []

    def _hook(self, module, inputs, output):
        x = inputs[0]
        f = x.feats.detach()
        with torch.no_grad():
            p1_out = module.p1_head(f).detach().cpu()
            p2_out = module.p2_head(f).detach().cpu()
            ef_out = module.ef_head(f).detach().cpu()
        self.records.append({
            "p1_out_norm": p1_out,
            "p2_out_norm": p2_out,
            "ef_out_norm": ef_out,
        })

    def close(self):
        self._handle.remove()


def run_val_forward_pass(
    encoder,
    decoder,
    val_dataset,
    stats_mean: torch.Tensor,
    stats_std: torch.Tensor,
    n_items: int,
    device: torch.device,
    seed: int = 0,
    indices=None,
) -> Dict[str, object]:
    """Forward N val items in fp32 with hooks; return raw captures.

    Args:
        indices: optional explicit list/array of dataset indices to forward.
            If None, generates `rng.permutation(len(val_dataset))[:n_items]`.
            Used by multi-rank sharding to give each rank a deterministic
            disjoint slice.

    Returns dict with: pred_norm (Tensor (Σ N_i, 18)), target_norm (Tensor),
    p1_contrib, p2_contrib, ef_contrib (each (Σ N_i, c_model)),
    p1_pred_head, p2_pred_head, ef_pred_head (each in normalised head-output
    space), and per-voxel partition masks `is_p2_zero`, `has_ef_signal`.
    """
    from trellis2.modules import sparse as sp

    from coart.common.dist_utils import unwrap
    from coart.data.stats import normalize

    enc = unwrap(encoder).eval()
    dec = unwrap(decoder).eval()

    enc_io = enc.input_layer
    dec_io = dec.output_layer
    enc_tap = _EncIOTap(enc_io)
    dec_tap = _DecIOTap(dec_io)

    if indices is None:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(val_dataset))[:n_items]

    pred_chunks: List[torch.Tensor] = []
    target_chunks: List[torch.Tensor] = []
    p2_zero_chunks: List[torch.Tensor] = []
    ef_signal_chunks: List[torch.Tensor] = []

    try:
        for idx in indices:
            sample = val_dataset[int(idx)]
            cube_indices = torch.from_numpy(
                sample["cube_indices"].astype(np.int32)
            ).to(device)
            feats_raw = torch.from_numpy(
                sample["feats"].astype(np.float32)
            ).to(device)
            feats_n = normalize(feats_raw, stats_mean, stats_std)

            N = cube_indices.shape[0]
            batch_col = torch.zeros((N, 1), dtype=torch.int32, device=device)
            coords_bn = torch.cat([batch_col, cube_indices], dim=1)
            x = sp.SparseTensor(feats=feats_n, coords=coords_bn)

            with torch.no_grad():
                z = enc(x, sample_posterior=False)
                pred = dec(z)
                pred = pred[0] if isinstance(pred, tuple) else pred

            pred_chunks.append(pred.feats.detach().float().cpu())
            target_chunks.append(feats_n.detach().float().cpu())

            p2_zero_chunks.append(
                (feats_raw[:, 3:6].abs().sum(dim=1) < 1e-6).cpu()
            )
            ef_signal_chunks.append(
                (feats_raw[:, 6:18].abs().sum(dim=1) > 1e-6).cpu()
            )
    finally:
        enc_tap.close()
        dec_tap.close()

    def _cat(records: List[Dict[str, torch.Tensor]], key: str) -> torch.Tensor:
        return torch.cat([r[key] for r in records], dim=0)

    return {
        "pred_norm": torch.cat(pred_chunks, dim=0),
        "target_norm": torch.cat(target_chunks, dim=0),
        "p1_contrib": _cat(enc_tap.records, "p1_contrib"),
        "p2_contrib": _cat(enc_tap.records, "p2_contrib"),
        "ef_contrib": _cat(enc_tap.records, "ef_contrib"),
        "is_p2_zero": torch.cat(p2_zero_chunks, dim=0),
        "has_ef_signal": torch.cat(ef_signal_chunks, dim=0),
    }


def analyze_activation_stats(
    captures: Dict[str, torch.Tensor],
    stats_mean: torch.Tensor,
    stats_std: torch.Tensor,
) -> Dict[str, pd.DataFrame]:
    """Turn raw captures into the two CSV-ready DataFrames.

    Returns {"contributions": branch_norm_df, "per_channel": pcs_df}.
    """
    from coart.data.stats import denormalize

    pred_raw = denormalize(
        captures["pred_norm"], stats_mean.cpu(), stats_std.cpu()
    )
    target_raw = denormalize(
        captures["target_norm"], stats_mean.cpu(), stats_std.cpu()
    )
    pcs = compute_per_channel_stats(pred_raw, target_raw, zero_eps=0.05)

    rows: List[Dict[str, object]] = []
    has_p2 = ~captures["is_p2_zero"]
    has_ef = captures["has_ef_signal"]
    all_true = torch.ones_like(has_p2, dtype=torch.bool)
    branch_specs = [
        ("p1_branch", captures["p1_contrib"], all_true),
        ("p2_branch", captures["p2_contrib"], has_p2),
        ("ef_branch", captures["ef_contrib"], has_ef),
    ]
    for name, contrib, mask in branch_specs:
        norms = compute_branch_norms(contrib, mask)
        rows.append({"branch": name, **norms})
    contrib_df = pd.DataFrame(rows)

    return {"contributions": contrib_df, "per_channel": pcs}


def plot_zero_rate_bars(per_channel_df: pd.DataFrame, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = per_channel_df.copy()
    fig, ax = plt.subplots(figsize=(7, 3))
    width = 0.4
    x = np.arange(len(df))
    ax.bar(x - width/2, df["target_zero_rate"], width=width, label="target")
    ax.bar(x + width/2, df["pred_zero_rate"], width=width, label="pred")
    ax.set_xticks(x)
    ax.set_xticklabels([f"ch{i}" for i in df["channel"]], rotation=90, fontsize=8)
    ax.set_ylabel("zero-rate (|x|<0.05)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_pred_vs_target_scatter(
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    channels: List[int],
    out_dir: str,
    max_points: int = 5000,
    seed: int = 0,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    N = pred_raw.shape[0]
    keep = rng.choice(N, size=min(max_points, N), replace=False)
    p_np = pred_raw[keep].cpu().numpy()
    t_np = target_raw[keep].cpu().numpy()
    os.makedirs(out_dir, exist_ok=True)
    for c in channels:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(t_np[:, c], p_np[:, c], s=2, alpha=0.4)
        lo = min(t_np[:, c].min(), p_np[:, c].min())
        hi = max(t_np[:, c].max(), p_np[:, c].max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_xlabel(f"target ch{c}")
        ax.set_ylabel(f"pred ch{c}")
        ax.set_title(f"channel {c}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"pred_vs_target_ch{c:02d}.png"), dpi=120)
        plt.close(fig)
