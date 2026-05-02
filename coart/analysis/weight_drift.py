"""See spec §5 in
docs/superpowers/specs/2026-05-02-vae-finetune-effectiveness-analysis-design.md.
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch


_TOP_K_SV = 5


def compute_drift_metrics(
    W_now: torch.Tensor,
    b_now: torch.Tensor,
    W_ref: torch.Tensor,
    b_ref: torch.Tensor,
) -> Dict[str, object]:
    """Static drift metrics between two `nn.Linear` weight/bias pairs.

    Both tensors must have matching shapes. Computation is fp64 internally
    to avoid catastrophic cancellation on near-zero drift.
    """
    if W_now.shape != W_ref.shape:
        raise ValueError(f"weight shape mismatch: {W_now.shape} vs {W_ref.shape}")
    if b_now.shape != b_ref.shape:
        raise ValueError(f"bias shape mismatch: {b_now.shape} vs {b_ref.shape}")

    W_now64 = W_now.detach().to(torch.float64)
    W_ref64 = W_ref.detach().to(torch.float64)
    b_now64 = b_now.detach().to(torch.float64)
    b_ref64 = b_ref.detach().to(torch.float64)

    diff = W_now64 - W_ref64
    rel_frob = (diff.norm() / W_ref64.norm().clamp_min(1e-12)).item()
    rms = (diff.norm() / math.sqrt(W_ref64.numel())).item()
    bias_drift = (
        (b_now64 - b_ref64).norm() / math.sqrt(max(b_ref64.numel(), 1))
    ).item()

    sv = torch.linalg.svdvals(W_now64)
    top = sv[:_TOP_K_SV].tolist()
    if len(top) < _TOP_K_SV:
        top = top + [0.0] * (_TOP_K_SV - len(top))

    s_sum = sv.sum().clamp_min(1e-12)
    p = (sv / s_sum).clamp_min(1e-30)
    H = -(p * p.log()).sum().item()
    eff_rank = math.exp(H)

    nonzero = sv[sv > 1e-9]
    cond = (sv[0] / nonzero[-1]).item() if nonzero.numel() > 0 else float("inf")

    cos = torch.nn.functional.cosine_similarity(
        W_now64, W_ref64, dim=1
    )
    mean_row_cos = cos.mean().item()

    return {
        "rel_frob_drift": rel_frob,
        "rms_elem_drift": rms,
        "bias_drift_per_dim": bias_drift,
        "singular_top5": top,
        "effective_rank": eff_rank,
        "cond_number": cond,
        "mean_row_cosine": mean_row_cos,
    }


from typing import Tuple


TARGET_LINEARS: List[Tuple[str, str]] = [
    ("encoder", "p1_branch"),
    ("encoder", "p2_branch"),
    ("encoder", "ef_branch"),
    ("decoder", "p1_head"),
    ("decoder", "p2_head"),
    ("decoder", "ef_head"),
]


def extract_io_linear_weights(
    state_dict: Dict[str, torch.Tensor],
    side: str,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Pull (W, b) for each three-branch IO Linear out of an enc/dec state_dict.

    Encoder Linears live under prefix `input_layer.<branch_name>.{weight,bias}`;
    decoder Linears live under `output_layer.<head_name>.{weight,bias}`.
    """
    if side == "encoder":
        prefix = "input_layer"
        names = ["p1_branch", "p2_branch", "ef_branch"]
    elif side == "decoder":
        prefix = "output_layer"
        names = ["p1_head", "p2_head", "ef_head"]
    else:
        raise ValueError(f"side must be 'encoder' or 'decoder', got {side!r}")

    out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for n in names:
        wkey = f"{prefix}.{n}.weight"
        bkey = f"{prefix}.{n}.bias"
        if wkey not in state_dict or bkey not in state_dict:
            raise KeyError(
                f"state_dict missing {wkey!r} or {bkey!r}; got keys "
                f"{list(state_dict.keys())[:8]}..."
            )
        out[n] = (state_dict[wkey], state_dict[bkey])
    return out


import json
import os
from pathlib import Path

import pandas as pd


def reconstruct_step0_state_dicts(
    cfg_json_path: str,
    seed: int = 0,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Re-build (encoder, decoder) at training-time init using the same cfg.

    Pins torch CPU/CUDA RNGs to `seed` *before* model construction so the
    xavier-initialised IO branches are reproducible across analysis runs.
    Note: this is NOT guaranteed to match the original training-launch seed,
    but provides a representative xavier draw under the same distribution
    (see spec §11 for caveats).
    """
    from coart.vae.build import build_models, load_pretrained_into

    with open(cfg_json_path) as fh:
        cfg = json.load(fh)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    enc, dec = build_models(
        latent_channels=cfg.get("latent_channels", 32),
        device="cpu",
        io_arch=cfg.get("io_arch", "three_branch"),
    )
    if cfg.get("from_pretrained", True):
        load_pretrained_into(
            enc, dec,
            enc_path=cfg["enc_pretrained"],
            dec_path=cfg["dec_pretrained"],
            io_arch=cfg.get("io_arch", "three_branch"),
            warmstart_io=cfg.get("warmstart_io", True),
            verbose=False,
        )
    return enc.state_dict(), dec.state_dict()


def _overlay_ema_shadow(model: torch.nn.Module, ema_blob: Dict[str, object]) -> None:
    """Overlay EMA shadow tensors onto `model.parameters()` in place.

    EMA file schema (saved by `coart.common.ema.EMAModel.state_dict`):
    `{"decay": float, "shadow": [Tensor, ...]}` where `shadow` is ordered to
    match `model.parameters()` at construction time. Mirrors the reference
    `_overlay_ema_params` in `scripts/coart_ema_eval.py`.
    """
    shadow = ema_blob["shadow"]
    live_params = list(model.parameters())
    if len(shadow) != len(live_params):
        raise RuntimeError(
            f"EMA shadow count mismatch: shadow={len(shadow)} live={len(live_params)}"
        )
    with torch.no_grad():
        for s, p in zip(shadow, live_params):
            p.data.copy_(s.data.to(p.device, dtype=p.dtype))


def _build_models_from_cfg(
    ckpt_dir: str,
    seed: int = 0,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Build encoder + decoder on CPU using `<ckpt_dir>/config.json` cfg.

    Pins torch CPU/CUDA RNG seeds so the xavier-initialised IO branches are
    reproducible across analysis runs. Does NOT load pretrained backbone —
    the caller overlays trained weights (online state_dict or EMA shadow)
    over every parameter slot, which covers all learnt tensors.
    """
    from coart.vae.build import build_models

    cfg_path = os.path.join(ckpt_dir, "config.json")
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    enc, dec = build_models(
        latent_channels=cfg.get("latent_channels", 32),
        device="cpu",
        io_arch=cfg.get("io_arch", "three_branch"),
    )
    return enc, dec


def _load_ckpt_state_dicts(
    ckpt_path: str,
    use_ema: bool,
    ckpt_dir: str,
    step: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Return (enc_sd, dec_sd) from either the online combined ckpt or the
    EMA shadow split ckpts at `step`.

    Online ckpt schema (per `coart.vae.train.save_ckpt`): a single .pt with
    keys {"encoder", "decoder", "optimizer", ...}; encoder/decoder are full
    state_dicts.
    EMA schema (per `coart.common.ema.EMAModel.state_dict`): separate
    `ema_<rate>_{enc,dec}_step<STEP>.pt`, each `{"decay": float, "shadow":
    [tensor, ...]}` where shadow matches `model.parameters()` order. We
    build a fresh model via `_build_models_from_cfg`, overlay shadow onto
    its parameters, and return the resulting state_dict.
    """
    if use_ema:
        ema_enc_path = os.path.join(ckpt_dir, f"ema_0.9999_enc_step{step:07d}.pt")
        ema_dec_path = os.path.join(ckpt_dir, f"ema_0.9999_dec_step{step:07d}.pt")
        ema_enc_blob = torch.load(ema_enc_path, map_location="cpu", weights_only=True)
        ema_dec_blob = torch.load(ema_dec_path, map_location="cpu", weights_only=True)
        enc, dec = _build_models_from_cfg(ckpt_dir)
        _overlay_ema_shadow(enc, ema_enc_blob)
        _overlay_ema_shadow(dec, ema_dec_blob)
        return enc.state_dict(), dec.state_dict()
    blob = torch.load(
        os.path.join(ckpt_dir, f"ckpt_step{step:07d}.pt"),
        map_location="cpu",
        weights_only=True,
    )
    return blob["encoder"], blob["decoder"]


def analyze_weight_drift(
    ckpt_dir: str,
    step: int,
    use_ema: bool = True,
    seed: int = 0,
    extra_steps: Tuple[int, ...] = (145000, 150000),
) -> pd.DataFrame:
    """Build the long-form drift CSV for the six target Linears.

    Columns: linear_name, side, ref_kind, rel_frob_drift, rms_elem_drift,
    bias_drift_per_dim, effective_rank, cond_number, mean_row_cosine,
    sv_top1, sv_top2, sv_top3, sv_top4, sv_top5.
    """
    cfg_path = os.path.join(ckpt_dir, "config.json")
    enc_now, dec_now = _load_ckpt_state_dicts(
        ckpt_dir, use_ema=use_ema, ckpt_dir=ckpt_dir, step=step,
    )
    enc_step0, dec_step0 = reconstruct_step0_state_dicts(cfg_path, seed=seed)

    refs = {"step0": (enc_step0, dec_step0)}
    for s in extra_steps:
        try:
            enc_s, dec_s = _load_ckpt_state_dicts(
                ckpt_dir, use_ema=False, ckpt_dir=ckpt_dir, step=s,
            )
            refs[f"step{s}"] = (enc_s, dec_s)
        except FileNotFoundError:
            continue

    rows: List[Dict[str, object]] = []
    for side, branch in TARGET_LINEARS:
        sd_now = enc_now if side == "encoder" else dec_now
        Wnow, bnow = extract_io_linear_weights(sd_now, side=side)[branch]
        for ref_kind, (enc_ref, dec_ref) in refs.items():
            sd_ref = enc_ref if side == "encoder" else dec_ref
            Wref, bref = extract_io_linear_weights(sd_ref, side=side)[branch]
            m = compute_drift_metrics(Wnow, bnow, Wref, bref)
            rows.append({
                "linear_name": branch,
                "side": side,
                "ref_kind": ref_kind,
                "rel_frob_drift": m["rel_frob_drift"],
                "rms_elem_drift": m["rms_elem_drift"],
                "bias_drift_per_dim": m["bias_drift_per_dim"],
                "effective_rank": m["effective_rank"],
                "cond_number": m["cond_number"],
                "mean_row_cosine": m["mean_row_cosine"],
                **{
                    f"sv_top{i+1}": m["singular_top5"][i]
                    for i in range(_TOP_K_SV)
                },
            })
    return pd.DataFrame(rows)


def plot_sv_spectrum(
    W_now: torch.Tensor,
    W_ref: torch.Tensor,
    title: str,
    out_path: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sv_now = torch.linalg.svdvals(W_now.to(torch.float64)).cpu().numpy()
    sv_ref = torch.linalg.svdvals(W_ref.to(torch.float64)).cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.semilogy(sv_now, label="now", marker=".")
    ax.semilogy(sv_ref, label="ref", marker=".", linestyle="--")
    ax.set_xlabel("index")
    ax.set_ylabel("singular value (log)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_perch_norm_hist(
    W_now: torch.Tensor,
    W_ref: torch.Tensor,
    title: str,
    out_path: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_now = W_now.detach().to(torch.float64).norm(dim=1).cpu().numpy()
    n_ref = W_ref.detach().to(torch.float64).norm(dim=1).cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(n_ref, bins=40, alpha=0.5, label="ref")
    ax.hist(n_now, bins=40, alpha=0.5, label="now")
    ax.set_xlabel("per-output-channel L2 norm")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
