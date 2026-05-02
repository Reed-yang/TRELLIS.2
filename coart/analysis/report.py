"""See spec §8 in
docs/superpowers/specs/2026-05-02-vae-finetune-effectiveness-analysis-design.md.
"""
from __future__ import annotations

import os
import textwrap
from typing import List

import pandas as pd


_LABEL_PRIORITY = {"dead": 0, "undertrained": 1, "alive": 2}


def derive_drift_label(
    rel_frob: float,
    bias_drift: float,
    *,
    dead_rel_frob: float = 0.05,
    dead_bias: float = 1e-4,
    alive_rel_frob: float = 0.5,
) -> str:
    if rel_frob < dead_rel_frob and bias_drift < dead_bias:
        return "dead"
    if rel_frob >= alive_rel_frob:
        return "alive"
    return "undertrained"


def derive_functional_label(
    pearson_r: float,
    pred_zero_rate: float,
    target_zero_rate: float,
    *,
    dead_r: float = 0.05,
    alive_r: float = 0.3,
    over_zero_margin: float = 0.5,
) -> str:
    if pearson_r < dead_r and pred_zero_rate >= target_zero_rate + over_zero_margin:
        return "dead"
    if pearson_r >= alive_r:
        return "alive"
    return "undertrained"


def derive_combined_label(parts: List[str]) -> str:
    """Worst-of-three among {alive, undertrained, dead}."""
    return min(parts, key=lambda x: _LABEL_PRIORITY.get(x, 99))


def _build_tldr_table(
    drift_df: pd.DataFrame,
    per_channel_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
) -> str:
    head_to_channels = {
        "p2_head": list(range(3, 6)),
        "ef_head": list(range(6, 18)),
    }
    head_to_branch = {"p2_head": "p2_branch", "ef_head": "ef_branch"}

    lines = ["| head | drift | functional | causal | combined |",
             "|------|-------|------------|--------|----------|"]
    for head, channels in head_to_channels.items():
        branch = head_to_branch[head]
        # Drift label uses the worst across the (head, branch) pair vs step0.
        d = drift_df[(drift_df["ref_kind"] == "step0")
                     & (drift_df["linear_name"].isin([head, branch]))]
        if d.empty:
            drift_label = "n/a"
        else:
            drift_label = derive_combined_label([
                derive_drift_label(r.rel_frob_drift, r.bias_drift_per_dim)
                for r in d.itertuples()
            ])
        # Functional label: worst across the head's channels.
        rows = per_channel_df[per_channel_df["channel"].isin(channels)]
        if rows.empty:
            func_label = "n/a"
        else:
            func_label = derive_combined_label([
                derive_functional_label(
                    r.pearson_r, r.pred_zero_rate, r.target_zero_rate,
                )
                for r in rows.itertuples()
            ])
        # Causal label.
        good = ablation_df[~ablation_df["skipped"].astype(bool)]
        kind = "ef" if head == "ef_head" else "p2"
        if good.empty or "cd" not in good.columns:
            causal_label = "n/a"
        else:
            full = good[good["condition"] == "full"]["cd"].mean()
            zero_h = good[good["condition"] == f"zero_{kind}"]["cd"].mean()
            oracle_h = good[good["condition"] == f"oracle_{kind}"]["cd"].mean()
            if pd.isna(full) or pd.isna(zero_h) or pd.isna(oracle_h) or full == 0:
                causal_label = "n/a"
            else:
                from coart.analysis.head_ablation import label_head_status
                causal_label = label_head_status(
                    delta_zero=(zero_h - full) / full,
                    delta_oracle=(oracle_h - full) / full,
                )
        valid = [l for l in (drift_label, func_label, causal_label) if l != "n/a"]
        combined = derive_combined_label(valid) if valid else "n/a"
        lines.append(
            f"| {head} | {drift_label} | {func_label} | {causal_label} | **{combined}** |"
        )
    return "\n".join(lines)


def write_report(
    drift_df: pd.DataFrame,
    per_channel_df: pd.DataFrame,
    contrib_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    figures_dir: str,
    out_path: str,
    *,
    ckpt_dir: str,
    step: int,
    use_ema: bool,
    n_val: int,
) -> None:
    tldr = _build_tldr_table(drift_df, per_channel_df, ablation_df)
    drift_md = drift_df.to_markdown(index=False)
    pcs_md = per_channel_df.to_markdown(index=False, floatfmt=".4f")
    contrib_md = contrib_df.to_markdown(index=False, floatfmt=".4f")
    abl_md = ablation_df.to_markdown(index=False, floatfmt=".4g")

    body = textwrap.dedent(f"""\
        # VAE Finetune Effectiveness Analysis - step {step}

        - ckpt_dir: `{ckpt_dir}`
        - weights: {"EMA 0.9999" if use_ema else "online"}
        - n_val (activation stats): {n_val}

        ## TL;DR

        {tldr}

        ## A. Weight drift

        Per-Linear drift vs step-0 reconstruction (and adjacent ckpts):

        {drift_md}

        Figures: `{figures_dir}/sv_spectrum_*.png`, `{figures_dir}/perch_norm_hist_*.png`.

        ## B. Activation / output stats

        ### Encoder branch contribution norms

        {contrib_md}

        ### Per-channel decoder pred vs target

        {pcs_md}

        Figure: `{figures_dir}/zero_rate_bars.png`, plus per-channel scatter plots.

        ## C. Head ablation (golden assets)

        {abl_md}

        Figure: `{figures_dir}/ablation_delta_per_asset.png`.

        ## D. Caveats

        - Forward pass in fp32; bf16 wandb-logged metrics may differ slightly.
        - Step-0 reference is a representative xavier draw under the same
          distribution, not the exact training-launch tensor.
        - Conditional MSE uses `target != 0` voxels only; rows with
          `n_target_nonzero < 50` should be treated as low-power.

        ## E. Recommendations

        See spec section 2 - diagnostic pass; tuning recommendations are out of scope.
        """)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(body)
