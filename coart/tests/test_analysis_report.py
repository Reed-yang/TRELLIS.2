"""Tests for coart.analysis.report."""
import pytest
import os
import pandas as pd

from coart.analysis.report import (
    derive_drift_label, derive_functional_label, derive_combined_label,
    write_report,
)


def test_derive_drift_label_dead():
    assert derive_drift_label(rel_frob=0.02, bias_drift=0.0) == "dead"


def test_derive_drift_label_alive():
    assert derive_drift_label(rel_frob=0.5, bias_drift=0.1) == "alive"


def test_derive_functional_label_thresholds():
    assert derive_functional_label(pearson_r=0.0, pred_zero_rate=1.0,
                                   target_zero_rate=0.5) == "dead"
    assert derive_functional_label(pearson_r=0.10, pred_zero_rate=0.5,
                                   target_zero_rate=0.5) == "undertrained"
    assert derive_functional_label(pearson_r=0.50, pred_zero_rate=0.5,
                                   target_zero_rate=0.5) == "alive"


def test_derive_combined_label_worst_of_three():
    assert derive_combined_label(["alive", "alive", "dead"]) == "dead"
    assert derive_combined_label(["alive", "undertrained", "alive"]) == "undertrained"
    assert derive_combined_label(["alive", "alive", "alive"]) == "alive"


def test_write_report_creates_file(tmp_path):
    drift = pd.DataFrame([{
        "linear_name": "ef_branch", "side": "encoder", "ref_kind": "step0",
        "rel_frob_drift": 0.5, "rms_elem_drift": 0.1,
        "bias_drift_per_dim": 0.05, "effective_rank": 3.0,
        "cond_number": 2.0, "mean_row_cosine": 0.8,
        "sv_top1": 1.0, "sv_top2": 0.5, "sv_top3": 0.3,
        "sv_top4": 0.1, "sv_top5": 0.0,
    }])
    pcs = pd.DataFrame([{
        "channel": 6, "pred_mean": 0.0, "pred_std": 0.1,
        "target_mean": 0.0, "target_std": 0.5,
        "pearson_r": 0.6, "mse_overall": 0.1, "mse_conditional": 0.2,
        "pred_zero_rate": 0.7, "target_zero_rate": 0.99,
        "n_target_nonzero": 100,
    }])
    contrib = pd.DataFrame([{
        "branch": "ef_branch", "norm_all_mean": 1.0, "norm_all_std": 0.1,
        "norm_signal_mean": 1.5, "norm_signal_std": 0.2,
        "norm_zero_mean": 0.1, "norm_zero_std": 0.01,
        "n_signal": 100, "n_zero": 900,
    }])
    abl = pd.DataFrame([
        {"asset": "helmet", "condition": "full", "skipped": False,
         "reason": "", "cube_mismatch": False, "cd": 1e-5},
        {"asset": "helmet", "condition": "zero_ef", "skipped": False,
         "reason": "", "cube_mismatch": False, "cd": 1.05e-5},
        {"asset": "helmet", "condition": "oracle_ef", "skipped": False,
         "reason": "", "cube_mismatch": False, "cd": 0.5e-5},
    ])
    out = tmp_path / "report.md"
    write_report(
        drift_df=drift, per_channel_df=pcs, contrib_df=contrib,
        ablation_df=abl, figures_dir="figures", out_path=str(out),
        ckpt_dir="dummy", step=155000, use_ema=True, n_val=200,
    )
    text = out.read_text()
    assert "TL;DR" in text
    assert "ef_branch" in text
    assert "helmet" in text
    assert os.path.exists(out)
