"""Tests for coart.analysis.activation_stats."""
import pytest
import torch

from coart.analysis.activation_stats import (
    compute_per_channel_stats,
    compute_branch_norms,
)


def test_per_channel_stats_shape_and_keys():
    pred = torch.randn(100, 18)
    target = torch.randn(100, 18)
    df = compute_per_channel_stats(pred, target, zero_eps=0.05)
    assert len(df) == 18
    for col in (
        "channel", "pred_mean", "pred_std",
        "target_mean", "target_std",
        "pearson_r", "mse_overall", "mse_conditional",
        "pred_zero_rate", "target_zero_rate",
        "n_target_nonzero",
    ):
        assert col in df.columns


def test_per_channel_stats_zero_target_handles_conditional_mse():
    # For a channel where target is identically zero, conditional MSE is NaN
    # and n_target_nonzero == 0.
    pred = torch.randn(50, 18)
    target = torch.randn(50, 18)
    target[:, 6] = 0.0
    df = compute_per_channel_stats(pred, target, zero_eps=0.05)
    row = df.iloc[6]
    assert row["n_target_nonzero"] == 0
    assert (row["mse_conditional"] != row["mse_conditional"])  # NaN check


def test_per_channel_stats_pearson_perfect_correlation():
    pred = torch.zeros(64, 18)
    target = torch.zeros(64, 18)
    target[:, 0] = torch.linspace(-1, 1, 64)
    pred[:, 0] = target[:, 0] * 2.0 + 0.5
    df = compute_per_channel_stats(pred, target, zero_eps=0.05)
    assert df.iloc[0]["pearson_r"] == pytest.approx(1.0, abs=1e-5)


def test_compute_branch_norms_partitions():
    # 10 voxels of c_model=8 contributions; signal_mask = first 4 are non-zero.
    contrib = torch.randn(10, 8)
    contrib[:4] *= 5.0
    mask = torch.tensor([True]*4 + [False]*6)
    out = compute_branch_norms(contrib, mask)
    assert "norm_all_mean" in out
    assert "norm_signal_mean" in out
    assert "norm_zero_mean" in out
    # Signal partition norm should be larger than zero partition norm.
    assert out["norm_signal_mean"] > out["norm_zero_mean"]
