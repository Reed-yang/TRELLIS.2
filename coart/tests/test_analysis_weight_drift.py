"""Tests for coart.analysis.weight_drift."""
import pytest
import torch

from coart.analysis.weight_drift import compute_drift_metrics


def test_zero_drift_when_identical():
    W = torch.eye(4)
    b = torch.zeros(4)
    m = compute_drift_metrics(W, b, W, b)
    assert m["rel_frob_drift"] == 0.0
    assert m["rms_elem_drift"] == 0.0
    assert m["bias_drift_per_dim"] == 0.0
    assert m["mean_row_cosine"] == pytest.approx(1.0, abs=1e-6)


def test_relative_frob_drift_formula():
    torch.manual_seed(0)
    W_ref = torch.randn(16, 8)
    b_ref = torch.randn(16)
    W_now = W_ref + 0.1 * torch.randn_like(W_ref)
    b_now = b_ref + 0.05 * torch.randn_like(b_ref)
    m = compute_drift_metrics(W_now, b_now, W_ref, b_ref)
    expected_rel = (W_now - W_ref).norm() / W_ref.norm()
    assert m["rel_frob_drift"] == pytest.approx(expected_rel.item(), rel=1e-6)
    expected_rms = (W_now - W_ref).norm() / (W_ref.numel() ** 0.5)
    assert m["rms_elem_drift"] == pytest.approx(expected_rms.item(), rel=1e-6)
    expected_b = (b_now - b_ref).norm() / (b_ref.numel() ** 0.5)
    assert m["bias_drift_per_dim"] == pytest.approx(expected_b.item(), rel=1e-6)


def test_singular_spectrum_and_effective_rank():
    # Diagonal matrix with known SVs.
    W = torch.diag(torch.tensor([3.0, 2.0, 1.0, 0.0, 0.0]))
    b = torch.zeros(5)
    m = compute_drift_metrics(W, b, W, b)
    sv = m["singular_top5"]
    assert sv[:3] == pytest.approx([3.0, 2.0, 1.0], abs=1e-5)
    # Effective rank = exp(H(s/sum(s))). Probabilities = [0.5, 1/3, 1/6].
    import math
    p = [0.5, 1 / 3, 1 / 6]
    H = -sum(pi * math.log(pi) for pi in p if pi > 0)
    assert m["effective_rank"] == pytest.approx(math.exp(H), rel=1e-5)
    # Condition number uses smallest non-zero SV.
    assert m["cond_number"] == pytest.approx(3.0, rel=1e-5)


def test_mean_row_cosine_orthogonal_rows_gives_zero():
    W_ref = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    W_rotated = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # each row 90 deg from ref row
    b = torch.zeros(2)
    m = compute_drift_metrics(W_rotated, b, W_ref, b)
    assert abs(m["mean_row_cosine"]) < 1e-6
