"""Tests for coart.analysis.head_ablation."""
import pytest
import torch

from coart.analysis.head_ablation import apply_head_substitution, ABLATION_CONDITIONS


def test_conditions_constant():
    assert ABLATION_CONDITIONS == [
        "full", "zero_ef", "zero_p2", "oracle_ef", "oracle_p2",
    ]


def test_full_returns_clone_unchanged():
    pred = torch.randn(8, 18)
    target = torch.randn(8, 18)
    out = apply_head_substitution(pred, target, "full")
    assert torch.equal(out, pred)
    assert out is not pred  # ensure clone, not alias


def test_zero_ef_zeros_channels_6_to_18():
    pred = torch.randn(5, 18)
    target = torch.randn(5, 18)
    out = apply_head_substitution(pred, target, "zero_ef")
    assert torch.equal(out[:, 0:6], pred[:, 0:6])
    assert torch.equal(out[:, 6:18], torch.zeros(5, 12))


def test_zero_p2_zeros_channels_3_to_6():
    pred = torch.randn(5, 18)
    target = torch.randn(5, 18)
    out = apply_head_substitution(pred, target, "zero_p2")
    assert torch.equal(out[:, 0:3], pred[:, 0:3])
    assert torch.equal(out[:, 3:6], torch.zeros(5, 3))
    assert torch.equal(out[:, 6:18], pred[:, 6:18])


def test_oracle_ef_copies_target_channels_6_to_18():
    pred = torch.randn(5, 18)
    target = torch.randn(5, 18)
    out = apply_head_substitution(pred, target, "oracle_ef")
    assert torch.equal(out[:, 0:6], pred[:, 0:6])
    assert torch.equal(out[:, 6:18], target[:, 6:18])


def test_oracle_p2_copies_target_channels_3_to_6():
    pred = torch.randn(5, 18)
    target = torch.randn(5, 18)
    out = apply_head_substitution(pred, target, "oracle_p2")
    assert torch.equal(out[:, 0:3], pred[:, 0:3])
    assert torch.equal(out[:, 3:6], target[:, 3:6])
    assert torch.equal(out[:, 6:18], pred[:, 6:18])


def test_unknown_condition_raises():
    pred = torch.randn(5, 18)
    target = torch.randn(5, 18)
    with pytest.raises(ValueError):
        apply_head_substitution(pred, target, "garbage")


def test_pred_target_shape_mismatch_raises():
    with pytest.raises(ValueError):
        apply_head_substitution(torch.randn(4, 18), torch.randn(5, 18), "oracle_ef")


from coart.analysis.head_ablation import label_head_status


def test_label_dead_when_zero_delta_below_threshold():
    label = label_head_status(delta_zero=0.01, delta_oracle=-0.50)
    assert label == "dead"


def test_label_undertrained_when_oracle_helps_a_lot():
    label = label_head_status(delta_zero=0.20, delta_oracle=-0.40)
    assert label == "undertrained"


def test_label_alive_when_zero_hurts_oracle_doesnt_help_much():
    label = label_head_status(delta_zero=0.20, delta_oracle=-0.10)
    assert label == "alive"
