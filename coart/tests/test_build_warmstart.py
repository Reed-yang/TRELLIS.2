"""Unit tests for load_pretrained_into warm-start matrix (4 combinations).

Verifies that each (io_arch, warmstart_io) combination produces the expected
initialisation. Does NOT load real pretrained weights — uses a mock pretrained
state dict with known contents and asserts the copy logic is correct.
"""
import pytest
import torch
import torch.nn as nn

from coart.vae.io_stems import Feat18EncIO, Feat18DecIO
from coart.vae.build import _apply_warmstart_three_branch


def _fake_pretrained_sd(c0=64, c_end=64):
    """Mock pretrained encoder + decoder state dicts."""
    return (
        {
            "input_layer.weight": torch.randn(c0, 6),
            "input_layer.bias": torch.randn(c0),
        },
        {
            "output_layer.weight": torch.randn(7, c_end),
            "output_layer.bias": torch.randn(7),
        },
    )


def test_three_branch_warmstart_copies_p1_full_strength():
    """p1_branch.weight must equal pretrained W[:, 0:3] exactly."""
    pre_enc_sd, pre_dec_sd = _fake_pretrained_sd()
    enc_stem = Feat18EncIO(c_model=64)
    dec_stem = Feat18DecIO(c_model=64)

    _apply_warmstart_three_branch(
        enc_stem, dec_stem, pre_enc_sd, pre_dec_sd
    )

    expected_p1_w = pre_enc_sd["input_layer.weight"][:, 0:3]
    assert torch.allclose(enc_stem.p1_branch.weight, expected_p1_w)
    assert torch.allclose(enc_stem.p1_branch.bias, pre_enc_sd["input_layer.bias"])


def test_three_branch_warmstart_p2_weight_matches_p1_but_bias_zero():
    """p2_branch.weight = pretrained vertex cols too; p2_branch.bias = 0."""
    pre_enc_sd, pre_dec_sd = _fake_pretrained_sd()
    enc_stem = Feat18EncIO(c_model=64)
    dec_stem = Feat18DecIO(c_model=64)

    _apply_warmstart_three_branch(enc_stem, dec_stem, pre_enc_sd, pre_dec_sd)

    expected_p2_w = pre_enc_sd["input_layer.weight"][:, 0:3]
    assert torch.allclose(enc_stem.p2_branch.weight, expected_p2_w)
    assert torch.allclose(
        enc_stem.p2_branch.bias,
        torch.zeros_like(enc_stem.p2_branch.bias),
    )


def test_three_branch_warmstart_ef_stays_untouched():
    """ef_branch keeps its xavier_uniform init (not zeroed, not copied)."""
    pre_enc_sd, pre_dec_sd = _fake_pretrained_sd()
    enc_stem = Feat18EncIO(c_model=64)
    dec_stem = Feat18DecIO(c_model=64)

    ef_w_before = enc_stem.ef_branch.weight.clone()
    _apply_warmstart_three_branch(enc_stem, dec_stem, pre_enc_sd, pre_dec_sd)
    assert torch.allclose(enc_stem.ef_branch.weight, ef_w_before), \
        "ef_branch.weight must NOT be modified by three_branch warm-start"


def test_three_branch_warmstart_decoder_p1_head_copies_rows_0_2():
    """Decoder p1_head.weight = pretrained W[0:3, :]."""
    pre_enc_sd, pre_dec_sd = _fake_pretrained_sd()
    enc_stem = Feat18EncIO(c_model=64)
    dec_stem = Feat18DecIO(c_model=64)

    _apply_warmstart_three_branch(enc_stem, dec_stem, pre_enc_sd, pre_dec_sd)

    assert torch.allclose(dec_stem.p1_head.weight, pre_dec_sd["output_layer.weight"][0:3, :])
    assert torch.allclose(dec_stem.p1_head.bias, pre_dec_sd["output_layer.bias"][0:3])


def test_three_branch_warmstart_decoder_p2_head_weight_matches_p1_but_bias_zero():
    """Decoder p2_head.weight = W[0:3, :] (same as p1); p2_head.bias = 0."""
    pre_enc_sd, pre_dec_sd = _fake_pretrained_sd()
    enc_stem = Feat18EncIO(c_model=64)
    dec_stem = Feat18DecIO(c_model=64)

    _apply_warmstart_three_branch(enc_stem, dec_stem, pre_enc_sd, pre_dec_sd)

    assert torch.allclose(dec_stem.p2_head.weight, pre_dec_sd["output_layer.weight"][0:3, :])
    assert torch.allclose(dec_stem.p2_head.bias, torch.zeros_like(dec_stem.p2_head.bias))
