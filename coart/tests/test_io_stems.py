"""Unit tests for coart.vae.io_stems."""
import pytest
import torch

from coart.vae.io_stems import Feat18EncIO, Feat18DecIO


def _fake_sparse_input(N=32, C=18):
    """Mock SparseTensor-like object exposing .feats and .replace()."""
    class _FakeST:
        def __init__(self, feats):
            self.feats = feats
        def replace(self, new_feats):
            return _FakeST(new_feats)
    return _FakeST(torch.randn(N, C))


def test_enc_io_output_shape():
    stem = Feat18EncIO(c_model=64)
    x = _fake_sparse_input(N=10, C=18)
    out = stem(x)
    assert out.feats.shape == (10, 64)


def test_enc_io_three_branches_independent_gradients():
    """Verify p1_branch, p2_branch, ef_branch are independent nn.Linear modules."""
    stem = Feat18EncIO(c_model=64)
    assert stem.p1_branch is not stem.p2_branch, "p1 and p2 must be independent Linears"
    assert stem.p1_branch.weight is not stem.p2_branch.weight
    assert stem.p1_branch.in_features == 3
    assert stem.p2_branch.in_features == 3
    assert stem.ef_branch.in_features == 12
    assert stem.p1_branch.out_features == 64
    assert stem.p2_branch.out_features == 64
    assert stem.ef_branch.out_features == 64


def test_enc_io_forward_decomposes_18ch():
    """Forward = p1_branch(f[:,0:3]) + p2_branch(f[:,3:6]) + ef_branch(f[:,6:18])."""
    stem = Feat18EncIO(c_model=64)
    f = torch.randn(5, 18)
    x = _fake_sparse_input(N=5, C=18)
    x.feats = f
    out = stem(x)
    expected = (
        stem.p1_branch(f[:, 0:3])
        + stem.p2_branch(f[:, 3:6])
        + stem.ef_branch(f[:, 6:18])
    )
    assert torch.allclose(out.feats, expected, atol=1e-6)


def test_dec_io_output_shape_and_decomposition():
    stem = Feat18DecIO(c_model=64)
    x = _fake_sparse_input(N=7, C=64)
    out = stem(x)
    assert out.feats.shape == (7, 18)
    # Verify split: first 3 = p1_head, 3:6 = p2_head, 6:18 = ef_head
    expected = torch.cat([
        stem.p1_head(x.feats),
        stem.p2_head(x.feats),
        stem.ef_head(x.feats),
    ], dim=-1)
    assert torch.allclose(out.feats, expected, atol=1e-6)


def test_dec_io_three_heads_independent():
    stem = Feat18DecIO(c_model=64)
    assert stem.p1_head is not stem.p2_head
    assert stem.p1_head.out_features == 3
    assert stem.p2_head.out_features == 3
    assert stem.ef_head.out_features == 12
