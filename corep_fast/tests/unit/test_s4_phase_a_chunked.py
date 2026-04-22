"""Parity tests for chunked Phase-A tiers in s4_phase_a_chunked.py.

Ensures dense GPU, chunked GPU, and chunked CPU all produce bit-exact
canonical_idx output, and that the full _count_uturns_from_packed output
is identical across tiers for the same input.
"""
import os

import numpy as np
import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA")


def _synthetic(G=4, P=256, seed=0):
    rng = np.random.default_rng(seed)
    # Make some near-duplicate points so the 1e-8 tolerance actually catches pairs.
    pts_np = rng.random((G, P, 3)).astype(np.float64)
    # Force a small fraction to have exact duplicates.
    dup_pairs = rng.integers(0, P, size=(G, P // 8, 2))
    for g in range(G):
        for a, b in dup_pairs[g]:
            if a != b:
                pts_np[g, b] = pts_np[g, a] + 1e-9  # well within 1e-8
    pts_valid_np = np.ones((G, P), dtype=bool)
    # Mark a random 5% as invalid padding.
    for g in range(G):
        inv = rng.choice(P, size=P // 20, replace=False)
        pts_valid_np[g, inv] = False
    return pts_np, pts_valid_np


def test_tier0_vs_tier1_tier2_parity():
    from corep_fast.stages.s4_phase_a_chunked import (
        _phase_a_dense_gpu, _phase_a_chunked_gpu, _phase_a_chunked_cpu,
    )
    for seed in range(5):
        pts_np, pv_np = _synthetic(G=4, P=512, seed=seed)
        pts = torch.from_numpy(pts_np).cuda()
        pts_valid = torch.from_numpy(pv_np).cuda()

        out_dense = _phase_a_dense_gpu(pts, pts_valid)
        out_chunk = _phase_a_chunked_gpu(pts, pts_valid, chunk_rows=64)
        out_cpu = _phase_a_chunked_cpu(pts, pts_valid, chunk_rows=64)

        assert torch.equal(out_dense, out_chunk), (
            f"seed={seed} dense vs chunked GPU mismatch")
        assert torch.equal(out_dense, out_cpu), (
            f"seed={seed} dense vs chunked CPU mismatch")


def test_end_to_end_count_uturns_parity():
    """Full _count_uturns_from_packed output should be identical for all tiers."""
    from corep_fast.stages.s4_face_point import _count_uturns_from_packed
    for seed in range(3):
        G, P = 2, 128
        pts_np, pv_np = _synthetic(G=G, P=P, seed=seed)
        pts = torch.from_numpy(pts_np).cuda()
        pts_valid = torch.from_numpy(pv_np).cuda()
        fv = torch.from_numpy(
            np.random.default_rng(seed).random((G, 3, 3))).cuda()
        ed = torch.zeros(G, 3, dtype=torch.int64, device="cuda")

        results = {}
        for mode in ("dense", "chunked", "cpu"):
            if mode == "chunked":
                os.environ["COREP_FAST_S4_PHASE_A_FORCE_CHUNKED"] = "1"
                os.environ.pop("COREP_FAST_S4_PHASE_A_FORCE_CPU", None)
            elif mode == "cpu":
                os.environ["COREP_FAST_S4_PHASE_A_FORCE_CPU"] = "1"
                os.environ.pop("COREP_FAST_S4_PHASE_A_FORCE_CHUNKED", None)
            else:
                os.environ.pop("COREP_FAST_S4_PHASE_A_FORCE_CHUNKED", None)
                os.environ.pop("COREP_FAST_S4_PHASE_A_FORCE_CPU", None)
            results[mode] = _count_uturns_from_packed(pts, pts_valid, fv, ed).cpu()

        # Cleanup env so other tests aren't affected.
        os.environ.pop("COREP_FAST_S4_PHASE_A_FORCE_CHUNKED", None)
        os.environ.pop("COREP_FAST_S4_PHASE_A_FORCE_CPU", None)

        assert torch.equal(results["dense"], results["chunked"]), (
            f"seed={seed} dense vs chunked u-turn count mismatch")
        assert torch.equal(results["dense"], results["cpu"]), (
            f"seed={seed} dense vs cpu u-turn count mismatch")
