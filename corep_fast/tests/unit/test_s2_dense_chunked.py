"""Parity tests for chunked dense label-propagation in s2_dense_chunked.py.

Ensures full-N GPU, chunked GPU, and chunked CPU tiers all produce bit-exact
`labels (N, M) int64` output. Uses env-var forcing to exercise each tier.
"""
import os
import pytest
import numpy as np
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA")


def _synthetic(N=8, M=32, seed=0):
    """Build a synthetic (padded_faces, neighbors_of, mask) triple.

    Each cube has a random number of "registered" faces in [1, M]. Face IDs
    are drawn from a pool of F_max distinct IDs. neighbors_of is a per-cube
    gather of a random face_adj table, so label-prop has real structure
    (not all disjoint, not all one component).
    """
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, M + 1, size=N).astype(np.int64)
    padded = np.full((N, M), -1, dtype=np.int32)
    F_max = int(counts.sum() * 2)
    face_ids = np.arange(F_max, dtype=np.int32)
    for n in range(N):
        c = int(counts[n])
        perm = rng.permutation(F_max)[:c]
        padded[n, :c] = face_ids[perm]
    face_adj = rng.integers(0, F_max, size=(F_max, 3)).astype(np.int32)
    drop = rng.random((F_max, 3)) > 0.4
    face_adj[drop] = -1
    mask_np = np.arange(M)[None, :] < counts[:, None]

    padded_t = torch.from_numpy(padded).cuda()
    mask_t = torch.from_numpy(mask_np).cuda()
    face_adj_t = torch.from_numpy(face_adj).cuda()
    nbr_of = face_adj_t[padded_t.long().clamp(min=0)]  # (N, M, 3)
    return padded_t, nbr_of, mask_t


def test_tier0_vs_tier1_parity():
    from corep_fast.stages.s2_dense_chunked import _s2_dense_label_prop_chunked
    for seed in range(5):
        padded, nbr, mask = _synthetic(N=16, M=32, seed=seed)
        N, M = padded.shape

        # Tier 0: natural dispatch — expected to fit full-N on GPU.
        os.environ.pop("COREP_FAST_S2_DENSE_FORCE_CHUNKED", None)
        os.environ.pop("COREP_FAST_S2_DENSE_FORCE_CPU", None)
        out_dense = _s2_dense_label_prop_chunked(
            padded, nbr, mask, N, M, torch.device("cuda"))

        # Tier 1: forced chunked GPU.
        os.environ["COREP_FAST_S2_DENSE_FORCE_CHUNKED"] = "1"
        out_chunk = _s2_dense_label_prop_chunked(
            padded, nbr, mask, N, M, torch.device("cuda"))
        os.environ.pop("COREP_FAST_S2_DENSE_FORCE_CHUNKED", None)

        # Tier 2: forced chunked CPU.
        os.environ["COREP_FAST_S2_DENSE_FORCE_CPU"] = "1"
        out_cpu = _s2_dense_label_prop_chunked(
            padded, nbr, mask, N, M, torch.device("cuda"))
        os.environ.pop("COREP_FAST_S2_DENSE_FORCE_CPU", None)

        # All three tiers must produce bit-exact identical labels.
        assert torch.equal(out_dense, out_chunk), (
            f"seed={seed}: dense vs chunked GPU mismatch")
        assert torch.equal(out_dense, out_cpu), (
            f"seed={seed}: dense vs chunked CPU mismatch")


def test_tiny_chunk_bytes_forces_single_cube_chunks():
    """With CHUNK_BYTES = 1, chunk size floors to 1 cube — still bit-exact."""
    from corep_fast.stages.s2_dense_chunked import _s2_dense_label_prop_chunked
    padded, nbr, mask = _synthetic(N=12, M=32, seed=42)
    N, M = padded.shape
    os.environ.pop("COREP_FAST_S2_DENSE_FORCE_CHUNKED", None)
    os.environ.pop("COREP_FAST_S2_DENSE_FORCE_CPU", None)
    ref = _s2_dense_label_prop_chunked(
        padded, nbr, mask, N, M, torch.device("cuda"))

    os.environ["COREP_FAST_S2_DENSE_FORCE_CHUNKED"] = "1"
    os.environ["COREP_FAST_S2_DENSE_CHUNK_BYTES"] = "1"
    try:
        got = _s2_dense_label_prop_chunked(
            padded, nbr, mask, N, M, torch.device("cuda"))
    finally:
        os.environ.pop("COREP_FAST_S2_DENSE_FORCE_CHUNKED", None)
        os.environ.pop("COREP_FAST_S2_DENSE_CHUNK_BYTES", None)
    assert torch.equal(ref, got), (
        "single-cube-per-chunk dispatch mismatch vs natural dispatch")


def test_end_to_end_s2_components_parity():
    """Invoking s2_components on a synthetic batch must give the same result
    with chunked dispatch as with the original non-chunked dense path."""
    # This relies on the integration into s2_components.py. The parametric
    # forcing via env-var lets us compare Tier 0 vs Tier 1 outputs through
    # the top-level s2_components call.
    pytest.skip(
        "End-to-end parity covered by F1/F2/F3 regression goldens, which exercise "
        "the integrated dispatch.")
