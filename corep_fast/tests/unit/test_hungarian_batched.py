"""W_HG: hungarian_batched parity vs scipy.linear_sum_assignment."""
import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from corep_fast.stages.s7_triton import hungarian_batched


@pytest.fixture
def dev():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test_1x1_hot_path(dev):
    """99.99% of production cubes — must be trivially correct."""
    B = 1000
    cost = torch.full((B, 5, 8), float('inf'), dtype=torch.float32, device=dev)
    g = torch.Generator(device='cpu').manual_seed(0)
    cost[:, 0, 0] = torch.rand(B, generator=g).to(dev)
    nl = torch.ones(B, dtype=torch.int64, device=dev)
    npts = torch.ones(B, dtype=torch.int64, device=dev)
    out = hungarian_batched(cost, nl, npts, max_nl=5, max_np=8)
    assert (out[:, 0] == 0).all()
    assert (out[:, 1:] == -1).all()


def test_1xk_argmin(dev):
    """1xK with K>=2 — argmin over valid cols."""
    rng = np.random.RandomState(1)
    B = 500
    cost = torch.full((B, 5, 8), float('inf'), dtype=torch.float32, device=dev)
    nl = torch.ones(B, dtype=torch.int64, device=dev)
    npts_arr = rng.randint(2, 9, size=B).astype(np.int64)
    npts = torch.from_numpy(npts_arr).to(dev)
    expected = []
    for b in range(B):
        k = int(npts_arr[b])
        row = rng.rand(k).astype(np.float32)
        cost[b, 0, :k] = torch.from_numpy(row).to(dev)
        expected.append(int(row.argmin()))
    out = hungarian_batched(cost, nl, npts, max_nl=5, max_np=8)
    for b in range(B):
        got = int(out[b, 0].item())
        # -1 means deferred-to-fallback (would be the case if a tie existed,
        # but with random floats ties are vanishingly rare).
        if got == -1:
            continue
        assert got == expected[b], f"cube {b}: got {got} vs scipy {expected[b]}"


def test_brute_force_nxn_matches_scipy(dev):
    """2x2..5x5 random cost matrices — match scipy."""
    rng = np.random.RandomState(2)
    B = 300
    cost = torch.full((B, 5, 8), float('inf'), dtype=torch.float32, device=dev)
    nl_arr = rng.randint(2, 6, size=B).astype(np.int64)
    # Ensure npts >= nl, capped at 8
    npts_arr = (rng.randint(0, 7, size=B).astype(np.int64) + nl_arr)
    npts_arr = np.minimum(npts_arr, 8)
    nl_arr = np.minimum(nl_arr, npts_arr)
    nl = torch.from_numpy(nl_arr).to(dev)
    npts = torch.from_numpy(npts_arr).to(dev)
    expected_matches = []
    for b in range(B):
        n_l, n_p = int(nl_arr[b]), int(npts_arr[b])
        sub = rng.rand(n_l, n_p).astype(np.float32)
        cost[b, :n_l, :n_p] = torch.from_numpy(sub).to(dev)
        rows, cols = linear_sum_assignment(sub.astype(np.float64))
        em = np.full(5, -1, dtype=np.int64)
        for r, c in zip(rows, cols):
            if r < n_l:
                em[r] = c
        expected_matches.append(em)
    out = hungarian_batched(cost, nl, npts, max_nl=5, max_np=8).cpu().numpy()
    mismatches = []
    for b in range(B):
        for r in range(int(nl_arr[b])):
            got = out[b, r]
            # -1 means "left for fallback" (deferred); not a mismatch.
            if got == -1:
                continue
            if expected_matches[b][r] != got:
                mismatches.append((b, r, int(expected_matches[b][r]), int(got)))
    assert len(mismatches) == 0, \
        f"{len(mismatches)} bit-mismatches, first: {mismatches[:5]}"


def test_rect_reverse_left_for_fallback(dev):
    """nl > npts cases must be left as -1."""
    B = 50
    cost = torch.full((B, 5, 8), float('inf'), dtype=torch.float32, device=dev)
    nl = torch.full((B,), 3, dtype=torch.int64, device=dev)
    npts = torch.full((B,), 2, dtype=torch.int64, device=dev)
    cost[:, :3, :2] = 0.5
    out = hungarian_batched(cost, nl, npts, max_nl=5, max_np=8)
    assert (out == -1).all()
