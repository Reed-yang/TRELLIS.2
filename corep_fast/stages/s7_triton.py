"""Triton kernels for s7_rank_assign stage.

W_BAF: single fused kernel replacing the 12*3*W Python-loop scatter dispatch
in _build_adjacency_gpu. Spike phase covers fast-path cubes only
(uturn_assignment[:, 0, 0] == -1 -> no U-turn correction).
"""
from __future__ import annotations
import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit
    def _build_adj_fast_kernel(
        edge_weights_ptr,        # (N, 18) int64
        eA_tab_ptr,              # (12, 3) int64 constant
        eB_tab_ptr,              # (12, 3) int64 constant
        eC_tab_ptr,              # (12, 3) int64 constant
        a_at_v0_tab_ptr,         # (12, 3) int8 constant
        b_at_v0_tab_ptr,         # (12, 3) int8 constant
        adj_ptr,                 # (N, NODES, 2) int32 output (init -1)
        fill_count_ptr,          # (N, NODES) int32 output (init 0)
        N: tl.constexpr,
        NODES: tl.constexpr,
        W: tl.constexpr,
    ):
        """One program per (cube_id, t_idx, pi, jj) tuple. Fast-path only."""
        pid = tl.program_id(0)
        jj = pid % W
        pid_r = pid // W
        pi = pid_r % 3
        pid_r2 = pid_r // 3
        t_idx = pid_r2 % 12
        cube_id = pid_r2 // 12

        if cube_id >= N:
            return

        # Load constant table entries
        eA = tl.load(eA_tab_ptr + t_idx * 3 + pi)
        eB = tl.load(eB_tab_ptr + t_idx * 3 + pi)
        eC = tl.load(eC_tab_ptr + t_idx * 3 + pi)
        a_at_v0 = tl.load(a_at_v0_tab_ptr + t_idx * 3 + pi)
        b_at_v0 = tl.load(b_at_v0_tab_ptr + t_idx * 3 + pi)

        # Fast-path: ew_eff == ew (no u-correction)
        w_a = tl.load(edge_weights_ptr + cube_id * 18 + eA)
        w_b = tl.load(edge_weights_ptr + cube_id * 18 + eB)
        w_c = tl.load(edge_weights_ptr + cube_id * 18 + eC)

        # k_pair in int64 (w_a/b/c are int64)
        k_pair = (w_a + w_b - w_c) // 2
        zero_i64 = tl.zeros((), dtype=tl.int64)
        W_i64 = zero_i64 + W
        k_pair = tl.where(k_pair < 0, zero_i64, k_pair)
        k_pair = tl.where(k_pair > W_i64, W_i64, k_pair)
        jj_i64 = zero_i64 + jj
        if jj_i64 >= k_pair:
            return

        # Endpoint flipping
        pts_A = tl.where(a_at_v0 != 0, jj_i64, w_a - 1 - jj_i64)
        pts_B = tl.where(b_at_v0 != 0, jj_i64, w_b - 1 - jj_i64)

        node_A = eA * W + pts_A
        node_B = eB * W + pts_B

        # A -> B (atomic slot assignment)
        slot_A = tl.atomic_add(fill_count_ptr + cube_id * NODES + node_A, 1)
        if slot_A < 2:
            tl.store(adj_ptr + cube_id * NODES * 2 + node_A * 2 + slot_A, node_B)
        # B -> A
        slot_B = tl.atomic_add(fill_count_ptr + cube_id * NODES + node_B, 1)
        if slot_B < 2:
            tl.store(adj_ptr + cube_id * NODES * 2 + node_B * 2 + slot_B, node_A)


def build_adjacency_triton_fast_only(
    edge_weights: torch.Tensor,       # (N, 18) int64
    eA_tab: torch.Tensor,             # (12, 3) int64
    eB_tab: torch.Tensor,             # (12, 3) int64
    eC_tab: torch.Tensor,             # (12, 3) int64
    a_at_v0_tab: torch.Tensor,        # (12, 3) bool/int
    b_at_v0_tab: torch.Tensor,
    NODES: int,
    W: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast-path-only Triton adjacency builder. Returns (adj, fill_count)."""
    assert _TRITON_AVAILABLE, "Triton not available"
    device = edge_weights.device
    N = int(edge_weights.shape[0])
    adj = torch.full((N, NODES, 2), -1, dtype=torch.int32, device=device)
    fill_count = torch.zeros((N, NODES), dtype=torch.int32, device=device)
    grid = (N * 12 * 3 * W,)
    _build_adj_fast_kernel[grid](
        edge_weights.contiguous(),
        eA_tab.contiguous(), eB_tab.contiguous(), eC_tab.contiguous(),
        a_at_v0_tab.to(torch.int8).contiguous(),
        b_at_v0_tab.to(torch.int8).contiguous(),
        adj, fill_count,
        N=N, NODES=NODES, W=W,
    )
    return adj, fill_count


# ----------------------------------------------------------------------------
# W_HG (Tasks 17+18): batched brute-force Hungarian assignment.
#
# Production data (see tmp/followup_design/w_hg_tie_scan.md):
#   - 99.9953% of Phase 3 cubes are 1x1 (cost = 0; trivial col=0)
#   - rect-reverse (npts < nl) is ~0.005%, deferred to scipy fallback
#   - 0 ties observed across 100k sampled cubes
#
# We implement in PyTorch (NOT Triton) because kernel launch overhead would
# exceed savings on these micro-shapes.
# ----------------------------------------------------------------------------

# Cache permutation tables on host once; ship to device per-batch as needed.
_PERM_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def _get_perms(npts: int, nl: int, device: torch.device) -> torch.Tensor:
    """Return (K, nl) int64 tensor of all P(npts, nl) permutations on `device`."""
    from itertools import permutations
    key = (npts, nl)
    base = _PERM_CACHE.get(key)
    if base is None:
        perms = list(permutations(range(npts), nl))
        base = torch.tensor(perms, dtype=torch.int64)
        _PERM_CACHE[key] = base
    if base.device != device:
        return base.to(device)
    return base


def hungarian_batched(
    cost_padded: torch.Tensor,    # (B, max_nl, max_np) float32, +inf for padded slots
    n_loops: torch.Tensor,         # (B,) int64
    n_points: torch.Tensor,        # (B,) int64
    max_nl: int = 5,
    max_np: int = 8,
) -> torch.Tensor:
    """Batched Hungarian assignment for s7 Phase 3.

    Returns (B, max_nl) int64 of column matches.
    Output[b, r] = matched column for row r in cube b, or -1 if r >= n_loops[b]
    OR the cube needs scipy fallback (callers handle by checking output==-1).

    Strategy:
      - 1x1 cubes (>=99.99%): output[b, 0] = 0 trivially
      - 1xK (K>1): argmin over valid columns
      - Other (nl, npts) buckets where nl <= npts <= max_np: brute-force
        permutation enumeration, take min-cost (unique-min only).
      - nl > npts (rect-reverse) or nl > max_nl or npts > max_np: leave -1
        (caller fallback).
      - If a tie between permutations is detected: leave -1 (caller fallback).
    """
    device = cost_padded.device
    B = cost_padded.shape[0]
    output = torch.full((B, max_nl), -1, dtype=torch.int64, device=device)

    if B == 0:
        return output

    # Hot path: 1x1 — trivially col 0
    is_1x1 = (n_loops == 1) & (n_points == 1)
    output[is_1x1, 0] = 0

    # 1xK with K>=2 — vectorized argmin over K valid cols
    is_1xk = (n_loops == 1) & (n_points >= 2) & (n_points <= max_np)
    if is_1xk.any():
        idx_1xk = is_1xk.nonzero(as_tuple=True)[0]
        sub = cost_padded[idx_1xk, 0, :max_np]   # (M, max_np)
        # Padded columns are +inf, so global argmin is over valid columns.
        cols = sub.argmin(dim=-1)
        mins = sub.gather(1, cols.unsqueeze(1))
        tie_count = (sub == mins).sum(dim=-1)
        is_unique = tie_count == 1
        for_update = idx_1xk[is_unique]
        output[for_update, 0] = cols[is_unique]
        # tied ones remain -1 → fallback

    # Brute-force buckets: 2 <= nl <= max_nl, nl <= npts <= max_np
    for nl in range(2, max_nl + 1):
        for npts in range(nl, max_np + 1):
            mask = (n_loops == nl) & (n_points == npts)
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=True)[0]
            sub = cost_padded[idx, :nl, :npts]   # (M, nl, npts)
            perms_t = _get_perms(npts, nl, device)   # (K, nl)
            K = perms_t.shape[0]
            M = sub.shape[0]
            # gathered[m, k, r] = sub[m, r, perms_t[k, r]]
            perms_exp = perms_t.view(1, K, nl).expand(M, K, nl)
            sub_exp = sub.unsqueeze(1).expand(M, K, nl, npts)
            gathered = sub_exp.gather(3, perms_exp.unsqueeze(3)).squeeze(3)
            cost_per_perm = gathered.sum(dim=-1)   # (M, K)
            best_cost, best_k = cost_per_perm.min(dim=-1)
            tie_count = (cost_per_perm == best_cost.unsqueeze(1)).sum(dim=-1)
            is_unique = tie_count == 1
            best_perms = perms_t[best_k]   # (M, nl)
            valid_rows = idx[is_unique]
            valid_perms = best_perms[is_unique]
            output[valid_rows, :nl] = valid_perms

    return output
