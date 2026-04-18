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
