"""Sparse label-propagation alternative to the dense (N, M, M) path in
s2_components. Use when max_faces > 32 and COREP_FAST_S2_SPARSE=1.

Design
------
For each cube with M_n valid faces, we build a CSR edge list of size
O(M_n * 3) (face_adj has 3 neighbors per face). Label propagation is
performed via torch.scatter_reduce_(reduce='amin') over these edges;
convergence is bounded by M_n iterations (graph diameter).

This replaces the O(N * M_max^2) memory allocation of the dense path with
O(E_total) where E_total = sum_n M_n * 3 ~= N * mean_M * 3, typically
10-100x smaller on pathological meshes.
"""
from __future__ import annotations

import torch


def _build_csr_edges(padded_faces: torch.Tensor,     # (N, M) int32
                    face_adj: torch.Tensor,          # (F, 3) int32
                    mask: torch.Tensor,              # (N, M) bool
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten per-cube face adjacency into a global CSR edge list.

    Returns:
        edge_src: (E,) int64 — global slot id src (cube * M + slot)
        edge_dst: (E,) int64 — global slot id dst (cube * M + slot), or -1
                               if the referenced face is not in the same cube.
    """
    N, M = padded_faces.shape
    dev = padded_faces.device

    # Each valid slot produces up to 3 candidate edges (one per face_adj entry).
    # Look up neighbors for every slot, then membership-test against the same
    # cube's padded_faces.
    safe_faces = torch.where(mask, padded_faces.long(), torch.zeros_like(padded_faces.long()))
    neighbors = face_adj[safe_faces]                    # (N, M, 3) int32
    neighbors_valid = (neighbors >= 0) & mask.unsqueeze(-1)  # face_adj sentinel = -1

    # Compare neighbors against each cube's face set. For very large M this is
    # where memory would blow up — so we loop over the 3 neighbor edges
    # independently, each producing (N, M, M) temporarily.
    E_rows = []
    E_cols = []
    for e_idx in range(3):
        nbr = neighbors[..., e_idx]                     # (N, M) int32
        nbr_valid = neighbors_valid[..., e_idx]         # (N, M)
        # Membership test: does nbr[n, j] equal padded_faces[n, k] for some k?
        match = (nbr.unsqueeze(-1) == padded_faces.unsqueeze(1))   # (N, M, M)
        match = match & mask.unsqueeze(1) & nbr_valid.unsqueeze(-1)
        # For each match, record edge (n*M + j, n*M + k).
        n_idx, j_idx, k_idx = torch.nonzero(match, as_tuple=True)
        src = (n_idx * M + j_idx).long()
        dst = (n_idx * M + k_idx).long()
        E_rows.append(src); E_cols.append(dst)
    edge_src = torch.cat(E_rows) if E_rows else torch.zeros(0, dtype=torch.int64, device=dev)
    edge_dst = torch.cat(E_cols) if E_cols else torch.zeros(0, dtype=torch.int64, device=dev)
    # P2 (2026-04-21): `_label_propagation_sequential` uses undirected
    # Union-Find; `scatter_reduce_('amin')` only propagates labels in the
    # src <- dst direction, so we must symmetrise the edge list to match
    # undirected merge semantics. Doubles E but E = O(N*M*3), still far
    # cheaper than the dense (N, M, M) path we're replacing.
    edge_src, edge_dst = (
        torch.cat([edge_src, edge_dst]),
        torch.cat([edge_dst, edge_src]),
    )
    return edge_src, edge_dst


def _sparse_num_components(padded_faces: torch.Tensor,
                           face_adj: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
    """Compute num_components per cube via scatter-min label propagation on
    the sparse edge list.

    Returns:
        num_components: (N,) int32
    """
    N, M = padded_faces.shape
    dev = padded_faces.device

    labels = torch.arange(N * M, dtype=torch.int64, device=dev).view(N, M)
    labels = torch.where(mask, labels, torch.full_like(labels, -1))

    edge_src, edge_dst = _build_csr_edges(padded_faces, face_adj, mask)

    if edge_src.numel() == 0:
        per_cube_counts = torch.zeros(N, dtype=torch.int32, device=dev)
        if mask.any():
            # No edges: every valid slot is its own component.
            per_cube_counts = mask.sum(dim=1).to(torch.int32)
        return per_cube_counts

    flat_labels = labels.view(-1).clone()
    # Converge: at most M iterations.
    max_iters = M
    for _ in range(max_iters):
        nbr_lbl = flat_labels.index_select(0, edge_dst)
        # Filter: sentinel-labeled slots (-1) must not propagate.
        nbr_lbl = torch.where(nbr_lbl < 0,
                              torch.full_like(nbr_lbl, flat_labels.numel()),
                              nbr_lbl)
        new = flat_labels.clone()
        new.scatter_reduce_(0, edge_src, nbr_lbl, reduce='amin', include_self=True)
        if torch.equal(new, flat_labels):
            break
        flat_labels = new

    labels = flat_labels.view(N, M)
    # Count distinct valid labels per cube.
    out = torch.zeros(N, dtype=torch.int32, device=dev)
    for n in range(N):
        lbl_n = labels[n][mask[n]]
        if lbl_n.numel() > 0:
            out[n] = lbl_n.unique().numel()
    return out
