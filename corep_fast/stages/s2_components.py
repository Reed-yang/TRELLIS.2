"""
Stage 2: Connected Components per cube via GPU label propagation.

For each occupied cube, counts the number of connected components among
its registered mesh faces (using the mesh face adjacency graph) and the
number of boundary edge components.

Public API:
    s2_components(batch, mesh) -> CubeBatch
"""
from __future__ import annotations

import torch

from corep_fast.containers import MeshTensors, CubeBatch, _replace_fields


def s2_components(batch: CubeBatch, mesh: MeshTensors) -> CubeBatch:
    """Compute num_components and num_boundary per cube via GPU Union-Find.

    For each cube, the registered faces form a subgraph of the mesh face
    adjacency.  Connected components are found via iterative label
    propagation on padded per-cube face arrays.

    num_boundary is set to 0 for all cubes (boundary edge registration
    is not yet implemented in s1_voxelize).

    Args:
        batch: CubeBatch after s1_voxelize (tri_offsets/tri_values populated).
        mesh:  MeshTensors with face_adj (F, 3).

    Returns:
        Updated CubeBatch with num_components and num_boundary filled in.
    """
    N = batch.num_cubes
    device = batch.device

    if N == 0:
        return batch

    # --- Step 1: Compute per-cube face counts and max faces ---
    tri_offsets = batch.tri_offsets  # (N+1,) int64
    tri_values = batch.tri_values   # (T,) int32
    face_adj = mesh.face_adj        # (F, 3) int32

    counts = (tri_offsets[1:] - tri_offsets[:-1]).to(torch.int64)  # (N,)
    max_faces = int(counts.max().item())

    if max_faces == 0:
        # No faces registered -- should not happen for occupied cubes
        return batch

    # --- Step 2: Pad registered faces into (N, max_faces) ---
    # Sentinel value -1 for padding
    padded_faces = torch.full(
        (N, max_faces), -1, dtype=torch.int32, device=device
    )
    # Build column indices for scatter
    col_idx = torch.arange(max_faces, device=device).unsqueeze(0).expand(N, -1)  # (N, max_faces)
    mask = col_idx < counts.unsqueeze(1)  # (N, max_faces) bool

    # Flat indices into tri_values for valid entries
    # For cube i, valid faces are tri_values[tri_offsets[i] : tri_offsets[i] + counts[i]]
    # We need a flat index: tri_offsets[i] + local_col
    offsets_expanded = tri_offsets[:-1].unsqueeze(1).expand(N, max_faces)  # (N, max_faces)
    flat_src_idx = (offsets_expanded + col_idx).long()  # (N, max_faces)
    flat_src_idx = flat_src_idx.clamp(max=tri_values.shape[0] - 1) if tri_values.shape[0] > 0 else flat_src_idx

    # Gather face IDs
    padded_faces[mask] = tri_values[flat_src_idx[mask]]

    # --- Step 3: Build per-cube local adjacency via face_adj lookup ---
    # For each face in each cube, check if any of its 3 mesh neighbors
    # are also in the same cube. We use label propagation.

    # Initialize labels: each face gets its column index as label
    # (so label[i,j] = j initially, sentinel positions get max_faces)
    labels = torch.arange(max_faces, device=device).unsqueeze(0).expand(N, -1).clone()  # (N, max_faces)
    labels[~mask] = max_faces  # sentinel label for padding

    # Precompute: for each (cube, slot), which other slots in the same cube
    # are mesh-adjacent?
    # face_adj[f, :] gives 3 neighbor face IDs for face f.
    # We need to check if any of those neighbors appear in padded_faces[cube, :].

    # For valid entries, look up their face_adj neighbors
    valid_face_ids = padded_faces.clone().long()
    valid_face_ids[~mask] = 0  # dummy index for gather (will be masked out)

    # neighbors_of[i, j, k] = face_adj[padded_faces[i,j], k] for k in {0,1,2}
    # Shape: (N, max_faces, 3)
    neighbors_of = face_adj[valid_face_ids]  # (N, max_faces, 3) int32

    # For each neighbor, find if it matches any slot in the same cube.
    # This is the key connectivity check.
    # We'll do iterative label propagation: if face j and face k in the
    # same cube share a mesh edge, propagate the minimum label.

    # Build adjacency: adj_matrix[i, j, k] = True if faces at slots j and k
    # in cube i are mesh-adjacent.
    # Compare neighbors_of[i, j, :] against padded_faces[i, :] for all pairs.

    # Efficient approach: for each (cube i, slot j, neighbor_edge e),
    # find the slot k in cube i where padded_faces[i, k] == neighbors_of[i, j, e]
    # This is a batched set-membership / index lookup.

    # Expand for comparison:
    # neighbors_of: (N, max_faces, 3) -> (N, max_faces, 3, 1)
    # padded_faces: (N, max_faces) -> (N, 1, 1, max_faces)
    # Match: (N, max_faces, 3, max_faces) -- could be large but max_faces is small

    # QW5 (2026-04-21): DISABLED — see logs/findings_qw5_regression.md.
    # Original intent: lower dense threshold 64 -> 32 under VRAM_RESCUE to
    # cap the (N, M, M) match tensor at M^2=1024 and avoid a 62 GB p99 VRAM
    # spike at res=512. Outcome on real meshes: cubes with max_faces in the
    # [33, 64] band got redirected to `_label_propagation_sequential`, a
    # Python `for cube_idx in range(N)` loop that performs ~3*n_valid .item()
    # D2H syncs per cube. Measured 100x-400x s2 slowdown on affected meshes
    # in the 2026-04-21 10-min A/B (s2 p99 2.7s -> 162s). The 62 GB spike
    # the rescue was supposed to mitigate actually comes from max_faces > 64
    # cubes that already use sequential regardless of threshold, so QW5
    # provided no VRAM benefit on those and only hurt the [33, 64] band.
    # Pinning threshold at 64 (legacy) until `_label_propagation_sequential`
    # is vectorised; VRAM_RESCUE remains import'd for future re-enablement
    # and to avoid a commit-log churn of removing then restoring.
    from corep_fast.config import VRAM_RESCUE as _VRAM_RESCUE  # noqa: F401
    from corep_fast.config import S2_SPARSE as _S2_SPARSE
    _dense_threshold = 64
    _overrule_num_components = None
    if max_faces <= _dense_threshold:
        # OOM-resilient chunked dense label-propagation (2026-04-21 s2 OOM
        # fallback track). The dense path used to build `(N, M, M)` tensors
        # directly — adj_labels clone = 8*N*M^2 bytes — which OOMs on 80 GiB
        # HBM for res=512 meshes with N in the millions. The helper below
        # chunks N into batches sized to 75% of free VRAM, with reactive
        # OOM retry and a CPU fallback. Tier 0 (full-N) runs when the whole
        # thing fits, preserving bit-exact output.
        # See corep_fast/stages/s2_dense_chunked.py.
        from corep_fast.stages.s2_dense_chunked import _s2_dense_label_prop_chunked
        labels = _s2_dense_label_prop_chunked(
            padded_faces, neighbors_of, mask, N, max_faces, device)
    elif _S2_SPARSE:
        # P2 (2026-04-21): sparse scatter_min path -- avoids the (N, M, M)
        # match-tensor spike that produced the 62 GB p99 VRAM on pathological
        # meshes at res=512. Labels for the comp_face CSR step still come
        # from _label_propagation_sequential; only the num_components count
        # uses the sparse algorithm.
        from corep_fast.stages.s2_components_sparse import _sparse_num_components
        _overrule_num_components = _sparse_num_components(
            padded_faces, face_adj, mask)
        labels = _label_propagation_sequential(
            padded_faces, neighbors_of, mask, N, max_faces, device
        )
    else:
        # Fallback for very large max_faces: per-cube sequential processing
        labels = _label_propagation_sequential(
            padded_faces, neighbors_of, mask, N, max_faces, device
        )

    # --- Step 5: Count unique labels per cube ---
    # For each cube, count distinct labels among valid slots
    # Use a trick: sort labels per cube, count transitions
    labels_for_count = labels.clone()
    labels_for_count[~mask] = max_faces  # ensure padding is ignored

    sorted_labels, _ = labels_for_count.sort(dim=1)  # (N, max_faces)
    # Count transitions: where sorted_labels[i, j] != sorted_labels[i, j-1]
    # First valid slot always counts as 1 component
    transitions = torch.zeros(N, max_faces, dtype=torch.bool, device=device)
    transitions[:, 0] = mask[:, 0]  # first slot counts if valid
    if max_faces > 1:
        diff = sorted_labels[:, 1:] != sorted_labels[:, :-1]  # (N, max_faces-1)
        valid_transition = diff & mask[:, 1:]  # only count valid slots
        # Also exclude transitions from padding to padding
        not_padding = sorted_labels[:, 1:] < max_faces
        transitions[:, 1:] = valid_transition & not_padding

    num_components = transitions.sum(dim=1).to(torch.int32)  # (N,)
    if _S2_SPARSE and max_faces > _dense_threshold and _overrule_num_components is not None:
        num_components = _overrule_num_components

    # --- Step 6: Build comp_face CSR (face ids grouped by component per cube) ---
    # Sort each cube's face slots by component label so faces of the same
    # component are contiguous.  Padding slots (label == max_faces) sort last.
    sort_keys = labels_for_count  # (N, max_faces) — padding has max_faces
    _, sort_order = sort_keys.sort(dim=1, stable=True)  # (N, max_faces)

    # Gather face ids in component-grouped order
    sorted_faces = padded_faces.gather(1, sort_order)  # (N, max_faces) int32
    sorted_mask = mask.gather(1, sort_order)            # (N, max_faces) bool

    # Per-cube valid face counts (same as `counts` computed earlier)
    per_cube_counts = counts  # (N,) int64

    # Build CSR offsets from per-cube counts
    comp_face_off = torch.zeros(N + 1, dtype=torch.int64, device=device)
    torch.cumsum(per_cube_counts, dim=0, out=comp_face_off[1:])

    # Flatten valid entries into comp_face_val
    comp_face_val = sorted_faces[sorted_mask].to(torch.int32)  # (total_faces,)

    # --- Step 7: num_boundary = 0 for now ---
    num_boundary = torch.zeros(N, dtype=torch.int32, device=device)

    return _replace_fields(
        batch,
        num_components=num_components,
        num_boundary=num_boundary,
        comp_face_off=comp_face_off,
        comp_face_val=comp_face_val,
    )


def _label_propagation_sequential(
    padded_faces: torch.Tensor,  # (N, M) int32
    neighbors_of: torch.Tensor,  # (N, M, 3) int32
    mask: torch.Tensor,          # (N, M) bool
    N: int,
    max_faces: int,
    device: torch.device,
) -> torch.Tensor:
    """Fallback: per-cube label propagation for large max_faces."""
    labels = torch.arange(max_faces, device=device).unsqueeze(0).expand(N, -1).clone()
    labels[~mask] = max_faces

    for cube_idx in range(N):
        cube_mask = mask[cube_idx]  # (M,)
        if not cube_mask.any():
            continue
        n_valid = int(cube_mask.sum().item())
        face_ids = padded_faces[cube_idx, cube_mask]  # (n_valid,)
        cube_labels = torch.arange(n_valid, device=device)

        # Build local adjacency
        face_set = set(face_ids.tolist())
        nbrs = neighbors_of[cube_idx, cube_mask]  # (n_valid, 3)

        # Union-Find
        parent = list(range(n_valid))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        face_to_slot = {int(f): s for s, f in enumerate(face_ids.tolist())}
        for slot_j in range(n_valid):
            for e in range(3):
                nbr_face = int(nbrs[slot_j, e].item())
                if nbr_face in face_to_slot:
                    union(slot_j, face_to_slot[nbr_face])

        # Write labels back
        valid_indices = torch.where(cube_mask)[0]
        for s in range(n_valid):
            labels[cube_idx, valid_indices[s]] = find(s)

    return labels
