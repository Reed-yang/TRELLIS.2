"""
Convert a CubeBatch back into custom/'s list-of-dict format.

Used for:
1. Partial A/B runs where custom/ consumes a corep_fast stage output.
2. Dumping CubeBatch snapshots for offline analysis with existing custom/ tooling.

Reference: spec §14.D.
"""
from __future__ import annotations

import torch

from corep_fast.containers import CubeBatch, MeshTensors


def custom_from_cube_batch(
    cb: CubeBatch,
    mesh: MeshTensors,
) -> list[dict]:
    result = []
    for i in range(cb.num_cubes):
        d: dict = {}

        d['cube_indices'] = tuple(cb.cube_indices[i].cpu().tolist())

        lo = int(cb.tri_offsets[i].item())
        hi = int(cb.tri_offsets[i + 1].item())
        d['face_indices'] = cb.tri_values[lo:hi].cpu().tolist()

        d['num_components'] = int(cb.num_components[i].item())
        d['num_boundary'] = int(cb.num_boundary[i].item())
        d['status'] = int(cb.status[i].item())

        d['edge_weights'] = cb.edge_weights[i].cpu().tolist()
        d['face_weights'] = cb.face_weights[i].cpu().tolist()

        p_lo = int(cb.point_offsets[i].item())
        p_hi = int(cb.point_offsets[i + 1].item())
        d['component_points'] = cb.point_values[p_lo:p_hi].cpu().tolist()

        l_lo = int(cb.loop_cube_off[i].item())
        l_hi = int(cb.loop_cube_off[i + 1].item())
        loops = []
        for l_idx in range(l_lo, l_hi):
            e_lo = int(cb.loop_edge_off[l_idx].item())
            e_hi = int(cb.loop_edge_off[l_idx + 1].item())
            edges = cb.loop_edge_val[e_lo:e_hi].cpu().tolist()
            ranks = cb.loop_edge_rank[e_lo:e_hi].cpu().tolist()
            loops.append({'loop': edges, 'rank': ranks})
        if loops:
            d['sorted_loops'] = loops

        result.append(d)
    return result
