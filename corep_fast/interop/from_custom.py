"""
Convert custom/'s list-of-dict face_registers into a corep_fast CubeBatch.

Reference: spec S14.D.
"""
from __future__ import annotations

from typing import Optional

import torch

from corep_fast.containers import CubeBatch, MeshTensors


def cube_batch_from_custom(
    face_registers: list[dict],
    mesh: MeshTensors,
    *,
    include: Optional[set[str]] = None,
    device: str | torch.device = 'cpu',
) -> CubeBatch:
    device = torch.device(device)
    N = len(face_registers)
    cb = CubeBatch.empty(num_cubes=N, resolution=mesh.resolution, device=device)

    _all = include is None

    # --- cube_indices ---
    if _all or 'cube_indices' in include:
        indices = torch.tensor(
            [r['cube_indices'] for r in face_registers],
            dtype=torch.int32, device=device,
        )
        cb = cb.with_cube_indices(indices)

    # --- tri CSR (face_indices) ---
    if _all or 'face_indices' in include:
        per_cube = []
        for r in face_registers:
            fids = r.get('face_indices', [])
            per_cube.append(torch.tensor(fids, dtype=torch.int32, device=device))
        cb = cb.set_tri_csr(per_cube)

    # --- scalar integer fields ---
    if _all or 'num_components' in include:
        cb.num_components = torch.tensor(
            [r.get('num_components', 0) for r in face_registers],
            dtype=torch.int32, device=device,
        )

    if _all or 'num_boundary' in include:
        cb.num_boundary = torch.tensor(
            [r.get('num_boundary', 0) for r in face_registers],
            dtype=torch.int32, device=device,
        )

    if _all or 'status' in include:
        cb.status = torch.tensor(
            [r.get('status', 0) for r in face_registers],
            dtype=torch.int32, device=device,
        )

    # --- edge_weights ---
    if _all or 'edge_weights' in include:
        ew_list = [r.get('edge_weights', [0] * 18) for r in face_registers]
        cb.edge_weights = torch.tensor(ew_list, dtype=torch.int32, device=device)

    # --- face_weights ---
    if _all or 'face_weights' in include:
        fw_list = [r.get('face_weights', [0] * 12) for r in face_registers]
        cb.face_weights = torch.tensor(fw_list, dtype=torch.int32, device=device)

    # --- component_points (CSR) ---
    if _all or 'component_points' in include:
        offsets = [0]
        all_pts = []
        for r in face_registers:
            pts = r.get('component_points', [])
            offsets.append(offsets[-1] + len(pts))
            all_pts.extend(pts)
        cb.point_offsets = torch.tensor(offsets, dtype=torch.int64, device=device)
        if all_pts:
            cb.point_values = torch.tensor(all_pts, dtype=torch.float32, device=device)
        else:
            cb.point_values = torch.zeros((0, 3), dtype=torch.float32, device=device)

    # --- loops (two-level CSR) ---
    if _all or 'loops' in include:
        loop_cube_off = [0]
        loop_edge_off = [0]
        loop_edge_val = []
        loop_edge_rank = []
        loop_point_match = []

        for r in face_registers:
            loops = r.get('sorted_loops', r.get('loops', []))
            loop_cube_off.append(loop_cube_off[-1] + len(loops))
            for loop_data in loops:
                if isinstance(loop_data, dict):
                    edges = loop_data.get('loop', [])
                    ranks = loop_data.get('rank', [-1] * len(edges))
                else:
                    edges = list(loop_data) if not isinstance(loop_data, list) else loop_data
                    ranks = [-1] * len(edges)
                loop_edge_off.append(loop_edge_off[-1] + len(edges))
                loop_edge_val.extend(edges)
                loop_edge_rank.extend(ranks)
            match = r.get('loop_point_match', list(range(len(loops))))
            loop_point_match.extend(match)

        cb.loop_cube_off = torch.tensor(loop_cube_off, dtype=torch.int64, device=device)
        cb.loop_edge_off = torch.tensor(loop_edge_off, dtype=torch.int64, device=device)
        cb.loop_edge_val = torch.tensor(loop_edge_val, dtype=torch.int32, device=device)
        cb.loop_edge_rank = torch.tensor(loop_edge_rank, dtype=torch.int32, device=device)
        cb.loop_point_match = torch.tensor(loop_point_match, dtype=torch.int32, device=device)

    return cb
