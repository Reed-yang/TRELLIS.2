# corep_fast/interop/custom_runner.py
"""Run custom/ pipeline stages individually and return intermediate registers.

Used by A/B regression tests to get ground-truth output from each stage.
"""
from __future__ import annotations

import os
import sys
import tempfile
import shutil
from pathlib import Path

import trimesh


def _ensure_custom_importable() -> None:
    project_root = str(Path(__file__).resolve().parents[2])
    custom_dir = os.path.join(project_root, 'custom')
    if custom_dir not in sys.path:
        sys.path.insert(0, custom_dir)


def run_custom_through_stage(
    mesh_path: str,
    resolution: int,
    up_to: str,
) -> list[dict]:
    """Run custom/ pipeline up to a specified stage, return registers.

    Args:
        mesh_path: Path to input PLY mesh.
        resolution: Voxel grid resolution.
        up_to: Stage to stop after. One of:
            's1' - after voxelization (cube_indices, face_indices)
            's2' - after feature_volume (+ num_components, num_boundary)
            's3' - after feature_edge (+ edge_weights)
            's4' - after feature_face + feature_point (+ face_weights, component_points)
            's6' - after collapse_face (+ loops, status)
            's7' - after collapse_point (+ sorted_loops with ranks + point assignment)

    Returns:
        list[dict] — the face_registers at the requested stage.
    """
    _ensure_custom_importable()

    from voxelize import voxelize
    from feature_volume import feature_volume
    from feature_edge import feature_edge
    from feature_face import feature_face
    from feature_point import feature_point
    from collapse_face import collapse_face_inner, collapse_face_boundary
    from collapse_point import collapse_point_inner, collapse_point_boundary
    from collapse import mark_exception
    from utils import fetch_np_array

    mesh = trimesh.load(mesh_path)
    tmp_dir = tempfile.mkdtemp(prefix='custom_runner_')

    try:
        norm_mesh, boundaries, face_regs, bnd_regs, nm_regs = \
            voxelize(mesh, tmp_dir, resolution)
        if up_to == 's1':
            return face_regs

        face_regs, bnd_regs = feature_volume(
            face_regs, bnd_regs, norm_mesh, boundaries, tmp_dir)
        if up_to == 's2':
            return face_regs

        face_regs = feature_edge(norm_mesh, resolution, face_regs, tmp_dir, debug=False)
        if up_to == 's3':
            return face_regs

        face_regs = feature_face(
            norm_mesh, resolution, face_regs, boundaries, bnd_regs, tmp_dir, debug=False)
        face_regs = feature_point(norm_mesh, resolution, face_regs, tmp_dir, debug=False)
        if up_to == 's4':
            return face_regs

        inner_mask = fetch_np_array(face_regs, 'num_boundary') == 0
        boundary_mask = ~inner_mask
        inner_regs = [face_regs[i] for i in range(len(face_regs)) if inner_mask[i]]
        boundary_regs_split = [face_regs[i] for i in range(len(face_regs)) if boundary_mask[i]]

        solved_u, ambig_u, unsolv_u = collapse_face_inner(inner_regs)
        solved_b, ambig_b, unsolv_b = collapse_face_boundary(boundary_regs_split)
        exception_regs = mark_exception([*ambig_u, *unsolv_u, *ambig_b, *unsolv_b])
        all_regs = [*solved_u, *solved_b, *exception_regs]
        if up_to == 's6':
            return all_regs

        point_regs = collapse_point_inner(solved_u, resolution, debug=False,
                                           output_directory=tmp_dir)
        point_regs_bnd = collapse_point_boundary(solved_b, resolution, debug=False,
                                                   output_directory=tmp_dir)
        all_regs = [*point_regs, *point_regs_bnd, *exception_regs]
        if up_to == 's7':
            return all_regs

        raise ValueError(f"Unknown stage: {up_to!r}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
