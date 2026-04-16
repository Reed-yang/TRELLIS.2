"""
Baseline runner: drive the custom/ CoReP pipeline on a set of meshes,
collect per-stage timings, and save results as JSON.

Usage:
    .venv/bin/python -m corep_fast.profiling.baseline_runner \
        --impl custom \
        --meshes results/baseline_experiments/data/*.ply \
        --resolution 512 \
        --output profiling/runs/baseline_custom_v0.json

Reference: spec §7.4.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import trimesh


def _run_custom_pipeline(mesh_path: str, resolution: int) -> dict:
    """
    Run custom/ pipeline on a single mesh and return timing + status dict.
    """
    from corep_fast.profiling.harness import ProfilingCollector, stage_timer

    # Add custom/ to sys.path if needed
    project_root = str(Path(__file__).resolve().parents[2])
    custom_dir = os.path.join(project_root, 'custom')
    if custom_dir not in sys.path:
        sys.path.insert(0, custom_dir)

    # Dynamic imports
    from voxelize import voxelize
    from feature_volume import feature_volume
    from feature_edge import feature_edge
    from feature_face import feature_face
    from feature_point import feature_point
    from collapse_face import collapse_face_inner, collapse_face_boundary
    from collapse_point import collapse_point_inner, collapse_point_boundary
    from collapse import reconstruct_mesh, mark_exception
    from utils import fetch_np_array

    pc = ProfilingCollector(mesh_name=os.path.basename(mesh_path),
                            resolution=resolution, impl='custom')
    mesh = trimesh.load(mesh_path)

    import tempfile
    output_dir = tempfile.mkdtemp(prefix='corep_baseline_')

    try:
        with stage_timer('s1_voxelize', pc):
            norm_mesh, boundaries, face_regs, bnd_regs, nm_regs = \
                voxelize(mesh, output_dir, resolution)

        with stage_timer('s2_feature_volume', pc):
            face_regs, bnd_regs = feature_volume(
                face_regs, bnd_regs, norm_mesh, boundaries, output_dir)

        with stage_timer('s3_feature_edge', pc):
            face_regs = feature_edge(norm_mesh, resolution, face_regs, output_dir, debug=False)

        with stage_timer('s4_feature_face', pc):
            face_regs = feature_face(
                norm_mesh, resolution, face_regs, boundaries, bnd_regs, output_dir, debug=False)

        with stage_timer('s4_feature_point', pc):
            face_regs = feature_point(norm_mesh, resolution, face_regs, output_dir, debug=False)

        # Split inner vs boundary
        inner_mask = fetch_np_array(face_regs, 'num_boundary') == 0
        boundary_mask = ~inner_mask
        inner_regs = [face_regs[i] for i in range(len(face_regs)) if inner_mask[i]]
        boundary_regs_split = [face_regs[i] for i in range(len(face_regs)) if boundary_mask[i]]

        with stage_timer('s6_collapse_face', pc):
            solved_u, ambig_u, unsolv_u = collapse_face_inner(inner_regs)
            solved_b, ambig_b, unsolv_b = collapse_face_boundary(boundary_regs_split)

        with stage_timer('s7_collapse_point', pc):
            point_regs = collapse_point_inner(solved_u, resolution, debug=False,
                                              output_directory=output_dir)
            point_regs_bnd = collapse_point_boundary(solved_b, resolution, debug=False,
                                                     output_directory=output_dir)

        exception_regs = mark_exception([*ambig_u, *unsolv_u, *ambig_b, *unsolv_b])
        all_regs = [*point_regs, *point_regs_bnd, *exception_regs]

        with stage_timer('s8_collapse', pc):
            reconstruct_mesh(resolution, all_regs,
                             output_filepath=os.path.join(output_dir, 'out.ply'))

    finally:
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)

    return {
        'mesh': os.path.basename(mesh_path),
        'resolution': resolution,
        'status': 'ok',
        'num_cubes': len(face_regs),
        'timings': pc,
    }


def run_baseline(
    mesh_paths: list[str],
    resolution: int,
    impl: str,
    output_path: str,
) -> None:
    results = []
    for mesh_path in mesh_paths:
        print(f"[baseline_runner] Processing {os.path.basename(mesh_path)} @ {resolution}...")
        t0 = time.time()
        try:
            if impl == 'custom':
                r = _run_custom_pipeline(mesh_path, resolution)
            else:
                raise NotImplementedError(f"impl={impl!r} not yet supported (Phase 1)")
            r['total_wall_s'] = time.time() - t0
            results.append(r)
            print(f"  → OK in {r['total_wall_s']:.1f}s, {r['num_cubes']} cubes")
        except Exception as e:
            tb = traceback.format_exc()
            results.append({
                'mesh': os.path.basename(mesh_path),
                'resolution': resolution,
                'status': 'error',
                'error': str(e),
                'traceback': tb,
            })
            print(f"  → ERROR: {e}")

    # Serialize
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for r in results:
        entry = {k: v for k, v in r.items() if k != 'timings'}
        if 'timings' in r:
            pc = r['timings']
            entry['stages'] = {
                name: {'wall_time_s': rec.wall_time_s, 'peak_gpu_mem_bytes': rec.peak_gpu_mem_bytes}
                for name, rec in pc._records.items()
            }
            entry['total_pipeline_s'] = pc.total_wall_time_s
        serializable.append(entry)

    out.write_text(json.dumps(serializable, indent=2))
    print(f"\n[baseline_runner] Saved {len(serializable)} results to {out}")


def main():
    parser = argparse.ArgumentParser(description='CoReP baseline runner')
    parser.add_argument('--impl', required=True, choices=['custom', 'corep_fast', 'ab'])
    parser.add_argument('--meshes', nargs='+', required=True, help='Glob patterns or paths')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--output', required=True, help='Output JSON path')
    args = parser.parse_args()

    # Expand globs
    all_paths = []
    for pattern in args.meshes:
        expanded = sorted(glob.glob(pattern))
        if not expanded:
            print(f"Warning: no files matched {pattern!r}")
        all_paths.extend(expanded)

    if not all_paths:
        parser.error("No mesh files found")

    run_baseline(all_paths, args.resolution, args.impl, args.output)


if __name__ == '__main__':
    main()
