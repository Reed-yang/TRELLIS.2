"""
EXP-4: O-Voxel Structural Deficiency Diagnosis using CoReP.

Phase 4a: Run CoReP Stages 1-3 to extract per-cube topology labels.
Phase 4b: Cross-analysis with EXP-1/2 data (if available).

Elastic: gracefully handles CoReP failures per model.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'custom'))

import csv
import numpy as np
import trimesh
import tempfile

OUTPUT_ROOT = "results/baseline_experiments"


def run_exp4(model_id, resolution):
    """Run EXP-4 CoReP diagnosis for a single model × resolution."""
    from scripts.eval.baseline_experiments import load_gt_mesh
    from voxelize import voxelize
    from feature_volume import volume_feature_face, volume_feature_boundary
    from feature_edge import calculate_edge_crossings

    gt_mesh = load_gt_mesh(model_id, resolution)

    # Use a temporary directory for CoReP intermediate files
    work_dir = os.path.join(OUTPUT_ROOT, "EXP4_structural_diagnosis", f"work_{model_id}_{resolution}")
    os.makedirs(work_dir, exist_ok=True)

    # Stage 1: Voxelize (CoReP normalizes internally to [0,1]^3)
    norm_mesh, boundaries, face_registers, boundary_registers, nm_registers = voxelize(
        gt_mesh, work_dir, resolution
    )
    total_cubes = len(face_registers)

    # Stage 2: Volume features — per-cube connected component counts
    face_result = volume_feature_face(face_registers, norm_mesh, work_dir)
    boundary_result = volume_feature_boundary(boundary_registers, boundaries, work_dir)

    # Stage 3: Edge features — 18 edge crossing weights per cube
    edge_result = calculate_edge_crossings(norm_mesh, resolution, face_result, work_dir)

    # Derive statistics
    multi_surface_count = sum(1 for r in face_result if r.get('num_components', 1) > 1)
    boundary_cube_count = len(boundary_registers)
    multi_crossing_count = sum(
        1 for r in edge_result
        if any(w > 1 for w in r.get('edge_weights', []))
    )

    # Component distribution
    comp_counts = [r.get('num_components', 1) for r in face_result]
    max_components = max(comp_counts) if comp_counts else 0
    avg_components = float(np.mean(comp_counts)) if comp_counts else 0

    result = {
        "model_id": model_id,
        "resolution": resolution,
        "total_cubes": total_cubes,
        "multi_surface_count": multi_surface_count,
        "multi_surface_pct": multi_surface_count / total_cubes if total_cubes > 0 else 0,
        "boundary_cube_count": boundary_cube_count,
        "boundary_cube_pct": boundary_cube_count / total_cubes if total_cubes > 0 else 0,
        "multi_crossing_count": multi_crossing_count,
        "multi_crossing_pct": multi_crossing_count / total_cubes if total_cubes > 0 else 0,
        "max_components": max_components,
        "avg_components": avg_components,
    }

    csv_path = os.path.join(OUTPUT_ROOT, "EXP4_structural_diagnosis", "deficiency_rates.csv")
    _append_csv(csv_path, result)

    print(f"  [EXP-4] {model_id}@{resolution}: "
          f"cubes={total_cubes}, multi_surface={multi_surface_count}({result['multi_surface_pct']:.1%}), "
          f"boundary={boundary_cube_count}({result['boundary_cube_pct']:.1%}), "
          f"multi_crossing={multi_crossing_count}({result['multi_crossing_pct']:.1%}), "
          f"max_comp={max_components}, avg_comp={avg_components:.2f}")
    return result


def _append_csv(csv_path, row_dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)
