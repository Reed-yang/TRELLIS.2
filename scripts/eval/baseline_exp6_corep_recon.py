"""
EXP-6: CoReP Full Reconstruction — Layer C baseline.

Runs the complete CoReP pipeline (Stages 1-8) to reconstruct mesh,
then computes geometric metrics (NC, CD, F-score) vs GT mesh.

This gives "Layer C" — the representation fidelity upper bound of CoReP,
comparable to Layer R (O-Voxel) and Layer V (O-Voxel + VAE).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'custom'))

import csv
import numpy as np
import trimesh

OUTPUT_ROOT = "results/baseline_experiments"


def run_exp6(model_id, resolution):
    """Run full CoReP reconstruction and compute metrics for a single model × resolution."""
    from scripts.eval.baseline_experiments import load_gt_mesh
    from scripts.eval.baseline_exp5_metrics import compute_all_metrics
    from voxelize import voxelize
    from feature_volume import feature_volume
    from feature_edge import feature_edge
    from feature_face import feature_face
    from feature_point import feature_point
    from collapse_face import collapse_face_inner, collapse_face_boundary
    from collapse_point import collapse_point_inner, collapse_point_boundary
    from collapse import reconstruct_mesh, mark_exception
    from utils import fetch_np_array

    gt_mesh = load_gt_mesh(model_id, resolution)

    work_dir = os.path.join(OUTPUT_ROOT, "EXP6_corep_recon", f"work_{model_id}_{resolution}")
    os.makedirs(work_dir, exist_ok=True)

    print(f"  [EXP-6] {model_id}@{resolution}: Stage 1 — Voxelize")
    norm_mesh, boundaries, face_registers, boundary_registers, nm_registers = voxelize(
        gt_mesh, work_dir, resolution
    )

    print(f"  [EXP-6] {model_id}@{resolution}: Stage 2 — Volume Features")
    face_registers, boundary_registers = feature_volume(
        face_registers, boundary_registers, norm_mesh, boundaries, work_dir
    )

    print(f"  [EXP-6] {model_id}@{resolution}: Stage 3 — Edge Features")
    face_registers = feature_edge(norm_mesh, resolution, face_registers, work_dir)

    print(f"  [EXP-6] {model_id}@{resolution}: Stage 4 — Face + Point Features")
    face_registers = feature_face(norm_mesh, resolution, face_registers, boundaries, boundary_registers, work_dir)
    face_registers = feature_point(norm_mesh, resolution, face_registers, work_dir)

    print(f"  [EXP-6] {model_id}@{resolution}: Stage 5 — Collapse Face (loop extraction)")
    inner_mask = fetch_np_array(face_registers, 'num_boundary') == 0
    boundary_mask = fetch_np_array(face_registers, 'num_boundary') > 0
    inner_registers = [face_registers[i] for i in range(len(face_registers)) if inner_mask[i]]
    boundary_regs = [face_registers[i] for i in range(len(face_registers)) if boundary_mask[i]]

    solved_inner, ambiguous_inner, unsolvable_inner = collapse_face_inner(inner_registers)
    solved_boundary, ambiguous_boundary, unsolvable_boundary = collapse_face_boundary(boundary_regs)

    print(f"  [EXP-6] {model_id}@{resolution}: Stage 6-7 — Collapse Point (loop sorting)")
    collapsed_inner = collapse_point_inner(solved_inner, resolution)
    collapsed_boundary = collapse_point_boundary(solved_boundary, resolution)

    exception_registers = mark_exception([
        *ambiguous_inner, *unsolvable_inner,
        *ambiguous_boundary, *unsolvable_boundary
    ])

    all_registers = [*collapsed_inner, *collapsed_boundary, *exception_registers]

    print(f"  [EXP-6] {model_id}@{resolution}: Stage 8 — Reconstruct Mesh")
    recon_path = os.path.join(work_dir, "corep_recon.ply")
    reconstruct_mesh(resolution, all_registers, output_filepath=recon_path)

    # Load reconstructed mesh and compute metrics
    recon_mesh = trimesh.load(recon_path)
    if isinstance(recon_mesh, trimesh.Scene):
        meshes = [g for g in recon_mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        recon_mesh = trimesh.util.concatenate(meshes) if meshes else trimesh.Trimesh()

    # Align CoReP recon to GT coordinate system via bounding box matching
    # CoReP's normalize_mesh uses asymmetric offsets, so simple -0.5 shift is insufficient
    recon_center = (recon_mesh.bounds[0] + recon_mesh.bounds[1]) / 2
    gt_center = (gt_mesh.bounds[0] + gt_mesh.bounds[1]) / 2
    recon_extent = (recon_mesh.bounds[1] - recon_mesh.bounds[0]).max()
    gt_extent = (gt_mesh.bounds[1] - gt_mesh.bounds[0]).max()
    scale = gt_extent / recon_extent if recon_extent > 0 else 1.0
    recon_mesh.vertices = (recon_mesh.vertices - recon_center) * scale + gt_center

    # Compute metrics (same as EXP-5)
    metrics = compute_all_metrics(gt_mesh, recon_mesh)

    area_ratio = recon_mesh.area / gt_mesh.area if gt_mesh.area > 0 else 0

    result = {
        "model_id": model_id,
        "resolution": resolution,
        "layer": "C",
        "cd": metrics.get("cd", 0),
        "nc": metrics.get("nc", 0),
        "fscore_0.005": metrics.get("fscore_0.005", 0),
        "gt_components": metrics.get("gt_components", 0),
        "recon_components": metrics.get("recon_components", 0),
        "area_ratio": area_ratio,
        "solved_inner": len(solved_inner),
        "solved_boundary": len(solved_boundary),
        "ambiguous": len(ambiguous_inner) + len(ambiguous_boundary),
        "unsolvable": len(unsolvable_inner) + len(unsolvable_boundary),
        "exceptions": len(exception_registers),
    }

    csv_path = os.path.join(OUTPUT_ROOT, "EXP6_corep_recon", "corep_recon_metrics.csv")
    _append_csv(csv_path, result)

    print(f"  [EXP-6] {model_id}@{resolution}: NC={result['nc']:.4f}, CD={result['cd']:.6f}, "
          f"F@0.005={result['fscore_0.005']:.4f}, "
          f"comp={result['gt_components']}→{result['recon_components']}, area={area_ratio:.4f}, "
          f"solved={len(solved_inner)+len(solved_boundary)}, "
          f"exceptions={len(exception_registers)}")
    return result


def _append_csv(csv_path, row_dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)
