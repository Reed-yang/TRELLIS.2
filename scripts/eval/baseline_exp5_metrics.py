"""
EXP-5: Full Metric Baseline (Layer R and Layer V).

Layer R: mesh -> O-Voxel QEF -> mesh (no VAE)
Layer V: mesh -> O-Voxel -> SC-VAE encode -> decode -> mesh (from cache)

Computes geometric + topological + rendering metrics.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import csv
import torch
import numpy as np
import trimesh
import o_voxel
from o_voxel.convert import flexible_dual_grid_to_mesh

OUTPUT_ROOT = "results/baseline_experiments"
AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
NUM_SAMPLE_POINTS = 100_000
F_SCORE_THRESHOLDS = [0.005, 0.001]


def compute_all_metrics(gt_mesh, recon_mesh):
    """Compute geometric + topological metrics."""
    from scripts.eval.eval_metrics import (
        sample_points_and_normals, chamfer_distance,
        f_score_multi, normal_consistency,
    )
    from scripts.eval.ovoxel_repr_test import _compute_topo_metrics

    gt_pts, gt_nrm = sample_points_and_normals(gt_mesh, NUM_SAMPLE_POINTS)
    recon_pts, recon_nrm = sample_points_and_normals(recon_mesh, NUM_SAMPLE_POINTS)
    gt_pts, gt_nrm = gt_pts.cuda(), gt_nrm.cuda()
    recon_pts, recon_nrm = recon_pts.cuda(), recon_nrm.cuda()

    cd = chamfer_distance(recon_pts, gt_pts)
    nc = normal_consistency(recon_pts, recon_nrm, gt_pts, gt_nrm)
    fscores = f_score_multi(recon_pts, gt_pts, F_SCORE_THRESHOLDS)

    gt_topo = _compute_topo_metrics(gt_mesh)
    recon_topo = _compute_topo_metrics(recon_mesh)

    return {
        "cd": cd, "nc": nc,
        **{f"fscore_{t}": v for t, v in fscores.items()},
        "gt_components": gt_topo["n_components"],
        "recon_components": recon_topo["n_components"],
        "gt_boundary_edges": gt_topo["n_boundary_edges"],
        "recon_boundary_edges": recon_topo["n_boundary_edges"],
        "gt_euler": gt_topo["euler_number"],
        "recon_euler": recon_topo["euler_number"],
        "gt_area": gt_topo["surface_area"],
        "recon_area": recon_topo["surface_area"],
        "area_ratio": recon_topo["surface_area"] / gt_topo["surface_area"]
                      if gt_topo["surface_area"] > 0 else 0,
    }


def run_layer_r(model_id, resolution):
    """Layer R: O-Voxel roundtrip (no VAE)."""
    from scripts.eval.baseline_experiments import load_gt_mesh

    cache_dir = os.path.join(OUTPUT_ROOT, "cache", f"{model_id}_{resolution}")
    os.makedirs(cache_dir, exist_ok=True)
    gt_mesh = load_gt_mesh(model_id, resolution)

    # O-Voxel encode
    vertices = torch.from_numpy(gt_mesh.vertices.copy()).float()
    faces = torch.from_numpy(gt_mesh.faces.copy()).long()
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices, faces=faces,
        grid_size=resolution, aabb=AABB,
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2,
    )

    # Decode with split_weight=None (heuristic)
    out_verts, out_faces = flexible_dual_grid_to_mesh(
        voxel_indices.cuda(), dual_vertices.cuda(), intersected.cuda(),
        split_weight=None, grid_size=resolution, aabb=AABB,
    )

    recon_mesh = trimesh.Trimesh(
        vertices=out_verts.detach().cpu().numpy(),
        faces=out_faces.detach().cpu().numpy(),
        process=False,
    )

    # Persist Layer R mesh for downstream visualization (idempotent).
    layer_r_dir = os.path.join(OUTPUT_ROOT, "EXP6_corep_recon", f"work_{model_id}_{resolution}")
    os.makedirs(layer_r_dir, exist_ok=True)
    layer_r_path = os.path.join(layer_r_dir, "layer_r.ply")
    if not os.path.exists(layer_r_path):
        recon_mesh.export(layer_r_path)

    metrics = compute_all_metrics(gt_mesh, recon_mesh)
    result = {"model_id": model_id, "resolution": resolution, "layer": "R",
              "n_voxels": int(voxel_indices.shape[0]), **metrics}

    # Append to CSV
    csv_path = os.path.join(OUTPUT_ROOT, "EXP5_full_baseline", "geometric_metrics.csv")
    _append_csv(csv_path, result)
    print(f"  [Layer R] {model_id}@{resolution}: CD={metrics['cd']:.6f}, NC={metrics['nc']:.4f}")
    return result


def run_layer_v(model_id, resolution):
    """Layer V: VAE roundtrip (from cache)."""
    from scripts.eval.baseline_experiments import load_gt_mesh

    cache_dir = os.path.join(OUTPUT_ROOT, "cache", f"{model_id}_{resolution}")
    gt_mesh = load_gt_mesh(model_id, resolution)

    qef = torch.load(os.path.join(cache_dir, "qef.pt"), weights_only=True)
    dec = torch.load(os.path.join(cache_dir, "decoder.pt"), weights_only=True)

    # Reconstruct mesh from decoder output (standard VAE inference = C4)
    coords = qef['coords'].cuda()
    out_verts, out_faces = flexible_dual_grid_to_mesh(
        coords, dec['dec_verts'].cuda(), dec['dec_intersected'].bool().cuda(),
        split_weight=dec['dec_split_weight'].cuda(),
        grid_size=resolution, aabb=AABB,
    )

    recon_mesh = trimesh.Trimesh(
        vertices=out_verts.detach().cpu().numpy(),
        faces=out_faces.detach().cpu().numpy(),
        process=False,
    )

    # Persist Layer V mesh for downstream visualization (idempotent).
    layer_v_dir = os.path.join(OUTPUT_ROOT, "EXP6_corep_recon", f"work_{model_id}_{resolution}")
    os.makedirs(layer_v_dir, exist_ok=True)
    layer_v_path = os.path.join(layer_v_dir, "layer_v.ply")
    if not os.path.exists(layer_v_path):
        recon_mesh.export(layer_v_path)

    metrics = compute_all_metrics(gt_mesh, recon_mesh)
    result = {"model_id": model_id, "resolution": resolution, "layer": "V",
              "n_voxels": int(coords.shape[0]), **metrics}

    csv_path = os.path.join(OUTPUT_ROOT, "EXP5_full_baseline", "geometric_metrics.csv")
    _append_csv(csv_path, result)
    print(f"  [Layer V] {model_id}@{resolution}: CD={metrics['cd']:.6f}, NC={metrics['nc']:.4f}")
    return result


def _append_csv(csv_path, row_dict):
    """Append a single row to CSV, creating file with header if needed."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)
