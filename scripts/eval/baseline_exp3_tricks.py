"""
EXP-3: Code Trick Quantification.

Ablation 3a: voxel_margin (0.5 -> 0.0)
Ablation 3b: split_weight (learned -> heuristic)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import csv
import torch
import trimesh
from o_voxel.convert import flexible_dual_grid_to_mesh

OUTPUT_ROOT = "results/baseline_experiments"
AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]


def run_exp3(model_id, resolution):
    """Run EXP-3 trick quantification for a single model × resolution."""
    from scripts.eval.baseline_experiments import load_gt_mesh
    from scripts.eval.baseline_exp5_metrics import compute_all_metrics

    cache_dir = os.path.join(OUTPUT_ROOT, "cache", f"{model_id}_{resolution}")
    qef = torch.load(os.path.join(cache_dir, "qef.pt"), weights_only=True)
    dec = torch.load(os.path.join(cache_dir, "decoder.pt"), weights_only=True)
    gt_mesh = load_gt_mesh(model_id, resolution)

    coords = qef['coords'].cuda()
    dec_intersected = dec['dec_intersected'].bool().cuda()
    dec_sw = dec['dec_split_weight'].cuda()
    dec_verts_m05 = dec['dec_verts'].cuda()  # margin=0.5 (standard)

    # Recompute vertices with margin=0.0
    # We need the raw logits. dec_verts = (1 + 2*0.5)*sigmoid(raw) - 0.5 = 2*sigmoid(raw) - 0.5
    # So sigmoid(raw) = (dec_verts + 0.5) / 2
    # For margin=0.0: dec_verts_m00 = sigmoid(raw) = (dec_verts + 0.5) / 2
    dec_verts_m00 = (dec_verts_m05 + 0.5) / 2.0

    # --- Ablation 3a: margin ---
    # Standard (margin=0.5)
    out_v_m05, out_f_m05 = flexible_dual_grid_to_mesh(
        coords, dec_verts_m05, dec_intersected, dec_sw,
        grid_size=resolution, aabb=AABB,
    )
    mesh_m05 = trimesh.Trimesh(vertices=out_v_m05.cpu().numpy(),
                                faces=out_f_m05.cpu().numpy(), process=False)
    metrics_m05 = compute_all_metrics(gt_mesh, mesh_m05)

    # Ablated (margin=0.0)
    out_v_m00, out_f_m00 = flexible_dual_grid_to_mesh(
        coords, dec_verts_m00, dec_intersected, dec_sw,
        grid_size=resolution, aabb=AABB,
    )
    mesh_m00 = trimesh.Trimesh(vertices=out_v_m00.cpu().numpy(),
                                faces=out_f_m00.cpu().numpy(), process=False)
    metrics_m00 = compute_all_metrics(gt_mesh, mesh_m00)

    # Vertex outside [0,1] analysis
    outside_mask = (dec_verts_m05 < 0).any(-1) | (dec_verts_m05 > 1).any(-1)
    pct_outside = outside_mask.float().mean().item()

    margin_row = {
        "model_id": model_id, "resolution": resolution,
        "NC_m05": metrics_m05["nc"], "NC_m00": metrics_m00["nc"],
        "delta_NC": metrics_m05["nc"] - metrics_m00["nc"],
        "CD_m05": metrics_m05["cd"], "CD_m00": metrics_m00["cd"],
        "delta_CD": metrics_m05["cd"] - metrics_m00["cd"],
        "pct_outside_01": pct_outside,
    }
    _append_csv(os.path.join(OUTPUT_ROOT, "EXP3_trick_quantification", "margin_ablation.csv"),
                margin_row)

    # --- Ablation 3b: split_weight ---
    # Standard (learned split_weight) — reuse mesh_m05 metrics
    # Ablated (heuristic, split_weight=None)
    out_v_heur, out_f_heur = flexible_dual_grid_to_mesh(
        coords, dec_verts_m05, dec_intersected, split_weight=None,
        grid_size=resolution, aabb=AABB,
    )
    mesh_heur = trimesh.Trimesh(vertices=out_v_heur.cpu().numpy(),
                                 faces=out_f_heur.cpu().numpy(), process=False)
    metrics_heur = compute_all_metrics(gt_mesh, mesh_heur)

    sw_row = {
        "model_id": model_id, "resolution": resolution,
        "NC_learned": metrics_m05["nc"], "NC_heuristic": metrics_heur["nc"],
        "delta_NC": metrics_m05["nc"] - metrics_heur["nc"],
        "CD_learned": metrics_m05["cd"], "CD_heuristic": metrics_heur["cd"],
        "delta_CD": metrics_m05["cd"] - metrics_heur["cd"],
    }
    _append_csv(os.path.join(OUTPUT_ROOT, "EXP3_trick_quantification", "split_weight_ablation.csv"),
                sw_row)

    # Vertex distribution stats
    v = dec_verts_m05.cpu()
    dist_row = {
        "model_id": model_id, "resolution": resolution,
        "mean_v": v.mean().item(), "std_v": v.std().item(),
        "min_v": v.min().item(), "max_v": v.max().item(),
        "pct_lt0": (v < 0).float().mean().item(),
        "pct_gt1": (v > 1).float().mean().item(),
        "pct_outside_01": pct_outside,
    }
    _append_csv(os.path.join(OUTPUT_ROOT, "EXP3_trick_quantification", "vertex_distribution.csv"),
                dist_row)

    print(f"  [EXP-3] {model_id}@{resolution}: "
          f"margin \u0394NC={margin_row['delta_NC']:+.4f}, "
          f"split_weight \u0394NC={sw_row['delta_NC']:+.4f}, "
          f"outside={pct_outside:.3%}")
    return margin_row, sw_row


def _append_csv(csv_path, row_dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)
