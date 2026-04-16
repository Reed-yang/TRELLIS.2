"""
EXP-1: VAE Topology Degradation.

Compares QEF GT intersected flags vs decoder predicted intersected flags.
Computes precision/recall/F1 globally and per-axis.
Also analyzes vertex position deviation (voxel_margin utilization).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import csv
import torch
import numpy as np

OUTPUT_ROOT = "results/baseline_experiments"


def compute_intersected_metrics(gt_intersected, pred_intersected):
    """Compute precision, recall, F1 for binary intersected flags.

    Args:
        gt_intersected: [N, 3] float tensor (0 or 1)
        pred_intersected: [N, 3] float tensor (0 or 1)

    Returns:
        dict with global and per-axis metrics
    """
    results = {}

    # Global (flattened across all 3 axes)
    gt_flat = gt_intersected.flatten().bool()
    pred_flat = pred_intersected.flatten().bool()
    tp = (gt_flat & pred_flat).sum().item()
    fp = (~gt_flat & pred_flat).sum().item()
    fn = (gt_flat & ~pred_flat).sum().item()
    tn = (~gt_flat & ~pred_flat).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn)

    results['precision'] = precision
    results['recall'] = recall
    results['f1'] = f1
    results['accuracy'] = accuracy
    results['flipped_count'] = int((gt_flat != pred_flat).sum().item())

    # Per-axis
    for axis_idx, axis_name in enumerate(['X', 'Y', 'Z']):
        gt_ax = gt_intersected[:, axis_idx].bool()
        pred_ax = pred_intersected[:, axis_idx].bool()
        tp_ax = (gt_ax & pred_ax).sum().item()
        fp_ax = (~gt_ax & pred_ax).sum().item()
        fn_ax = (gt_ax & ~pred_ax).sum().item()
        p = tp_ax / (tp_ax + fp_ax) if (tp_ax + fp_ax) > 0 else 0.0
        r = tp_ax / (tp_ax + fn_ax) if (tp_ax + fn_ax) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        results[f'precision_{axis_name}'] = p
        results[f'recall_{axis_name}'] = r
        results[f'f1_{axis_name}'] = f

    return results


def compute_vertex_deviation(qef_verts, dec_verts):
    """Analyze decoder vertex positions vs QEF ground truth.

    Args:
        qef_verts: [N, 3] float, in [0, 1] local coords
        dec_verts: [N, 3] float, in [-margin, 1+margin] local coords
    """
    diff = dec_verts - qef_verts
    l2 = torch.norm(diff, dim=-1)
    outside_mask = (dec_verts < 0).any(-1) | (dec_verts > 1).any(-1)
    below_zero = (dec_verts < 0).float().mean(dim=0)
    above_one = (dec_verts > 1).float().mean(dim=0)

    return {
        'mean_vertex_l2': l2.mean().item(),
        'median_vertex_l2': l2.median().item(),
        'max_vertex_l2': l2.max().item(),
        'pct_outside_01': outside_mask.float().mean().item(),
        'pct_below0_X': below_zero[0].item(),
        'pct_below0_Y': below_zero[1].item(),
        'pct_below0_Z': below_zero[2].item(),
        'pct_above1_X': above_one[0].item(),
        'pct_above1_Y': above_one[1].item(),
        'pct_above1_Z': above_one[2].item(),
    }


def run_exp1(model_id, resolution):
    """Run EXP-1 for a single model × resolution."""
    cache_dir = os.path.join(OUTPUT_ROOT, "cache", f"{model_id}_{resolution}")
    qef = torch.load(os.path.join(cache_dir, "qef.pt"), weights_only=True)
    dec = torch.load(os.path.join(cache_dir, "decoder.pt"), weights_only=True)

    gt_intersected = qef['intersected']      # [N, 3]
    pred_intersected = dec['dec_intersected']  # [N, 3]

    # Intersected F1
    int_metrics = compute_intersected_metrics(gt_intersected, pred_intersected)

    # QEF verts are in grid coords; convert to local [0,1]
    qef_verts_local = qef['dual_vertices'] * resolution - qef['coords'].float()
    qef_verts_local = torch.clamp(qef_verts_local, 0, 1)
    dec_verts = dec['dec_verts']

    vert_metrics = compute_vertex_deviation(qef_verts_local, dec_verts)

    result = {
        "model_id": model_id, "resolution": resolution,
        "n_voxels": int(gt_intersected.shape[0]),
        **int_metrics, **vert_metrics,
    }

    # Write CSVs
    csv_f1 = os.path.join(OUTPUT_ROOT, "EXP1_topology_degradation", "intersected_f1_results.csv")
    _append_csv(csv_f1, {k: v for k, v in result.items()
                         if not k.startswith('pct_below') and not k.startswith('pct_above')})

    csv_vert = os.path.join(OUTPUT_ROOT, "EXP1_topology_degradation", "vertex_position_analysis.csv")
    _append_csv(csv_vert, {"model_id": model_id, "resolution": resolution, **vert_metrics})

    print(f"  [EXP-1] {model_id}@{resolution}: F1={int_metrics['f1']:.4f}, "
          f"Flipped={int_metrics['flipped_count']}, Outside={vert_metrics['pct_outside_01']:.3%}")
    return result


def _append_csv(csv_path, row_dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)
