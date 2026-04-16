"""
EXP-2: Factored Ablation (C0-C4).

Constructs 5 mesh configurations by mixing QEF and decoder components,
computes metrics for each to decompose contributions.
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

CONFIGS = {
    "C0": {"verts": "qef", "intersected": "qef", "split_weight": "none"},
    "C1": {"verts": "dec", "intersected": "qef", "split_weight": "none"},
    "C2": {"verts": "qef", "intersected": "qef", "split_weight": "dec"},
    "C3": {"verts": "dec", "intersected": "qef", "split_weight": "dec"},
    "C4": {"verts": "dec", "intersected": "dec", "split_weight": "dec"},
}


def run_exp2(model_id, resolution):
    """Run EXP-2 factored ablation for a single model × resolution."""
    from scripts.eval.baseline_experiments import load_gt_mesh
    from scripts.eval.baseline_exp5_metrics import compute_all_metrics

    cache_dir = os.path.join(OUTPUT_ROOT, "cache", f"{model_id}_{resolution}")
    qef = torch.load(os.path.join(cache_dir, "qef.pt"), weights_only=True)
    dec = torch.load(os.path.join(cache_dir, "decoder.pt"), weights_only=True)
    gt_mesh = load_gt_mesh(model_id, resolution)

    coords = qef['coords'].cuda()
    qef_verts = qef['dual_vertices'].cuda()
    qef_intersected = qef['intersected'].cuda()
    dec_verts = dec['dec_verts'].cuda()
    dec_intersected = dec['dec_intersected'].bool().cuda()
    dec_sw = dec['dec_split_weight'].cuda()

    results = {}
    for config_name, cfg in CONFIGS.items():
        v = dec_verts if cfg["verts"] == "dec" else qef_verts
        i = dec_intersected if cfg["intersected"] == "dec" else qef_intersected
        sw = dec_sw if cfg["split_weight"] == "dec" else None

        out_verts, out_faces = flexible_dual_grid_to_mesh(
            coords, v, i, split_weight=sw, grid_size=resolution, aabb=AABB,
        )

        recon = trimesh.Trimesh(
            vertices=out_verts.detach().cpu().numpy(),
            faces=out_faces.detach().cpu().numpy(),
            process=False,
        )
        metrics = compute_all_metrics(gt_mesh, recon)
        results[config_name] = metrics

        row = {"model_id": model_id, "resolution": resolution, "config": config_name,
               **metrics}
        csv_path = os.path.join(OUTPUT_ROOT, "EXP2_factored_ablation", "ablation_all_metrics.csv")
        _append_csv(csv_path, row)

    # Print summary
    c0_nc = results["C0"]["nc"]
    c3_nc = results["C3"]["nc"]
    c4_nc = results["C4"]["nc"]
    print(f"  [EXP-2] {model_id}@{resolution}: "
          f"C0={c0_nc:.4f}, C1={results['C1']['nc']:.4f}, "
          f"C2={results['C2']['nc']:.4f}, C3={c3_nc:.4f}, C4={c4_nc:.4f}, "
          f"C3-C4={c3_nc - c4_nc:+.4f}")
    return results


def _append_csv(csv_path, row_dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)
