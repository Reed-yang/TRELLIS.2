"""
Aggregate all baseline experiment CSVs into a summary.md report.
"""
import os
import csv

OUTPUT_ROOT = "results/baseline_experiments"


def read_csv_rows(csv_path):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def generate_summary():
    """Generate summary.md from all experiment results."""
    lines = ["# Baseline Experiment Results Summary\n"]

    # EXP-1
    rows = read_csv_rows(os.path.join(OUTPUT_ROOT, "EXP1_topology_degradation", "intersected_f1_results.csv"))
    if rows:
        lines.append("## EXP-1: VAE Topology Degradation\n")
        lines.append("| Model | Res | F1 | Precision | Recall | Flipped |\n")
        lines.append("|-------|-----|----|-----------|--------|---------|\n")
        for r in rows:
            lines.append(f"| {r['model_id']} | {r['resolution']} | {float(r['f1']):.4f} | "
                         f"{float(r['precision']):.4f} | {float(r['recall']):.4f} | {r['flipped_count']} |\n")
        lines.append("\n")

    # EXP-2
    rows = read_csv_rows(os.path.join(OUTPUT_ROOT, "EXP2_factored_ablation", "ablation_all_metrics.csv"))
    if rows:
        lines.append("## EXP-2: Factored Ablation\n")
        lines.append("| Model | Res | Config | NC | CD |\n")
        lines.append("|-------|-----|--------|----|----|\n")
        for r in rows:
            lines.append(f"| {r['model_id']} | {r['resolution']} | {r['config']} | "
                         f"{float(r['nc']):.4f} | {float(r['cd']):.6f} |\n")
        lines.append("\n")

    # EXP-3
    margin_rows = read_csv_rows(os.path.join(OUTPUT_ROOT, "EXP3_trick_quantification", "margin_ablation.csv"))
    if margin_rows:
        lines.append("## EXP-3: Trick Quantification\n")
        lines.append("### Margin Ablation\n")
        lines.append("| Model | Res | NC_m05 | NC_m00 | \u0394NC | Pct_outside |\n")
        lines.append("|-------|-----|--------|--------|-----|-------------|\n")
        for r in margin_rows:
            lines.append(f"| {r['model_id']} | {r['resolution']} | {float(r['NC_m05']):.4f} | "
                         f"{float(r['NC_m00']):.4f} | {float(r['delta_NC']):+.4f} | "
                         f"{float(r['pct_outside_01']):.3%} |\n")
        lines.append("\n")

    sw_rows = read_csv_rows(os.path.join(OUTPUT_ROOT, "EXP3_trick_quantification", "split_weight_ablation.csv"))
    if sw_rows:
        lines.append("### Split Weight Ablation\n")
        lines.append("| Model | Res | NC_learned | NC_heuristic | \u0394NC |\n")
        lines.append("|-------|-----|------------|--------------|-----|\n")
        for r in sw_rows:
            lines.append(f"| {r['model_id']} | {r['resolution']} | {float(r['NC_learned']):.4f} | "
                         f"{float(r['NC_heuristic']):.4f} | {float(r['delta_NC']):+.4f} |\n")
        lines.append("\n")

    # EXP-4
    rows = read_csv_rows(os.path.join(OUTPUT_ROOT, "EXP4_structural_diagnosis", "deficiency_rates.csv"))
    if rows:
        lines.append("## EXP-4: O-Voxel Structural Deficiency (CoReP)\n")
        lines.append("| Model | Res | Cubes | Multi_surface(%) | Boundary(%) | Multi_crossing(%) | Max_comp | Avg_comp |\n")
        lines.append("|-------|-----|-------|------------------|-------------|-------------------|----------|----------|\n")
        for r in rows:
            bnd_pct = r.get('boundary_cube_pct', r.get('boundary_pct', '0'))
            lines.append(f"| {r['model_id']} | {r['resolution']} | {r['total_cubes']} | "
                         f"{float(r['multi_surface_pct']):.1%} | {float(bnd_pct):.1%} | "
                         f"{float(r['multi_crossing_pct']):.1%} | "
                         f"{r.get('max_components', '?')} | {float(r.get('avg_components', 0)):.2f} |\n")
        lines.append("\n")

    # EXP-5
    rows = read_csv_rows(os.path.join(OUTPUT_ROOT, "EXP5_full_baseline", "geometric_metrics.csv"))
    if rows:
        lines.append("## EXP-5: Full Metric Baseline\n")
        lines.append("| Model | Res | Layer | CD | NC | F@0.005 | Components(GT\u2192Recon) | Area_ratio |\n")
        lines.append("|-------|-----|-------|----|----|---------|---------------------|------------|\n")
        for r in rows:
            lines.append(f"| {r['model_id']} | {r['resolution']} | {r['layer']} | "
                         f"{float(r['cd']):.6f} | {float(r['nc']):.4f} | "
                         f"{float(r.get('fscore_0.005', 0)):.4f} | "
                         f"{r.get('gt_components', '?')}\u2192{r.get('recon_components', '?')} | "
                         f"{float(r.get('area_ratio', 0)):.4f} |\n")
        lines.append("\n")

    summary_path = os.path.join(OUTPUT_ROOT, "summary.md")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(summary_path, "w") as f:
        f.writelines(lines)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    generate_summary()
