"""
Report generator for component evaluation pipeline CSV results.

Generates markdown summary reports for Phase A (VAE), Phase B (DiT),
and Phase C (stage attribution) evaluation results.

Usage:
    python scripts/report_gen.py --phase a --csv results/phase_a.csv
    python scripts/report_gen.py --phase b --csv results/phase_b.csv --phase_a_csv results/phase_a.csv
    python scripts/report_gen.py --phase c --csv results/phase_c.csv
    python scripts/report_gen.py --phase a --csv results/phase_a.csv --output_dir reports/
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _fmt(val, decimals=4):
    """Format a float for markdown tables."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"


def _pct(val, decimals=1):
    """Format a float as percentage string."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}%"


def _write_report(path, content):
    """Write report content to file and print confirmation."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"  Written: {path}")


def _filter_valid(df):
    """Filter out rows with errors."""
    if "error" in df.columns:
        return df[df["error"].isna() | (df["error"] == "")]
    return df


def _agg_stats(series):
    """Compute mean, std, median, min, max for a numeric series."""
    return {
        "mean": series.mean(),
        "std": series.std(),
        "median": series.median(),
        "min": series.min(),
        "max": series.max(),
        "count": len(series),
    }


# ---------------------------------------------------------------------------
# Phase A: VAE Reconstruction
# ---------------------------------------------------------------------------

PHASE_A_METRICS = ["cd_100k", "cd_1m", "nc", "psnr", "ssim", "lpips"]
PHASE_A_FSCORE_THRESHOLDS = [0.005, 0.01, 0.05, 0.1, 0.2]


def _phase_a_summary(df, output_dir):
    """Generate summary.md for Phase A results."""
    valid = _filter_valid(df)
    total = len(df)
    n_valid = len(valid)
    n_error = total - n_valid

    lines = []
    lines.append("# Phase A: VAE Reconstruction — Summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"- **Total samples:** {total}")
    lines.append(f"- **Valid:** {n_valid}")
    lines.append(f"- **Errors:** {n_error}")
    lines.append("")

    # Aggregate metrics table
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Metric | Mean | Std | Median | Min | Max |")
    lines.append("|--------|------|-----|--------|-----|-----|")

    for metric in PHASE_A_METRICS:
        if metric not in valid.columns:
            continue
        s = _agg_stats(valid[metric].dropna())
        lines.append(
            f"| {metric} | {_fmt(s['mean'])} | {_fmt(s['std'])} | "
            f"{_fmt(s['median'])} | {_fmt(s['min'])} | {_fmt(s['max'])} |"
        )

    lines.append("")

    # F-score table
    fscore_cols = [f"fscore_{t}" for t in PHASE_A_FSCORE_THRESHOLDS]
    available_fscore = [c for c in fscore_cols if c in valid.columns]
    if available_fscore:
        lines.append("## F-Score by Threshold")
        lines.append("")
        lines.append("| Threshold | Mean | Std | Median |")
        lines.append("|-----------|------|-----|--------|")
        for col in available_fscore:
            threshold = col.replace("fscore_", "")
            s = _agg_stats(valid[col].dropna())
            lines.append(
                f"| {threshold} | {_fmt(s['mean'])} | {_fmt(s['std'])} | {_fmt(s['median'])} |"
            )
        lines.append("")

    _write_report(os.path.join(output_dir, "summary.md"), "\n".join(lines))


def _phase_a_by_tier(df, output_dir):
    """Generate by_tier.md for Phase A results."""
    valid = _filter_valid(df)

    if "tier" not in valid.columns:
        print("  Warning: 'tier' column not found, skipping by_tier.md")
        return

    lines = []
    lines.append("# Phase A: VAE Reconstruction — By Tier")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    tiers = sorted(valid["tier"].dropna().unique())

    for metric in PHASE_A_METRICS:
        if metric not in valid.columns:
            continue
        lines.append(f"## {metric}")
        lines.append("")
        lines.append("| Tier | Count | Mean | Std | Median |")
        lines.append("|------|-------|------|-----|--------|")
        for tier in tiers:
            subset = valid[valid["tier"] == tier][metric].dropna()
            s = _agg_stats(subset)
            lines.append(
                f"| {int(tier)} | {int(s['count'])} | {_fmt(s['mean'])} | "
                f"{_fmt(s['std'])} | {_fmt(s['median'])} |"
            )
        lines.append("")

    # F-score by tier
    fscore_cols = [f"fscore_{t}" for t in PHASE_A_FSCORE_THRESHOLDS]
    available_fscore = [c for c in fscore_cols if c in valid.columns]
    if available_fscore:
        lines.append("## F-Score by Tier")
        lines.append("")
        header = "| Tier | Count |"
        sep = "|------|-------|"
        for col in available_fscore:
            threshold = col.replace("fscore_", "")
            header += f" F@{threshold} |"
            sep += "------|"
        lines.append(header)
        lines.append(sep)
        for tier in tiers:
            subset = valid[valid["tier"] == tier]
            row = f"| {int(tier)} | {len(subset)} |"
            for col in available_fscore:
                val = subset[col].dropna().mean()
                row += f" {_fmt(val)} |"
            lines.append(row)
        lines.append("")

    _write_report(os.path.join(output_dir, "by_tier.md"), "\n".join(lines))


def generate_phase_a(csv_path, output_dir):
    """Generate all Phase A reports."""
    print(f"Phase A report generation from: {csv_path}")
    df = pd.read_csv(csv_path)
    _phase_a_summary(df, output_dir)
    _phase_a_by_tier(df, output_dir)
    print("Phase A reports complete.")


# ---------------------------------------------------------------------------
# Phase B: DiT Generation
# ---------------------------------------------------------------------------

PHASE_B_METRICS = ["cd_filled", "nc_filled", "psnr", "ssim", "lpips"]
PHASE_B_FSCORE_THRESHOLDS = PHASE_A_FSCORE_THRESHOLDS


def _phase_b_summary(df, output_dir, phase_a_df=None):
    """Generate summary.md for Phase B results."""
    valid = _filter_valid(df)
    total = len(df)
    n_valid = len(valid)
    n_error = total - n_valid

    lines = []
    lines.append("# Phase B: DiT Generation — Summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"- **Total samples:** {total}")
    lines.append(f"- **Valid:** {n_valid}")
    lines.append(f"- **Errors:** {n_error}")
    lines.append("")

    # Aggregate metrics table
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Metric | Mean | Std | Median | Min | Max |")
    lines.append("|--------|------|-----|--------|-----|-----|")

    for metric in PHASE_B_METRICS:
        if metric not in valid.columns:
            continue
        s = _agg_stats(valid[metric].dropna())
        lines.append(
            f"| {metric} | {_fmt(s['mean'])} | {_fmt(s['std'])} | "
            f"{_fmt(s['median'])} | {_fmt(s['min'])} | {_fmt(s['max'])} |"
        )

    lines.append("")

    # F-score table (filled variant)
    fscore_cols = [f"fscore_{t}_filled" for t in PHASE_B_FSCORE_THRESHOLDS]
    available_fscore = [c for c in fscore_cols if c in valid.columns]
    if available_fscore:
        lines.append("## F-Score (filled)")
        lines.append("")
        lines.append("| Threshold | Mean | Std | Median |")
        lines.append("|-----------|------|-----|--------|")
        for col in available_fscore:
            threshold = col.replace("fscore_", "").replace("_filled", "")
            s = _agg_stats(valid[col].dropna())
            lines.append(
                f"| {threshold} | {_fmt(s['mean'])} | {_fmt(s['std'])} | {_fmt(s['median'])} |"
            )
        lines.append("")

    # Gap ratio vs Phase A
    if phase_a_df is not None:
        lines.append("## Gap Ratio vs Phase A (DiT / VAE)")
        lines.append("")
        a_valid = _filter_valid(phase_a_df)

        # Metric mapping: Phase B metric -> Phase A metric
        gap_mapping = {
            "cd_filled": "cd_100k",
            "nc_filled": "nc",
            "psnr": "psnr",
            "ssim": "ssim",
            "lpips": "lpips",
        }

        lines.append("| Metric | Phase A Mean | Phase B Mean | Ratio (B/A) |")
        lines.append("|--------|-------------|-------------|-------------|")

        for b_metric, a_metric in gap_mapping.items():
            if b_metric not in valid.columns or a_metric not in a_valid.columns:
                continue
            a_mean = a_valid[a_metric].dropna().mean()
            b_mean = valid[b_metric].dropna().mean()
            if a_mean != 0 and not np.isnan(a_mean):
                ratio = b_mean / a_mean
                lines.append(
                    f"| {b_metric} | {_fmt(a_mean)} | {_fmt(b_mean)} | {_fmt(ratio, 2)}x |"
                )
            else:
                lines.append(
                    f"| {b_metric} | {_fmt(a_mean)} | {_fmt(b_mean)} | N/A |"
                )

        lines.append("")

    # Best/worst 10 cases by cd_filled
    if "cd_filled" in valid.columns and len(valid) > 0:
        sorted_df = valid.sort_values("cd_filled")

        lines.append("## Best 10 Cases (lowest cd_filled)")
        lines.append("")
        uid_col = "uid" if "uid" in valid.columns else valid.columns[0]
        cols_to_show = [uid_col, "cd_filled", "nc_filled", "psnr"]
        cols_to_show = [c for c in cols_to_show if c in valid.columns]
        header = "| " + " | ".join(cols_to_show) + " |"
        sep = "|" + "|".join(["------" for _ in cols_to_show]) + "|"
        lines.append(header)
        lines.append(sep)
        for _, row in sorted_df.head(10).iterrows():
            vals = []
            for c in cols_to_show:
                v = row[c]
                if isinstance(v, float):
                    vals.append(_fmt(v))
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

        lines.append("## Worst 10 Cases (highest cd_filled)")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for _, row in sorted_df.tail(10).iloc[::-1].iterrows():
            vals = []
            for c in cols_to_show:
                v = row[c]
                if isinstance(v, float):
                    vals.append(_fmt(v))
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    _write_report(os.path.join(output_dir, "summary.md"), "\n".join(lines))


def _phase_b_postprocess(df, output_dir):
    """Generate postprocess_comparison.md comparing filled vs raw metrics."""
    valid = _filter_valid(df)

    lines = []
    lines.append("# Phase B: Post-processing Comparison (fill_holes)")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Compare filled vs raw for CD and NC
    comparisons = [
        ("cd_filled", "cd_raw", "Chamfer Distance"),
        ("nc_filled", "nc_raw", "Normal Consistency"),
    ]

    lines.append("## Metric Comparison: Filled vs Raw")
    lines.append("")
    lines.append("| Metric | Filled Mean | Raw Mean | Improvement | Improved % |")
    lines.append("|--------|------------|----------|-------------|------------|")

    for filled_col, raw_col, label in comparisons:
        if filled_col not in valid.columns or raw_col not in valid.columns:
            continue
        both_valid = valid[[filled_col, raw_col]].dropna()
        filled_mean = both_valid[filled_col].mean()
        raw_mean = both_valid[raw_col].mean()
        # For CD, lower is better; for NC, higher is better
        if "cd" in filled_col:
            improvement = raw_mean - filled_mean
            pct_improved = (both_valid[filled_col] < both_valid[raw_col]).mean() * 100
        else:
            improvement = filled_mean - raw_mean
            pct_improved = (both_valid[filled_col] > both_valid[raw_col]).mean() * 100
        lines.append(
            f"| {label} | {_fmt(filled_mean)} | {_fmt(raw_mean)} | "
            f"{_fmt(improvement)} | {_pct(pct_improved)} |"
        )

    lines.append("")

    # F-score comparison
    fscore_filled = [f"fscore_{t}_filled" for t in PHASE_B_FSCORE_THRESHOLDS]
    fscore_raw = [f"fscore_{t}_raw" for t in PHASE_B_FSCORE_THRESHOLDS]
    available_pairs = [
        (f, r, t)
        for f, r, t in zip(fscore_filled, fscore_raw, PHASE_B_FSCORE_THRESHOLDS)
        if f in valid.columns and r in valid.columns
    ]

    if available_pairs:
        lines.append("## F-Score Comparison: Filled vs Raw")
        lines.append("")
        lines.append("| Threshold | Filled Mean | Raw Mean | Delta |")
        lines.append("|-----------|------------|----------|-------|")
        for filled_col, raw_col, threshold in available_pairs:
            both_valid = valid[[filled_col, raw_col]].dropna()
            filled_mean = both_valid[filled_col].mean()
            raw_mean = both_valid[raw_col].mean()
            delta = filled_mean - raw_mean
            lines.append(
                f"| {threshold} | {_fmt(filled_mean)} | {_fmt(raw_mean)} | "
                f"{_fmt(delta)} |"
            )
        lines.append("")

    _write_report(os.path.join(output_dir, "postprocess_comparison.md"), "\n".join(lines))


def _phase_b_by_tier(df, output_dir):
    """Generate by_tier.md for Phase B results."""
    valid = _filter_valid(df)

    if "tier" not in valid.columns:
        print("  Warning: 'tier' column not found, skipping by_tier.md")
        return

    lines = []
    lines.append("# Phase B: DiT Generation — By Tier")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    tiers = sorted(valid["tier"].dropna().unique())

    for metric in PHASE_B_METRICS:
        if metric not in valid.columns:
            continue
        lines.append(f"## {metric}")
        lines.append("")
        lines.append("| Tier | Count | Mean | Std | Median |")
        lines.append("|------|-------|------|-----|--------|")
        for tier in tiers:
            subset = valid[valid["tier"] == tier][metric].dropna()
            s = _agg_stats(subset)
            lines.append(
                f"| {int(tier)} | {int(s['count'])} | {_fmt(s['mean'])} | "
                f"{_fmt(s['std'])} | {_fmt(s['median'])} |"
            )
        lines.append("")

    _write_report(os.path.join(output_dir, "by_tier.md"), "\n".join(lines))


def generate_phase_b(csv_path, output_dir, phase_a_csv=None):
    """Generate all Phase B reports."""
    print(f"Phase B report generation from: {csv_path}")
    df = pd.read_csv(csv_path)
    phase_a_df = pd.read_csv(phase_a_csv) if phase_a_csv else None
    _phase_b_summary(df, output_dir, phase_a_df)
    _phase_b_postprocess(df, output_dir)
    _phase_b_by_tier(df, output_dir)
    print("Phase B reports complete.")


# ---------------------------------------------------------------------------
# Phase C: Stage Attribution
# ---------------------------------------------------------------------------

PHASE_C_CONDITIONS = [
    "baseline",
    "c1_gt_struct",
    "c2_gt_struct_shape",
    "c3_gt_material",
    "c4_gt_struct_material",
]

# Human-readable labels for conditions
CONDITION_LABELS = {
    "baseline": "Baseline (no GT)",
    "c1_gt_struct": "C1: GT Structure",
    "c2_gt_struct_shape": "C2: GT Structure + Shape",
    "c3_gt_material": "C3: GT Material",
    "c4_gt_struct_material": "C4: GT Structure + Material",
}


def _phase_c_stage_attribution(df, output_dir):
    """Generate stage_attribution.md for Phase C results."""
    valid = _filter_valid(df)

    lines = []
    lines.append("# Phase C: Stage Attribution Analysis")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"- **Total samples:** {len(df)}")
    lines.append(f"- **Valid:** {len(valid)}")
    lines.append("")

    # CD by condition
    lines.append("## Chamfer Distance by Condition")
    lines.append("")
    lines.append("| Condition | Description | Mean CD | Std | Median |")
    lines.append("|-----------|-------------|---------|-----|--------|")

    cd_means = {}
    for cond in PHASE_C_CONDITIONS:
        col = f"{cond}_cd"
        if col not in valid.columns:
            continue
        s = _agg_stats(valid[col].dropna())
        cd_means[cond] = s["mean"]
        label = CONDITION_LABELS.get(cond, cond)
        lines.append(
            f"| {cond} | {label} | {_fmt(s['mean'], 6)} | "
            f"{_fmt(s['std'], 6)} | {_fmt(s['median'], 6)} |"
        )

    lines.append("")

    # NC by condition
    lines.append("## Normal Consistency by Condition")
    lines.append("")
    lines.append("| Condition | Description | Mean NC | Std | Median |")
    lines.append("|-----------|-------------|---------|-----|--------|")

    for cond in PHASE_C_CONDITIONS:
        col = f"{cond}_nc"
        if col not in valid.columns:
            continue
        s = _agg_stats(valid[col].dropna())
        label = CONDITION_LABELS.get(cond, cond)
        lines.append(
            f"| {cond} | {label} | {_fmt(s['mean'])} | "
            f"{_fmt(s['std'])} | {_fmt(s['median'])} |"
        )

    lines.append("")

    # Error attribution analysis
    # Attribution logic:
    #   Structure DiT contribution: baseline_cd - c1_cd (giving GT structure reduces error by this much)
    #   Shape DiT contribution: c1_cd - c2_cd (adding GT shape on top of GT structure)
    #   Material DiT contribution: baseline_cd - c3_cd (giving GT material reduces error by this much)
    #   Structure+Material combined: baseline_cd - c4_cd
    if "baseline" in cd_means and len(cd_means) > 1:
        lines.append("## Error Attribution (CD-based)")
        lines.append("")
        lines.append(
            "Analysis of how much each DiT stage contributes to the total error, "
            "measured by CD reduction when replacing that stage's output with GT."
        )
        lines.append("")

        baseline_cd = cd_means.get("baseline", None)
        attributions = []

        if baseline_cd is not None and baseline_cd > 0:
            lines.append("| Attribution | CD Reduction | % of Baseline |")
            lines.append("|-------------|-------------|----------------|")

            # Structure contribution: baseline - C1
            if "c1_gt_struct" in cd_means:
                reduction = baseline_cd - cd_means["c1_gt_struct"]
                pct = (reduction / baseline_cd) * 100
                attributions.append(("Structure DiT", reduction, pct))
                lines.append(
                    f"| Structure DiT (baseline -> C1) | {_fmt(reduction, 6)} | {_pct(pct)} |"
                )

            # Shape DiT contribution: C1 - C2
            if "c1_gt_struct" in cd_means and "c2_gt_struct_shape" in cd_means:
                reduction = cd_means["c1_gt_struct"] - cd_means["c2_gt_struct_shape"]
                pct = (reduction / baseline_cd) * 100
                attributions.append(("Shape DiT", reduction, pct))
                lines.append(
                    f"| Shape DiT (C1 -> C2) | {_fmt(reduction, 6)} | {_pct(pct)} |"
                )

            # Material contribution: baseline - C3
            if "c3_gt_material" in cd_means:
                reduction = baseline_cd - cd_means["c3_gt_material"]
                pct = (reduction / baseline_cd) * 100
                attributions.append(("Material DiT", reduction, pct))
                lines.append(
                    f"| Material DiT (baseline -> C3) | {_fmt(reduction, 6)} | {_pct(pct)} |"
                )

            # Structure + Material combined: baseline - C4
            if "c4_gt_struct_material" in cd_means:
                reduction = baseline_cd - cd_means["c4_gt_struct_material"]
                pct = (reduction / baseline_cd) * 100
                attributions.append(("Structure + Material", reduction, pct))
                lines.append(
                    f"| Structure + Material (baseline -> C4) | {_fmt(reduction, 6)} | {_pct(pct)} |"
                )

            lines.append("")

            # Summary
            if attributions:
                lines.append("### Summary")
                lines.append("")
                # Sort by contribution percentage descending
                attributions.sort(key=lambda x: x[2], reverse=True)
                for name, reduction, pct in attributions:
                    lines.append(f"- **{name}**: {_pct(pct)} of total error")
                lines.append("")

    # F-score by condition
    fscore_thresholds = PHASE_A_FSCORE_THRESHOLDS
    sample_col = f"{PHASE_C_CONDITIONS[0]}_fscore_{fscore_thresholds[0]}"
    if sample_col in valid.columns:
        lines.append("## F-Score by Condition")
        lines.append("")
        header = "| Condition |"
        sep = "|-----------|"
        for t in fscore_thresholds:
            header += f" F@{t} |"
            sep += "------|"
        lines.append(header)
        lines.append(sep)
        for cond in PHASE_C_CONDITIONS:
            row = f"| {cond} |"
            for t in fscore_thresholds:
                col = f"{cond}_fscore_{t}"
                if col in valid.columns:
                    val = valid[col].dropna().mean()
                    row += f" {_fmt(val)} |"
                else:
                    row += " N/A |"
            lines.append(row)
        lines.append("")

    _write_report(os.path.join(output_dir, "stage_attribution.md"), "\n".join(lines))


def _phase_c_by_tier(df, output_dir):
    """Generate by_tier.md for Phase C results."""
    valid = _filter_valid(df)

    if "tier" not in valid.columns:
        print("  Warning: 'tier' column not found, skipping by_tier.md")
        return

    lines = []
    lines.append("# Phase C: Stage Attribution — By Tier")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    tiers = sorted(valid["tier"].dropna().unique())

    # CD breakdown per condition per tier
    lines.append("## Chamfer Distance: Condition x Tier")
    lines.append("")
    header = "| Condition |"
    sep = "|-----------|"
    for tier in tiers:
        header += f" Tier {int(tier)} |"
        sep += "------|"
    lines.append(header)
    lines.append(sep)

    for cond in PHASE_C_CONDITIONS:
        col = f"{cond}_cd"
        if col not in valid.columns:
            continue
        row = f"| {cond} |"
        for tier in tiers:
            subset = valid[valid["tier"] == tier][col].dropna()
            if len(subset) > 0:
                row += f" {_fmt(subset.mean(), 6)} |"
            else:
                row += " N/A |"
        lines.append(row)

    lines.append("")

    # NC breakdown per condition per tier
    lines.append("## Normal Consistency: Condition x Tier")
    lines.append("")
    header = "| Condition |"
    sep = "|-----------|"
    for tier in tiers:
        header += f" Tier {int(tier)} |"
        sep += "------|"
    lines.append(header)
    lines.append(sep)

    for cond in PHASE_C_CONDITIONS:
        col = f"{cond}_nc"
        if col not in valid.columns:
            continue
        row = f"| {cond} |"
        for tier in tiers:
            subset = valid[valid["tier"] == tier][col].dropna()
            if len(subset) > 0:
                row += f" {_fmt(subset.mean())} |"
            else:
                row += " N/A |"
        lines.append(row)

    lines.append("")

    # Sample counts per tier
    lines.append("## Sample Counts")
    lines.append("")
    lines.append("| Tier | Count |")
    lines.append("|------|-------|")
    for tier in tiers:
        count = len(valid[valid["tier"] == tier])
        lines.append(f"| {int(tier)} | {count} |")
    lines.append("")

    _write_report(os.path.join(output_dir, "by_tier.md"), "\n".join(lines))


def generate_phase_c(csv_path, output_dir):
    """Generate all Phase C reports."""
    print(f"Phase C report generation from: {csv_path}")
    df = pd.read_csv(csv_path)
    _phase_c_stage_attribution(df, output_dir)
    _phase_c_by_tier(df, output_dir)
    print("Phase C reports complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate markdown summary reports from evaluation CSV results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/report_gen.py --phase a --csv results/phase_a.csv
  python scripts/report_gen.py --phase b --csv results/phase_b.csv --phase_a_csv results/phase_a.csv
  python scripts/report_gen.py --phase c --csv results/phase_c.csv
  python scripts/report_gen.py --phase a --csv results/phase_a.csv --output_dir reports/
        """,
    )
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["a", "b", "c"],
        help="Evaluation phase: a (VAE), b (DiT), c (stage attribution)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the evaluation results CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for reports (default: CSV parent directory)",
    )
    parser.add_argument(
        "--phase_a_csv",
        type=str,
        default=None,
        help="Path to Phase A CSV for gap computation (Phase B only)",
    )

    args = parser.parse_args()

    # Validate CSV exists
    if not os.path.isfile(args.csv):
        print(f"Error: CSV file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    # Default output dir = CSV parent directory
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.csv))

    if args.phase == "a":
        generate_phase_a(args.csv, output_dir)
    elif args.phase == "b":
        if args.phase_a_csv and not os.path.isfile(args.phase_a_csv):
            print(f"Error: Phase A CSV not found: {args.phase_a_csv}", file=sys.stderr)
            sys.exit(1)
        generate_phase_b(args.csv, output_dir, args.phase_a_csv)
    elif args.phase == "c":
        generate_phase_c(args.csv, output_dir)


if __name__ == "__main__":
    main()
