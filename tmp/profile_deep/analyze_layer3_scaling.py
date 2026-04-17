"""Compare per-stage ops between res=256 and res=128, emit scaling table.

Computes t_256 / t_128 ratio per stage by summing device_us from the
per-stage top-30 CSVs produced by analyze_layer1_trace.py.

Usage:
    python tmp/profile_deep/analyze_layer3_scaling.py \\
        tmp/profile_deep/results/per_stage_ops_res256.csv \\
        tmp/profile_deep/results/per_stage_ops_res128.csv \\
        tmp/profile_deep/results/scaling_table.csv
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path


def load_stage_totals(csv_path: str) -> dict:
    """Sum device_us per stage from the per-stage top-N CSV."""
    totals = defaultdict(float)
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            totals[row["stage"]] += float(row["device_us"])
    return dict(totals)


def main():
    hi_csv, lo_csv, out_csv = sys.argv[1:4]
    hi = load_stage_totals(hi_csv)   # res=256
    lo = load_stage_totals(lo_csv)   # res=128

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "device_us_res256", "device_us_res128", "ratio", "flag"])
        for stage in sorted(set(hi) | set(lo)):
            h = hi.get(stage, 0)
            l = lo.get(stage, 0)
            if l > 0:
                ratio = h / l
            else:
                ratio = float("inf")
            if l == 0:
                flag = "no_res128_data"
            elif ratio < 2:
                flag = "below_2x_fixed_cost_dominant"
            elif ratio > 10:
                flag = "above_10x_superlinear"
            else:
                flag = "ok_2x_to_10x"
            w.writerow([stage, f"{h:.1f}", f"{l:.1f}", f"{ratio:.2f}", flag])
    print(f"[write] {out_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
