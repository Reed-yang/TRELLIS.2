#!/usr/bin/env python
"""Compare N profile result.json files.

Renders a markdown table of key metrics with delta-vs-baseline columns. The
first --runs argument is treated as the baseline.

Usage:
    python scripts/profiling/profile_compare.py \
        --runs logs/profile_results/v4_baseline_*.json \
               logs/profile_results/split1_*.json \
               logs/profile_results/fa3_*.json \
        [--out logs/profile_compare/cmp_<ts>.md]

If --out is omitted, the markdown is printed to stdout.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional

REPO = pathlib.Path(__file__).resolve().parents[2]


METRIC_TABLE = [
    # (label,                key in summary.metrics,        unit,  bigger_is_better)
    ("step time mean",       "time/step",                   "s",    False),
    ("step time p95",        "time/step",                   "s",    False),  # uses .p95
    ("steps/h (mean)",       None,                          "/h",   True),   # derived from time/step.mean
    ("throughput tok/s",     "perf/throughput_tok_per_s",   "tok/s",True),
    ("dataloader wait",      "perf/dataloader_wait_s",      "s",    False),
    ("mem peak",             "perf/mem_peak_gb",            "GB",   False),
    ("mem alloc",            "perf/mem_alloc_gb",           "GB",   False),
    ("loss (train, mean)",   "loss/loss",                   "",     False),
    ("grad_norm (mean)",     "status/grad_norm",            "",     False),
    ("input_size mean",      "elastic/input_size",          "tok",  None),
    ("mem_ratio mean",       "elastic/mem_ratio",           "",     None),
]


def _get(metrics: Dict[str, Any], key: str, stat: str = "mean") -> Optional[float]:
    if key not in metrics:
        return None
    return metrics[key].get(stat)


def _fmt(x: Optional[float], unit: str) -> str:
    if x is None:
        return "—"
    if unit == "s" and x < 1:
        return f"{x*1000:.1f}ms"
    if unit == "GB":
        return f"{x:.2f}GB"
    if unit == "/h":
        return f"{x:.0f}"
    if unit == "tok/s":
        return f"{x:.0f}"
    if unit == "tok":
        return f"{x:.0f}"
    return f"{x:.4f}"


def _delta(cur: Optional[float], base: Optional[float], bigger_better: Optional[bool]) -> str:
    if cur is None or base is None or base == 0:
        return ""
    pct = (cur - base) / base * 100.0
    if bigger_better is None:
        return f"({pct:+.1f}%)"
    arrow = "✅" if (pct > 0) == bigger_better else "❌"
    if abs(pct) < 0.5:
        arrow = "≈"
    return f"({pct:+.1f}% {arrow})"


def render_table(runs: List[Dict[str, Any]]) -> str:
    if not runs:
        return "(no runs)"
    base = runs[0]
    base_metrics = base["summary"]["metrics"]

    headers = ["metric"]
    for r in runs:
        headers.append(f"{r['label']}")
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    out = ["| " + " | ".join(headers) + " |", sep]

    for label, key, unit, bigger_better in METRIC_TABLE:
        row = [label + (f" ({unit})" if unit else "")]
        # Compute baseline
        if key is None:  # derived
            base_step = _get(base_metrics, "time/step", "mean")
            base_val = (3600.0 / base_step) if base_step else None
        elif "p95" in label:
            base_val = _get(base_metrics, key, "p95")
        else:
            base_val = _get(base_metrics, key, "mean")

        for r in runs:
            m = r["summary"]["metrics"]
            if key is None:
                step_mean = _get(m, "time/step", "mean")
                cur = (3600.0 / step_mean) if step_mean else None
            elif "p95" in label:
                cur = _get(m, key, "p95")
            else:
                cur = _get(m, key, "mean")
            cell = _fmt(cur, unit)
            if r is base:
                row.append(cell)
            else:
                d = _delta(cur, base_val, bigger_better)
                row.append(f"{cell} {d}".rstrip())
        out.append("| " + " | ".join(row) + " |")

    # Loss bins (just mean per bin)
    out.append("")
    out.append("### Per-t-bin train loss (mean)")
    headers = ["t-bin"]
    for r in runs:
        headers.append(r["label"])
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for i in range(10):
        k = f"loss/bin_{i}/mse"
        row = [f"bin_{i}"]
        base_v = _get(base["summary"].get("loss_bins", {}), k, "mean")
        for r in runs:
            v = _get(r["summary"].get("loss_bins", {}), k, "mean")
            cell = _fmt(v, "")
            if r is not base and base_v is not None and v is not None:
                pct = (v - base_v) / base_v * 100.0 if base_v else 0
                arrow = "↓" if v < base_v else "↑"
                cell = f"{cell} ({pct:+.1f}%{arrow})"
            row.append(cell)
        out.append("| " + " | ".join(row) + " |")

    # Eval (if any run has it)
    has_eval = any("loss" in r["summary"].get("eval", {}) or "loss/loss" in r["summary"].get("eval", {}) for r in runs)
    if has_eval:
        out.append("")
        out.append("### Hold-out eval loss (mean)")
        keys = sorted(set().union(*[set(r["summary"].get("eval", {}).keys()) for r in runs]))
        headers = ["metric"]
        for r in runs:
            headers.append(r["label"])
        out.append("| " + " | ".join(headers) + " |")
        out.append("|" + "|".join(["---"] * len(headers)) + "|")
        for k in keys:
            row = [k]
            for r in runs:
                v = _get(r["summary"].get("eval", {}), k, "mean")
                row.append(_fmt(v, ""))
            out.append("| " + " | ".join(row) + " |")

    # Override summary (helps reading the table)
    out.append("")
    out.append("### Run config overrides")
    for r in runs:
        ovs = r.get("overrides") or []
        if not ovs and r.get("kind") == "baseline_extract":
            ovs = ["(baseline-extract from a live run; no overrides)"]
        elif not ovs:
            ovs = ["(none)"]
        out.append(f"- **{r['label']}** ({r.get('kind','?')}, n={r['summary'].get('n_rows_after_warmup','?')} steps): {ovs}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Result JSON paths. First one is baseline.")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="Markdown output path (default: stdout)")
    args = ap.parse_args()

    runs: List[Dict[str, Any]] = []
    for p in args.runs:
        path = pathlib.Path(p)
        if not path.exists():
            print(f"[err] missing {path}", file=sys.stderr)
            sys.exit(1)
        runs.append(json.loads(path.read_text()))

    md = render_table(runs)
    header = (
        f"# Profile Compare ({time.strftime('%Y-%m-%d %H:%M:%S')})\n\n"
        f"Baseline: **{runs[0]['label']}**\n\n"
    )
    body = header + md + "\n"

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body)
        print(f"[ok] wrote {args.out}")
    else:
        print(body)


if __name__ == "__main__":
    main()
