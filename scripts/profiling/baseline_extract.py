#!/usr/bin/env python
"""Extract a profile-result.json baseline from an existing/running training log.

Reads ``<output_dir>/log.txt`` (one JSON-per-step format produced by
trellis2 BasicTrainer.save_logs) and produces the same result schema as
``profile_dit.py`` so it can be fed into ``profile_compare.py``.

This avoids spinning up a fresh profile run when we already have a
training in flight whose stable-region steps are themselves a baseline.

Usage:
    python scripts/profiling/baseline_extract.py \
        --output-dir results/coart_dit_shape_20260502_214128_v4 \
        --label v4_baseline \
        --warmup-steps 100 --tail-steps 200

The result JSON is written to logs/profile_results/{label}_{ts}.json.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

REPO = pathlib.Path(__file__).resolve().parents[2]


def _flatten(d: Dict[str, Any], prefix: str = "", out: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    if out is None:
        out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            _flatten(v, key, out)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _agg(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "min": min(values),
        "max": max(values),
        "stddev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def parse_log(log_path: pathlib.Path) -> List[Dict[str, Any]]:
    """Yield list of {step:int, **flat_metrics}."""
    rows: List[Dict[str, Any]] = []
    with log_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                step_str, payload = line.split(": ", 1)
                step = int(step_str)
                d = json.loads(payload)
            except (ValueError, json.JSONDecodeError):
                continue
            flat = _flatten(d)
            flat["step"] = step
            rows.append(flat)
    return rows


def summarise(rows: List[Dict[str, Any]], warmup_steps: int, tail_steps: Optional[int] = None) -> Dict[str, Any]:
    if not rows:
        raise RuntimeError("no rows parsed from log.txt")
    rows.sort(key=lambda r: r["step"])
    keep = [r for r in rows if r["step"] > warmup_steps]
    if tail_steps:
        keep = keep[-tail_steps:]
    if not keep:
        raise RuntimeError(
            f"no rows after warmup={warmup_steps} (got {len(rows)} total, max step={rows[-1]['step']})"
        )

    metrics_keys = [
        "time/step",
        "loss/loss",
        "loss/mse",
        "status/grad_norm",
        "elastic/input_size",
        "elastic/memory",
        "elastic/mem_ratio",
        "perf/throughput_step_per_h",
        "perf/throughput_tok_per_s",
        "perf/dataloader_wait_s",
        "perf/mem_peak_gb",
        "perf/mem_alloc_gb",
    ]
    bin_keys = [f"loss/bin_{i}/mse" for i in range(10)]
    eval_keys = [f"eval/{k}" for k in ["loss", "mse"] + [f"bin_{i}/mse" for i in range(10)]]

    summary: Dict[str, Any] = {
        "n_rows_total": len(rows),
        "n_rows_after_warmup": len(keep),
        "step_range": [keep[0]["step"], keep[-1]["step"]],
        "warmup_steps": warmup_steps,
        "tail_steps": tail_steps,
        "metrics": {},
        "loss_bins": {},
        "eval": {},
    }
    for k in metrics_keys:
        vals = [r[k] for r in keep if k in r]
        if vals:
            summary["metrics"][k] = _agg(vals)
    for k in bin_keys:
        vals = [r[k] for r in keep if k in r]
        if vals:
            summary["loss_bins"][k] = _agg(vals)
    for k in eval_keys:
        vals = [r[k] for r in keep if k in r]
        if vals:
            summary["eval"][k] = _agg(vals)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=pathlib.Path, required=True,
                    help="Training output dir containing log.txt + config.json")
    ap.add_argument("--label", required=True, help="Short label for this baseline (used in result filename + compare table)")
    ap.add_argument("--warmup-steps", type=int, default=100,
                    help="Skip the first N step rows (cold-start, NCCL init, autotune)")
    ap.add_argument("--tail-steps", type=int, default=None,
                    help="Optionally restrict to the last N rows after warmup")
    ap.add_argument("--result-dir", type=pathlib.Path, default=REPO / "logs" / "profile_results")
    args = ap.parse_args()

    log_path = args.output_dir / "log.txt"
    cfg_path = args.output_dir / "config.json"
    if not log_path.exists():
        print(f"[err] missing {log_path}", file=sys.stderr)
        sys.exit(1)

    rows = parse_log(log_path)
    summary = summarise(rows, args.warmup_steps, args.tail_steps)
    config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    args.result_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = args.result_dir / f"{args.label}_{ts}.json"
    payload = {
        "label": args.label,
        "kind": "baseline_extract",
        "source_output_dir": str(args.output_dir),
        "extracted_at": ts,
        "config": config,
        "summary": summary,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"[ok] wrote {out}")
    if "time/step" in summary["metrics"]:
        m = summary["metrics"]["time/step"]
        thpt_h = 3600.0 / m["mean"] if m["mean"] > 0 else float("nan")
        print(f"  step_time mean={m['mean']:.3f}s p95={m['p95']:.3f}s -> {thpt_h:.1f} steps/h")
    if "perf/mem_peak_gb" in summary["metrics"]:
        m = summary["metrics"]["perf/mem_peak_gb"]
        print(f"  mem_peak  mean={m['mean']:.1f}GB p95={m['p95']:.1f}GB")
    if "loss/loss" in summary["metrics"]:
        m = summary["metrics"]["loss/loss"]
        print(f"  loss      mean={m['mean']:.4f}  range=[{m['min']:.4f}, {m['max']:.4f}]")


if __name__ == "__main__":
    main()
