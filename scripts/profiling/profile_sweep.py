#!/usr/bin/env python
"""Run a matrix of profile_dit configurations sequentially and aggregate.

The sweep YAML schema:

    base_config: coart/dit/configs/coart_dit_shape_512_ft.json   # optional, defaults to this
    defaults:                  # optional shared knobs applied to every cell
      warmup_steps: 30
      active_steps: 100
      num_gpus: 8
      host: host-10-240-99-118
    runs:
      - name: baseline
        overrides: {}
      - name: split1
        overrides:
          trainer.args.batch_split: 1
      - name: split1_target082
        overrides:
          trainer.args.batch_split: 1
          trainer.args.elastic.args.target_ratio: 0.82
      - name: bs12_zero1     # would need ZeRO-1 patch first; placeholder
        overrides:
          trainer.args.batch_size_per_gpu: 12

Sequentially calls scripts/profiling/profile_dit.py for each cell, then
runs profile_compare.py over the result list (baseline first).

Usage:
    python scripts/profiling/profile_sweep.py --sweep scripts/profiling/sweeps/example.yaml \
        [--out logs/profile_sweeps/{ts}/]

Each cell becomes:
    logs/profile_results/{cell_name}_{ts}.json
And the final markdown comparison + per-cell stdout logs land under --out.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List

REPO = pathlib.Path(__file__).resolve().parents[2]
PROFILE_DIT = REPO / "scripts" / "profiling" / "profile_dit.py"
PROFILE_COMPARE = REPO / "scripts" / "profiling" / "profile_compare.py"


def _load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "PyYAML required for sweep YAML loading. Install: "
            ".venv/bin/pip install pyyaml"
        ) from e
    return yaml.safe_load(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=pathlib.Path, required=True,
                    help="Sweep YAML path")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="Output dir (default: logs/profile_sweeps/{ts}/)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--continue-on-fail", action="store_true",
                    help="Don't abort the sweep if a single cell crashes")
    args = ap.parse_args()

    spec = _load_yaml(args.sweep)
    base_config = spec.get("base_config")
    defaults = spec.get("defaults", {}) or {}
    runs: List[Dict[str, Any]] = spec.get("runs") or []
    if not runs:
        print("[err] sweep has no 'runs' list", file=sys.stderr)
        sys.exit(1)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or REPO / "logs" / "profile_sweeps" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sweep] {len(runs)} cells -> {out_dir}")
    result_paths: List[pathlib.Path] = []

    for i, run in enumerate(runs, 1):
        name = run["name"]
        overrides = run.get("overrides") or {}
        merged = {**defaults, **{k: v for k, v in run.items() if k not in ("name", "overrides")}}
        warmup = int(merged.get("warmup_steps", 30))
        active = int(merged.get("active_steps", 100))
        num_gpus = int(merged.get("num_gpus", 8))
        host = merged.get("host")

        cmd = [str(REPO / ".venv" / "bin" / "python"), str(PROFILE_DIT),
               "--label", name,
               "--warmup-steps", str(warmup),
               "--active-steps", str(active),
               "--num-gpus", str(num_gpus)]
        if base_config:
            cmd += ["--base-config", base_config]
        if host:
            cmd += ["--host", host]
        for k, v in overrides.items():
            cmd += ["--override", f"{k}={json.dumps(v)}"]
        if args.dry_run:
            cmd += ["--dry-run"]

        cell_log = out_dir / f"{name}_stdout.log"
        print(f"\n[sweep {i}/{len(runs)}] {name} -> {cell_log}")
        print("  " + " ".join(cmd))

        with cell_log.open("w") as fp:
            proc = subprocess.run(cmd, stdout=fp, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            msg = f"[sweep] cell {name} FAILED (exit={proc.returncode}); see {cell_log}"
            print(msg)
            if not args.continue_on_fail:
                sys.exit(proc.returncode)
            continue

        # Discover the result file written by profile_dit (label_* most recent)
        cands = sorted((REPO / "logs" / "profile_results").glob(f"{name}_*.json"))
        if not cands:
            print(f"[sweep] cell {name} produced no result.json; skipping in compare")
            continue
        result_paths.append(cands[-1])
        print(f"  result -> {cands[-1].name}")

    # Aggregate via profile_compare.py
    if not result_paths or args.dry_run:
        print("\n[sweep] no results to compare (dry-run or all failed)")
        return
    cmp_md = out_dir / "compare.md"
    cmd = [str(REPO / ".venv" / "bin" / "python"), str(PROFILE_COMPARE),
           "--runs", *[str(p) for p in result_paths],
           "--out", str(cmp_md)]
    print(f"\n[sweep] aggregating -> {cmp_md}")
    subprocess.run(cmd, check=True)
    print(f"[sweep] done: {cmp_md}")


if __name__ == "__main__":
    main()
