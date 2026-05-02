#!/usr/bin/env python
"""Run a short, instrumented coart DiT training pass and emit a result.json.

Workflow:
  1. Load base config JSON.
  2. Apply --override key.path=value (dotted JSON-path; values JSON-decoded).
  3. Force max_steps = warmup + active. Set i_log=1 (every step → log.txt).
     Disable expensive periodic ops (i_save / i_eval / wandb) unless asked.
  4. Persist tmp config to scripts/profiling/.tmp_configs/{label}_{ts}.json.
  5. Build train command (auto_retry=0 → mp.spawn surfaces tracebacks).
  6. Run locally OR over SSH (--host).
  7. After exit, parse <output_dir>/log.txt → call baseline_extract.summarise.
  8. Write logs/profile_results/{label}_{ts}.json.

Examples:
  # Local single-GPU sanity check (warmup 5, active 20)
  python scripts/profiling/profile_dit.py \
      --label local_smoke --num-gpus 1 \
      --warmup-steps 5 --active-steps 20

  # Remote 8-GPU profile of a config tweak
  python scripts/profiling/profile_dit.py \
      --label split1_target082 --host host-10-240-99-118 --num-gpus 8 \
      --warmup-steps 30 --active-steps 100 \
      --override trainer.args.batch_split=1 \
      --override trainer.args.elastic.args.target_ratio=0.82

The script DOES NOT touch a live training session. It writes to its own
output dir under results/profile_dit_runs/.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.profiling.baseline_extract import parse_log, summarise  # noqa: E402


def deep_set(obj: Any, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = obj
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            raise KeyError(f"override path {dotted!r}: segment {p!r} missing or not a dict")
        cur = cur[p]
    leaf = parts[-1]
    if leaf not in cur:
        # Allow adding new leaves (e.g. wandb_project that isn't in base).
        cur[leaf] = value
    else:
        cur[leaf] = value


def parse_override(s: str) -> tuple[str, Any]:
    if "=" not in s:
        raise ValueError(f"--override expects key.path=value, got {s!r}")
    k, v = s.split("=", 1)
    # Try JSON-decode (so 0.82 → float, true → bool, [1,2] → list); fall back to str.
    try:
        v_parsed = json.loads(v)
    except json.JSONDecodeError:
        v_parsed = v
    return k.strip(), v_parsed


def build_cmd(
    *,
    config_path: pathlib.Path,
    output_dir: pathlib.Path,
    load_dir: pathlib.Path,
    ckpt: str,
    num_gpus: int,
    extra_argv: List[str],
) -> List[str]:
    py = str(REPO / ".venv" / "bin" / "python")
    entry = str(REPO / "scripts" / "coart_train_dit_entry.py")
    return [
        py, entry,
        "--config", str(config_path),
        "--output_dir", str(output_dir),
        "--data_dir", os.environ.get(
            "DATA_ROOT",
            "/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0",
        ),
        "--auto_retry", "0",
        "--num_gpus", str(num_gpus),
        "--load_dir", str(load_dir),
        "--ckpt", ckpt,
        *extra_argv,
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="Short identifier; used in run dir + result file")
    ap.add_argument("--base-config", type=pathlib.Path,
                    default=REPO / "coart" / "dit" / "configs" / "coart_dit_shape_512_ft.json")
    ap.add_argument("--load-dir", type=pathlib.Path,
                    default=REPO / "results" / "coart_dit_warmstart_stage")
    ap.add_argument("--ckpt", default="0")
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--warmup-steps", type=int, default=30,
                    help="Steps to discard at the start (NCCL init, autotune, mem-controller bootstrap)")
    ap.add_argument("--active-steps", type=int, default=100,
                    help="Steps to measure after warmup")
    ap.add_argument("--override", action="append", default=[],
                    help="Repeatable: dotted.json.path=jsonValue. Values JSON-decoded.")
    ap.add_argument("--host", default=None,
                    help="If set, run via ssh on this host (e.g. host-10-240-99-118)")
    ap.add_argument("--keep-eval", action="store_true",
                    help="Don't suppress i_eval (default: suppressed since profile is short)")
    ap.add_argument("--keep-save", action="store_true",
                    help="Don't suppress i_save (default: suppressed)")
    ap.add_argument("--keep-wandb", action="store_true",
                    help="Don't strip wandb_project (default: stripped to avoid polluting wandb runs)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the command + tmp config but don't execute.")
    ap.add_argument("--output-root", type=pathlib.Path,
                    default=REPO / "results" / "profile_dit_runs",
                    help="Where the per-run output_dir is created")
    ap.add_argument("--result-dir", type=pathlib.Path,
                    default=REPO / "logs" / "profile_results")
    args = ap.parse_args()

    # 1. Load + clone base config.
    cfg = json.loads(args.base_config.read_text())

    # 2. Apply user overrides.
    for s in args.override:
        k, v = parse_override(s)
        deep_set(cfg, k, v)

    # 3. Force profile-friendly knobs.
    total_steps = args.warmup_steps + args.active_steps
    cfg["trainer"]["args"]["max_steps"] = total_steps
    cfg["trainer"]["args"]["i_log"] = 1
    cfg["trainer"]["args"]["i_print"] = max(args.warmup_steps // 2, 5)
    if not args.keep_save:
        cfg["trainer"]["args"]["i_save"] = total_steps + 1  # never trigger
    if not args.keep_eval:
        cfg["trainer"]["args"]["i_eval"] = total_steps + 1
    if not args.keep_wandb:
        cfg["trainer"]["args"]["wandb_project"] = None

    # 4. Persist tmp config + create output dir.
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.label}_{ts}"
    tmp_dir = REPO / "scripts" / "profiling" / ".tmp_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg = tmp_dir / f"{run_id}.json"
    tmp_cfg.write_text(json.dumps(cfg, indent=2))
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = REPO / "logs" / f"profile_run_{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 5. Build cmd.
    cmd = build_cmd(
        config_path=tmp_cfg,
        output_dir=output_dir,
        load_dir=args.load_dir,
        ckpt=args.ckpt,
        num_gpus=args.num_gpus,
        extra_argv=[],
    )
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO))
    env["COART_AUTO_REGISTER_DIT"] = "1"
    env["TRITON_CACHE_DIR"] = env.get("TRITON_CACHE_DIR", "/tmp/trellis2_triton_cache")
    env["PYTHONFAULTHANDLER"] = "1"

    if args.host:
        # Pack command as a remote bash invocation. We chain `cd` + env vars so
        # the relative paths in the entry script (it does its own cd via repo
        # root resolution) work uniformly.
        env_str = " ".join(f"{k}={v!r}" for k, v in [
            ("PYTHONPATH", env["PYTHONPATH"]),
            ("COART_AUTO_REGISTER_DIT", env["COART_AUTO_REGISTER_DIT"]),
            ("TRITON_CACHE_DIR", env["TRITON_CACHE_DIR"]),
            ("PYTHONFAULTHANDLER", env["PYTHONFAULTHANDLER"]),
        ])
        remote_cmd = f"cd {REPO} && env {env_str} " + " ".join(_shquote(c) for c in cmd)
        full = ["ssh", args.host, remote_cmd]
    else:
        full = cmd

    print(f"[profile_dit] label        = {args.label}")
    print(f"[profile_dit] tmp_config   = {tmp_cfg}")
    print(f"[profile_dit] output_dir   = {output_dir}")
    print(f"[profile_dit] log_path     = {log_path}")
    print(f"[profile_dit] num_gpus     = {args.num_gpus}")
    print(f"[profile_dit] warmup/active= {args.warmup_steps}/{args.active_steps} (total={total_steps})")
    print(f"[profile_dit] host         = {args.host or '<local>'}")
    print(f"[profile_dit] cmd:")
    print("  " + " ".join(_shquote(c) for c in full))

    if args.dry_run:
        print("[profile_dit] dry-run: not launching.")
        return

    # 6. Execute, mirror stdout to log.
    t0 = time.time()
    with log_path.open("w") as logfp:
        proc = subprocess.run(full, env=env, stdout=logfp, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    print(f"[profile_dit] subprocess exit={proc.returncode} elapsed={elapsed:.1f}s")

    if proc.returncode != 0:
        print(f"[profile_dit] FAILED — see {log_path} for full output")
        # Still try to parse partial log; tail of last failed run useful.

    # 7. Parse log.txt → result.
    log_txt = output_dir / "log.txt"
    if not log_txt.exists():
        print(f"[profile_dit] err: no log.txt at {log_txt}; can't summarise")
        sys.exit(2)
    rows = parse_log(log_txt)
    summary = summarise(rows, warmup_steps=args.warmup_steps, tail_steps=None)

    args.result_dir.mkdir(parents=True, exist_ok=True)
    out = args.result_dir / f"{run_id}.json"
    payload = {
        "label": args.label,
        "kind": "profile_run",
        "run_id": run_id,
        "host": args.host or "<local>",
        "num_gpus": args.num_gpus,
        "wallclock_s": elapsed,
        "subprocess_exit": proc.returncode,
        "warmup_steps": args.warmup_steps,
        "active_steps": args.active_steps,
        "tmp_config_path": str(tmp_cfg),
        "output_dir": str(output_dir),
        "stdout_log": str(log_path),
        "config": cfg,
        "overrides": args.override,
        "summary": summary,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"[profile_dit] wrote result: {out}")

    if "time/step" in summary["metrics"]:
        m = summary["metrics"]["time/step"]
        thpt_h = 3600.0 / m["mean"] if m["mean"] > 0 else float("nan")
        print(f"  step_time mean={m['mean']:.3f}s p95={m['p95']:.3f}s -> {thpt_h:.1f} steps/h")
    if "perf/mem_peak_gb" in summary["metrics"]:
        m = summary["metrics"]["perf/mem_peak_gb"]
        print(f"  mem_peak  mean={m['mean']:.1f}GB p95={m['p95']:.1f}GB")


def _shquote(s: str) -> str:
    """Minimal shell quote (handles spaces + single quotes)."""
    if not s or any(ch in s for ch in " \t\n\"'$\\&|;<>(){}*?#~"):
        return "'" + s.replace("'", "'\\''") + "'"
    return s


if __name__ == "__main__":
    main()
