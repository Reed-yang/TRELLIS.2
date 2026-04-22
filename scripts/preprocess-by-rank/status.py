"""
Any-time progress snapshot for the full-rank precompute run.

Safe to run while jobs are live. Does no writes; only reads:
  - OUT_DIR/data/*.npz               → already-encoded
  - OUT_DIR/data/*.npz.failed        → failure sentinels
  - OUT_DIR/failed_rank*.txt         → canonical failure log
  - LOG_DIR/node*local*global*.log   → per-rank live logs (any run_stamp)
  - full_ranked.csv                  → authoritative work-list with tier info

Outputs:
  - human-readable table on stdout
  - (optional) --emit_csv per_sha status snapshot for downstream tools

Usage:
  python scripts/preprocess-by-rank/status.py
  python scripts/preprocess-by-rank/status.py --watch 30      # refresh every 30s
  python scripts/preprocess-by-rank/status.py --emit_csv /tmp/per_sha.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from collections import defaultdict

import pandas as pd


DEFAULT_OUT_DIR = "/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512"
DEFAULT_RANKED_CSV = "/mnt/novita2/siyuan/workspace/TRELLIS.2/scripts/preprocess-by-rank/out/full_ranked.csv"


_OK_LINE_RE = re.compile(r"\[ ok  \]\s+([0-9a-f]{16})\s+voxels=\s*(\d+)\s+enc=\s*([\d.]+)s")
_ENCODE_RE = re.compile(r"\[encode\]\s+([0-9a-f]{16})\s+([\d.]+)\s*MB\s+(\S+)")


def _scan_out_dir(out_dir: str) -> tuple[set[str], set[str]]:
    """Return (set of done sha, set of failed-sentinel sha)."""
    data_dir = os.path.join(out_dir, "data")
    if not os.path.isdir(data_dir):
        return set(), set()
    done = set()
    failed = set()
    # os.scandir is ~3x faster than glob for large dirs
    with os.scandir(data_dir) as it:
        for e in it:
            name = e.name
            if name.endswith(".npz.failed"):
                failed.add(name[:-len(".npz.failed")])
            elif name.endswith(".npz"):
                done.add(name[:-len(".npz")])
    return done, failed


def _scan_failed_txt(out_dir: str) -> dict[str, str]:
    """Map sha256 -> last error message from failed_rank*.txt files."""
    out: dict[str, str] = {}
    for fp in sorted(glob.glob(os.path.join(out_dir, "failed_rank*.txt"))):
        try:
            with open(fp, "r", errors="replace") as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t", 1)
                    if len(parts) == 2 and len(parts[0]) == 64:
                        out[parts[0]] = parts[1][:180]
        except OSError:
            continue
    return out


def _scan_rank_logs(log_dir: str, path_to_sha: dict[str, str]) -> tuple[list[float], int, str | None]:
    """Walk all node*local*global*.log files under LOG_DIR.

    Returns (all ok enc_time_s collected, total ok-line count, newest_mtime_iso).
    Used to display an instantaneous throughput estimate + per-rank liveness.
    """
    enc_times: list[float] = []
    n_ok = 0
    latest = 0.0
    if not os.path.isdir(log_dir):
        return enc_times, n_ok, None
    for fp in glob.glob(os.path.join(log_dir, "node*local*global*.log")):
        try:
            mt = os.path.getmtime(fp)
            if mt > latest:
                latest = mt
            with open(fp, "r", errors="replace") as fh:
                for line in fh:
                    for ln in line.split("\r"):
                        m = _OK_LINE_RE.search(ln)
                        if m:
                            enc_times.append(float(m.group(3)))
                            n_ok += 1
        except OSError:
            continue
    iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(latest)) if latest else None
    return enc_times, n_ok, iso


def _load_ranked(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=[
        "sha256", "tier", "rank", "pred_enc_s", "pred_oom_prob"
    ])
    return df


def _one_snapshot(args):
    out_dir = args.out_dir
    ranked_path = args.ranked_csv

    ranked = _load_ranked(ranked_path)
    done, failed_sentinel = _scan_out_dir(out_dir)
    failed_txt = _scan_failed_txt(out_dir)
    enc_times, n_ok_log, latest_log_time = _scan_rank_logs(args.log_dir, {})

    ranked["status"] = "pending"
    ranked.loc[ranked["sha256"].isin(done), "status"] = "done"
    # sentinel ∪ failed_txt: the union is the set of definitively-failed sha
    failed_all = failed_sentinel | set(failed_txt.keys())
    # A sha might be both done (npz exists) and previously failed (sentinel
    # left over from earlier attempt). done wins.
    pending_mask = (ranked["status"] == "pending") & (ranked["sha256"].isin(failed_all))
    ranked.loc[pending_mask, "status"] = "failed"

    # Per-tier breakdown
    breakdown = ranked.groupby("tier")["status"].value_counts().unstack(fill_value=0)
    for col in ("done", "failed", "pending"):
        if col not in breakdown.columns:
            breakdown[col] = 0
    breakdown["total"] = breakdown[["done", "failed", "pending"]].sum(axis=1)
    breakdown["done_pct"] = (breakdown["done"] / breakdown["total"] * 100).round(1)
    breakdown = breakdown[["done", "failed", "pending", "total", "done_pct"]]

    # Overall counters
    total = len(ranked)
    n_done = int((ranked["status"] == "done").sum())
    n_fail = int((ranked["status"] == "failed").sum())
    n_pending = int((ranked["status"] == "pending").sum())

    # Throughput / ETA from per-rank log enc_times (median-of-last-N)
    recent = sorted(enc_times, reverse=False)[-1000:]  # last 1k ok encodes seen
    if recent:
        median_enc = sorted(recent)[len(recent) // 2]
        mean_enc = sum(recent) / len(recent)
    else:
        median_enc = mean_enc = 0.0

    # ETA for pending based on predicted enc sum
    pending_df = ranked[ranked["status"] == "pending"].copy()
    pending_pred_sum = float(pending_df["pred_enc_s"].sum())
    eta_h = pending_pred_sum / max(args.n_gpus, 1) / 3600.0

    print("=" * 72)
    print(f"[status] snapshot at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[status] out_dir     = {out_dir}")
    print(f"[status] ranked_csv  = {ranked_path}")
    print(f"[status] log_dir     = {args.log_dir}")
    print(f"[status] latest log  = {latest_log_time or 'none'}")
    print("-" * 72)
    print(f"  total:    {total:>7d}")
    print(f"  done:     {n_done:>7d}   ({n_done/total*100:>5.1f}%)")
    print(f"  failed:   {n_fail:>7d}   ({n_fail/total*100:>5.1f}%)")
    print(f"  pending:  {n_pending:>7d}   ({n_pending/total*100:>5.1f}%)")
    print("-" * 72)
    print("Per-tier breakdown:")
    print(breakdown.to_string())
    print("-" * 72)
    print(f"Observed enc-time (last {len(recent)} ok encodes): "
          f"median={median_enc:.1f}s  mean={mean_enc:.1f}s")
    print(f"Pending predicted enc-seconds total: {pending_pred_sum:.0f}s")
    print(f"Pending ETA @ {args.n_gpus} GPUs: {eta_h:.2f} h")
    # Show top-N pending first rows (hardest samples that haven't run yet)
    if args.show_next and n_pending > 0:
        head = pending_df.sort_values("rank").head(args.show_next)
        print("-" * 72)
        print(f"Next {args.show_next} pending by rank:")
        print(head[["rank", "tier", "pred_enc_s", "pred_oom_prob", "sha256"]].to_string(index=False))
    print("=" * 72, flush=True)

    if args.emit_csv:
        ranked[["sha256", "rank", "tier", "status"]].to_csv(args.emit_csv, index=False)
        print(f"[status] wrote per-sha snapshot to {args.emit_csv}", flush=True)

    return {"total": total, "done": n_done, "failed": n_fail, "pending": n_pending,
            "eta_h": eta_h}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR,
                   help="feat18_<R> output dir to scan")
    p.add_argument("--ranked_csv", default=DEFAULT_RANKED_CSV,
                   help="full_ranked.csv produced by fit_and_rank.py")
    p.add_argument("--log_dir",
                   default=None,
                   help="per-rank log dir (defaults to <out_dir>/../logs/precompute_feat18_<R>_full_*)")
    p.add_argument("--n_gpus", type=int, default=8, help="for ETA math")
    p.add_argument("--show_next", type=int, default=0,
                   help="also print the next N pending rows by rank")
    p.add_argument("--emit_csv", default=None,
                   help="also write per-sha (rank, tier, status) CSV")
    p.add_argument("--watch", type=int, default=0,
                   help="refresh every N seconds; 0 = one shot")
    args = p.parse_args()

    # Auto-discover log_dir if unset
    if args.log_dir is None:
        candidates = sorted(glob.glob(
            os.path.join(args.out_dir, "..", "logs", "precompute_feat18_*full*")
        ))
        if candidates:
            args.log_dir = candidates[-1]
        else:
            args.log_dir = "/tmp/noexist"
        print(f"[status] log_dir auto = {args.log_dir}", file=sys.stderr)

    if args.watch <= 0:
        _one_snapshot(args)
        return

    try:
        while True:
            _one_snapshot(args)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("[status] interrupted; exiting", flush=True)


if __name__ == "__main__":
    main()
