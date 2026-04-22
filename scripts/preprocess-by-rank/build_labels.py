"""
Extract (sha256, enc_time_s, status) labels from the existing precompute run.

Sources:
  - logs/precompute_feat18_r512_g8/rank*.log — [encode]/[ ok  ] line pairs
    (sha16 + local_path from [encode], enc_time from paired [ ok  ])
  - feat18_512/failed_rank*.txt                — full sha256 + OOM/other error
  - feat18_512/data/*.npz                      — set of successful full sha256

The enc_time logs use only the first 16 chars of the sha256; we resolve
back to the full sha by joining on local_path against scan_168k.parquet.

Output: out/labels.parquet with columns:
    sha256, status (ok|oom|fail|none), enc_time_s (nullable), err (str)
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm


LOG_DIR = "/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/logs/precompute_feat18_r512_g8"
FEAT_DIR = "/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512"

# Sample: "[encode] 0002f8675afaa4c6    0.8 MB  raw/hf-objaverse-v1/glbs/000-078/a1307a74fb2a4673af99ad9f2fbfbb7c.glb"
_ENCODE_RE = re.compile(
    r"\[encode\]\s+([0-9a-f]{16})\s+([\d.]+)\s*MB\s+(\S+)"
)
# Sample: "[ ok  ] 0f78c989fffb0545 voxels=1653053  enc=  20.0s"
_OK_RE = re.compile(
    r"\[ ok  \]\s+([0-9a-f]{16})\s+voxels=\s*(\d+)\s+enc=\s*([\d.]+)s"
)
# Sample: "[oom ] 000a7093f9a25952f8f9c5e148d75d2158d2421e3b3e95173a7022e0cdd146da CUDA out ..."
_OOM_RE = re.compile(r"\[oom \]\s+([0-9a-f]{64})\s+(.*)")
_FAIL_RE = re.compile(r"\[fail\]\s+([0-9a-f]{64})\s+(.*)")


def _parse_rank_log(path: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (ok_rows, oom_rows, fail_rows).

    ok_rows:  {sha16, local_path, file_mb, enc_time_s}
    oom_rows: {sha256, err}
    fail_rows: {sha256, err}

    Log lines may arrive as rendered tqdm output (single physical line
    contains carriage-returns and embedded tqdm bars). The regexes look
    for the bracket markers, so tqdm clutter is silently skipped.
    """
    ok_rows: list[dict] = []
    oom_rows: list[dict] = []
    fail_rows: list[dict] = []

    pending_encode: dict | None = None
    try:
        fh = open(path, "r", errors="replace")
    except OSError:
        return ok_rows, oom_rows, fail_rows

    with fh:
        for raw in fh:
            # Split on both \n and \r so we see each logical line.
            for line in raw.split("\r"):
                m = _ENCODE_RE.search(line)
                if m:
                    pending_encode = {
                        "sha16": m.group(1),
                        "file_mb": float(m.group(2)),
                        "local_path": m.group(3),
                    }
                    continue

                m = _OK_RE.search(line)
                if m and pending_encode is not None and m.group(1) == pending_encode["sha16"]:
                    ok_rows.append({
                        **pending_encode,
                        "enc_time_s": float(m.group(3)),
                    })
                    pending_encode = None
                    continue

                m = _OOM_RE.search(line)
                if m:
                    oom_rows.append({
                        "sha256": m.group(1),
                        "err": m.group(2).strip()[:200],
                    })
                    pending_encode = None
                    continue

                m = _FAIL_RE.search(line)
                if m:
                    fail_rows.append({
                        "sha256": m.group(1),
                        "err": m.group(2).strip()[:200],
                    })
                    pending_encode = None
                    continue

    return ok_rows, oom_rows, fail_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan", default="scripts/preprocess-by-rank/out/scan_168k.parquet")
    p.add_argument("--out", default="scripts/preprocess-by-rank/out/labels.parquet")
    args = p.parse_args()

    scan = pd.read_parquet(args.scan)[["sha256", "local_path"]]
    # path-keyed lookup for sha16 -> sha256 via local_path match
    scan["sha16"] = scan["sha256"].str.slice(0, 16)
    path_to_sha = dict(zip(scan["local_path"], scan["sha256"]))

    print(f"[labels] parsing rank logs in {LOG_DIR} ...", flush=True)
    all_ok: list[dict] = []
    all_oom: list[dict] = []
    all_fail: list[dict] = []
    log_paths = sorted(Path(LOG_DIR).glob("rank*.log"))
    for lp in tqdm(log_paths, desc="rank log"):
        ok, oom, fail = _parse_rank_log(str(lp))
        all_ok.extend(ok)
        all_oom.extend(oom)
        all_fail.extend(fail)

    print(f"[labels] parsed ok={len(all_ok)}  oom={len(all_oom)}  fail={len(all_fail)}",
          flush=True)

    # Resolve sha16 -> sha256 for ok rows via local_path
    ok_df = pd.DataFrame(all_ok)
    if len(ok_df):
        ok_df["sha256"] = ok_df["local_path"].map(path_to_sha)
        missed = ok_df["sha256"].isna().sum()
        if missed:
            print(f"[labels] WARN: {missed}/{len(ok_df)} ok rows failed path lookup; dropping", flush=True)
            ok_df = ok_df.dropna(subset=["sha256"])
        ok_df = ok_df[["sha256", "enc_time_s", "file_mb"]].copy()
        ok_df["status"] = "ok"
        ok_df["err"] = ""
    else:
        ok_df = pd.DataFrame(columns=["sha256", "enc_time_s", "file_mb", "status", "err"])

    oom_df = pd.DataFrame(all_oom) if all_oom else pd.DataFrame(columns=["sha256", "err"])
    fail_df = pd.DataFrame(all_fail) if all_fail else pd.DataFrame(columns=["sha256", "err"])
    if len(oom_df):
        oom_df["status"] = "oom"
        oom_df["enc_time_s"] = pd.NA
        oom_df["file_mb"] = pd.NA
    if len(fail_df):
        fail_df["status"] = "fail"
        fail_df["enc_time_s"] = pd.NA
        fail_df["file_mb"] = pd.NA

    cols = ["sha256", "status", "enc_time_s", "file_mb", "err"]
    labels = pd.concat([ok_df[cols], oom_df[cols], fail_df[cols]], ignore_index=True)

    # Also ingest failed_rank*.txt as a second oom/fail source (these are
    # written synchronously so they may contain entries the rank log did
    # not flush before a crash).
    failed_txt = []
    for fp in sorted(Path(FEAT_DIR).glob("failed_rank*.txt")):
        with open(fp, "r", errors="replace") as fh:
            for line in fh:
                parts = line.strip().split("\t", 1)
                if len(parts) != 2 or len(parts[0]) != 64:
                    continue
                sha, msg = parts
                status = "oom" if msg.startswith("OOM") else "fail"
                failed_txt.append({
                    "sha256": sha, "status": status,
                    "enc_time_s": pd.NA, "file_mb": pd.NA,
                    "err": msg[:200],
                })
    if failed_txt:
        ft = pd.DataFrame(failed_txt)
        # If a sha already exists with status ok, keep ok; else merge
        existing_ok = set(labels[labels["status"] == "ok"]["sha256"])
        ft = ft[~ft["sha256"].isin(existing_ok)]
        labels = pd.concat([labels, ft], ignore_index=True)
        labels = labels.drop_duplicates(subset="sha256", keep="first")

    print(f"[labels] final: {len(labels)} labelled sha256", flush=True)
    print(labels["status"].value_counts().to_string(), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    labels.to_parquet(args.out, index=False)
    print(f"[labels] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
