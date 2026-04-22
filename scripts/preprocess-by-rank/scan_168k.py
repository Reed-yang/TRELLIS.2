"""
Fast CPU-only GLB header scan for all 168k ObjaverseXL Sketchfab assets.

Extracts lightweight complexity features without running trimesh.load:
  - file_mb          : on-disk size
  - n_meshes         : glTF meshes count
  - n_primitives     : sum of primitives across all meshes
  - n_faces          : sum of (indices count / 3) across primitives; falls
                       back to POSITION count / 3 for non-indexed geometry
  - n_vertices       : sum of POSITION accessor counts across primitives
  - n_nodes          : scene nodes (proxy for instancing complexity)
  - parse_error      : non-empty string iff the GLB header could not be read

Input:  merged_records/*.csv  (sha256, local_path columns; 168k unique sha)
Output: out/scan_168k.parquet

Usage:
    python scan_168k.py                          # 168k, all CPUs - 4
    python scan_168k.py --limit 500              # smoke test
    python scan_168k.py --workers 32             # cap workers
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import struct
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm


DATASET_ROOT = "/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab"
MERGED_DIR = f"{DATASET_ROOT}/raw/merged_records"


def _peek_glb(path: str) -> dict:
    """Parse just the JSON chunk of a GLB file; no buffer data read.

    Returns dict with keys file_mb, n_meshes, n_primitives, n_faces,
    n_vertices, n_nodes, parse_error.
    """
    out = {
        "file_mb": 0.0,
        "n_meshes": 0,
        "n_primitives": 0,
        "n_faces": 0,
        "n_vertices": 0,
        "n_nodes": 0,
        "parse_error": "",
    }
    try:
        size = os.path.getsize(path)
        out["file_mb"] = size / (1024.0 * 1024.0)
    except OSError as e:
        out["parse_error"] = f"stat:{e}"
        return out

    try:
        with open(path, "rb") as f:
            # 12-byte GLB header: magic (4) + version (4) + total_length (4)
            header = f.read(12)
            if len(header) < 12 or header[:4] != b"glTF":
                out["parse_error"] = "not_glb"
                return out
            # First chunk: length (4) + type (4) + data (length bytes)
            clen_bytes = f.read(4)
            ctype = f.read(4)
            if len(clen_bytes) < 4 or len(ctype) < 4 or ctype != b"JSON":
                out["parse_error"] = "no_json_chunk"
                return out
            clen = struct.unpack("<I", clen_bytes)[0]
            # Cap at 64 MiB to avoid loading pathological GLBs; if JSON is
            # bigger than 64 MiB the scan is useless anyway.
            if clen > 64 * 1024 * 1024:
                out["parse_error"] = f"json_too_large:{clen}"
                return out
            data = f.read(clen)
            if len(data) < clen:
                out["parse_error"] = "truncated_json"
                return out
    except OSError as e:
        out["parse_error"] = f"open:{e}"
        return out

    try:
        # JSON chunk may be padded with 0x20 to 4-byte boundary.
        gltf = json.loads(data.rstrip(b"\x00 "))
    except json.JSONDecodeError as e:
        out["parse_error"] = f"json_decode:{e.msg[:40]}"
        return out

    accessors = gltf.get("accessors") or []
    meshes = gltf.get("meshes") or []
    out["n_meshes"] = len(meshes)
    out["n_nodes"] = len(gltf.get("nodes") or [])

    n_prims = 0
    n_faces = 0
    n_vertices = 0
    for m in meshes:
        prims = m.get("primitives") or []
        n_prims += len(prims)
        for p in prims:
            attrs = p.get("attributes") or {}
            pos_idx = attrs.get("POSITION")
            pos_count = 0
            if pos_idx is not None and 0 <= pos_idx < len(accessors):
                pos_count = int(accessors[pos_idx].get("count") or 0)
            n_vertices += pos_count

            idx = p.get("indices")
            if idx is not None and 0 <= idx < len(accessors):
                tri_count = int(accessors[idx].get("count") or 0) // 3
            else:
                # Non-indexed geometry: assume triangles
                tri_count = pos_count // 3
            n_faces += tri_count
    out["n_primitives"] = n_prims
    out["n_faces"] = n_faces
    out["n_vertices"] = n_vertices
    return out


def _worker(args):
    sha, rel_path = args
    abs_path = os.path.join(DATASET_ROOT, rel_path)
    stats = _peek_glb(abs_path)
    stats["sha256"] = sha
    stats["local_path"] = rel_path
    return stats


def _load_sha_to_path() -> list[tuple[str, str]]:
    """Merge all merged_records CSVs into unique (sha256, local_path) list."""
    dfs = []
    for fp in sorted(Path(MERGED_DIR).glob("*.csv")):
        try:
            df = pd.read_csv(fp, usecols=["sha256", "local_path"], dtype=str)
        except (pd.errors.EmptyDataError, ValueError):
            continue
        if len(df) == 0:
            continue
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    # Keep first observed path per sha256 (arbitrary but stable after sort)
    all_df = all_df.drop_duplicates(subset="sha256", keep="first")
    return list(zip(all_df["sha256"].tolist(), all_df["local_path"].tolist()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="scripts/preprocess-by-rank/out/scan_168k.parquet")
    p.add_argument("--limit", type=int, default=None,
                   help="Only scan the first N entries (smoke test)")
    p.add_argument("--workers", type=int, default=None,
                   help="Process pool size (default cpu_count - 4)")
    p.add_argument(
        "--resume", action="store_true",
        help="Skip sha already present in --out; append new rows only.",
    )
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"[scan] loading sha→path mapping from {MERGED_DIR} ...", flush=True)
    work = _load_sha_to_path()
    print(f"[scan] total unique sha256: {len(work)}", flush=True)

    if args.resume and os.path.exists(args.out):
        done = pd.read_parquet(args.out)
        done_set = set(done["sha256"].tolist())
        print(f"[scan] resume: {len(done_set)} already done, skipping.", flush=True)
        work = [t for t in work if t[0] not in done_set]
        print(f"[scan] remaining: {len(work)}", flush=True)

    if args.limit is not None:
        work = work[: args.limit]
        print(f"[scan] --limit {args.limit}: processing {len(work)}", flush=True)

    n_workers = args.workers or max(1, (os.cpu_count() or 4) - 4)
    print(f"[scan] launching {n_workers} workers ...", flush=True)

    results = []
    t0 = time.time()
    chunksize = max(16, len(work) // (n_workers * 64) or 16)
    with mp.Pool(n_workers) as pool:
        for rec in tqdm(
            pool.imap_unordered(_worker, work, chunksize=chunksize),
            total=len(work),
            desc="scan",
            dynamic_ncols=True,
        ):
            results.append(rec)

    df = pd.DataFrame(results)
    cols = ["sha256", "local_path", "file_mb", "n_meshes", "n_primitives",
            "n_faces", "n_vertices", "n_nodes", "parse_error"]
    df = df[cols]

    if args.resume and os.path.exists(args.out):
        old = pd.read_parquet(args.out)
        df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset="sha256", keep="last")

    df.to_parquet(args.out, index=False)
    dt = time.time() - t0
    print(f"[scan] wrote {len(df)} rows to {args.out} in {dt:.1f}s "
          f"({len(work) / max(dt, 1e-6):.1f}/s)", flush=True)
    bad = df[df["parse_error"] != ""]
    print(f"[scan] parse_error count: {len(bad)}", flush=True)
    if len(bad):
        print(bad["parse_error"].value_counts().head(10).to_string(), flush=True)


if __name__ == "__main__":
    main()
