#!/usr/bin/env python3
"""Slurm-friendly wrapper for Objaverse-XL sketchfab download.

Wraps objaverse.xl.download_objects but adds:
  - --processes tuning (multiprocessing pool size inside oxl)
  - --offset / --limit (scale / tune experiments on a subslice)
  - --tag (label per-rank log/summary/part filenames)
  - Structured summary.json + failures.csv + rank_{N}.log
  - Shared annotations cache (one .parquet for all nodes via NFS)

Resume: oxl.download_objects already skips objects whose target file exists,
and we also filter rows whose local_path is already recorded in raw/metadata.csv.
Re-running the same command is idempotent.

Examples:

    # Tune 500 objs at processes=64, off the 1001 already downloaded
    python tools/data_download/download_sketchfab.py \\
        --root /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab \\
        --processes 64 --world_size 168 --rank 101 --limit 500 --tag tune-p64

    # Full-set shard of 6 nodes
    python tools/data_download/download_sketchfab.py \\
        --root /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab \\
        --processes 96 --world_size 6 --rank 0 --tag full
"""
import argparse
import json
import os
import random
import socket
import sys
import time
import urllib.error
import urllib.request

import pandas as pd


# Inject Authorization: Bearer <HF_TOKEN> on HF URLs before importing objaverse,
# so its urllib.request.urlopen calls become authenticated (higher 429 quota).
try:
    from huggingface_hub import get_token as _hf_get_token
    _HF_TOKEN = os.environ.get('HF_TOKEN') or _hf_get_token()
except Exception:
    _HF_TOKEN = os.environ.get('HF_TOKEN')

if _HF_TOKEN:
    os.environ['HF_TOKEN'] = _HF_TOKEN  # propagates into forked children
    _orig_urlopen = urllib.request.urlopen

    def _auth_urlopen(url, *args, **kwargs):
        if isinstance(url, str) and 'huggingface.co' in url:
            url = urllib.request.Request(
                url, headers={'Authorization': f'Bearer {_HF_TOKEN}'})
        elif isinstance(url, urllib.request.Request) \
                and 'huggingface.co' in url.full_url \
                and not url.has_header('Authorization'):
            url.add_header('Authorization', f'Bearer {_HF_TOKEN}')
        return _orig_urlopen(url, *args, **kwargs)

    urllib.request.urlopen = _auth_urlopen


import objaverse.xl as oxl
from objaverse.xl.sketchfab import SketchfabDownloader as _SFDownloader
from loguru import logger


DEFAULT_ANNOTATIONS_CACHE = '/mnt/novita2/data/video_obj/.objaverse_cache'


# Guard against multiprocessing.pool.MaybeEncodingError: urllib.HTTPError
# carries a non-pickleable BufferedReader, so when a worker fails the
# exception can't be shipped back through the Pool IPC and the whole
# Pool crashes (taking every in-flight download with it). Wrap the worker
# so failures become (file_identifier, None) — Pool stays alive, callers
# skip None paths and move on.
_orig_download_object = _SFDownloader.__dict__['_download_object'].__func__


_MAX_RETRIES = 5


def _safe_download_object(cls, file_identifier, *rest, **kwargs):
    """Retry HF CDN 429 / transient network errors; swallow fatal ones.

    Returns (file_identifier, local_path) on success or (file_identifier, None)
    on final failure — never raises into the Pool.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            return _orig_download_object(cls, file_identifier, *rest, **kwargs)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < _MAX_RETRIES - 1:
                # Exponential backoff with jitter, starting at ~2s.
                time.sleep(2 ** attempt + random.random())
                continue
            return file_identifier, None
        except urllib.error.URLError:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(1 + random.random())
                continue
            return file_identifier, None
        except Exception:
            return file_identifier, None
    return file_identifier, None


_SFDownloader._download_object = classmethod(_safe_download_object)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    parser.add_argument('--root', required=True,
                        help='Dataset root containing metadata.csv')
    parser.add_argument('--processes', type=int, required=True,
                        help='multiprocessing pool size inside oxl.download_objects')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--offset', type=int, default=0,
                        help='Skip first N objects inside this rank slice')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only download first N objects after offset')
    parser.add_argument('--tag', default='run',
                        help='Label appended to log/summary/part filenames')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='Batch size for each oxl.download_objects call. '
                             'Isolates MaybeEncodingError pool crashes to one chunk.')
    parser.add_argument('--instances', default=None,
                        help='Restrict to this set of sha256s. Accepts a file '
                             '(one sha256 per line OR csv with header) or comma-sep string')
    parser.add_argument('--annotations_cache', default=DEFAULT_ANNOTATIONS_CACHE,
                        help='Shared directory for the oxl annotations parquet')
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    log_dir = os.path.join(root, 'logs', 'download')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(root, 'raw', 'new_records'), exist_ok=True)
    os.makedirs(args.annotations_cache, exist_ok=True)

    tag = args.tag
    label = f"rank{args.rank}_{tag}"
    summary_path = os.path.join(log_dir, f"{label}_summary.json")
    failures_path = os.path.join(log_dir, f"{label}_failures.csv")
    log_file_path = os.path.join(log_dir, f"{label}.log")

    logger.remove()
    logger.add(sys.stdout, level='INFO',
               format='{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}')
    logger.add(log_file_path, level='DEBUG', enqueue=True, rotation=None)

    logger.info(f"args: {vars(args)}")
    logger.info(f"host={socket.gethostname()} cpu_count={os.cpu_count()} pid={os.getpid()}")
    logger.info(f"hf_token_loaded={'yes' if _HF_TOKEN else 'no'}")

    t0 = time.time()

    metadata = pd.read_csv(os.path.join(root, 'metadata.csv')).set_index('sha256')
    raw_meta_path = os.path.join(root, 'raw', 'metadata.csv')
    if os.path.exists(raw_meta_path):
        metadata = metadata.combine_first(
            pd.read_csv(raw_meta_path).set_index('sha256'))
    metadata = metadata.reset_index()
    if 'local_path' in metadata.columns:
        n_before = len(metadata)
        metadata = metadata[metadata['local_path'].isna()]
        logger.info(f"filtered {n_before - len(metadata)} already-downloaded "
                    f"→ {len(metadata)} remaining in metadata")

    if args.instances is not None:
        if os.path.exists(args.instances):
            with open(args.instances) as f:
                raw_lines = [line.strip() for line in f if line.strip()]
            if raw_lines and raw_lines[0].startswith('sha256'):
                raw_lines = raw_lines[1:]
            wanted = {line.split(',')[0] for line in raw_lines}
        else:
            wanted = {s.strip() for s in args.instances.split(',') if s.strip()}
        n_before = len(metadata)
        metadata = metadata[metadata['sha256'].isin(wanted)]
        logger.info(f"--instances filter: {n_before} → {len(metadata)} "
                    f"(wanted {len(wanted)})")

    start_idx = len(metadata) * args.rank // args.world_size
    end_idx = len(metadata) * (args.rank + 1) // args.world_size
    slice_df = metadata.iloc[start_idx:end_idx].copy()
    if args.offset:
        slice_df = slice_df.iloc[args.offset:]
    if args.limit is not None:
        slice_df = slice_df.head(args.limit)
    n_target = len(slice_df)
    logger.info(f"rank {args.rank}/{args.world_size} slice=[{start_idx},{end_idx}) "
                f"offset={args.offset} limit={args.limit} → {n_target} targets")

    t_meta = time.time() - t0

    t_ann_start = time.time()
    annotations = oxl.get_annotations(download_dir=args.annotations_cache)
    annotations = annotations[annotations['sha256'].isin(slice_df['sha256'].values)]
    t_ann = time.time() - t_ann_start
    logger.info(f"annotations load={t_ann:.1f}s, matched {len(annotations)}/{n_target}")

    t_dl_start = time.time()
    file_paths = {}
    n_chunks = (len(annotations) + args.chunk_size - 1) // args.chunk_size
    chunk_crashes = 0
    raw_dir = os.path.join(root, 'raw')
    for ci in range(n_chunks):
        chunk = annotations.iloc[ci * args.chunk_size:(ci + 1) * args.chunk_size]
        try:
            partial = oxl.download_objects(
                chunk,
                download_dir=raw_dir,
                processes=args.processes,
                save_repo_format='zip',
            )
            file_paths.update(partial)
            logger.info(f"chunk {ci + 1}/{n_chunks} ok "
                        f"(+{len(partial)} / target {len(chunk)}; "
                        f"total={len(file_paths)})")
        except Exception as exc:
            chunk_crashes += 1
            logger.warning(f"chunk {ci + 1}/{n_chunks} crashed: "
                           f"{type(exc).__name__}: {exc}")
    t_dl = time.time() - t_dl_start
    logger.info(f"download phase done: {len(file_paths)}/{len(annotations)} "
                f"objects, {chunk_crashes}/{n_chunks} chunks crashed")

    slice_indexed = slice_df.set_index('file_identifier')
    records = []
    downloaded_sha = set()
    total_bytes = 0
    worker_failed = 0
    for fid, local_path in file_paths.items():
        if local_path is None:
            worker_failed += 1
            continue
        sha = slice_indexed.loc[fid, 'sha256']
        rel = os.path.relpath(local_path, root)
        records.append({'sha256': sha, 'local_path': rel})
        downloaded_sha.add(sha)
        try:
            total_bytes += os.path.getsize(local_path)
        except OSError:
            pass
    if worker_failed:
        logger.warning(f"{worker_failed} per-worker download failures "
                       f"(likely HTTP 404 / 5xx / SSL). Will be listed in failures.csv")

    part_path = os.path.join(root, 'raw', 'new_records',
                             f'part_{args.rank}_{tag}.csv')
    pd.DataFrame(records).to_csv(part_path, index=False)

    failed = slice_df[~slice_df['sha256'].isin(downloaded_sha)]
    failed[['sha256', 'file_identifier']].to_csv(failures_path, index=False)

    n_success = len(records)
    n_failed = len(failed)
    rate = n_success / max(t_dl, 1e-6)
    mb_per_obj = (total_bytes / 1e6 / n_success) if n_success else 0
    mb_per_sec = (total_bytes / 1e6 / t_dl) if t_dl > 0 else 0

    summary = {
        'tag': tag,
        'host': socket.gethostname(),
        'rank': args.rank,
        'world_size': args.world_size,
        'processes': args.processes,
        'chunk_size': args.chunk_size,
        'n_chunks': n_chunks,
        'n_chunk_crashes': chunk_crashes,
        'n_worker_failed': worker_failed,
        'offset': args.offset,
        'limit': args.limit,
        'n_target': n_target,
        'n_success': n_success,
        'n_failed': n_failed,
        'metadata_load_sec': round(t_meta, 2),
        'annotations_load_sec': round(t_ann, 2),
        'download_sec': round(t_dl, 2),
        'total_sec': round(time.time() - t0, 2),
        'rate_obj_per_sec': round(rate, 2),
        'total_mb': round(total_bytes / 1e6, 1),
        'mb_per_obj': round(mb_per_obj, 2),
        'mb_per_sec': round(mb_per_sec, 2),
        'part_file': os.path.relpath(part_path, root),
        'failures_file': os.path.relpath(failures_path, root),
        'log_file': os.path.relpath(log_file_path, root),
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info('summary:\n' + json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
