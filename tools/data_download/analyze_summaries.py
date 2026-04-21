#!/usr/bin/env python3
"""Collect rank_*_summary.json files and print a comparison table.

Usage:
    python tools/data_download/analyze_summaries.py --tag tune-p
    python tools/data_download/analyze_summaries.py --tag scale-n2
    python tools/data_download/analyze_summaries.py --tag full
"""
import argparse
import glob
import json
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab')
    parser.add_argument('--tag', required=True,
                        help='substring to filter summary filenames')
    args = parser.parse_args()

    pattern = os.path.join(args.root, 'logs', 'download', f'rank*_summary.json')
    rows = []
    for p in sorted(glob.glob(pattern)):
        if args.tag not in p:
            continue
        with open(p) as f:
            rows.append(json.load(f))

    if not rows:
        print(f"No summaries match tag={args.tag!r} under {pattern}")
        return

    df = pd.DataFrame(rows)
    cols = ['tag', 'host', 'rank', 'processes', 'n_success', 'n_failed',
            'download_sec', 'rate_obj_per_sec', 'mb_per_sec', 'mb_per_obj',
            'annotations_load_sec']
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values(['processes', 'rank'])
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', None)
    print(df.to_string(index=False))

    if 'processes' in df.columns and df['processes'].nunique() > 1:
        agg = (df.groupby('processes')
                 .agg(n_runs=('rank', 'count'),
                      total_success=('n_success', 'sum'),
                      total_mb=('mb_per_sec', 'sum'),
                      mean_rate=('rate_obj_per_sec', 'mean'),
                      mean_mb_per_sec=('mb_per_sec', 'mean'))
                 .reset_index())
        print("\n=== Aggregated by processes ===")
        print(agg.to_string(index=False))


if __name__ == '__main__':
    main()
