#!/bin/bash
# Re-run M2 baseline on 119 GPU 0 to confirm starting point for pre-Triton work.
# Output: tmp/pretriton_baseline_res128.json + tmp/pretriton_baseline_res256.json (median of 3)
set -euo pipefail
cd /mnt/novita2/siyuan/workspace/TRELLIS.2

export CUDA_VISIBLE_DEVICES=0
echo "=== res=128 (3 runs, take median) ==="
for i in 1 2 3; do
    echo "--- run $i ---"
    .venv/bin/python tmp/e2e_profile_m2.py --res 128 --out tmp/pretriton_baseline_res128_run${i}.json
done
echo "=== res=256 (3 runs, take median) ==="
for i in 1 2 3; do
    echo "--- run $i ---"
    .venv/bin/python tmp/e2e_profile_m2.py --res 256 --out tmp/pretriton_baseline_res256_run${i}.json
done

# Pick the run with the median e2e for each resolution
.venv/bin/python -c "
import json, glob
for res in [128, 256]:
    runs = []
    for p in sorted(glob.glob(f'tmp/pretriton_baseline_res{res}_run*.json')):
        with open(p) as f: runs.append((p, json.load(f)))
    runs.sort(key=lambda x: x[1]['new']['e2e'])
    median_path, median_data = runs[len(runs)//2]
    final_path = f'tmp/pretriton_baseline_res{res}.json'
    with open(final_path, 'w') as f: json.dump(median_data, f, indent=2)
    print(f'{final_path}: e2e={median_data[\"new\"][\"e2e\"]:.2f}s (median, from {median_path})')
"
echo "DONE"
