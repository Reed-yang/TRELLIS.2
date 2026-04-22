#!/usr/bin/env bash
# Local smoke test for the OOM validation runner — no GPU required.
# Runs worker.py --dry_run sequentially for all 4 labels, then aggregates.
# Confirms argparse, JSON schema, and aggregator markdown are well-formed.
set -uo pipefail
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
export PYTHONPATH=.

RESULTS=scripts/oom_validation/results_smoke
rm -rf "$RESULTS"; mkdir -p "$RESULTS"

declare -a CASES=(
  "easy-40GiB:raw/hf-objaverse-v1/glbs/000-054/66130e3034dd49eba8680f24cdee6b70.glb"
  "median-90GiB:raw/hf-objaverse-v1/glbs/000-045/bec4ddb829fb4a4db9726d8c907e297e.glb"
  "medium-400GiB:raw/hf-objaverse-v1/glbs/000-121/8a8194eaec15434c9544c1c7d0aed663.glb"
  "hard-1800GiB:raw/hf-objaverse-v1/glbs/000-085/74198161b7c2409da7ed45402a56d883.glb"
)

for entry in "${CASES[@]}"; do
  IFS=':' read -r label rel_path <<< "$entry"
  .venv/bin/python -u scripts/oom_validation/worker.py \
    --label "$label" \
    --glb_path "datasets/ObjaverseXL_sketchfab/$rel_path" \
    --device "cpu" \
    --out_json "$RESULTS/${label}.json" \
    --dry_run
done

.venv/bin/python scripts/oom_validation/aggregate.py --results_dir "$RESULTS"
