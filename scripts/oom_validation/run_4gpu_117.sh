#!/usr/bin/env bash
# 4-GPU parallel OOM validation. Runs easy/median/medium/hard meshes on
# GPU 0/1/2/3 simultaneously. Expects the OOM fallback impl (chunked
# Phase A dispatcher in s4_face_point.py) to be committed on throughput-168k.
#
# Each inner process sets CUDA_VISIBLE_DEVICES=<rank> and sees its physical
# GPU as cuda:0 — keeps the worker GPU-agnostic.
#
# Usage:
#   bash scripts/oom_validation/run_4gpu_117.sh            # real run
#   bash scripts/oom_validation/run_4gpu_117.sh --dry_run  # stub JSONs only

set -uo pipefail
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
export PYTHONPATH=.

DRY_FLAG=""
if [[ "${1:-}" == "--dry_run" ]]; then
    DRY_FLAG="--dry_run"
    echo "[host] dry_run mode — skipping actual encodes"
fi

RESULTS=scripts/oom_validation/results
rm -rf "$RESULTS"; mkdir -p "$RESULTS"

# Confirm production-flag state (only meaningful outside --dry_run)
if [[ -z "$DRY_FLAG" ]]; then
    .venv/bin/python -c "import corep_fast.config as c; print(f'VRAM_RESCUE={c.VRAM_RESCUE} S2_SPARSE={c.S2_SPARSE} ASYNC_D2H={c.ASYNC_D2H}')"
fi

declare -a CASES=(
  "0:easy-40GiB:raw/hf-objaverse-v1/glbs/000-054/66130e3034dd49eba8680f24cdee6b70.glb"
  "1:median-90GiB:raw/hf-objaverse-v1/glbs/000-045/bec4ddb829fb4a4db9726d8c907e297e.glb"
  "2:medium-400GiB:raw/hf-objaverse-v1/glbs/000-121/8a8194eaec15434c9544c1c7d0aed663.glb"
  "3:hard-1800GiB:raw/hf-objaverse-v1/glbs/000-085/74198161b7c2409da7ed45402a56d883.glb"
)

pids=()
for entry in "${CASES[@]}"; do
  IFS=':' read -r gpu label rel_path <<< "$entry"
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    # Wall-clock timeout generous enough for CPU fallback (~60 min for hard).
    timeout 3600s .venv/bin/python -u scripts/oom_validation/worker.py \
      --label "$label" \
      --glb_path "datasets/ObjaverseXL_sketchfab/$rel_path" \
      --device "cuda:0" \
      --out_json "$RESULTS/${label}.json" \
      $DRY_FLAG \
      > "$RESULTS/${label}.log" 2>&1 || echo "[host] rank $gpu ($label) exited $?"
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || true; done

# Aggregate
.venv/bin/python scripts/oom_validation/aggregate.py --results_dir "$RESULTS"
