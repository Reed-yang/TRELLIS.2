#!/bin/bash
# 8-GPU sharded launcher for coart_analyze_finetune.py.
#
# Usage:
#   bash scripts/run_analyze_finetune_8gpu.sh <ckpt_dir> <step> [n_val]
# Defaults:
#   n_val=200
#
# Spawns 8 background python processes (CUDA_VISIBLE_DEVICES=0..7),
# each running --mode shard for its rank. Once all 8 finish, runs
# --mode merge once on GPU 0 to consolidate outputs and write report.md.
set -euo pipefail

CKPT_DIR=${1:?ckpt_dir required}
STEP=${2:?step required}
N_VAL=${3:-200}
W=8

REPO=/mnt/novita2/siyuan/workspace/TRELLIS.2
cd "$REPO"

OUT="${CKPT_DIR}/analysis_step${STEP}"
mkdir -p "${OUT}/shards"

echo "[launcher] starting ${W}-way shard sweep for step=${STEP} n_val=${N_VAL}"

declare -a PIDS
for r in $(seq 0 $((W-1))); do
  CUDA_VISIBLE_DEVICES=$r .venv/bin/python scripts/coart_analyze_finetune.py \
    --ckpt_dir "$CKPT_DIR" --step "$STEP" --use_ema \
    --n_val "$N_VAL" --ablate_assets golden \
    --rank "$r" --world_size "$W" --mode shard \
    > "${OUT}/shards/rank${r}.log" 2>&1 &
  PIDS[$r]=$!
  echo "[launcher] rank ${r} -> pid ${PIDS[$r]} (GPU ${r})"
done

FAIL=0
for r in $(seq 0 $((W-1))); do
  if ! wait "${PIDS[$r]}"; then
    echo "[launcher] rank ${r} (pid ${PIDS[$r]}) failed; see ${OUT}/shards/rank${r}.log"
    FAIL=1
  fi
done

if [ "$FAIL" -ne 0 ]; then
  echo "[launcher] one or more shards failed; aborting before merge"
  exit 1
fi

echo "[launcher] all shards done; running merge on GPU 0"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/coart_analyze_finetune.py \
  --ckpt_dir "$CKPT_DIR" --step "$STEP" --use_ema \
  --n_val "$N_VAL" --ablate_assets golden \
  --world_size "$W" --mode merge

echo "[launcher] DONE. report -> ${OUT}/report.md"
