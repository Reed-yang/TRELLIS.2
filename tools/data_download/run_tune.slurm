#!/bin/bash
# Sketchfab download processes-sweep: 32 / 64 / 96 / 128, 500 objects per bucket.
# Runs 4 array tasks sequentially (%1) on the same kind of node to avoid cross-node noise.
# Each task uses a distinct rank on the full 168-way shard so sha256 sets don't overlap.
#
# Submit:
#   sbatch tools/data_download/run_tune.slurm
# Check results:
#   ls /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/logs/download/rank*tune*_summary.json
#SBATCH --job-name=sf-tune
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-3%1
#SBATCH --output=/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/logs/download/sbatch_tune_%A_%a.log
#SBATCH --error=/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/logs/download/sbatch_tune_%A_%a.log

set -euo pipefail

PROCS_LIST=(32 64 96 128)
P="${PROCS_LIST[$SLURM_ARRAY_TASK_ID]}"
RANK=$((101 + SLURM_ARRAY_TASK_ID))

ROOT=/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab
REPO=/mnt/novita2/siyuan/workspace/TRELLIS.2

echo "[tune] host=$(hostname) array_id=$SLURM_ARRAY_TASK_ID processes=$P rank=$RANK"

cd "$REPO"
.venv/bin/python tools/data_download/download_sketchfab.py \
    --root "$ROOT" \
    --processes "$P" \
    --world_size 168 \
    --rank "$RANK" \
    --limit 500 \
    --tag "tune-p${P}"
