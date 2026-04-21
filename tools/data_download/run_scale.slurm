#!/bin/bash
# 2-node scale test: compare per-node throughput at 1-node vs 2-node concurrency
# to detect HF CDN rate-limiting before committing to 6-node full run.
# Uses offset to jump past the already-downloaded prefix, --limit 500 per rank.
#
# Submit (after choosing P from tune step):
#   sbatch --export=ALL,PROCESSES=96 tools/data_download/run_scale.slurm
#SBATCH --job-name=sf-scale
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/logs/download/sbatch_scale_%A_%a.log
#SBATCH --error=/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/logs/download/sbatch_scale_%A_%a.log

set -euo pipefail

P="${PROCESSES:-96}"
ROOT=/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab
REPO=/mnt/novita2/siyuan/workspace/TRELLIS.2

# Rank 0/1 on a 336-way virtual shard → ~501 objects each, far from the
# already-downloaded [0, 1001) and tune ranges (rank 101..104 on 168-way = index 101k..105k).
# 336-way rank 220/221 lands around index [110k, 111k] — safely untouched.
VIRT_RANK=$((220 + SLURM_ARRAY_TASK_ID))

echo "[scale] host=$(hostname) array_id=$SLURM_ARRAY_TASK_ID processes=$P virt_rank=$VIRT_RANK"

cd "$REPO"
.venv/bin/python tools/data_download/download_sketchfab.py \
    --root "$ROOT" \
    --processes "$P" \
    --world_size 336 \
    --rank "$VIRT_RANK" \
    --limit 500 \
    --tag "scale-n2-p${P}"
