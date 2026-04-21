#!/bin/bash
# Sketchfab full-set 6-way array download.
# Each of ranks 0..5 takes ~28k objects; idempotent re-runs skip already-downloaded.
#
# Submit with the tuned --processes value (default 96):
#   sbatch --export=ALL,PROCESSES=96 tools/data_download/run_full.slurm
#
# Monitor progress:
#   squeue --me
#   ls /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/logs/download/rank*_full_summary.json
#   watch -n 30 ".venv/bin/python data_toolkit/build_metadata.py ObjaverseXL \\
#       --source sketchfab --root /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab \\
#       | tail -5"
#SBATCH --job-name=sf-full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/logs/download/sbatch_full_%A_%a.log
#SBATCH --error=/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/logs/download/sbatch_full_%A_%a.log

set -euo pipefail

P="${PROCESSES:-96}"
ROOT=/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab
REPO=/mnt/novita2/siyuan/workspace/TRELLIS.2

echo "[full] host=$(hostname) array_id=$SLURM_ARRAY_TASK_ID processes=$P"

cd "$REPO"
.venv/bin/python tools/data_download/download_sketchfab.py \
    --root "$ROOT" \
    --processes "$P" \
    --world_size 6 \
    --rank "$SLURM_ARRAY_TASK_ID" \
    --tag full
