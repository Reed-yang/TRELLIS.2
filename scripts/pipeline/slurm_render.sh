#!/bin/bash
#SBATCH --job-name=blender_render
#SBATCH --partition=gpu
#SBATCH --nodes=6
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:0
#SBATCH --time=6:00:00
#SBATCH --output=experiments/component_eval/slurm_render_%j.log
#SBATCH --nodelist=host-10-240-99-[115-120]

# Blender CYCLES CPU rendering for Toys4k conditioning images
# 24 tasks (4 per node × 6 nodes), each rendering a shard of the manifest
# Force CPU-only (no GPU) via CUDA_VISIBLE_DEVICES=""

export CUDA_VISIBLE_DEVICES=""

# srun distributes tasks across nodes; each task gets unique SLURM_PROCID
srun bash -c '
WORKDIR=/mnt/novita2/siyuan/workspace/TRELLIS.2
cd $WORKDIR

RANK=$SLURM_PROCID
WORLD_SIZE=$SLURM_NTASKS

echo "[$(hostname)] Task $RANK/$WORLD_SIZE starting on $(date)"

$WORKDIR/.venv/bin/python scripts/render/render_blender_cond.py \
    --manifest $WORKDIR/experiments/component_eval/test_set/manifest.json \
    --render_output $WORKDIR/experiments/component_eval/renders_cond \
    --num_views 16 \
    --rank $RANK \
    --world_size $WORLD_SIZE

echo "[$(hostname)] Task $RANK/$WORLD_SIZE finished on $(date)"
'
