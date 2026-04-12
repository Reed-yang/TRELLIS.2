#!/bin/bash
#SBATCH --job-name=phase_a
#SBATCH --partition=gpu
#SBATCH --nodelist=host-10-240-99-120
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --output=experiments/component_eval/phase_a_slurm_%j.log

cd /mnt/novita2/siyuan/workspace/TRELLIS.2

# SLURM maps allocated GPUs to CUDA_VISIBLE_DEVICES=0,1 automatically
# Rank 9 and 10 (world_size=11 to avoid overlap with 115's ranks 0-8)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eval/component_eval.py --phase a \
    --manifest experiments/component_eval/test_set/manifest.json \
    --output_dir experiments/component_eval \
    --rank 9 --world_size 11 &

CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/eval/component_eval.py --phase a \
    --manifest experiments/component_eval/test_set/manifest.json \
    --output_dir experiments/component_eval \
    --rank 10 --world_size 11 &

wait
echo "Phase A ranks 9-10 complete"
