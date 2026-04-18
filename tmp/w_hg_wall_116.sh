#!/usr/bin/env bash
set -e
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
COREP_FAST_HUNGARIAN_GPU=1 CUDA_VISIBLE_DEVICES=4 \
  .venv/bin/python tmp/cpu_profile/t0_driver.py \
    --mode default --resolution 256 --trials 3 2>&1 \
    | tee tmp/cpu_profile/w_hg_wall.log
cp tmp/cpu_profile/t0_default.json tmp/cpu_profile/w_hg_wall.json
