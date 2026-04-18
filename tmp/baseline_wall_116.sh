#!/usr/bin/env bash
# Note: tmp/cpu_profile/t0_driver.py does not support --out flag; its real
# args are --mode/--trials/--resolution and it writes to a fixed path
# tmp/cpu_profile/t0_{mode}.json. We copy that output to the target path
# tmp/followup_baseline/t0_driver_clean.json after the run.
set -e
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
mkdir -p tmp/followup_baseline
CUDA_VISIBLE_DEVICES=4 .venv/bin/python tmp/cpu_profile/t0_driver.py \
    --mode default --resolution 256 --trials 3 2>&1 \
    | tee tmp/followup_baseline/wall.log
cp tmp/cpu_profile/t0_default.json tmp/followup_baseline/t0_driver_clean.json
echo "[copy] tmp/cpu_profile/t0_default.json -> tmp/followup_baseline/t0_driver_clean.json"
