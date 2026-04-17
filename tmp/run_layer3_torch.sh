#!/usr/bin/env bash
set -e
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
export CUDA_VISIBLE_DEVICES=0
.venv/bin/python tmp/profile_deep/driver.py --layer 3 --res 128
