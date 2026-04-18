#!/usr/bin/env bash
set -e
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
mkdir -p tmp/followup_design
CUDA_VISIBLE_DEVICES=4 PYTHONHASHSEED=0 .venv/bin/python tmp/w_baf_spike_real_116.py 2>&1 \
  | tee tmp/followup_design/w_baf_spike_real.txt
