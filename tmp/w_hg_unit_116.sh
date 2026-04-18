#!/usr/bin/env bash
set -e
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m pytest \
    corep_fast/tests/unit/test_hungarian_batched.py -v 2>&1 \
    | tee tmp/w_hg_unit.log
