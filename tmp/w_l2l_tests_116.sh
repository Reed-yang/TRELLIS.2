#!/usr/bin/env bash
set -e
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m pytest \
  corep_fast/tests/unit/test_labels_to_list_vectorized.py -v 2>&1 \
  | tee tmp/followup_baseline/w_l2l_tests.log
