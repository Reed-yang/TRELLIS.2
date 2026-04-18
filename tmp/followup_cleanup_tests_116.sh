#!/usr/bin/env bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  mkdir -p tmp/followup_baseline && \
  CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m pytest \
    corep_fast/tests/unit/test_labels_to_list_vectorized.py \
    corep_fast/tests/unit/test_stage_d_gpu_bfs.py \
    corep_fast/tests/unit/test_hungarian_batched.py -v 2>&1 \
    | tee tmp/followup_baseline/cleanup_unit_tests.log
