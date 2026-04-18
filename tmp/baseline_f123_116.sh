#!/usr/bin/env bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  mkdir -p tmp/followup_baseline && \
  CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m pytest \
    corep_fast/tests/regression/test_cpu_worker_optim.py -v 2>&1 \
    | tee tmp/followup_baseline/f123.log
