#!/usr/bin/env bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  COREP_FAST_HUNGARIAN_GPU=1 CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m pytest \
    corep_fast/tests/regression/test_cpu_worker_optim.py -v 2>&1 \
    | tee tmp/w_hg_f123.log
