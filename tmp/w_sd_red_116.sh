#!/usr/bin/env bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m pytest \
  corep_fast/tests/unit/test_stage_d_gpu_bfs.py -v 2>&1
