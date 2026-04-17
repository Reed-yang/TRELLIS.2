#!/usr/bin/env bash
set -e
cd /mnt/novita2/siyuan/workspace/TRELLIS.2/.claude/worktrees/pre-triton-s6
export CUDA_VISIBLE_DEVICES=0
/mnt/novita2/siyuan/workspace/TRELLIS.2/.venv/bin/python -m pytest \
    corep_fast/tests/regression/test_s6_fastpath_gpu_ab.py -v 2>&1 | tail -40
