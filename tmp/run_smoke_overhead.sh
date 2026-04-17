#!/usr/bin/env bash
set -e
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
export CUDA_VISIBLE_DEVICES=0
echo "=== UNPATCHED ==="
.venv/bin/python tmp/profile_deep/smoke_compare.py unpatched
echo "=== PATCHED ==="
.venv/bin/python tmp/profile_deep/smoke_compare.py patched
