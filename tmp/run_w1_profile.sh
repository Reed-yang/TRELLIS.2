#!/bin/bash
# Profile s8 W1 work: flag OFF vs ON at res=128 + 256
# Runs from pre-triton-s8 worktree to pick up worktree-local s8_collapse.py
set -euo pipefail
cd /mnt/novita2/siyuan/workspace/TRELLIS.2/.claude/worktrees/pre-triton-s8
export CUDA_VISIBLE_DEVICES=0
PY=/mnt/novita2/siyuan/workspace/TRELLIS.2/.venv/bin/python

echo "=== res=128 OFF ==="
"$PY" tmp/e2e_profile_m2.py --res 128 --out tmp/w1_off_res128.json | tail -10

echo "=== res=128 ON ==="
COREP_FAST_S8_4CUBE_VECTORIZED=1 "$PY" tmp/e2e_profile_m2.py --res 128 --out tmp/w1_on_res128.json | tail -10

echo "=== res=256 OFF ==="
"$PY" tmp/e2e_profile_m2.py --res 256 --out tmp/w1_off_res256.json | tail -10

echo "=== res=256 ON ==="
COREP_FAST_S8_4CUBE_VECTORIZED=1 "$PY" tmp/e2e_profile_m2.py --res 256 --out tmp/w1_on_res256.json | tail -10

echo "=== Summary ==="
"$PY" -c "
import json
print(f'{\"res\":<5}{\"stage\":<6}{\"OFF\":>8}{\"ON\":>8}{\"delta\":>8}')
for res in [128, 256]:
    o = json.load(open(f'tmp/w1_off_res{res}.json'))['new']
    n = json.load(open(f'tmp/w1_on_res{res}.json'))['new']
    for s in ['s4','s6','s7','s8','e2e']:
        print(f'{res:<5}{s:<6}{o[s]:>8.3f}{n[s]:>8.3f}{n[s]-o[s]:>+8.3f}')
"
echo "DONE"
