#!/bin/bash
# Profile s7 W2 work: flag OFF vs ON at res=128 + 256 (also includes W1 flag ON for cumulative effect)
set -euo pipefail
cd /mnt/novita2/siyuan/workspace/TRELLIS.2/.claude/worktrees/pre-triton-s7
export CUDA_VISIBLE_DEVICES=1
PY=/mnt/novita2/siyuan/workspace/TRELLIS.2/.venv/bin/python

echo "=== res=128 W2 OFF ==="
"$PY" tmp/e2e_profile_m2.py --res 128 --out tmp/w2_off_res128.json | tail -5

echo "=== res=128 W2 ON ==="
COREP_FAST_S7_PHASE1_GPU=1 "$PY" tmp/e2e_profile_m2.py --res 128 --out tmp/w2_on_res128.json | tail -5

echo "=== res=256 W2 OFF ==="
"$PY" tmp/e2e_profile_m2.py --res 256 --out tmp/w2_off_res256.json | tail -5

echo "=== res=256 W2 ON ==="
COREP_FAST_S7_PHASE1_GPU=1 "$PY" tmp/e2e_profile_m2.py --res 256 --out tmp/w2_on_res256.json | tail -5

echo "=== Summary (W2 alone, no W1 since we're on s7 worktree) ==="
"$PY" -c "
import json
print(f'{\"res\":<5}{\"stage\":<6}{\"OFF\":>8}{\"ON\":>8}{\"delta\":>8}')
for res in [128, 256]:
    o = json.load(open(f'tmp/w2_off_res{res}.json'))['new']
    n = json.load(open(f'tmp/w2_on_res{res}.json'))['new']
    for s in ['s4','s6','s7','s8','e2e']:
        print(f'{res:<5}{s:<6}{o[s]:>8.3f}{n[s]:>8.3f}{n[s]-o[s]:>+8.3f}')
"
echo "DONE"
