#!/usr/bin/env bash
set -e
cd /mnt/novita2/siyuan/workspace/TRELLIS.2/.claude/worktrees/pre-triton-s6
export CUDA_VISIBLE_DEVICES=0

# A/B test first
echo "=== A/B test (flag OFF) ==="
COREP_FAST_S6_FASTPATH_GPU=0 /mnt/novita2/siyuan/workspace/TRELLIS.2/.venv/bin/python \
    -m pytest corep_fast/tests/regression/test_s6_fastpath_gpu_ab.py -v 2>&1 | tail -10
echo "=== A/B test (flag ON) ==="
COREP_FAST_S6_FASTPATH_GPU=1 /mnt/novita2/siyuan/workspace/TRELLIS.2/.venv/bin/python \
    -m pytest corep_fast/tests/regression/test_s6_fastpath_gpu_ab.py -v 2>&1 | tail -10

# Run profile @ res=128 + 256, flag ON, 3 runs each
echo
echo "=== Profile (W3 O5+O7, flag ON) ==="
for res in 128 256; do
    for run in 1 2 3; do
        out=tmp/w3_o7_on_res${res}_run${run}.json
        env COREP_FAST_S6_FASTPATH_GPU=1 /mnt/novita2/siyuan/workspace/TRELLIS.2/.venv/bin/python \
            tmp/e2e_profile_m2.py --res $res --out $out 2>&1 | tail -3
    done
done

/mnt/novita2/siyuan/workspace/TRELLIS.2/.venv/bin/python -c "
import json, statistics
print()
print('w3_o7_on (after O5+O7) vs w3_o5_on (O5 only) baseline:')
for res in [128, 256]:
    o5_s6 = statistics.median([json.load(open(f'tmp/w3_o5_on_res{res}_run{r}.json'))['new']['s6'] for r in [1,2,3]])
    o7_s6 = statistics.median([json.load(open(f'tmp/w3_o7_on_res{res}_run{r}.json'))['new']['s6'] for r in [1,2,3]])
    o5_e2e = statistics.median([json.load(open(f'tmp/w3_o5_on_res{res}_run{r}.json'))['new']['e2e'] for r in [1,2,3]])
    o7_e2e = statistics.median([json.load(open(f'tmp/w3_o7_on_res{res}_run{r}.json'))['new']['e2e'] for r in [1,2,3]])
    print(f'  res={res}: O5-only s6={o5_s6:.3f}s -> O5+O7 s6={o7_s6:.3f}s ΔS6={o7_s6-o5_s6:+.3f}s   e2e {o5_e2e:.3f}->{o7_e2e:.3f}')
print()
print('Total W3 (O5+O7) gain vs original off-baseline:')
for res in [128, 256]:
    off_s6 = statistics.median([json.load(open(f'tmp/w3_o5_off_res{res}_run{r}.json'))['new']['s6'] for r in [1,2,3]])
    final_s6 = statistics.median([json.load(open(f'tmp/w3_o7_on_res{res}_run{r}.json'))['new']['s6'] for r in [1,2,3]])
    print(f'  res={res}: s6 OFF={off_s6:.3f}s -> ON+O7={final_s6:.3f}s   ΔS6={final_s6-off_s6:+.3f}s')
"
