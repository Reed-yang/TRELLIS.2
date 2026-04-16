#!/usr/bin/env bash
set -e
cd /mnt/novita2/siyuan/workspace/TRELLIS.2/.claude/worktrees/pre-triton-s6
export CUDA_VISIBLE_DEVICES=0

# Take 3-run median by running 3 times for each (off, on) at res=128 and res=256
for res in 128 256; do
    for setting in off on; do
        for run in 1 2 3; do
            if [ "$setting" = "on" ]; then
                FLAG="COREP_FAST_S6_FASTPATH_GPU=1"
            else
                FLAG="COREP_FAST_S6_FASTPATH_GPU=0"
            fi
            out=tmp/w3_o5_${setting}_res${res}_run${run}.json
            env $FLAG /mnt/novita2/siyuan/workspace/TRELLIS.2/.venv/bin/python \
                tmp/e2e_profile_m2.py --res $res --out $out 2>&1 | tail -3
        done
    done
done

# Aggregate medians
/mnt/novita2/siyuan/workspace/TRELLIS.2/.venv/bin/python -c "
import json, statistics
print('per-run + median:')
for res in [128, 256]:
    for setting in ['off', 'on']:
        s6 = []
        e2e = []
        for run in [1, 2, 3]:
            d = json.load(open(f'tmp/w3_o5_{setting}_res{res}_run{run}.json'))['new']
            s6.append(d['s6'])
            e2e.append(d['e2e'])
        med_s6 = statistics.median(s6)
        med_e2e = statistics.median(e2e)
        print(f'  res={res} {setting:>3}: s6 median={med_s6:.3f}s (runs={[round(x,3) for x in s6]})  e2e median={med_e2e:.3f}s')
print()
print('summary:')
for res in [128, 256]:
    off_s6 = statistics.median([json.load(open(f'tmp/w3_o5_off_res{res}_run{r}.json'))['new']['s6'] for r in [1,2,3]])
    on_s6 = statistics.median([json.load(open(f'tmp/w3_o5_on_res{res}_run{r}.json'))['new']['s6'] for r in [1,2,3]])
    off_e2e = statistics.median([json.load(open(f'tmp/w3_o5_off_res{res}_run{r}.json'))['new']['e2e'] for r in [1,2,3]])
    on_e2e = statistics.median([json.load(open(f'tmp/w3_o5_on_res{res}_run{r}.json'))['new']['e2e'] for r in [1,2,3]])
    print(f'  res={res}: ΔS6={on_s6-off_s6:+.3f}s ({off_s6:.3f} -> {on_s6:.3f})  Δe2e={on_e2e-off_e2e:+.3f}s ({off_e2e:.3f} -> {on_e2e:.3f})')
"
