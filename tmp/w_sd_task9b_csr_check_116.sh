#!/usr/bin/env bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -c "
import pstats
stats = pstats.Stats('tmp/cpu_profile/w_sd_task9b.prof')
for func, e in stats.stats.items():
    fn, ln, name = func
    if 'count_uturns' in name or 'gpu_batched_csr' in name:
        cc, nc, tt, ct, _ = e
        print(f'{name:<45} self_ms={tt*1000:.1f} cum_ms={ct*1000:.1f} calls={cc}')
"
