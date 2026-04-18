#!/usr/bin/env bash
set -e
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
COREP_FAST_STAGE_D_GPU=1 CUDA_VISIBLE_DEVICES=4 \
  .venv/bin/python -m tmp.cpu_profile.driver_main --res 256 2>&1 | tail -5
# Copy to task9 slot
cp tmp/cpu_profile/results_main/main_thread_res256.prof tmp/cpu_profile/w_sd_task9.prof
CUDA_VISIBLE_DEVICES=4 .venv/bin/python - <<'PYEOF'
import pstats
from pathlib import Path
stats = pstats.Stats("tmp/cpu_profile/w_sd_task9.prof")
items = [(func, e) for func, e in stats.stats.items()]
items.sort(key=lambda kv: kv[1][2], reverse=True)
lines = [f"{'rank':>4} {'self_ms':>10} {'cum_ms':>10} {'calls':>10}  function @ file:line", "-"*100]
for i, (func, e) in enumerate(items[:20], 1):
    cc, nc, tt, ct, _ = e
    fn, ln, name = func
    lines.append(f"{i:>4} {tt*1000:>10.1f} {ct*1000:>10.1f} {cc:>10}  {name} @ {fn}:{ln}")
Path("tmp/cpu_profile/w_sd_task9_hotspots.txt").write_text("\n".join(lines)+"\n")
print("\n".join(lines))
PYEOF
