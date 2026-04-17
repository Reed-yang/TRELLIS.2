#!/usr/bin/env bash
set -e
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m tmp.cpu_profile.driver_main --res 256
PROF=tmp/cpu_profile/results_main/main_thread_res256.prof
CUDA_VISIBLE_DEVICES=4 .venv/bin/python - <<'PYEOF'
import pstats
from pathlib import Path
stats = pstats.Stats("tmp/cpu_profile/results_main/main_thread_res256.prof")
items = [(func, entry) for func, entry in stats.stats.items()]
items.sort(key=lambda kv: kv[1][2], reverse=True)
lines = [f"{'rank':>4} {'self_ms':>10} {'cum_ms':>10} {'calls':>10}  function @ file:line", "-" * 100]
for i, (func, entry) in enumerate(items[:20], 1):
    cc, nc, tt, ct, _ = entry
    filename, lineno, fn_name = func
    lines.append(f"{i:>4} {tt*1000:>10.1f} {ct*1000:>10.1f} {cc:>10}  {fn_name} @ {filename}:{lineno}")
Path("tmp/cpu_profile/w_l2l_post_hotspots.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PYEOF
