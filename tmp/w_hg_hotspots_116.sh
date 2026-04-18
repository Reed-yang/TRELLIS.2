#!/usr/bin/env bash
set -e
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
COREP_FAST_HUNGARIAN_GPU=1 CUDA_VISIBLE_DEVICES=4 \
  .venv/bin/python -m tmp.cpu_profile.driver_main --res 256
PROF=tmp/cpu_profile/results_main/main_thread_res256.prof
COREP_FAST_HUNGARIAN_GPU=1 CUDA_VISIBLE_DEVICES=4 .venv/bin/python - <<PYEOF
import pstats
from pathlib import Path
prof = "${PROF}"
stats = pstats.Stats(prof)
items = [(func, entry) for func, entry in stats.stats.items()]
items.sort(key=lambda kv: kv[1][2], reverse=True)
lines = []
lines.append(f"# Top-20 by self-time from {prof}")
lines.append("")
lines.append(f"{'rank':>4} {'self_ms':>10} {'cum_ms':>10} {'calls':>8}  function @ file:line")
lines.append("-" * 100)
for i, (func, entry) in enumerate(items[:20], 1):
    cc, nc, tt, ct, _ = entry
    filename, lineno, fn_name = func
    lines.append(f"{i:>4} {tt*1000:>10.1f} {ct*1000:>10.1f} {cc:>8}  {fn_name} @ {filename}:{lineno}")
out = Path("tmp/cpu_profile/w_hg_hotspots.txt")
out.write_text("\n".join(lines) + "\n")
print(f"[write] {out}")
print("\n".join(lines))
PYEOF
