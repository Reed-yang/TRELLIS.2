#!/usr/bin/env bash
# Note: tmp/cpu_profile/t0_driver.py has no --cprofile* flags. Use
# driver_main.py which has built-in cProfile (dumps .prof file), then
# convert the .prof to a top-20 text listing via inline pstats.
set -e
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
mkdir -p tmp/followup_baseline
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m tmp.cpu_profile.driver_main --res 256
PROF=tmp/cpu_profile/results_main/main_thread_res256.prof
CUDA_VISIBLE_DEVICES=4 .venv/bin/python - <<PYEOF
import pstats
from pathlib import Path
prof = "${PROF}"
stats = pstats.Stats(prof)
items = [(func, entry) for func, entry in stats.stats.items()]
# sort by self-time (tt = entry[2])
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
out = Path("tmp/followup_baseline/hotspots_pre.txt")
out.write_text("\n".join(lines) + "\n")
print(f"[write] {out}")
print("\n".join(lines))
PYEOF
