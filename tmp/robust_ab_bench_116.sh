#!/usr/bin/env bash
# Robust A/B interleaved benchmark on 116 GPU 4.
# A = pre-followup state (all followup flags OFF)
# B = current production (STAGE_D_GPU=1, HUNGARIAN_GPU=1)
# 4 rounds × 2 modes × 3 trials = 24 trials.
set -e
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
OUT=tmp/followup_baseline/robust_ab
mkdir -p "$OUT"

echo "=== Robust A/B benchmark on 116 GPU 4 — $(date -u +%FT%TZ) ==="
echo "load_before=$(cat /proc/loadavg)"
echo ""

for round in 1 2 3 4; do
  echo "--- Round $round / A (all followup flags OFF; baseline-equivalent) ---"
  COREP_FAST_LABELS_TO_LIST_VECTORIZED=0 \
  COREP_FAST_STAGE_D_GPU=0 \
  COREP_FAST_HUNGARIAN_GPU=0 \
  CUDA_VISIBLE_DEVICES=4 \
    .venv/bin/python tmp/cpu_profile/t0_driver.py \
      --mode default --resolution 256 --trials 3 \
      2>&1 | tee "$OUT/round${round}_A.log"
  cp tmp/cpu_profile/t0_default.json "$OUT/round${round}_A.json"
  echo "load=$(cat /proc/loadavg)"
  echo ""

  echo "--- Round $round / B (production: STAGE_D_GPU=1 + HUNGARIAN_GPU=1) ---"
  COREP_FAST_LABELS_TO_LIST_VECTORIZED=0 \
  COREP_FAST_STAGE_D_GPU=1 \
  COREP_FAST_HUNGARIAN_GPU=1 \
  CUDA_VISIBLE_DEVICES=4 \
    .venv/bin/python tmp/cpu_profile/t0_driver.py \
      --mode default --resolution 256 --trials 3 \
      2>&1 | tee "$OUT/round${round}_B.log"
  cp tmp/cpu_profile/t0_default.json "$OUT/round${round}_B.json"
  echo "load=$(cat /proc/loadavg)"
  echo ""
done

echo "=== done $(date -u +%FT%TZ) ==="
echo "load_after=$(cat /proc/loadavg)"

# Aggregate
.venv/bin/python - <<'PYEOF'
import json, statistics
from pathlib import Path
out = Path("tmp/followup_baseline/robust_ab")
A = []; B = []
for r in (1, 2, 3, 4):
    for mode, acc in (("A", A), ("B", B)):
        data = json.loads((out / f"round{r}_{mode}.json").read_text())
        acc.extend(data["samples_sec"])

import math
def stats(name, x):
    n, mu, med, sd = len(x), statistics.mean(x), statistics.median(x), (statistics.stdev(x) if len(x) > 1 else 0.0)
    print(f"{name:20s} N={n:2d}  mean={mu:.4f}s  median={med:.4f}s  stdev={sd:.4f}s  min={min(x):.3f}  max={max(x):.3f}")
    return mu, med, sd

mA, medA, sA = stats("A (all-flags-OFF)", A)
mB, medB, sB = stats("B (production)", B)
print(f"Δ(B-A) mean   = {mB-mA:+.4f}s (negative = B faster)")
print(f"Δ(B-A) median = {medB-medA:+.4f}s")

if len(A) > 1 and len(B) > 1:
    se = math.sqrt(sA**2/len(A) + sB**2/len(B))
    t = (mB - mA) / se if se > 0 else 0.0
    print(f"Welch t = {t:+.3f}  (|t|>2 means stat-significant at 95% CI)")

print(f"\nA all: {[round(x,3) for x in A]}")
print(f"B all: {[round(x,3) for x in B]}")
PYEOF
