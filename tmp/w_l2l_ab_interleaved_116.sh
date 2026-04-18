#!/usr/bin/env bash
# W_L2L fair A/B interleaved re-test on 116 GPU 5.
# 3 rounds × (A=flag0 then B=flag1) × 3 trials each = 18 trials total.
# Interleaving cancels environmental drift.
set -e
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
OUT=tmp/followup_baseline/ab_test
mkdir -p "$OUT"

echo "=== W_L2L A/B interleaved, 116 GPU 5, $(date -u +%FT%TZ) ==="
echo "load_before=$(cat /proc/loadavg)"
echo ""

for round in 1 2 3; do
  echo "--- Round $round / A (flag=0, legacy) ---"
  COREP_FAST_LABELS_TO_LIST_VECTORIZED=0 CUDA_VISIBLE_DEVICES=5 \
    .venv/bin/python tmp/cpu_profile/t0_driver.py \
      --mode default --resolution 256 --trials 3 \
      2>&1 | tee "$OUT/round${round}_A.log"
  cp tmp/cpu_profile/t0_default.json "$OUT/round${round}_A.json"
  echo "load=$(cat /proc/loadavg)"
  echo ""

  echo "--- Round $round / B (flag=1, vectorized) ---"
  COREP_FAST_LABELS_TO_LIST_VECTORIZED=1 CUDA_VISIBLE_DEVICES=5 \
    .venv/bin/python tmp/cpu_profile/t0_driver.py \
      --mode default --resolution 256 --trials 3 \
      2>&1 | tee "$OUT/round${round}_B.log"
  cp tmp/cpu_profile/t0_default.json "$OUT/round${round}_B.json"
  echo "load=$(cat /proc/loadavg)"
  echo ""
done

echo "=== done $(date -u +%FT%TZ) ==="
echo "load_after=$(cat /proc/loadavg)"

# Aggregate stats
CUDA_VISIBLE_DEVICES=5 .venv/bin/python - <<'PYEOF'
import json
from pathlib import Path
import statistics
out = Path("tmp/followup_baseline/ab_test")
A = []; B = []
for r in (1, 2, 3):
    for mode, acc in (("A", A), ("B", B)):
        data = json.loads((out / f"round{r}_{mode}.json").read_text())
        # t0_driver.py structure: {"mode": "default", "trials": [{"wall": ...}, ...]}
        for t in data.get("trials", []):
            w = t.get("wall") or t.get("wall_s") or t.get("e2e")
            if w is None:
                # fallback: scan dict for single numeric value
                w = next((v for v in t.values() if isinstance(v, (int, float))), None)
            if w is not None:
                acc.append(float(w))
print(f"A (flag=0 legacy):    N={len(A)} mean={statistics.mean(A):.4f}s median={statistics.median(A):.4f}s stdev={statistics.stdev(A):.4f}s  min={min(A):.3f} max={max(A):.3f}")
print(f"B (flag=1 vectored):  N={len(B)} mean={statistics.mean(B):.4f}s median={statistics.median(B):.4f}s stdev={statistics.stdev(B):.4f}s  min={min(B):.3f} max={max(B):.3f}")
print(f"Δ(B-A) mean:     {statistics.mean(B)-statistics.mean(A):+.4f}s")
print(f"Δ(B-A) median:   {statistics.median(B)-statistics.median(A):+.4f}s")
print(f"A trials: {A}")
print(f"B trials: {B}")
PYEOF
