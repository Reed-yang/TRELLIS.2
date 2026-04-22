#!/usr/bin/env bash
# Single-mesh run on the worst historical OOM asset (51316 GiB requested).
# Uses a long timeout (3 hours) because CPU-fallback Phase A on P ~ 2.5M
# is expected to be minutes, and Phase B/C may still OOM — we want the
# captured failure location, not a mid-run timeout.
#
# Runs on host-10-240-99-117 GPU 0 (solo: GPU 1-3 reserved for concurrent
# retries or other work).
set -uo pipefail
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
export PYTHONPATH=.

RESULTS=scripts/oom_validation/results_extreme
rm -rf "$RESULTS"; mkdir -p "$RESULTS"

.venv/bin/python -c "import corep_fast.config as c; print(f'VRAM_RESCUE={c.VRAM_RESCUE} S2_SPARSE={c.S2_SPARSE} ASYNC_D2H={c.ASYNC_D2H}')" \
  | tee "$RESULTS/flag_state.log"

# Optional: expose PYTORCH CUDA alloc config for less fragmentation.
# Leave commented so we measure the vanilla behaviour first.
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LABEL="extreme-51TiB"
GLB="datasets/ObjaverseXL_sketchfab/raw/hf-objaverse-v1/glbs/000-134/b74a3e5b9e6c4547b2692299fdd740c7.glb"
GPU=0

export CUDA_VISIBLE_DEVICES=$GPU

echo "=== Launching $LABEL on GPU $GPU ==="
echo "GLB: $GLB"
echo "Results: $RESULTS"
echo "Expected behaviour:"
echo "  - Phase A chunked dispatch picks Tier 1 (chunked GPU) or Tier 2 (CPU)"
echo "  - Phase B edge_mask (G*P^2 bool ~ 6.25 TiB for G=1,P=2.5M) likely OOMs"
echo "  - Capture the exact failure phase in the log for follow-up spec"

# 3-hour timeout. The worker itself writes JSON incrementally; if the
# process SIGKILLs on OOM or timeout, the partial JSON is still there.
timeout 10800s .venv/bin/python -u scripts/oom_validation/worker.py \
    --label "$LABEL" \
    --glb_path "$GLB" \
    --device "cuda:0" \
    --out_json "$RESULTS/${LABEL}.json" \
  > "$RESULTS/${LABEL}.log" 2>&1 || echo "[host] worker exit $?"

echo ""
echo "=== Tail of worker log ==="
tail -40 "$RESULTS/${LABEL}.log" 2>/dev/null || true
echo ""
echo "=== JSON summary (if produced) ==="
if [ -f "$RESULTS/${LABEL}.json" ]; then
  cat "$RESULTS/${LABEL}.json"
else
  echo "(no JSON produced — worker crashed before write)"
fi
echo ""
echo "=== GPU state after run ==="
nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv
