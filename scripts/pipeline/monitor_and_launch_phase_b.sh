#!/bin/bash
# Monitor Blender rendering completion and validate Phase B readiness.
# Does NOT launch Phase B — returns status for Claude to act on.

set -euo pipefail

WORKDIR="/mnt/novita2/siyuan/workspace/TRELLIS.2"
RENDERS_DIR="$WORKDIR/experiments/component_eval/renders_cond"
MANIFEST="$WORKDIR/experiments/component_eval/test_set/manifest.json"
PYTHON="$WORKDIR/.venv/bin/python"
LOGFILE="$WORKDIR/logs/monitor_phase_b.log"
TOTAL_ASSETS=4000
CHECK_INTERVAL=1800  # 30 minutes

mkdir -p "$WORKDIR/logs"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

log "=== Monitor started. Checking every ${CHECK_INTERVAL}s for $TOTAL_ASSETS rendered assets ==="

# ---- Wait for rendering to complete ----
while true; do
    DONE_COUNT=$(find "$RENDERS_DIR" -name "transforms.json" 2>/dev/null | wc -l)
    SLURM_RUNNING=$(squeue -j 360 -h 2>/dev/null | wc -l)

    log "Rendering progress: $DONE_COUNT / $TOTAL_ASSETS (slurm job active: $SLURM_RUNNING)"

    if [ "$DONE_COUNT" -ge "$TOTAL_ASSETS" ]; then
        log "All $TOTAL_ASSETS assets rendered!"
        break
    fi

    if [ "$SLURM_RUNNING" -eq 0 ] && [ "$DONE_COUNT" -lt "$TOTAL_ASSETS" ]; then
        log "WARNING: Slurm job 360 finished but only $DONE_COUNT / $TOTAL_ASSETS rendered."
        break
    fi

    sleep "$CHECK_INTERVAL"
done

# ---- Validate readiness ----
FINAL_COUNT=$(find "$RENDERS_DIR" -name "transforms.json" 2>/dev/null | wc -l)

# Spot check view images
SPOT_FAIL=0
SPOT_DETAILS=""
for ASSET_DIR in $(ls -d "$RENDERS_DIR"/*/ | shuf -n 5); do
    for VIEW in 000 007 015; do
        if [ ! -f "$ASSET_DIR/${VIEW}.png" ]; then
            SPOT_FAIL=1
            SPOT_DETAILS="$SPOT_DETAILS MISSING:${ASSET_DIR}${VIEW}.png"
        fi
    done
done

# Check manifest
MANIFEST_OK=0
if [ -f "$MANIFEST" ]; then
    MANIFEST_COUNT=$($PYTHON -c "import json; print(len(json.load(open('$MANIFEST'))))" 2>/dev/null)
    MANIFEST_OK=1
fi

# ---- Output summary for Claude ----
echo ""
echo "========== RENDERING MONITOR RESULT =========="
echo "RENDERED_COUNT=$FINAL_COUNT"
echo "TOTAL_EXPECTED=$TOTAL_ASSETS"
echo "SPOT_CHECK_PASS=$( [ $SPOT_FAIL -eq 0 ] && echo YES || echo NO )"
echo "SPOT_DETAILS=$SPOT_DETAILS"
echo "MANIFEST_EXISTS=$MANIFEST_OK"
echo "MANIFEST_ENTRIES=$MANIFEST_COUNT"
echo "RENDERS_DIR=$RENDERS_DIR"
echo "MANIFEST_PATH=$MANIFEST"
echo "==============================================="

if [ "$FINAL_COUNT" -ge "$TOTAL_ASSETS" ] && [ "$SPOT_FAIL" -eq 0 ] && [ "$MANIFEST_OK" -eq 1 ]; then
    echo "STATUS=READY"
    log "Phase B readiness: READY"
else
    echo "STATUS=INCOMPLETE"
    log "Phase B readiness: INCOMPLETE (renders=$FINAL_COUNT, spot_ok=$([ $SPOT_FAIL -eq 0 ] && echo Y || echo N))"
fi
