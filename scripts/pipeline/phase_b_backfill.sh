#!/bin/bash
# Monitor local Phase B ranks; when one finishes, relaunch it with world_size=1
# to pick up remaining work (resume skips done UIDs).
WORKDIR="/mnt/novita2/siyuan/workspace/TRELLIS.2"
MANIFEST="$WORKDIR/experiments/component_eval/test_set/manifest_pbr.json"
PYTHON="$WORKDIR/.venv/bin/python"
LOGDIR="$WORKDIR/logs"
BACKFILL_RANK=100  # start backfill ranks at 100 to avoid CSV collision

check_pbr_remaining() {
    $PYTHON -c "
import csv, glob, os, json
with open('$WORKDIR/experiments/component_eval/test_set/pbr_strict_uids.json') as f:
    pbr = set(json.load(f))
done = set()
for f in glob.glob('$WORKDIR/experiments/component_eval/phase_b/results/per_sample_rank*.csv'):
    if os.path.getsize(f) > 0:
        with open(f) as fh:
            for row in csv.DictReader(fh):
                uid = row.get('uid','')
                if uid and uid in pbr: done.add(uid)
print(len(pbr) - len(done))
"
}

echo "[$(date)] Backfill monitor started"

while true; do
    REMAINING=$(check_pbr_remaining)
    echo "[$(date)] PBR remaining: $REMAINING"

    if [ "$REMAINING" -le 0 ]; then
        echo "[$(date)] All PBR UIDs done!"
        break
    fi

    # Check each local GPU (0-7)
    for GPU in 0 1 2 3 4 5 6 7; do
        # Check if any component_eval process is using this GPU
        ACTIVE=$(ps aux | grep "component_eval.py --phase b" | grep "CUDA_VISIBLE_DEVICES=$GPU " | grep -v grep | wc -l)
        if [ "$ACTIVE" -eq 0 ]; then
            # Check if there's still work to do
            REMAINING=$(check_pbr_remaining)
            if [ "$REMAINING" -le 0 ]; then
                break 2
            fi
            echo "[$(date)] GPU $GPU idle, launching backfill rank $BACKFILL_RANK ($REMAINING remaining)"
            cd "$WORKDIR"
            CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON scripts/eval/component_eval.py \
                --phase b \
                --manifest "$MANIFEST" \
                --output_dir experiments/component_eval \
                --rank $BACKFILL_RANK --world_size 1000 \
                > "$LOGDIR/phase_b_backfill_rank${BACKFILL_RANK}_gpu${GPU}.log" 2>&1 &
            BACKFILL_RANK=$((BACKFILL_RANK + 1))
        fi
    done

    sleep 60
done

echo "[$(date)] Backfill monitor done"
