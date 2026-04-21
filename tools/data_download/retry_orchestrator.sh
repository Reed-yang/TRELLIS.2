#!/bin/bash
# Self-contained retry loop.
#
# Behavior:
#   1) Wait for any in-flight download_sketchfab.py on the 6 nodes to exit.
#   2) Merge new_records into raw/metadata.csv (so resume works).
#   3) Aggregate + dedup all rank_*_failures.csv, subtract items already downloaded.
#   4) If non-empty, sleep 120s for HF 429 cooldown, then ssh-launch 6 ranks with
#      --instances pointing at the pending sha256 list. Tag = retry-N.
#   5) Goto 1. Max 3 retry attempts (plus the first pass that's already running).
#
# Usage:
#   nohup bash tools/data_download/retry_orchestrator.sh > /tmp/orchestrator.log 2>&1 &
set -euo pipefail

ROOT=/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab
REPO=/mnt/novita2/siyuan/workspace/TRELLIS.2
NODES=(
    host-10-240-99-115
    host-10-240-99-116
    host-10-240-99-117
    host-10-240-99-118
    host-10-240-99-119
    host-10-240-99-120
)
P=${PROCESSES:-20}
MAX_RETRIES=${MAX_RETRIES:-3}
COOLDOWN=${COOLDOWN:-120}

# HF token read once from the head node (115's ~/.cache/huggingface/token is
# not on shared NFS, so other nodes must receive it via env). With token,
# 429 frequency drops dramatically (anonymous quota vs authenticated quota).
HF_TOKEN_VAL=""
if [ -f "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN_VAL=$(tr -d '\n' < "$HOME/.cache/huggingface/token")
fi

LOG="$ROOT/logs/download/orchestrator.log"
mkdir -p "$(dirname "$LOG")"

log() { echo "[orch $(date -Is)] $*" | tee -a "$LOG"; }

wait_for_done() {
    while :; do
        local cnt=0
        for h in "${NODES[@]}"; do
            local c
            c=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" \
                    "pgrep -cf download_sketchfab.py" 2>/dev/null || echo 0)
            cnt=$((cnt + c))
        done
        if [ "$cnt" = "0" ]; then break; fi
        sleep 60
    done
}

log "orchestrator starting (P=$P, MAX_RETRIES=$MAX_RETRIES, COOLDOWN=$COOLDOWN)"
log "waiting for current ssh-full / retry-N run to finish..."
wait_for_done
log "first-pass (or prior) run has exited"

for attempt in $(seq 1 "$MAX_RETRIES"); do
    log "=== attempt $attempt / $MAX_RETRIES ==="

    log "merging part files into raw/metadata.csv..."
    "$REPO/.venv/bin/python" "$REPO/data_toolkit/build_metadata.py" ObjaverseXL \
        --source sketchfab --root "$ROOT" 2>&1 | tail -4 | tee -a "$LOG"

    FAIL_LIST="$ROOT/logs/download/retry_${attempt}_targets.csv"
    "$REPO/.venv/bin/python" - <<PY | tee -a "$LOG"
import glob, os, pandas as pd

fails_dfs = []
for f in sorted(glob.glob("$ROOT/logs/download/rank*_failures.csv")):
    try:
        df = pd.read_csv(f)
        if len(df): fails_dfs.append(df)
    except Exception:
        pass

if not fails_dfs:
    print("no failures.csv found")
    pd.DataFrame(columns=["sha256","file_identifier"]).to_csv("$FAIL_LIST", index=False)
    raise SystemExit(0)

fail = pd.concat(fails_dfs, ignore_index=True).drop_duplicates("sha256")
print(f"aggregated failures (pre-filter): {len(fail)}")

raw_meta = "$ROOT/raw/metadata.csv"
if os.path.exists(raw_meta):
    done = set(pd.read_csv(raw_meta)["sha256"].dropna())
    before = len(fail)
    fail = fail[~fail["sha256"].isin(done)]
    print(f"after subtracting raw/metadata.csv: {before} -> {len(fail)}")

fail.to_csv("$FAIL_LIST", index=False)
print(f"wrote $FAIL_LIST ({len(fail)} rows)")
PY

    PENDING=$(tail -n +2 "$FAIL_LIST" | wc -l)
    log "pending after aggregate+subtract = $PENDING"

    if [ "$PENDING" -le 0 ]; then
        log "no pending items, orchestrator done"
        break
    fi

    log "sleeping ${COOLDOWN}s for HF rate-limit cooldown..."
    sleep "$COOLDOWN"

    log "launching retry attempt $attempt across ${#NODES[@]} nodes (processes=$P)..."
    for i in "${!NODES[@]}"; do
        host="${NODES[$i]}"
        log_file="$ROOT/logs/download/ssh_retry${attempt}_rank${i}.log"
        ssh -o BatchMode=yes "$host" \
            "cd $REPO && HF_TOKEN='$HF_TOKEN_VAL' nohup .venv/bin/python tools/data_download/download_sketchfab.py \
                --root $ROOT \
                --processes $P \
                --world_size ${#NODES[@]} \
                --rank $i \
                --instances $FAIL_LIST \
                --tag retry-${attempt} \
                > $log_file 2>&1 < /dev/null &" &
    done
    wait

    log "retry-$attempt launched, waiting for completion..."
    wait_for_done
    log "retry-$attempt completed"
done

log "orchestrator finished"
