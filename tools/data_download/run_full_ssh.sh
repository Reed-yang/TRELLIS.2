#!/bin/bash
# Fallback launcher when slurm queue is congested.
# Directly ssh-nohups the 6-rank download across host-10-240-99-{115..120}.
#
# Usage:
#   bash tools/data_download/run_full_ssh.sh [PROCESSES]   # default 96
#
# Monitor:
#   ls /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/logs/download/rank*_ssh-full_summary.json
#   tail -F /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/logs/download/ssh_rank{0..5}.log
set -euo pipefail

P="${1:-96}"
NODES=(
    host-10-240-99-115
    host-10-240-99-116
    host-10-240-99-117
    host-10-240-99-118
    host-10-240-99-119
    host-10-240-99-120
)
ROOT=/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab
REPO=/mnt/novita2/siyuan/workspace/TRELLIS.2

# HF token from head node's ~/.cache (not shared NFS); passed via env so other
# nodes can do authenticated HF requests (much higher 429 quota).
HF_TOKEN_VAL=""
if [ -f "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN_VAL=$(tr -d '\n' < "$HOME/.cache/huggingface/token")
fi

mkdir -p "$ROOT/logs/download"

for i in "${!NODES[@]}"; do
    host="${NODES[$i]}"
    log_file="$ROOT/logs/download/ssh_rank${i}.log"
    echo "[launch] $host rank=$i processes=$P -> $log_file"
    # nohup daemonizes on the remote; inner & backgrounds on the remote shell,
    # outer & fans out ssh in parallel from the launcher.
    ssh -o StrictHostKeyChecking=no -o BatchMode=yes "$host" \
        "cd $REPO && HF_TOKEN='$HF_TOKEN_VAL' nohup .venv/bin/python tools/data_download/download_sketchfab.py \
            --root $ROOT \
            --processes $P \
            --world_size 6 \
            --rank $i \
            --tag ssh-full \
            > $log_file 2>&1 < /dev/null &" &
done
wait

echo
echo "All 6 ssh jobs launched. Monitor with:"
echo "  squeue --me         # slurm (shouldn't see anything)"
echo "  ls $ROOT/logs/download/rank*_ssh-full_summary.json"
echo "  tail -F $ROOT/logs/download/ssh_rank{0..5}.log"
