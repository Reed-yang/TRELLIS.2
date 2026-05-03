#!/usr/bin/env bash
# Run coart vae reconstruction on the underwater_plant_pack.glb mesh on 119.
# Invoked via:
#   ssh host-10-240-99-119 'bash /mnt/novita2/siyuan/workspace/TRELLIS.2/scripts/run_reconstruct_underwater.sh'
set -euo pipefail

REPO=/mnt/novita2/siyuan/workspace/TRELLIS.2
MESH="${MESH:-/mnt/novita2/siyuan/test2/TRELLIS.2/tmp/test_mesh/underwater_plant_pack.glb}"
OUT_DIR="$REPO/results/coart_feat18_20260423_three_branch_ws_v0"
MESH_STEM="$(basename "$MESH")"
MESH_STEM="${MESH_STEM%.*}"
PLY_OUT="${PLY_OUT:-$OUT_DIR/${MESH_STEM}_recon_ema_step0155000.ply}"

cd "$REPO"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[host] $(hostname)  GPU=$CUDA_VISIBLE_DEVICES"
echo "[mesh] $MESH"
echo "[out ] $PLY_OUT"

"$REPO/.venv/bin/python" "$REPO/scripts/coart_reconstruct_mesh.py" \
    --mesh "$MESH" \
    --output_dir "$OUT_DIR" \
    --ema_step 155000 \
    --out_ply "$PLY_OUT"
