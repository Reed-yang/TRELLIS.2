#!/usr/bin/env bash
# Fine-tune the TRELLIS.2 shape SC-VAE on precomputed 18-ch feature shards.
#
# Speed tricks enabled here (see bench A→E in train_finetune_feat18.py history):
#   --use_bf16                  : bfloat16 autocast on encoder/decoder forward
#                                 (losses kept in fp32 for stability).
#   --bucket_sampler            : group similarly-sized samples into the same
#                                 global batch so ranks don't wait on the
#                                 slowest one.
#   --bucket_sort_mode ascending: small → large within each epoch; makes
#                                 triton autotune hit monotonically and
#                                 removes minutes-long "big bucket" stalls.
#   --max_voxels 1500000        : deterministically subsample cubes on the
#                                 21/737 largest samples (sha-seeded, stable
#                                 across epochs, so rulebook cache stays hot).
#   TRITON_CACHE_DIR            : persist triton autotune results across runs.
#
# Override anything by passing extra CLI args to this script, e.g.:
#   bash scripts/train_finetune_feat18.sh --max_steps 100000 --lr 5e-5
set -euo pipefail

# ── cd to repo root (script lives in scripts/) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── config (override via env vars) ──
DATA_ROOT="${DATA_ROOT:-/mnt/novita2/siyuan/workspace/TRELLIS.2/datasets/ObjaverseXL_sketchfab/feat18_512}"
OUTPUT_DIR="${OUTPUT_DIR:-results/finetune_feat18_1k_ddp}"
MAX_STEPS="${MAX_STEPS:-50000}"
MAX_VOXELS="${MAX_VOXELS:-1500000}"
NPROC="${NPROC:-8}"

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/trellis2_triton_cache}"
mkdir -p "${TRITON_CACHE_DIR}" "${OUTPUT_DIR}"

echo "[run] repo        = ${REPO_ROOT}"
echo "[run] data_root   = ${DATA_ROOT}"
echo "[run] output_dir  = ${OUTPUT_DIR}"
echo "[run] max_steps   = ${MAX_STEPS}"
echo "[run] max_voxels  = ${MAX_VOXELS}"
echo "[run] nproc       = ${NPROC}"
echo "[run] triton cache= ${TRITON_CACHE_DIR}"

exec torchrun --standalone --nproc_per_node="${NPROC}" \
    train_finetune_feat18.py \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_steps "${MAX_STEPS}" \
    --use_bf16 \
    --bucket_sampler --bucket_sort_mode ascending \
    --max_voxels "${MAX_VOXELS}" \
    "$@"
