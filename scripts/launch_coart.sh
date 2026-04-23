#!/bin/bash
# Launch coart.vae training on a single node (default 8 GPUs via torchrun).
#
# Usage:
#   scripts/launch_coart.sh [torchrun-style args...] [coart.vae CLI args...]
#
# Sources <repo>/.coart.env (gitignored) for WANDB_API_KEY, OMP_NUM_THREADS, etc.
# Expects .coart.env to exist; if missing, exits with a clear error.
#
# Example:
#   scripts/launch_coart.sh --run_tag three_branch_ws_v0 \
#       --io_arch three_branch --warmstart_io \
#       --max_steps 200000 --max_voxels 500000
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

ENV_FILE="$REPO/.coart.env"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "[launch_coart] ERROR: $ENV_FILE not found." >&2
    echo "[launch_coart] Create it with WANDB_API_KEY + any other env exports." >&2
    echo "[launch_coart] File must be chmod 600 and is gitignored by default." >&2
    exit 1
fi

# shellcheck disable=SC1090
set -a
source "$ENV_FILE"
set +a

: "${OMP_NUM_THREADS:=8}"
export OMP_NUM_THREADS

# Determine NPROC (default = visible CUDA devices)
NPROC="${COART_NPROC_PER_NODE:-}"
if [[ -z "$NPROC" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NPROC=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    else
        NPROC=1
    fi
fi

echo "[launch_coart] repo=$REPO"
echo "[launch_coart] nproc_per_node=$NPROC"
echo "[launch_coart] OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "[launch_coart] wandb entity=${WANDB_ENTITY:-<default>} project=${WANDB_PROJECT:-<default>}"

exec "$REPO/.venv/bin/torchrun" \
    --nproc_per_node="$NPROC" \
    --standalone \
    -m coart.vae \
    "$@"
