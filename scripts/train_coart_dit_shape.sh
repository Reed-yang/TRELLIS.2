#!/usr/bin/env bash
# Finetune the 1.3B shape DiT on cached features (coart_dit_data_v0).
#
# Wraps the upstream train.py with our coart.dit JSON config. We launch via
# `python -c` so that `import coart.dit` runs *before* train.py's
# `from trellis2 import datasets, trainers`; that import side-effect-
# registers our two glue classes into the trellis2 namespaces, which is how
# train.py resolves them by string name from the JSON config.
#
# WARM-START TODO:
#   train.py's --load_dir/--ckpt expects misc_step*.pt under
#   <load_dir>/ckpts. The official pretrained DiT ships as a .safetensors
#   blob (see coart.dit.config.PRETRAINED_DIT_CKPT). For a true warm-start,
#   convert the safetensors -> denoiser_step0.pt + a matching misc_step0.pt
#   stub *once* and point --load_dir at the staging dir. See the
#   ``WARMSTART_HINT`` string in coart/dit/config.py for the recipe. Until
#   then, this script trains from scratch.
#
# Override behaviour with env vars (see config block below). Extra CLI args
# pass through to train.py.
set -euo pipefail

# --- locate repo root (script lives in scripts/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- configurable knobs ---
DATA_ROOT="${DATA_ROOT:-/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0}"
CONFIG="${CONFIG:-coart/dit/configs/coart_dit_shape_512_ft.json}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
RUN_TAG="${RUN_TAG:-shape_ft}"
OUTPUT_DIR="${OUTPUT_DIR:-results/coart_dit_shape_${DATE_TAG}_${RUN_TAG}}"
LOAD_DIR="${LOAD_DIR:-}"           # optional warm-start dir (see WARMSTART_HINT)
CKPT="${CKPT:-latest}"
NPROC="${NPROC:-8}"
AUTO_RETRY="${AUTO_RETRY:-3}"
PY="${PY:-${REPO_ROOT}/.venv/bin/python}"

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/trellis2_triton_cache}"
mkdir -p "${TRITON_CACHE_DIR}" "${OUTPUT_DIR}"

# Make sure `import coart.dit` works in subprocesses spawned by torchrun.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "[run] repo        = ${REPO_ROOT}"
echo "[run] data_root   = ${DATA_ROOT}"
echo "[run] config      = ${CONFIG}"
echo "[run] output_dir  = ${OUTPUT_DIR}"
echo "[run] load_dir    = ${LOAD_DIR:-<none>}"
echo "[run] ckpt        = ${CKPT}"
echo "[run] nproc       = ${NPROC}"
echo "[run] triton cache= ${TRITON_CACHE_DIR}"

# Build train.py CLI args. Using arrays keeps quoting sane.
TRAIN_ARGS=(
    --config "${CONFIG}"
    --output_dir "${OUTPUT_DIR}"
    --data_dir "${DATA_ROOT}"
    --auto_retry "${AUTO_RETRY}"
)
if [[ -n "${LOAD_DIR}" ]]; then
    TRAIN_ARGS+=(--load_dir "${LOAD_DIR}" --ckpt "${CKPT}")
fi
TRAIN_ARGS+=("$@")

# Side-effect-import coart.dit before train.py's `from trellis2 import ...`
# so our dataset + trainer classes are registered in the trellis2 namespaces.
SHIM='import sys, runpy; import coart.dit; sys.argv[0] = "train.py"; runpy.run_path("train.py", run_name="__main__")'

exec torchrun --standalone --nproc_per_node="${NPROC}" \
    --no-python \
    "${PY}" -c "${SHIM}" "${TRAIN_ARGS[@]}"
