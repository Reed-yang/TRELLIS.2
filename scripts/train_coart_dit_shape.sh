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

# NFS triton cache races on concurrent kernel writes from 8 mp.spawn workers
# (Errno 39 ENOTEMPTY in atomic_rename). Use local FS; export a persistent
# local path explicitly (e.g. /var/tmp/...) for cross-reboot reuse.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/trellis2_triton_cache}"
mkdir -p "${TRITON_CACHE_DIR}" "${OUTPUT_DIR}"

# Make sure `import coart.dit` works in subprocesses spawned by mp.spawn.
# The .pth file at .venv/lib/python3.10/site-packages/coart_dit_autoload.pth
# auto-imports coart.dit on every Python startup when this env var is set.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export COART_AUTO_REGISTER_DIT=1

# C3: FA3 default-on for self-attn (cross-attn still FA2 — see F5 future work).
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-flash_attn_3}"

# C4: larger NCCL bucket reduces #allreduce calls (DDP path).
export NCCL_BUCKET_CAP_MB="${NCCL_BUCKET_CAP_MB:-50}"

echo "[run] repo        = ${REPO_ROOT}"
echo "[run] data_root   = ${DATA_ROOT}"
echo "[run] config      = ${CONFIG}"
echo "[run] output_dir  = ${OUTPUT_DIR}"
echo "[run] load_dir    = ${LOAD_DIR:-<none>}"
echo "[run] ckpt        = ${CKPT}"
echo "[run] nproc       = ${NPROC}"
echo "[run] triton cache= ${TRITON_CACHE_DIR}"

# Build train.py CLI args. Using arrays keeps quoting sane.
# IMPORTANT: train.py manages its own multiprocessing via torch.multiprocessing
# spawn (see train.py:143 `mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus)`),
# so we must NOT wrap it in torchrun — running both produces a port conflict
# at master_port=12345 (train.py default) vs torchrun's random standalone port.
# Pass --num_gpus directly and let train.py handle the rest.
TRAIN_ARGS=(
    --config "${CONFIG}"
    --output_dir "${OUTPUT_DIR}"
    --data_dir "${DATA_ROOT}"
    --auto_retry "${AUTO_RETRY}"
    --num_gpus "${NPROC}"
)
if [[ -n "${LOAD_DIR}" ]]; then
    TRAIN_ARGS+=(--load_dir "${LOAD_DIR}" --ckpt "${CKPT}")
fi
TRAIN_ARGS+=("$@")

# Use the entry wrapper that registers coart.dit classes in BOTH the parent
# process AND each mp.spawn worker (workers don't inherit parent imports).
exec "${PY}" "${REPO_ROOT}/scripts/coart_train_dit_entry.py" "${TRAIN_ARGS[@]}"
