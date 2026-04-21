#!/usr/bin/env bash
# Precompute feat18 (.npz per mesh) for ObjaverseXL Sketchfab-style layout:
#   <dataset_root>/<metadata_csv>  with columns sha256, local_path
#
# Defaults target the video_obj copy; override with env vars. Extra CLI args
# are forwarded to precompute_feat18.py (e.g. --limit 20 --verbose).
#
# Examples:
#   bash scripts/precompute_feat18_objaverse_sketchfab.sh
#   NUM_GPUS=1 bash scripts/precompute_feat18_objaverse_sketchfab.sh --limit 10
#   NUM_GPUS=8 MAX_MESH_FILE_MB=80 bash scripts/precompute_feat18_objaverse_sketchfab.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASET_ROOT="${DATASET_ROOT:-/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab}"
METADATA_CSV="${METADATA_CSV:-raw/metadata.csv}"
RESOLUTION="${RESOLUTION:-512}"
NUM_GPUS="${NUM_GPUS:-8}"
LOG_DIR="${LOG_DIR:-${DATASET_ROOT}/logs/precompute_feat18_r${RESOLUTION}_g${NUM_GPUS}}"

COMMON=(
  precompute_feat18.py
  --dataset_root "${DATASET_ROOT}"
  --metadata_csv "${METADATA_CSV}"
  --resolution "${RESOLUTION}"
  --world_size "${NUM_GPUS}"
)

if [[ -n "${MAX_MESH_FILE_MB:-}" ]]; then
  COMMON+=(--max_mesh_file_mb "${MAX_MESH_FILE_MB}")
fi

if [[ "${VERBOSE:-0}" == "1" ]]; then
  COMMON+=(--verbose)
fi

mkdir -p "${LOG_DIR}"

echo "[precompute] repo          = ${REPO_ROOT}"
echo "[precompute] dataset_root  = ${DATASET_ROOT}"
echo "[precompute] metadata_csv  = ${METADATA_CSV}"
echo "[precompute] resolution    = ${RESOLUTION}"
echo "[precompute] num_gpus      = ${NUM_GPUS}"
echo "[precompute] log_dir       = ${LOG_DIR}"
if [[ -n "${MAX_MESH_FILE_MB:-}" ]]; then
  echo "[precompute] max_mesh_file_mb = ${MAX_MESH_FILE_MB}"
fi

if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "NUM_GPUS must be >= 1" >&2
  exit 1
fi

if [[ "${NUM_GPUS}" -eq 1 ]]; then
  python "${COMMON[@]}" --rank 0 "$@"
else
  pids=()
  for ((r = 0; r < NUM_GPUS; r++)); do
    CUDA_VISIBLE_DEVICES="${r}" python "${COMMON[@]}" --rank "${r}" "$@" \
      >"${LOG_DIR}/rank${r}.log" 2>&1 &
    pids+=("$!")
  done
  ec=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      ec=1
    fi
  done
  if [[ "${ec}" -ne 0 ]]; then
    echo "[precompute] one or more ranks failed; check ${LOG_DIR}/rank*.log" >&2
    exit "${ec}"
  fi
fi

echo "[precompute] done. Outputs under ${DATASET_ROOT}/feat18_${RESOLUTION}/ (unless --out_dir was passed)"
