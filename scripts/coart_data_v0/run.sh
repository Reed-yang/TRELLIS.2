#!/usr/bin/env bash
# Stage launcher for coart DiT data pipeline v0.
# Mirrors the env-var pattern of scripts/precompute_feat18_objaverse_sketchfab.sh.
#
# Required env vars:
#   STAGE                   render | dino | slat | manifest
#
# Common env vars (defaults shown):
#   COART_DATA_ROOT         /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0
#   DATASET_ROOT            /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab
#   INSTANCES               instances_10k.csv (relative to COART_DATA_ROOT)
#   NODES                   "117 118 119"
#   NUM_GPUS_PER_NODE       8
#   LIMIT                   "" (smoke runs use --limit)
#   PYTHON                  .venv/bin/python
#   REPO_ROOT               (auto-detected from this script's location)
#
# STAGE=slat additionally requires:
#   VAE_CKPT                results/coart_feat18_20260423_three_branch_ws_v0/ema_0.9999_enc_step0155000.pt
#   VAE_TAG                 vae_three_branch_ws_v0_ema_s0155000
#   VAE_IO_ARCH             three_branch
#   STATS_NPZ               <DATASET_ROOT>/feat18_512/stats_global.npz
#
# Examples:
#   STAGE=render NODES="117 118 119" NUM_GPUS_PER_NODE=8 bash scripts/coart_data_v0/run.sh
#   STAGE=dino NODES="118 119" bash scripts/coart_data_v0/run.sh
#   STAGE=slat VAE_CKPT=... VAE_TAG=... NODES="119" bash scripts/coart_data_v0/run.sh
#   STAGE=manifest bash scripts/coart_data_v0/run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

STAGE="${STAGE:?STAGE env var required (render | dino | slat | manifest)}"
COART_DATA_ROOT="${COART_DATA_ROOT:-/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab}"
INSTANCES_REL="${INSTANCES:-instances_10k.csv}"
INSTANCES="${COART_DATA_ROOT}/${INSTANCES_REL}"
NODES="${NODES:-117 118 119}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
# Resolve PYTHON to absolute path so it works after `cd` in stage templates.
_PYTHON_DEFAULT=".venv/bin/python"
PYTHON="${PYTHON:-${_PYTHON_DEFAULT}}"
case "${PYTHON}" in
    /*) ;;  # already absolute
    *)  PYTHON="${REPO_ROOT}/${PYTHON}" ;;
esac
LIMIT="${LIMIT:-}"
LIMIT_ARG=""
[ -n "${LIMIT}" ] && LIMIT_ARG="--limit ${LIMIT}"

LOG_DIR="${COART_DATA_ROOT}/logs"
mkdir -p "${LOG_DIR}"

NODES_ARR=( ${NODES} )
WORLD_SIZE=$(( ${#NODES_ARR[@]} * NUM_GPUS_PER_NODE ))
echo "[run] STAGE=${STAGE} NODES=(${NODES}) NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE} WORLD_SIZE=${WORLD_SIZE}"

dispatch_python_per_rank() {
    local cmd_template="$1"
    local stage_log="$2"
    local pids=()
    local rank=0
    for node in "${NODES_ARR[@]}"; do
        for gpu in $(seq 0 $((NUM_GPUS_PER_NODE - 1))); do
            local log_file="${LOG_DIR}/${stage_log}_rank$(printf '%02d' ${rank}).log"
            local cmd="$(printf "${cmd_template}" "${gpu}" "${rank}" "${WORLD_SIZE}")"
            local script="${REPO_ROOT}/tmp/run_${STAGE}_rank${rank}.sh"
            mkdir -p "${REPO_ROOT}/tmp"
            cat > "${script}" <<EOF
#!/usr/bin/env bash
cd "${REPO_ROOT}"
${cmd} > "${log_file}" 2>&1
EOF
            chmod +x "${script}"
            ssh "host-10-240-99-${node}" "bash ${script}" &
            pids+=($!)
            rank=$((rank + 1))
        done
    done
    echo "[run] launched ${#pids[@]} ranks; waiting..."
    local exits=()
    for pid in "${pids[@]}"; do
        if wait ${pid}; then exits+=(0); else exits+=($?); fi
    done
    echo "[run] per-rank exit codes: ${exits[@]}"
    local n_fail=0
    for code in "${exits[@]}"; do [ ${code} -ne 0 ] && n_fail=$((n_fail+1)); done
    [ ${n_fail} -ne 0 ] && echo "[run] WARN: ${n_fail}/${#exits[@]} ranks failed"
}

case "${STAGE}" in
    render)
        SHA_LIST="${COART_DATA_ROOT}/_render_sha_list.txt"
        cut -d, -f1 "${INSTANCES}" | tail -n +2 > "${SHA_LIST}"
        # --max_workers 1: each rank runs ONE Blender at a time. With 24 ranks
        # cluster-wide and Blender CYCLES eating all CPU cores by default, more
        # workers per rank causes thread-oversubscription on 128-core boxes
        # (192-way oversubscription dropped throughput to 13/min instead of the
        # uncontended 19/min). Override via MAX_WORKERS env if needed.
        MAX_WORKERS_ARG="--max_workers ${MAX_WORKERS:-1}"
        TPL="cd ${REPO_ROOT}/data_toolkit && CUDA_VISIBLE_DEVICES=%s ${PYTHON} render_cond.py ObjaverseXL --root ${DATASET_ROOT} --download_root ${DATASET_ROOT} --render_cond_root ${COART_DATA_ROOT} --rank %s --world_size %s --num_cond_views 16 ${MAX_WORKERS_ARG} --instances ${SHA_LIST}"
        dispatch_python_per_rank "${TPL}" "render"
        ;;
    dino)
        TPL="CUDA_VISIBLE_DEVICES=%s ${PYTHON} scripts/coart_data_v0/coart_cache_dino.py --instances ${INSTANCES} --renders_dir ${COART_DATA_ROOT}/renders_cond --out_dir ${COART_DATA_ROOT}/dino_l16_s512 --rank %s --world_size %s ${LIMIT_ARG}"
        dispatch_python_per_rank "${TPL}" "dino"
        ;;
    slat)
        VAE_CKPT="${VAE_CKPT:?VAE_CKPT env var required for STAGE=slat}"
        VAE_TAG="${VAE_TAG:?VAE_TAG env var required for STAGE=slat}"
        VAE_IO_ARCH="${VAE_IO_ARCH:-three_branch}"
        STATS_NPZ="${STATS_NPZ:-${DATASET_ROOT}/feat18_512/stats_global.npz}"
        TPL="CUDA_VISIBLE_DEVICES=%s ${PYTHON} scripts/coart_data_v0/coart_cache_slat.py --instances ${INSTANCES} --feat18_dir ${DATASET_ROOT}/feat18_512/data --vae_ckpt ${VAE_CKPT} --vae_tag ${VAE_TAG} --vae_io_arch ${VAE_IO_ARCH} --stats_npz ${STATS_NPZ} --out_root ${COART_DATA_ROOT}/slat --rank %s --world_size %s ${LIMIT_ARG}"
        dispatch_python_per_rank "${TPL}" "slat_${VAE_TAG}"
        ;;
    manifest)
        cd "${REPO_ROOT}"
        ${PYTHON} scripts/coart_data_v0/build_manifest.py \
            --instances "${INSTANCES}" \
            --renders_dir "${COART_DATA_ROOT}/renders_cond" \
            --dino_dir "${COART_DATA_ROOT}/dino_l16_s512" \
            --slat_root "${COART_DATA_ROOT}/slat" \
            --out "${COART_DATA_ROOT}/manifest.csv"
        ;;
    *)
        echo "Unknown STAGE=${STAGE}"; exit 2
        ;;
esac

echo "[run] STAGE=${STAGE} complete"
