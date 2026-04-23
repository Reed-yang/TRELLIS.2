#!/usr/bin/env bash
# Launch precompute_feat18.py across the full easy→hard ranked list.
# Supports multi-node + multi-GPU-per-node parallelism, and is fully
# resume-safe thanks to precompute_feat18.py's own per-mesh npz + .failed
# sentinel logic.
#
# Per-node invocation (run on EACH node separately):
#
#   NODE_IDX=0 NODE_COUNT=4 GPUS_PER_NODE=8 \
#     bash scripts/preprocess-by-rank/launch_full_rank.sh
#
#   NODE_IDX=1 NODE_COUNT=4 GPUS_PER_NODE=8 \
#     bash scripts/preprocess-by-rank/launch_full_rank.sh
#   ...
#
# Single node (default):
#
#   NODE_IDX=0 NODE_COUNT=1 GPUS_PER_NODE=8 \
#     bash scripts/preprocess-by-rank/launch_full_rank.sh
#
# Resume: just re-run the exact same command. Already-encoded npz files
# in OUT_DIR/data/ are skipped; .failed sentinels are skipped. No flag
# needed, no state file to manage.
#
# Env vars (all optional except NODE_IDX / NODE_COUNT when multi-node):
#   NODE_IDX         — 0-based node index                   (default 0)
#   NODE_COUNT       — total number of nodes                (default 1)
#   GPUS_PER_NODE    — GPUs per node                        (default 8)
#   DATASET_ROOT     — absolute path containing raw/...     (default /mnt/novita2/...)
#   METADATA_CSV     — CSV with sha256,local_path columns   (default full_ranked.csv)
#   RESOLUTION       — voxel resolution                     (default 512)
#   OUT_DIR          — output feat18 dir                    (default DATASET_ROOT/feat18_R)
#   LOG_DIR          — rank log dir                         (default OUT_DIR/../logs/precompute_feat18_...)
#   NUM_WORKERS      — override per-rank worker count       (default auto = (nproc-4)/GPUS_PER_NODE)
#   MAX_MESH_FILE_MB — skip meshes larger than this on disk (default unset = no limit)
#   EXTRA_ARGS       — extra args forwarded to precompute_feat18.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

NODE_IDX="${NODE_IDX:-0}"
NODE_COUNT="${NODE_COUNT:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
# Asymmetric-cluster knobs (all optional):
#   WORLD_SIZE    — total ranks across all nodes (default NODE_COUNT*GPUS_PER_NODE).
#                   Override when nodes have different GPUS_PER_NODE.
#   RANK_START    — global rank index this node's local_rank=0 maps to
#                   (default NODE_IDX*GPUS_PER_NODE). Override when running
#                   an uneven split.
#   CUDA_DEVICES  — comma-separated CUDA device indices for local_rank 0..N-1
#                   (default 0,1,...,GPUS_PER_NODE-1). Override to cherry-pick
#                   idle GPUs on a shared node.
WORLD_SIZE="${WORLD_SIZE:-$((NODE_COUNT * GPUS_PER_NODE))}"
RANK_START="${RANK_START:-$((NODE_IDX * GPUS_PER_NODE))}"
CUDA_DEVICES="${CUDA_DEVICES:-}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab}"
METADATA_CSV="${METADATA_CSV:-${REPO_ROOT}/scripts/preprocess-by-rank/out/full_ranked.csv}"
RESOLUTION="${RESOLUTION:-512}"
OUT_DIR="${OUT_DIR:-${DATASET_ROOT}/feat18_${RESOLUTION}}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/../logs/precompute_feat18_r${RESOLUTION}_full_n${NODE_COUNT}x${GPUS_PER_NODE}}"

# Validate integers so we fail loud on NODE_IDX=abc etc.
for v in NODE_IDX NODE_COUNT GPUS_PER_NODE RESOLUTION; do
  if [[ ! "${!v}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ${v}='${!v}' must be a non-negative integer" >&2
    exit 2
  fi
done

if [[ "${NODE_IDX}" -ge "${NODE_COUNT}" ]]; then
  echo "ERROR: NODE_IDX=${NODE_IDX} must be < NODE_COUNT=${NODE_COUNT}" >&2
  exit 2
fi
if [[ "${GPUS_PER_NODE}" -lt 1 ]]; then
  echo "ERROR: GPUS_PER_NODE must be >= 1" >&2
  exit 2
fi
if [[ "${WORLD_SIZE}" -lt "$((RANK_START + GPUS_PER_NODE))" ]]; then
  echo "ERROR: WORLD_SIZE=${WORLD_SIZE} must be >= RANK_START+GPUS_PER_NODE=$((RANK_START + GPUS_PER_NODE))" >&2
  exit 2
fi

# Parse CUDA_DEVICES into an array; default to 0,1,...,GPUS_PER_NODE-1
if [[ -n "${CUDA_DEVICES}" ]]; then
  IFS=',' read -ra CUDA_ARR <<<"${CUDA_DEVICES}"
  if [[ "${#CUDA_ARR[@]}" -ne "${GPUS_PER_NODE}" ]]; then
    echo "ERROR: CUDA_DEVICES (${CUDA_DEVICES}) has ${#CUDA_ARR[@]} entries but GPUS_PER_NODE=${GPUS_PER_NODE}" >&2
    exit 2
  fi
else
  CUDA_ARR=()
  for ((i = 0; i < GPUS_PER_NODE; i++)); do
    CUDA_ARR+=("${i}")
  done
fi

if [[ ! -f "${METADATA_CSV}" ]]; then
  echo "ERROR: METADATA_CSV not found: ${METADATA_CSV}" >&2
  exit 2
fi

# Fall back to PATH's python if the venv binary is missing — useful for
# one-off smoke tests on a machine without the venv provisioned.
if [[ ! -x "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null; then
    echo "WARN: ${PYTHON_BIN} not found; falling back to $(command -v python)" >&2
    PYTHON_BIN="$(command -v python)"
  else
    echo "ERROR: no usable python interpreter (tried ${PYTHON_BIN})" >&2
    exit 2
  fi
fi

# Verify we can actually write to OUT_DIR before spawning 8 worker procs
# that will all crash at finalisation time. If OUT_DIR doesn't exist we
# create it; if it does, we probe a tmp file.
if [[ -d "${OUT_DIR}" ]]; then
  probe="${OUT_DIR}/.launch_probe.$$"
  if ! (touch "${probe}" && rm -f "${probe}") 2>/dev/null; then
    owner="$(stat -c '%U:%G' "${OUT_DIR}" 2>/dev/null || echo unknown)"
    cat >&2 <<EOM
ERROR: Cannot write to OUT_DIR=${OUT_DIR}
       Directory is owned by ${owner}; current user is $(id -un).

       Options:
         1) Fix perms:   sudo chown -R $(id -un):$(id -gn) "${OUT_DIR}"
                      OR sudo chmod -R g+rwX "${OUT_DIR}" && sudo chgrp -R \$(id -gn) "${OUT_DIR}"
         2) Use a different OUT_DIR (loses reuse of existing npz unless you
            copy/move /mnt/novita2/data/.../feat18_${RESOLUTION}/data/*.npz
            into the new location):
              OUT_DIR=/path/you/own  bash \$0
         3) Re-run this launcher as user ${owner%%:*}.
EOM
    exit 3
  fi
else
  if ! mkdir -p "${OUT_DIR}/data" 2>/dev/null; then
    echo "ERROR: Cannot mkdir OUT_DIR=${OUT_DIR}" >&2
    exit 3
  fi
fi

# CPU budget across the local node's GPUs (ignore other nodes' budget).
# precompute_feat18.py's default formula divides by world_size which is
# incorrect under multi-node, so we pass --num_workers explicitly.
if [[ -z "${NUM_WORKERS:-}" ]]; then
  CPU_TOTAL="$(nproc --all 2>/dev/null || echo 16)"
  PER_RANK=$(( (CPU_TOTAL - 4) / GPUS_PER_NODE ))
  if [[ "${PER_RANK}" -lt 1 ]]; then PER_RANK=1; fi
  NUM_WORKERS="${PER_RANK}"
fi

mkdir -p "${LOG_DIR}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_DIR}/node${NODE_IDX}_launch_${RUN_STAMP}.log"

echo "==================================================================="
echo "[full-rank-launch] repo_root     = ${REPO_ROOT}"
echo "[full-rank-launch] dataset_root  = ${DATASET_ROOT}"
echo "[full-rank-launch] metadata_csv  = ${METADATA_CSV}"
echo "[full-rank-launch] resolution    = ${RESOLUTION}"
echo "[full-rank-launch] out_dir       = ${OUT_DIR}"
echo "[full-rank-launch] log_dir       = ${LOG_DIR}"
echo "[full-rank-launch] run_stamp     = ${RUN_STAMP}"
echo "[full-rank-launch] node          = ${NODE_IDX} / ${NODE_COUNT}"
echo "[full-rank-launch] gpus/node     = ${GPUS_PER_NODE}"
echo "[full-rank-launch] world_size    = ${WORLD_SIZE}"
echo "[full-rank-launch] rank_start    = ${RANK_START}  (local_rank 0..${#CUDA_ARR[@]}-1 → global ${RANK_START}..$((RANK_START + GPUS_PER_NODE - 1)))"
echo "[full-rank-launch] cuda_devices  = ${CUDA_ARR[*]}"
echo "[full-rank-launch] num_workers   = ${NUM_WORKERS} per rank (local cpu_total=${CPU_TOTAL:-NA})"
echo "[full-rank-launch] python_bin    = ${PYTHON_BIN}"
echo "==================================================================="

# Mirror the above banner to the launch log for audit trail.
exec > >(tee -a "${RUN_LOG}") 2>&1

# One-time resume snapshot so the user sees what's already done.
"${PYTHON_BIN}" - <<PY || true
import os, glob
d = os.path.join("${OUT_DIR}", "data")
if not os.path.isdir(d):
    print(f"[resume-scan] out_dir has no data/ yet; this is a fresh start")
else:
    npz = len(glob.glob(os.path.join(d, "*.npz")))
    fail = len(glob.glob(os.path.join(d, "*.npz.failed")))
    print(f"[resume-scan] already-done npz      = {npz}")
    print(f"[resume-scan] failed sentinels      = {fail}")
PY

COMMON=(
  "${REPO_ROOT}/precompute_feat18.py"
  --dataset_root "${DATASET_ROOT}"
  --metadata_csv "${METADATA_CSV}"
  --resolution "${RESOLUTION}"
  --world_size "${WORLD_SIZE}"
  --out_dir "${OUT_DIR}"
  --num_workers "${NUM_WORKERS}"
  --skip_done_fast
)
if [[ -n "${MAX_MESH_FILE_MB:-}" ]]; then
  COMMON+=(--max_mesh_file_mb "${MAX_MESH_FILE_MB}")
fi
# Per-mesh wall-time guard. 0 / unset = disabled; typical long-tail
# killer at res=512 is 180 (3 min). Meshes exceeding this are marked
# .failed and skipped without retry — prevents one pathological sample
# from stalling a rank for hours inside the OOM shrink-budget loop.
if [[ -n "${PER_MESH_TIMEOUT_S:-}" && "${PER_MESH_TIMEOUT_S}" != "0" ]]; then
  COMMON+=(--per_mesh_timeout_s "${PER_MESH_TIMEOUT_S}")
fi
# Forward any user-supplied extra args (e.g. --limit 1000 for smoke tests).
EXTRA_ARGS_ARR=()
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARR=(${EXTRA_ARGS})
fi

# Launch GPUS_PER_NODE parallel worker processes on this node.
pids=()
for ((local_rank = 0; local_rank < GPUS_PER_NODE; local_rank++)); do
  global_rank=$((RANK_START + local_rank))
  cuda_dev="${CUDA_ARR[$local_rank]}"
  # Rotate rank log per launch so resume restarts don't clobber the
  # previous session's tail. Past logs remain for post-mortem /
  # aggregation via status.py.
  rank_log="${LOG_DIR}/node${NODE_IDX}_local${local_rank}_global${global_rank}_cuda${cuda_dev}.${RUN_STAMP}.log"
  echo "[launch] local=${local_rank} global=${global_rank} cuda=${cuda_dev} → ${rank_log}"
  CUDA_VISIBLE_DEVICES="${cuda_dev}" nohup "${PYTHON_BIN}" \
    "${COMMON[@]}" "${EXTRA_ARGS_ARR[@]}" "$@" \
    --rank "${global_rank}" \
    >"${rank_log}" 2>&1 &
  pids+=("$!")
done

echo "[launch] ${#pids[@]} worker pids on this node: ${pids[*]}"

# Wait for all local ranks; report which (if any) failed.
ec=0
fail_ranks=()
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  if ! wait "${pid}"; then
    ec=1
    fail_ranks+=("$i")
  fi
done

if [[ "${ec}" -ne 0 ]]; then
  echo "[launch] FAILED local ranks: ${fail_ranks[*]}"
  echo "[launch] check per-rank logs under ${LOG_DIR}"
fi

echo "[launch] node ${NODE_IDX}/${NODE_COUNT} complete (exit=${ec})"
exit "${ec}"
