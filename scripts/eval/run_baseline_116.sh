#!/bin/bash
#
# Run baseline experiments on node 116.
# Tries slurm first; if node is occupied, falls back to SSH direct execution.
#
# Usage:
#   bash scripts/eval/run_baseline_116.sh              # Run all steps
#   bash scripts/eval/run_baseline_116.sh --step 1     # Run step 1 only
#   bash scripts/eval/run_baseline_116.sh --ssh         # Force SSH mode (skip slurm)
#

set -e

PROJECT_DIR="/mnt/novita2/siyuan/workspace/TRELLIS.2"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
SCRIPT="${PROJECT_DIR}/scripts/eval/baseline_experiments.py"
NODE="host-10-240-99-116"
LOG_DIR="${PROJECT_DIR}/results/baseline_experiments/logs"

# Parse args
USE_SSH=false
EXTRA_ARGS=""
for arg in "$@"; do
    if [ "$arg" = "--ssh" ]; then
        USE_SSH=true
    else
        EXTRA_ARGS="${EXTRA_ARGS} ${arg}"
    fi
done

# Default: run all steps
if [ -z "$EXTRA_ARGS" ]; then
    EXTRA_ARGS="--step all"
fi

mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# --- Generate synthetic models first (local, no GPU needed) ---
echo "=== Generating synthetic test models ==="
cd "${PROJECT_DIR}"
${PYTHON} scripts/eval/baseline_generate_models.py --output-dir results/baseline_experiments/data

# --- Try slurm ---
if [ "$USE_SSH" = false ]; then
    echo "=== Trying slurm submission to ${NODE} ==="

    SLURM_SCRIPT=$(mktemp /tmp/baseline_slurm_XXXXXX.sh)
    cat > "${SLURM_SCRIPT}" << 'SLURM_EOF'
#!/bin/bash
#SBATCH --job-name=baseline_exp
#SBATCH --nodelist=host-10-240-99-116
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=4:00:00
#SBATCH --output=LOGDIR/slurm_%j.out
#SBATCH --error=LOGDIR/slurm_%j.err

cd PROJECT_DIR
export HF_HUB_CACHE="${PROJECT_DIR}/pretrained"

echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L | wc -l)"

PYTHON SCRIPT --num-gpus 8 EXTRA_ARGS

echo "Job finished at $(date)"
SLURM_EOF

    # Replace placeholders
    sed -i "s|PROJECT_DIR|${PROJECT_DIR}|g" "${SLURM_SCRIPT}"
    sed -i "s|PYTHON|${PYTHON}|g" "${SLURM_SCRIPT}"
    sed -i "s|SCRIPT|${SCRIPT}|g" "${SLURM_SCRIPT}"
    sed -i "s|LOGDIR|${LOG_DIR}|g" "${SLURM_SCRIPT}"
    sed -i "s|EXTRA_ARGS|${EXTRA_ARGS}|g" "${SLURM_SCRIPT}"

    if sbatch "${SLURM_SCRIPT}" 2>/dev/null; then
        echo "Slurm job submitted. Check logs at ${LOG_DIR}/"
        echo "Monitor: squeue -u \$USER"
        rm -f "${SLURM_SCRIPT}"
        exit 0
    else
        echo "Slurm submission failed. Falling back to SSH..."
        rm -f "${SLURM_SCRIPT}"
    fi
fi

# --- Fallback: SSH direct execution ---
echo "=== Running via SSH on ${NODE} ==="
LOG_FILE="${LOG_DIR}/ssh_${TIMESTAMP}.log"

ssh "${NODE}" "cd ${PROJECT_DIR} && \
    export HF_HUB_CACHE=${PROJECT_DIR}/pretrained && \
    nohup ${PYTHON} ${SCRIPT} --num-gpus 8 ${EXTRA_ARGS} \
    > ${LOG_FILE} 2>&1 &
    echo \"PID: \$!\"
    echo \"Log: ${LOG_FILE}\"
"

echo ""
echo "=== Experiment launched in background ==="
echo "Log file: ${LOG_FILE}"
echo "Monitor:  tail -f ${LOG_FILE}"
echo "Check:    ssh ${NODE} 'ps aux | grep baseline_experiments'"
