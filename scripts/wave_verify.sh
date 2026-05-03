#!/usr/bin/env bash
# Run a short profile-style training to produce metrics for wave gate comparison.
# Usage: bash scripts/wave_verify.sh <wave_name> [--steps N] [--bs B] [--mode MODE] [--host HOST]
#
# NOTES (2026-05-03 / W1 baseline):
#   * profile_dit.py exposes --result-dir (result JSON target) and --output-root
#     (per-run training output_dir parent), NOT --output-dir. We pass --result-dir
#     so the wave's JSON lands under logs/wave_verify/<wave>_<ts>/.
#   * The result file is named <label>_<ts>.json (not result.json) — see
#     scripts/profiling/profile_dit.py main() step 7.
#   * --mode propagates as trainer.args.parallel_mode override (W2.1+).
#     Valid: "ddp" (default), "zro1", "fsdp2_zero2".
set -euo pipefail
WAVE="${1:?wave name required}"; shift
STEPS=30; BS=8; MODE=ddp; HOST=host-10-240-99-119
while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps) STEPS="$2"; shift 2;;
    --bs)    BS="$2"; shift 2;;
    --mode)  MODE="$2"; shift 2;;
    --host)  HOST="$2"; shift 2;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
TS=$(date +%Y%m%d_%H%M%S)
OUT="logs/wave_verify/${WAVE}_${TS}"
mkdir -p "$OUT"
.venv/bin/python scripts/profiling/profile_dit.py \
    --label "${WAVE}" --host "${HOST}" --num-gpus 8 \
    --warmup-steps 5 --active-steps "${STEPS}" \
    --override "trainer.args.batch_split=2" \
    --override "trainer.args.batch_size_per_gpu=${BS}" \
    --override "trainer.args.parallel_mode=${MODE}" \
    --extra-env SPARSE_ATTN_BACKEND=flash_attn_3 \
    --extra-env NCCL_BUCKET_CAP_MB="${NCCL_BUCKET_CAP_MB:-50}" \
    --result-dir "${OUT}"

# Canonicalize: downstream wave-compare tasks reference ${OUT}/result.json.
# profile_dit writes ${WAVE}_<ts>.json; symlink to result.json for plan-spec compat.
PRODUCED=$(ls -t "${OUT}/${WAVE}_"*.json 2>/dev/null | head -1)
if [[ -n "${PRODUCED}" ]]; then
  ln -sf "$(basename "${PRODUCED}")" "${OUT}/result.json"
  echo "[wave_verify] wrote ${OUT}/result.json -> ${PRODUCED}"
else
  echo "[wave_verify] WARN: no ${WAVE}_*.json found in ${OUT}" >&2
  exit 1
fi
echo "[wave_verify] mode=${MODE} (trainer.args.parallel_mode=${MODE})"
