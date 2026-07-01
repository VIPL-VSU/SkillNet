#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLNET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SKILLNET_ROOT}"

CONFIG_NAME="${CONFIG_NAME:-pi05_libero_moe_skill_4_90}"
EXP_NAME="${EXP_NAME:-libero90_moe_skill}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
KEEP_PERIOD="${KEEP_PERIOD:-1000}"
COMPUTE_NORM_STATS="${COMPUTE_NORM_STATS:-1}"
SKILLNET_RELEASE_HF_NAMESPACE="${SKILLNET_RELEASE_HF_NAMESPACE:-jsw19}"
LIBERO_REPO_ID="${SKILLNET_LIBERO90_REPO_ID:-${SKILLNET_RELEASE_HF_NAMESPACE}/libero_90_v1}"
NORM_STATS_PATH="${NORM_STATS_PATH:-assets/${CONFIG_NAME}/${LIBERO_REPO_ID}/norm_stats.json}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON_CMD=(python)
else
  PYTHON_CMD=(uv run python)
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ "${COMPUTE_NORM_STATS}" == "1" && ! -f "${NORM_STATS_PATH}" ]]; then
  "${PYTHON_CMD[@]}" scripts/compute_norm_stats_moe_skill.py "${CONFIG_NAME}"
fi

"${PYTHON_CMD[@]}" scripts/train_moe_skill.py "${CONFIG_NAME}" \
  --exp-name="${EXP_NAME}" \
  --save-interval="${SAVE_INTERVAL}" \
  --keep-period="${KEEP_PERIOD}" \
  --no-wandb-enabled \
  "$@"
