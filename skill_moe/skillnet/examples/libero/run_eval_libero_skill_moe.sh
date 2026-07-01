#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLNET_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SKILLNET_ROOT}"

CONFIG_NAME="${CONFIG_NAME:-pi05_libero_moe_skill_4_90}"
CKPT_DIR="${CKPT_DIR:-}"
PORT="${PORT:-8056}"
HOST="${HOST:-127.0.0.1}"
SERVER_GPU="${SERVER_GPU:-0}"
TASK_SUITE="${TASK_SUITE:-libero_skill}"
NUM_TRIALS="${NUM_TRIALS:-50}"
VIDEO_OUT_PATH="${VIDEO_OUT_PATH:-data/libero/videos/skill_moe_libero_skill_obj}"
SKILL_ANNOTATION_PATH="${SKILL_ANNOTATION_PATH:-examples/libero/annotations/libero_skill_obj_annotations.json}"
START_SERVER="${START_SERVER:-1}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-60}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON_CMD=(python)
else
  PYTHON_CMD=(uv run python)
fi

export PYTHONPATH="${PYTHONPATH:-}:${SKILLNET_ROOT}/third_party/libero"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "${START_SERVER}" == "1" ]]; then
  if [[ -z "${CKPT_DIR}" ]]; then
    echo "CKPT_DIR must point to a trained checkpoint when START_SERVER=1." >&2
    exit 2
  fi

  CUDA_VISIBLE_DEVICES="${SERVER_GPU}" "${PYTHON_CMD[@]}" scripts/serve_policy_moe_skill.py \
    --port="${PORT}" \
    policy:checkpoint \
    --policy.config="${CONFIG_NAME}" \
    --policy.dir="${CKPT_DIR}" &
  SERVER_PID="$!"
  sleep "${SERVER_WAIT_SECONDS}"
fi

"${PYTHON_CMD[@]}" examples/libero/main_skill_obj_test_moe_skill.py \
  --host="${HOST}" \
  --port="${PORT}" \
  --task-suite-name="${TASK_SUITE}" \
  --num-trials-per-task="${NUM_TRIALS}" \
  --video-out-path="${VIDEO_OUT_PATH}" \
  --skill-annotation-path="${SKILL_ANNOTATION_PATH}" \
  "$@"
