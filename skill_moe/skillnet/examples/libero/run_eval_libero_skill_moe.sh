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
TASK_SUITE="${TASK_SUITE:-libero_skill_obj}"
NUM_TRIALS="${NUM_TRIALS:-50}"
VIDEO_OUT_PATH="${VIDEO_OUT_PATH:-data/libero/videos/skill_moe_libero_skill_obj}"
SKILL_ANNOTATION_PATH="${SKILL_ANNOTATION_PATH:-examples/libero/annotations/libero_skill_obj_annotations.json}"
START_SERVER="${START_SERVER:-1}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-60}"
SERVER_LOG_PATH="${SERVER_LOG_PATH:-data/libero/server_logs/${CONFIG_NAME}_${PORT}.log}"
FAIL_FAST="${FAIL_FAST:-1}"
ZERO_SHOT="${ZERO_SHOT:-1}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON_CMD=(python)
else
  PYTHON_CMD=(uv run python)
fi

export PYTHONPATH="${SKILLNET_ROOT}/third_party/libero:${PYTHONPATH:-}"
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
  if [[ ! -e "${CKPT_DIR}" ]]; then
    echo "CKPT_DIR does not exist: ${CKPT_DIR}" >&2
    exit 2
  fi

  mkdir -p "$(dirname "${SERVER_LOG_PATH}")"
  CUDA_VISIBLE_DEVICES="${SERVER_GPU}" "${PYTHON_CMD[@]}" scripts/serve_policy_moe_skill.py \
    --port="${PORT}" \
    policy:checkpoint \
    --policy.config="${CONFIG_NAME}" \
    --policy.dir="${CKPT_DIR}" >"${SERVER_LOG_PATH}" 2>&1 &
  SERVER_PID="$!"
  echo "Policy server log: ${SERVER_LOG_PATH}"

  SERVER_READY=0
  START_TIME=${SECONDS}
  while (( SECONDS - START_TIME < SERVER_WAIT_SECONDS )); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "Policy server exited before becoming ready. Last log lines:" >&2
      tail -n 80 "${SERVER_LOG_PATH}" >&2 || true
      exit 1
    fi
    if HOST="${HOST}" PORT="${PORT}" "${PYTHON_CMD[@]}" - <<'PY' >/dev/null 2>&1
import os
import socket

host = os.environ["HOST"]
port = int(os.environ["PORT"])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(1.0)
    raise SystemExit(0 if sock.connect_ex((host, port)) == 0 else 1)
PY
    then
      SERVER_READY=1
      break
    fi
    sleep 2
  done

  if [[ "${SERVER_READY}" != "1" ]]; then
    echo "Policy server did not accept connections on ${HOST}:${PORT} within ${SERVER_WAIT_SECONDS}s." >&2
    echo "Last log lines:" >&2
    tail -n 80 "${SERVER_LOG_PATH}" >&2 || true
    exit 1
  fi
fi

EVAL_ARGS=(
  examples/libero/main_skill_obj_test_moe_skill.py
  --host="${HOST}"
  --port="${PORT}"
  --task-suite-name="${TASK_SUITE}"
  --num-trials-per-task="${NUM_TRIALS}"
  --video-out-path="${VIDEO_OUT_PATH}"
  --skill-annotation-path="${SKILL_ANNOTATION_PATH}"
)
if [[ "${FAIL_FAST}" == "1" ]]; then
  EVAL_ARGS+=(--fail-fast)
fi
if [[ "${ZERO_SHOT}" == "0" || "${ZERO_SHOT}" == "false" || "${ZERO_SHOT}" == "False" ]]; then
  EVAL_ARGS+=(--no-zero-shot)
else
  EVAL_ARGS+=(--zero-shot)
fi

"${PYTHON_CMD[@]}" "${EVAL_ARGS[@]}" "$@"
