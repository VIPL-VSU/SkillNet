#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLNET_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SKILLNET_REPO_ROOT="$(cd "${SKILLNET_ROOT}/../.." && pwd)"
cd "${SKILLNET_ROOT}"

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-}"
CONFIG_NAME="${CONFIG_NAME:-pi05_robotwin_moe_skill_transfer}"
CKPT_DIR="${CKPT_DIR:-}"
TRANSFER_TASK="${TRANSFER_TASK:-}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
TASK_SET="${TASK_SET:-transfer}"
TASKS="${TASKS:-}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8098}"
SERVER_GPU="${SERVER_GPU:-0}"
START_SERVER="${START_SERVER:-1}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-60}"
SERVER_LOG_PATH="${SERVER_LOG_PATH:-data/robotwin/server_logs/${CONFIG_NAME}_${PORT}.log}"
ACTION_HORIZON="${ACTION_HORIZON:-10}"
RESULT_DIR="${RESULT_DIR:-data/robotwin/eval_results}"
SKILL_PLAN="${SKILL_PLAN:-${SKILLNET_REPO_ROOT}/data_process/robotwin/robotwin_plan.json}"

if [[ -z "${ROBOTWIN_ROOT}" ]]; then
  echo "ROBOTWIN_ROOT must point to a local RoboTwin-2.0 checkout." >&2
  exit 2
fi
if [[ ! -d "${ROBOTWIN_ROOT}" ]]; then
  echo "ROBOTWIN_ROOT does not exist: ${ROBOTWIN_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${SKILL_PLAN}" ]]; then
  echo "SKILL_PLAN does not exist: ${SKILL_PLAN}" >&2
  exit 2
fi

if [[ -n "${TRANSFER_TASK}" && -z "${SKILLNET_ROBOTWIN_TRANSFER_REPO_ID:-}" ]]; then
  export SKILLNET_ROBOTWIN_TRANSFER_REPO_ID="jsw19/robotwin_${TRANSFER_TASK}_v1"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON_CMD=(python)
else
  PYTHON_CMD=(uv run python)
fi

export PYTHONPATH="${SKILLNET_ROOT}/src:${SKILLNET_ROOT}/packages/openpi-client/src:${ROBOTWIN_ROOT}:${ROBOTWIN_ROOT}/policy:${ROBOTWIN_ROOT}/description/utils:${PYTHONPATH:-}"
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
    echo "CKPT_DIR must point to a trained SkillNet checkpoint when START_SERVER=1." >&2
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
  --robotwin-root "${ROBOTWIN_ROOT}"
  --host "${HOST}"
  --port "${PORT}"
  --task-config "${TASK_CONFIG}"
  --num-trials "${NUM_TRIALS}"
  --seed "${SEED}"
  --action-horizon "${ACTION_HORIZON}"
  --skill-plan "${SKILL_PLAN}"
  --result-dir "${RESULT_DIR}"
)

if [[ -n "${TASKS}" ]]; then
  read -r -a TASK_ARRAY <<< "${TASKS}"
  EVAL_ARGS+=(--tasks "${TASK_ARRAY[@]}")
else
  EVAL_ARGS+=(--task-set "${TASK_SET}")
fi

"${PYTHON_CMD[@]}" examples/robotwin/eval_robotwin_moe_skill.py "${EVAL_ARGS[@]}" "$@"
