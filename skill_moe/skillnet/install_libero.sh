#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -n "${SKILLNET_PROXY:-}" ]]; then
  export https_proxy="${SKILLNET_PROXY}"
  export http_proxy="${SKILLNET_PROXY}"
fi

VENV_PATH="${SKILLNET_VENV_PATH:-examples/libero/.venv}"
if [[ -n "${VIRTUAL_ENV:-}" && -z "${SKILLNET_VENV_PATH:-}" ]]; then
  VENV_PATH="${VIRTUAL_ENV}"
fi

if [[ -z "${VIRTUAL_ENV:-}" || "${VIRTUAL_ENV}" != "${VENV_PATH}" ]]; then
  uv venv --python "${PYTHON_VERSION:-3.10}" "${VENV_PATH}"
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
fi

if [[ "${SKILLNET_SKIP_CORE_INSTALL:-0}" != "1" ]]; then
  uv pip install -e .
  uv pip install -e packages/openpi-client
fi

REQ_FILES=()
if [[ -f examples/libero/requirements.txt ]]; then
  REQ_FILES+=(examples/libero/requirements.txt)
fi
if [[ -f third_party/libero/requirements.txt ]]; then
  REQ_FILES+=(third_party/libero/requirements.txt)
fi

if [[ "${#REQ_FILES[@]}" -gt 0 ]]; then
  REQ_ARGS=()
  for req in "${REQ_FILES[@]}"; do
    REQ_ARGS+=("-r" "${req}")
  done
  uv pip install "${REQ_ARGS[@]}" \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --index-strategy=unsafe-best-match
else
  echo "No bundled LIBERO requirements.txt files were found."
  echo "Install the full LIBERO/MuJoCo/Robosuite simulator dependencies required by your environment."
fi

cat > examples/libero/skillnet_env.sh <<EOF
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src:$PWD/third_party/libero:\${PYTHONPATH:-}"
EOF

if python -c "import libero" >/dev/null 2>&1; then
  echo "LIBERO import check passed."
  if python - <<'PY'
from libero.libero import benchmark

benchmarks = benchmark.get_benchmark_dict()
if "libero_skill" not in benchmarks and "libero_skill_obj" not in benchmarks:
    raise SystemExit(1)
PY
  then
    echo "LIBERO-Skill benchmark registration check passed."
  else
    echo "Warning: LIBERO is importable, but neither 'libero_skill' nor 'libero_skill_obj' is registered." >&2
    echo "Ensure examples/libero/skillnet_env.sh is sourced so the bundled third_party/libero tree is on PYTHONPATH." >&2
    echo "If you use an external LIBERO install, copy or register the bundled libero_skill_obj bddl/init/map files before evaluation." >&2
    if [[ "${SKILLNET_REQUIRE_LIBERO:-0}" == "1" ]]; then
      exit 2
    fi
  fi
else
  echo "Warning: Python cannot import the full 'libero' simulator package yet." >&2
  echo "This release bundles LIBERO-Skill task/init files, not the complete simulator dependency stack." >&2
  echo "Install LIBERO, MuJoCo, Robosuite, and BDDL in this environment before running evaluation." >&2
  if [[ "${SKILLNET_REQUIRE_LIBERO:-0}" == "1" ]]; then
    exit 2
  fi
fi

echo "Prepared ${VENV_PATH}"
echo "Run: source ${VENV_PATH}/bin/activate"
echo "Run: source examples/libero/skillnet_env.sh"
