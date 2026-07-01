#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Compatibility shim. Public docs use examples/libero/run_eval_libero_skill_moe.sh directly.
exec "${SCRIPT_DIR}/examples/libero/run_eval_libero_skill_moe.sh" "$@"
