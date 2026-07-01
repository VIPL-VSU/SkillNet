#!/usr/bin/env bash
set -euo pipefail

TASK_SET="${1:-pretrain}"
GPU_ID="${GPU_ID:-${2:-0}}"
DEMO_KIND="${DEMO_KIND:-demo_clean}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${PWD}}"

PRETRAIN_TASKS=(
  "adjust bottle"
  "beat block hammer"
  "click alarmclock"
  "click bell"
  "grab roller"
  "handover block"
  "lift pot"
  "move can pot"
  "move playingcard away"
  "open microwave"
  "place burger fries"
  "place object basket"
  "rotate qrcode"
  "shake bottle horizontally"
  "stack blocks two"
)

TRANSFER_TASKS=(
  "blocks ranking size"
  "hanging mug"
  "move pillbottle pad"
  "open laptop"
  "place a2b left"
  "place bread basket"
  "place bread skillet"
  "place cans plasticbox"
  "place fan"
  "press stapler"
  "scan object"
  "shake bottle"
  "stack blocks three"
  "stack bowls two"
  "stamp seal"
)

case "${TASK_SET}" in
  pretrain)
    TASKS=("${PRETRAIN_TASKS[@]}")
    ;;
  transfer)
    TASKS=("${TRANSFER_TASKS[@]}")
    ;;
  paper)
    TASKS=("${PRETRAIN_TASKS[@]}" "${TRANSFER_TASKS[@]}")
    ;;
  *)
    echo "Unknown task set: ${TASK_SET}. Use pretrain, transfer, or paper." >&2
    exit 2
    ;;
esac

if [[ ! -f "${ROBOTWIN_ROOT}/collect_data.sh" ]]; then
  echo "ROBOTWIN_ROOT must point to a RoboTwin checkout containing collect_data.sh." >&2
  exit 2
fi

cd "${ROBOTWIN_ROOT}"

for task in "${TASKS[@]}"; do
  bash collect_data.sh "${task}" "${DEMO_KIND}" "${GPU_ID}"
done
