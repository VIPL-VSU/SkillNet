# Quick Start

This guide gives the shortest public path from a fresh SkillNet clone to the
released data-processing, training, and evaluation entrypoints.

## Clone

```bash
# On Windows, clone under a short path such as C:\sn because LIBERO-Skill
# task filenames are long. The -c flag enables Git long-path checkout.
git -c core.longpaths=true clone --branch skillnet-public-release --depth 1 https://github.com/VIPL-VSU/SkillNet.git SkillNet
cd SkillNet
export SKILLNET_REPO_ROOT="${PWD}"
```

The runnable SkillNet source root is:

```bash
cd "${SKILLNET_REPO_ROOT}/skill_moe/skillnet"
```

The Python package namespace under `src/openpi` is kept for compatibility with
the pi0.5 model and checkpoint format. User-facing commands, environment
variables, docs, and checkpoints use the SkillNet name.

## Prerequisites

Install these command-line tools before the workflows below:

- Python 3.10 for the SkillNet runtime environment.
- `git` for cloning the repository.
- `uv` for creating the local environment and installing editable packages.
- Hugging Face Hub CLI (`hf`) for checkpoint and dataset downloads.

For example:

```bash
python -m pip install -U uv "huggingface_hub[cli]"
hf --help
```

GPU training additionally needs a CUDA/JAX installation that matches your
machine. LIBERO and RoboTwin evaluation also require their simulator assets and
system dependencies; install those in the same environment that runs the client
or environment adapter.

The lightweight package smoke checks in this repository do not install GPU or
simulator stacks. For training, install a JAX CUDA wheel compatible with your
driver and CUDA runtime, then rerun the config import smoke check below. For
LIBERO evaluation, use a Linux or WSL environment with MuJoCo, Robosuite, BDDL,
and LIBERO importable. For RoboTwin evaluation, use a local RoboTwin-2.0
checkout and install the simulator dependencies required by that checkout.

## Environment

SkillNet currently ships the Skill-MoE model/config/data/evaluation additions,
LIBERO-Skill benchmark files, and data-processing utilities. Install the local
`skillnet` runtime package and the bundled client package:

```bash
cd "${SKILLNET_REPO_ROOT}/skill_moe/skillnet"
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -e .
uv pip install -e packages/openpi-client
export PYTHONPATH="${PWD}/src:${PWD}/packages/openpi-client/src:${PWD}/third_party/libero:${PYTHONPATH:-}"
```

For GPU training, install the JAX wheel that matches your CUDA setup in this
same environment.

Example CUDA installation command, to be adapted to your machine:

```bash
uv pip install "jax[cuda12]"
```

For LIBERO evaluation dependencies:

```bash
# Reuses the currently active venv by default. Set SKILLNET_VENV_PATH to create
# a separate LIBERO environment.
bash install_libero.sh
source examples/libero/skillnet_env.sh
```

`install_libero.sh` installs the editable `skillnet` and `openpi-client`
packages unless `SKILLNET_SKIP_CORE_INSTALL=1` is set. It also installs bundled
requirements when present and writes `examples/libero/skillnet_env.sh` for the
repo-local source paths. If your simulator setup needs additional
LIBERO/MuJoCo/Robosuite packages, install them in the same environment. The
script prints a warning if Python still cannot import the full `libero`
simulator package after setup. When LIBERO is importable, the script also checks
that `libero_skill` or `libero_skill_obj` is registered; set
`SKILLNET_REQUIRE_LIBERO=1` to make missing LIBERO/LIBERO-Skill registration a
hard setup error.

If MuJoCo EGL fails on your machine, retry with:

```bash
export MUJOCO_GL=glx
```

## Checkpoints

Download released SkillNet checkpoints:

```bash
hf download jsw19/SkillNet-LIBERO-40 \
  --local-dir checkpoints/pi05_libero_moe_skill_4_40/moe_add_balance_4_128_30000/29999

hf download jsw19/SkillNet-LIBERO-90 \
  --local-dir checkpoints/pi05_libero_moe_skill_4_90/moe_balance_4_32_90_20000/19999
```

If Hugging Face is slow:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

The pi0.5 base checkpoint path can be overridden with:

```bash
export SKILLNET_PI05_BASE_PARAMS=checkpoints/pi05_base/params
```

## Public Release Check

From the repository root, run the lightweight release check before installing
the heavy simulator stacks:

```bash
cd "${SKILLNET_REPO_ROOT}"
python scripts/check_public_release.py
```

This verifies key files, JSON/JSONL artifacts, selected CLI help commands,
Python syntax for public data/eval helpers, and common private-path or token
patterns. It does not import simulator packages or load model checkpoints.

To additionally test editable package metadata in a temporary Python 3.10
environment without downloading heavyweight runtime dependencies:

```bash
python scripts/check_public_release.py --install-smoke
```

To check that the public Hugging Face checkpoints and LIBERO/RoboTwin source
datasets are reachable from your machine:

```bash
python scripts/check_public_release.py --hub-smoke
```

This network check respects `HF_ENDPOINT` and `HF_TOKEN`. Add
`--include-derived-datasets` only when you expect the derived LeRobot output
repo ids such as `jsw19/libero_40_v1` and `jsw19/libero_90_v1` to be visible
from the current account. Otherwise, rebuild them locally and keep the same
repo_id layout under `LEROBOT_HOME`. `docs/libero_data_processing.md` includes
metadata verification and Hub publishing commands for those derived datasets.

## Smoke Checks

Run the hierarchy tokenizer from the repository root:

```bash
cd "${SKILLNET_REPO_ROOT}"
python data_process/skill_hierarchy/skill_hierarchy_tokenizer.py \
  --phrase "open drawer" \
  --motion-code 200200
```

Generate RoboTwin paper-split metadata:

```bash
cd "${SKILLNET_REPO_ROOT}"
python data_process/robotwin/build_robotwin_skill_metadata.py \
  --task-set paper \
  --output robotwin_skill_metadata.jsonl
```

After the base runtime dependencies are available, check that SkillNet configs
are importable from the SkillNet source root:

```bash
cd "${SKILLNET_REPO_ROOT}/skill_moe/skillnet"
python -c "import openpi.training.config_moe_skill as c; print(c.get_config('pi05_libero_moe_skill_4_40').name)"
```

## Main Workflows

LIBERO data processing:

```bash
cd "${SKILLNET_REPO_ROOT}"
python data_process/libero/download_libero_sources.py \
  --dataset all \
  --hf-endpoint https://hf-mirror.com

python data_process/libero/convert_libero_40_to_lerobot.py \
  --data-dir "$LIBERO40_RLDS_DIR" \
  --plan-root "$LIBERO40_PLAN_ROOT" \
  --output-repo-id jsw19/libero_40_v1

python data_process/libero/convert_libero_90_to_lerobot.py \
  --data-dir "$LIBERO90_RLDS_DIR" \
  --plan-root "$LIBERO90_PLAN_ROOT" \
  --output-repo-id jsw19/libero_90_v1
```

LIBERO-40 training:

```bash
cd "${SKILLNET_REPO_ROOT}/skill_moe/skillnet"
bash scripts/run_train_libero40_moe_skill.sh
```

LIBERO-90 training for LIBERO-Skill zero-shot evaluation:

```bash
cd "${SKILLNET_REPO_ROOT}/skill_moe/skillnet"
bash scripts/run_train_libero90_moe_skill.sh
```

LIBERO-Skill evaluation:

```bash
cd "${SKILLNET_REPO_ROOT}/skill_moe/skillnet"
CONFIG_NAME=pi05_libero_moe_skill_4_90 \
CKPT_DIR=checkpoints/pi05_libero_moe_skill_4_90/moe_balance_4_32_90_20000/19999 \
bash examples/libero/run_eval_libero_skill_moe.sh
```

RoboTwin few-shot data processing from the repository root:

```bash
cd "${SKILLNET_REPO_ROOT}"
python data_process/robotwin/download_robotwin_sources.py \
  --task-set paper \
  --output-dir ./robotwin_datasets

python data_process/robotwin/convert_robotwin_to_lerobot.py \
  --source-dir ./robotwin_datasets \
  --task-set pretrain \
  --output-repo-id jsw19/robotwin_pretrain_v1

python data_process/robotwin/convert_robotwin_to_lerobot.py \
  --source-dir ./robotwin_datasets \
  --tasks blocks_ranking_size \
  --output-repo-id jsw19/robotwin_blocks_ranking_size_v1
```

RoboTwin few-shot training:

```bash
cd "${SKILLNET_REPO_ROOT}/skill_moe/skillnet"
bash scripts/run_train_robotwin_pretrain_moe_skill.sh

export SKILLNET_ROBOTWIN_TRANSFER_INIT_PARAMS=checkpoints/pi05_robotwin_moe_skill_pretrain/robotwin_moe_skill_pretrain/19999/params
TRANSFER_TASK=blocks_ranking_size \
bash scripts/run_train_robotwin_transfer_moe_skill.sh
```

For paper-style few-shot transfer, keep
`SKILLNET_ROBOTWIN_TRANSFER_INIT_PARAMS` pointed at a RoboTwin pretraining
checkpoint. If it is unset, the transfer config falls back to the pi0.5 base
checkpoint and the launcher prints a warning.

RoboTwin few-shot evaluation against a local RoboTwin-2.0 checkout:

```bash
cd "${SKILLNET_REPO_ROOT}/skill_moe/skillnet"
ROBOTWIN_ROOT="$HOME/RoboTwin_eval" \
CKPT_DIR=checkpoints/pi05_robotwin_moe_skill_transfer/robotwin_moe_skill_transfer_blocks_ranking_size/999 \
TRANSFER_TASK=blocks_ranking_size \
TASKS=blocks_ranking_size \
bash examples/robotwin/run_eval_robotwin_moe_skill.sh
```

See the detailed workflow docs for dataset-specific setup:

- `docs/skill_hierarchy.md`
- `docs/libero_data_processing.md`
- `docs/training_and_evaluation.md`
- `docs/robotwin_few_shot.md`
