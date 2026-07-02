# Quick Start

This guide gives the shortest public path from a fresh SkillNet clone to the
released data-processing, training, and evaluation entrypoints.

## Clone

Command blocks in this guide use Bash syntax (`export`, `source`, `bash`).
On Windows, use WSL for the simulator and training workflows below. Git Bash or
PowerShell can run the clone command and lightweight release checks, but native
Windows environment-variable and venv activation syntax differs, and
LIBERO/RoboTwin simulator evaluation is Linux-oriented.

```bash
# On Windows, clone under a short, non-user-specific directory because
# LIBERO-Skill task filenames are long. The -c flag handles checkout; the
# local config keeps later git status/diff commands from hitting path limits.
git -c core.longpaths=true clone --branch skillnet-public-release --depth 1 https://github.com/VIPL-VSU/SkillNet.git SkillNet
cd SkillNet
git config core.longpaths true
export SKILLNET_REPO_ROOT="${PWD}"
```

If HTTPS cloning is blocked by your network, use the same release branch over
SSH:

```bash
git -c core.longpaths=true clone --branch skillnet-public-release --depth 1 git@github.com:VIPL-VSU/SkillNet.git SkillNet
cd SkillNet
git config core.longpaths true
export SKILLNET_REPO_ROOT="${PWD}"
```

The runnable SkillNet source root is:

```bash
cd "${SKILLNET_REPO_ROOT}/skill_moe/skillnet"
```

The SkillNet runtime keeps a few pi0.5-compatible import paths internally so
the released checkpoints and websocket policy clients load without conversion.
Treat those package paths as implementation details; user-facing commands,
environment variables, docs, checkpoints, and release checks use the SkillNet
name.

## Prerequisites

Install these command-line tools before the workflows below:

- Python 3.10 for the SkillNet runtime environment.
- `git` for cloning the repository.
- `uv` for creating the local environment and installing editable packages.
- Hugging Face Hub CLI (`hf`) for checkpoint and dataset downloads.
- Optional Google Cloud SDK (`gcloud`) if you want to mirror the pi0.5 base
  checkpoint locally instead of reading the default GCS asset at training time.

For example:

```bash
python -m pip install -U uv "huggingface_hub[cli]"
hf --help
```

GPU training additionally needs a CUDA/JAX installation that matches your
machine. LIBERO and RoboTwin evaluation also require their simulator assets and
system dependencies; install those in the same environment that runs the client
or environment adapter.

External simulator entrypoints:

- LIBERO official repository: <https://github.com/Lifelong-Robot-Learning/LIBERO>
- LIBERO documentation: <https://lifelong-robot-learning.github.io/LIBERO/html/index.html>
- RoboTwin-2.0 official repository: <https://github.com/RoboTwin-Platform/RoboTwin>
- RoboTwin-2.0 documentation: <https://robotwin-platform.github.io/doc/index.html>

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
uv pip install -e packages/skillnet-client
export PYTHONPATH="${PWD}/src:${PWD}/packages/skillnet-client/src:${PWD}/third_party/libero:${PYTHONPATH:-}"
```

For a native Windows venv, activate with `.venv/Scripts/activate` instead of
`.venv/bin/activate`; simulator workflows should still be run from Linux or
WSL.

For GPU training, install the JAX wheel that matches your CUDA setup in this
same environment.

Example CUDA installation command, to be adapted to your machine:

```bash
uv pip install "jax[cuda12]"
```

For SkillNet LIBERO-Skill registration and preflight:

```bash
# Reuses the currently active venv by default. Set SKILLNET_VENV_PATH to create
# a separate LIBERO environment.
bash install_libero.sh
source examples/libero/skillnet_env.sh
```

`install_libero.sh` installs the editable `skillnet` and `skillnet-client`
packages unless `SKILLNET_SKIP_CORE_INSTALL=1` is set. It also installs bundled
requirements when present, writes `examples/libero/skillnet_env.sh` for the
repo-local source paths, and checks LIBERO-Skill benchmark registration.
It is not a full simulator installer: install LIBERO, MuJoCo, Robosuite, BDDL,
and any machine-specific rendering dependencies in the same environment.

When an external LIBERO package is importable but does not yet know about
SkillNet's `libero_skill_obj` benchmark, the script runs:

```bash
python examples/libero/install_libero_skill_assets.py --install
```

That helper copies the bundled bddl/init files and patches benchmark
registration with `.skillnet.bak` backups where text files are changed. Use
`--dry-run` first to inspect planned changes, or set
`SKILLNET_SKIP_LIBERO_SKILL_ASSET_INSTALL=1` to skip the automatic attempt.
Set `SKILLNET_REQUIRE_LIBERO=1` to make missing LIBERO/LIBERO-Skill
registration a hard setup error.

If MuJoCo EGL fails on your machine, retry with:

```bash
export MUJOCO_GL=glx
```

## Checkpoints

Download released SkillNet checkpoints from the runnable SkillNet source root:

```bash
cd "${SKILLNET_REPO_ROOT}/skill_moe/skillnet"

hf download jsw19/SkillNet-LIBERO-40 \
  --local-dir checkpoints/pi05_libero_moe_skill_4_40/moe_add_balance_4_128_30000/29999

hf download jsw19/SkillNet-LIBERO-90 \
  --local-dir checkpoints/pi05_libero_moe_skill_4_90/moe_balance_4_32_90_20000/19999
```

The default Hub namespace for released datasets and locally generated LeRobot
repo ids is:

```bash
export SKILLNET_RELEASE_HF_NAMESPACE=jsw19
```

If you mirror the datasets or checkpoints under an organization namespace, set
`SKILLNET_RELEASE_HF_NAMESPACE` to that namespace before running conversion,
training, or evaluation. Dataset-specific variables such as
`SKILLNET_LIBERO40_REPO_ID`, `SKILLNET_LIBERO90_REPO_ID`,
`SKILLNET_ROBOTWIN_PRETRAIN_REPO_ID`, and
`SKILLNET_ROBOTWIN_TRANSFER_REPO_ID` override the namespace-derived defaults.

If Hugging Face is slow:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

The pi0.5 base checkpoint path can be overridden with:

```bash
export SKILLNET_PI05_BASE_PARAMS=checkpoints/pi05_base/params
```

If your training machine can read the default remote asset
`gs://openpi-assets/checkpoints/pi05_base/params`, no local mirror is required.
To mirror it locally, use a GCS-capable tool and point SkillNet at the copied
`params` directory:

```bash
cd "${SKILLNET_REPO_ROOT}/skill_moe/skillnet"
mkdir -p checkpoints/pi05_base
gcloud storage cp -r gs://openpi-assets/checkpoints/pi05_base/params checkpoints/pi05_base/
export SKILLNET_PI05_BASE_PARAMS=checkpoints/pi05_base/params
test -e "${SKILLNET_PI05_BASE_PARAMS}" && echo "pi0.5 base checkpoint is visible"
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
See `docs/release_status.md` for the current release-readiness checklist and
external asset gates.

To additionally test editable package metadata in a temporary Python 3.10
environment without downloading heavyweight runtime dependencies, run the
no-deps install smoke:

```bash
python scripts/check_public_release.py --install-smoke
```

If your Python 3.10 executable is not discoverable as `3.10`, provide it
explicitly:

```bash
python scripts/check_public_release.py --install-smoke --install-python /path/to/python3.10
```

To check that the public Hugging Face checkpoints and LIBERO/RoboTwin source
datasets are reachable from your machine:

```bash
python scripts/check_public_release.py --hub-smoke
```

This network check respects `HF_ENDPOINT` and intentionally does not use
`HF_TOKEN` by default, so it checks what external users can read publicly. Add
`--include-libero-derived-datasets` when you expect `jsw19/libero_40_v1` and
`jsw19/libero_90_v1` to be publicly visible. A plain `--hub-smoke` pass means
the public checkpoints and source datasets are reachable; it does not prove that
the derived LIBERO LeRobot datasets are public. If the derived dataset check is
not expected to pass, rebuild them locally and keep the same repo_id layout
under `LEROBOT_HOME`. Use `--hub-authenticated` only when checking private or
staging assets.
`docs/libero_data_processing.md` includes metadata verification and Hub
publishing commands for those derived datasets.

## Smoke Checks

Run the hierarchy tokenizer from the repository root:

```bash
cd "${SKILLNET_REPO_ROOT}"
python data_process/skill_hierarchy/skill_hierarchy_tokenizer.py \
  --phrase "open drawer" \
  --motion-code 200100
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

Exact LIBERO training uses derived LeRobot datasets with frame-level skill
boundaries. The repository includes compact slice-index metadata under
`data_process/libero/slice_indices/`, so the converter commands below can
rebuild the v1 labels from public RLDS source frames. If the derived
`jsw19/libero_40_v1` and
`jsw19/libero_90_v1` Hub repos are accessible, you can use those directly under
`LEROBOT_HOME` instead of rebuilding. If you have an existing local v1 LeRobot
copy, `data_process/libero/export_libero_skill_slices.py` can export a compact
slice-index JSON, and the converter can consume it with `--slice-index`.

```bash
cd "${SKILLNET_REPO_ROOT}"
export SKILLNET_LIBERO_DATA_ROOT="${SKILLNET_LIBERO_DATA_ROOT:-data/libero}"
export LIBERO40_RLDS_DIR="${LIBERO40_RLDS_DIR:-$SKILLNET_LIBERO_DATA_ROOT/libero40_rlds}"
export LIBERO90_RLDS_DIR="${LIBERO90_RLDS_DIR:-$SKILLNET_LIBERO_DATA_ROOT/libero90_rlds}"

python data_process/libero/download_libero_sources.py \
  --dataset all \
  --hf-endpoint https://hf-mirror.com

python data_process/libero/convert_libero_40_to_lerobot.py \
  --data-dir "$LIBERO40_RLDS_DIR" \
  --slice-index data_process/libero/slice_indices/libero40_slice_index.json \
  --output-repo-id jsw19/libero_40_v1

python data_process/libero/convert_libero_90_to_lerobot.py \
  --data-dir "$LIBERO90_RLDS_DIR" \
  --slice-index data_process/libero/slice_indices/libero90_slice_index.json \
  --output-repo-id jsw19/libero_90_v1

python scripts/verify_lerobot_dataset.py "$LEROBOT_HOME/jsw19/libero_40_v1" \
  --expected-tasks 40 \
  --expected-episodes 3862 \
  --expected-frames 273465 \
  --require-feature class \
  --require-feature all_classes

python scripts/verify_lerobot_dataset.py "$LEROBOT_HOME/jsw19/libero_90_v1" \
  --expected-tasks 73 \
  --expected-episodes 7874 \
  --expected-frames 574571 \
  --require-feature class \
  --require-feature all_classes
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

The `jsw19/robotwin_*_v1` names above are local LeRobot `repo_id` values used
by the released configs and launchers. Set `LEROBOT_HOME` or
`HF_LEROBOT_HOME` before conversion if you want the datasets written to a
specific cache, and keep the same value when training. These derived RoboTwin
datasets are generated locally in this release rather than treated as public Hub
download dependencies.

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
checkpoint. Set `TRANSFER_TASK` for the paper-style per-task dataset and
checkpoint naming convention; without it, the transfer config uses the aggregate
`SKILLNET_ROBOTWIN_TRANSFER_REPO_ID` value. The transfer launcher exits if the
initialization checkpoint is unset. For a debugging-only pi0.5-base
initialization run, set `ALLOW_PI05_TRANSFER_INIT=1` explicitly.

RoboTwin few-shot evaluation against a local RoboTwin-2.0 checkout:

```bash
cd "${SKILLNET_REPO_ROOT}/skill_moe/skillnet"
ROBOTWIN_ROOT="<path-to-robotwin-checkout>" \
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
