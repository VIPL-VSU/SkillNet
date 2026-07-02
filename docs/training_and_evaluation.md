# Skill-MoE Training and Evaluation

This document describes the public launch path for the Skill-MoE LIBERO training
configs and the LIBERO-Skill OOD evaluation entrypoint.

## Prepare SkillNet

Clone SkillNet and use `skill_moe/skillnet` as the runnable source root:

```bash
git -c core.longpaths=true clone --branch skillnet-public-release --depth 1 https://github.com/VIPL-VSU/SkillNet.git SkillNet
cd SkillNet/skill_moe/skillnet
git -C ../.. config core.longpaths true
```

If HTTPS cloning is blocked by your network, use the same release branch over
SSH:

```bash
git -c core.longpaths=true clone --branch skillnet-public-release --depth 1 git@github.com:VIPL-VSU/SkillNet.git SkillNet
cd SkillNet/skill_moe/skillnet
git -C ../.. config core.longpaths true
```

For prerequisites such as Python 3.10, `uv`, and Hugging Face Hub CLI, see
`docs/quick_start.md` from the repository root. From this runtime root, that
file is `../../docs/quick_start.md`. GPU training also needs a CUDA/JAX
installation that matches your machine.

Install the editable SkillNet runtime package and bundled client package:

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -e .
uv pip install -e packages/skillnet-client
export PYTHONPATH="${PWD}/src:${PWD}/packages/skillnet-client/src:${PWD}/third_party/libero:${PYTHONPATH:-}"
```

For a native Windows venv, use `.venv/Scripts/activate` instead of
`.venv/bin/activate`. LIBERO simulator evaluation remains Linux/WSL-oriented.

The distribution name and public workflow name are `skillnet`. A few internal
package names are retained only for pi0.5 checkpoint/runtime and policy-client
compatibility, so release commands should still be documented and launched as
SkillNet workflows.

For GPU training, install the JAX wheel that matches your CUDA setup in this
same environment.

Prepare SkillNet LIBERO-Skill registration in the environment that runs the
client:

```bash
# Reuses the currently active venv by default. Set SKILLNET_VENV_PATH to create
# a separate LIBERO environment.
bash install_libero.sh
source examples/libero/skillnet_env.sh
```

`install_libero.sh` installs the editable `skillnet` and `skillnet-client`
packages, installs bundled LIBERO requirement files when they are available,
writes `examples/libero/skillnet_env.sh` for the repo-local source paths, and
checks LIBERO-Skill benchmark registration. It is not a full simulator
installer: install LIBERO, MuJoCo, Robosuite, BDDL, and any machine-specific
rendering dependencies in the same environment. Set `SKILLNET_SKIP_CORE_INSTALL=1`
if the core packages are already installed.

Use the public LIBERO project as the simulator source of truth:

- Repository: <https://github.com/Lifelong-Robot-Learning/LIBERO>
- Documentation: <https://lifelong-robot-learning.github.io/LIBERO/html/index.html>

When an external LIBERO package is importable but does not register
`libero_skill_obj`, the setup script attempts:

```bash
python examples/libero/install_libero_skill_assets.py --install
```

The helper copies SkillNet's bundled bddl/init files into the active LIBERO
package and patches benchmark registration with `.skillnet.bak` backups where
text files are changed. Use `--dry-run` first to inspect planned changes, or set
`SKILLNET_SKIP_LIBERO_SKILL_ASSET_INSTALL=1` to skip the automatic attempt.
To fail early instead of warning, run setup with `SKILLNET_REQUIRE_LIBERO=1`.

If MuJoCo EGL fails on your machine, retry with:

```bash
export MUJOCO_GL=glx
```

## Data and Assets

The training configs use LeRobot-format LIBERO datasets:

- `jsw19/libero_40_v1`
- `jsw19/libero_90_v1`

These are the canonical `repo_id` values expected by the released configs. If
the derived dataset repos are not accessible from your Hugging Face account,
build them locally from the public RLDS sources and the included compact
slice-index metadata under `../../data_process/libero/slice_indices/`, then set
`LEROBOT_HOME` so LeRobot can find the same `repo_id` layout. From this source
root, `../../docs/libero_data_processing.md` documents the conversion workflow.
The default release namespace can be changed with
`SKILLNET_RELEASE_HF_NAMESPACE`; dataset-specific overrides
`SKILLNET_LIBERO40_REPO_ID` and `SKILLNET_LIBERO90_REPO_ID` take precedence when
using mirrored or renamed LeRobot repos.

Exact training reproduction needs accessible copies of those derived LeRobot
datasets, or a local rebuild using the included `libero40_slice_index.json` and
`libero90_slice_index.json` files. RLDS source frames alone do not contain
SkillNet's frame-level skill boundaries.

Before launching training, verify that the derived datasets are present in the
same local `repo_id` layout used by the configs:

```bash
python ../../scripts/verify_lerobot_dataset.py "$LEROBOT_HOME/jsw19/libero_40_v1" \
  --expected-tasks 40 \
  --expected-episodes 3862 \
  --expected-frames 273465 \
  --require-feature class \
  --require-feature all_classes

python ../../scripts/verify_lerobot_dataset.py "$LEROBOT_HOME/jsw19/libero_90_v1" \
  --expected-tasks 73 \
  --expected-episodes 7874 \
  --expected-frames 574571 \
  --require-feature class \
  --require-feature all_classes
```

Before training, each config needs normalization statistics under
`assets/<config_name>/<repo_id>/norm_stats.json`. The public launch scripts below
compute the stats automatically if they are missing.

The pi0.5 base checkpoint defaults to the upstream base checkpoint path:

```bash
gs://openpi-assets/checkpoints/pi05_base/params
```

To use a local mirror, set:

```bash
export SKILLNET_PI05_BASE_PARAMS=checkpoints/pi05_base/params
```

Skill annotations used by training and evaluation are included in SkillNet:

- `examples/libero/annotations/instruct2plan_40.json`
- `examples/libero/annotations/instruct2plan_obj_90.json`
- `examples/libero/annotations/libero_skill_obj_annotations.json`

The model receives integer skill ids plus a mask, for example
`skills: (batch, 4)@int32` and `skill_mask: (batch, 4)@bool`. This is a sequence
of up to four flat skill ids, not a one-hot vector. The LIBERO LeRobot datasets
store frame labels as `class` and `all_classes`; the SkillNet transform converts
them into model inputs named `skills` and `skill_mask`. The hierarchy tokenizer
in `docs/skill_hierarchy.md` publishes `[motion_cluster_id, verbnet_class_id,
verb_id]` metadata for hierarchy construction and analysis; the released LIBERO
checkpoint configs still consume the flat skill-id sequence.

## Released Checkpoints

Download the released checkpoints into the layout expected by the eval scripts:

```bash
hf download jsw19/SkillNet-LIBERO-40 \
  --local-dir checkpoints/pi05_libero_moe_skill_4_40/moe_add_balance_4_128_30000/29999

hf download jsw19/SkillNet-LIBERO-90 \
  --local-dir checkpoints/pi05_libero_moe_skill_4_90/moe_balance_4_32_90_20000/19999
```

If Hugging Face access is slow in your network, use a mirror endpoint:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

From the repository root, `python scripts/check_public_release.py --hub-smoke`
verifies the public checkpoint and source-dataset links. Add
`--include-libero-derived-datasets` only when the derived LIBERO LeRobot dataset
repos should be visible to the current account.

## Model and Config Summary

| Config | Dataset | Batch | Steps | LR peak / final | EMA |
| --- | --- | --- | --- | --- | --- |
| `pi05_libero_moe_skill_4_40` | `jsw19/libero_40_v1` | 128 | 30,000 | `5e-5` / `5e-6` | `0.999` |
| `pi05_libero_moe_skill_4_90` | `jsw19/libero_90_v1` | 32 | 20,000 | `2.5e-5` / `2.5e-6` | `0.99` |

Both release configs use pi0.5 with `action_expert_variant="gemma_300m_moe_4"`,
4 routed MoE experts, top-1 routing, router balance loss scale `0.01`,
`skill_num=6`, `skill_embed_dim=64`, and action horizon 10.

## LIBERO-40 Training

The launchers pass trailing arguments to `scripts/train_moe_skill.py`; use
`--fsdp-devices=1` for a single-GPU smoke run, and drop or adjust it for the
multi-GPU release setting.

Run:

```bash
bash scripts/run_train_libero40_moe_skill.sh
```

Useful overrides:

```bash
EXP_NAME=open_source_libero40 \
PYTHON_BIN=python \
SKILLNET_PI05_BASE_PARAMS=checkpoints/pi05_base/params \
bash scripts/run_train_libero40_moe_skill.sh --fsdp-devices=1
```

This launches `scripts/train_moe_skill.py` with config
`pi05_libero_moe_skill_4_40`, batch size 128, 30k steps, peak learning rate
`5e-5`, checkpoint interval 1000, and keep period 1000.
Fresh training runs use the script `EXP_NAME` in the checkpoint path. The
released checkpoint download path shown above keeps the original experiment
directory name used for the published weights.

## LIBERO-90 Training

This is the training stage used before LIBERO-Skill zero-shot evaluation. No
LIBERO-Skill benchmark task trajectories are used for this training run.

Run:

```bash
bash scripts/run_train_libero90_moe_skill.sh
```

Useful overrides:

```bash
EXP_NAME=open_source_libero90 \
PYTHON_BIN=python \
SKILLNET_PI05_BASE_PARAMS=checkpoints/pi05_base/params \
bash scripts/run_train_libero90_moe_skill.sh --fsdp-devices=1
```

This launches `scripts/train_moe_skill.py` with config
`pi05_libero_moe_skill_4_90`, batch size 32, 20k steps, peak learning rate
`2.5e-5`, checkpoint interval 1000, and keep period 1000.
Fresh training runs use the script `EXP_NAME` in the checkpoint path. The
released checkpoint download path shown above keeps the original experiment
directory name used for the published weights.

## LIBERO-40 Evaluation

`examples/libero/run_eval_libero_skill_moe.sh` defaults to the
LIBERO-90-to-LIBERO-Skill zero-shot setting with `TASK_SUITE=libero_skill_obj`.
For LIBERO-40 evaluation, override
`CONFIG_NAME`, `TASK_SUITE`, and `SKILL_ANNOTATION_PATH` as shown below.

Evaluate the LIBERO-40 checkpoint over the four standard LIBERO suites:

```bash
for suite in libero_spatial libero_object libero_goal libero_10; do
  CONFIG_NAME=pi05_libero_moe_skill_4_40 \
  CKPT_DIR=checkpoints/pi05_libero_moe_skill_4_40/moe_add_balance_4_128_30000/29999 \
  TASK_SUITE="${suite}" \
  SKILL_ANNOTATION_PATH=examples/libero/annotations/instruct2plan_40.json \
  NUM_TRIALS=50 \
  ZERO_SHOT=0 \
  VIDEO_OUT_PATH="data/libero/videos/skillnet_libero40_${suite}" \
  bash examples/libero/run_eval_libero_skill_moe.sh
done
```

`ZERO_SHOT=0` keeps the standard in-domain LIBERO rollout budgets: 220 control
steps for `libero_spatial`, 280 for `libero_object`, 300 for `libero_goal`, and
520 for `libero_10`, plus 10 initial stabilization steps in the simulator.

Expected outputs:

- The evaluator writes rollout videos under `VIDEO_OUT_PATH`.
- The console logs one success/failure outcome per rollout and reports task or
  suite-level success rates at the end of evaluation.
- For paper-style reporting, average success rates over the four standard
  LIBERO suites after running the same `NUM_TRIALS` for every task.

## LIBERO-Skill Evaluation

LIBERO-Skill task definitions and initial states are included under:

```bash
third_party/libero/libero/libero/bddl_files/libero_skill_obj
third_party/libero/libero/libero/init_files/libero_skill_obj
```

The LIBERO-Skill tasks live under the `libero_skill_obj` bddl/init-state folder.
The public benchmark contains 9 tasks: the first task from `libero_10` followed
by 8 skill-composition tasks. The checked-in bddl/init filenames use compact
`skill_obj_XX` aliases so the repository checks out cleanly on platforms with
conservative path-length limits. Each bddl file still stores the original
natural-language task in its `:language` field.
The authoritative machine-readable task order is:

```bash
third_party/libero/libero/libero/bddl_files/libero_skill_obj/public_task_manifest.json
```

The manifest also includes `task_details` entries with the original generated
LIBERO task names and language strings. The same folder contains auxiliary
candidate bddl files from benchmark construction. `tasks_info.txt` is an
asset inventory for those bundled files; do not use it as the reported
LIBERO-Skill evaluation list.
The evaluation wrapper defaults to `libero_skill_obj` to avoid colliding with
unrelated external benchmark names. The evaluator still accepts `libero_skill`
when the installed LIBERO registers that key, but it validates that the
registered suite exactly matches SkillNet's public 9-task manifest before
rollout. After setup, `install_libero.sh` checks that at least one of those keys
is visible from `benchmark.get_benchmark_dict()`.

Evaluate a LIBERO-90 Skill-MoE checkpoint:

```bash
CKPT_DIR=checkpoints/pi05_libero_moe_skill_4_90/moe_balance_4_32_90_20000/19999 \
bash examples/libero/run_eval_libero_skill_moe.sh
```

Useful overrides:

```bash
CONFIG_NAME=pi05_libero_moe_skill_4_90 \
CKPT_DIR=checkpoints/pi05_libero_moe_skill_4_90/moe_balance_4_32_90_20000/19999 \
SERVER_GPU=0 \
PORT=8056 \
NUM_TRIALS=50 \
VIDEO_OUT_PATH=data/libero/videos/skillnet_libero_skill_obj \
bash examples/libero/run_eval_libero_skill_moe.sh
```

If a policy server is already running:

```bash
START_SERVER=0 \
HOST=127.0.0.1 \
PORT=8056 \
bash examples/libero/run_eval_libero_skill_moe.sh
```

The default LIBERO-Skill skill/object annotation file is:

```bash
examples/libero/annotations/libero_skill_obj_annotations.json
```

Override it with `SKILL_ANNOTATION_PATH` or the eval script argument
`--skill-annotation-path` when testing another benchmark annotation file.

`examples/libero/main_skill_obj_test_moe_skill.py` defaults to zero-shot mode.
For LIBERO-Skill, the base rollout limit is 800 control steps; zero-shot mode
doubles it to 1600 control steps, plus 10 initial stabilization steps in the
simulator.
The wrapper writes policy-server logs under `data/libero/server_logs/`, waits
for the server port to become reachable, and passes `--fail-fast` by default.
Set `FAIL_FAST=0` only when you intentionally want to keep evaluating after a
rollout exception. Set `ZERO_SHOT=0` only for in-domain LIBERO evaluation or an
ablation that intentionally uses the base rollout budget.

The 9 registered LIBERO-Skill task ids and language strings are:

1. `skill_obj_01`: put both the alphabet soup and the tomato sauce in the basket
2. `skill_obj_02`: open the top drawer of the cabinet and put the bowl on the plate
3. `skill_obj_03`: put the black bowl in the bottom drawer of the cabinet and close the bottom drawer of the cabinet
4. `skill_obj_04`: close the top drawer of the cabinet and put the black bowl on the plate
5. `skill_obj_05`: close the top drawer of the cabinet and close the microwave
6. `skill_obj_06`: stack the middle black bowl on the back black bowl and open the top drawer of the cabinet
7. `skill_obj_07`: put the black bowl on the plate and close the microwave
8. `skill_obj_08`: close the drawer of the cabinet and turn off the stove
9. `skill_obj_09`: put the black bowl on the plate and open the microwave

Expected outputs:

- Rollout videos are written to `VIDEO_OUT_PATH`.
- Per-task success rates are printed by the LIBERO evaluator.
- The reported LIBERO-Skill number should be computed as the mean success rate
  over the 9 tasks above with the same `NUM_TRIALS` for every task.

## Reported Results

The SkillNet paper reports LIBERO in-domain success rates with 50 evaluation
trials per task. The `LONG` column corresponds to the standard `libero_10`
suite in the evaluation loop above.

| Method | Spatial | Object | Goal | Long / LIBERO-10 | Avg. |
| --- | --- | --- | --- | --- | --- |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| pi0 | 96.4 | 98.8 | 95.8 | 85.2 | 94.2 |
| OpenVLA-OFT | 97.7 | 98.0 | 96.1 | 95.3 | 96.8 |
| GR00T-N1.6 | 97.7 | 97.5 | 98.5 | 94.4 | 97.0 |
| AtomicVLA | 98.8 | 98.8 | 97.2 | 96.2 | 97.8 |
| pi0.5 | 99.4 | 97.8 | 98.2 | 96.6 | 98.0 |
| SkillNet | 99.6 | 99.2 | 99.0 | 98.4 | 99.1 |

For LIBERO-Skill zero-shot evaluation, the paper reports 50 trials per task on
the 9-task manifest listed above:

| Method | Avg. |
| --- | --- |
| OpenVLA-OFT | 0.0 |
| OpenVLA | 2.0 |
| pi0 | 3.3 |
| GR00T-N1.6 | 12.2 |
| pi0.5 | 31.8 |
| Vanilla MoE | 33.1 |
| SkillNet | 47.8 |

Use the released checkpoints
`checkpoints/pi05_libero_moe_skill_4_40/moe_add_balance_4_128_30000/29999`
and
`checkpoints/pi05_libero_moe_skill_4_90/moe_balance_4_32_90_20000/19999`
when comparing against these paper-reported numbers. Freshly trained
checkpoints can vary with hardware, random seeds, simulator versions, and
dataset rebuild details; report the exact git SHA, dataset verification counts,
checkpoint path, `NUM_TRIALS`, and simulator environment with new results.

## Release Scope

The release tree is pruned to the public SkillNet reproduction path. Internal
ablation, visualization, and real-robot launchers are intentionally excluded.
For reproducible LIBERO training and LIBERO-Skill evaluation, use the public
launch scripts in this document; they use repo-relative paths and require
explicit checkpoints.
