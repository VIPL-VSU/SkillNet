# Skill-MoE Training and Evaluation

This document describes the public launch path for the Skill-MoE LIBERO training
configs and the LIBERO-Skill OOD evaluation entrypoint.

## Prepare SkillNet

Clone SkillNet and use `skill_moe/skillnet` as the runnable source root:

```bash
git -c core.longpaths=true clone --branch skillnet-public-release --depth 1 https://github.com/VIPL-VSU/SkillNet.git SkillNet
cd SkillNet/skill_moe/skillnet
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
uv pip install -e packages/openpi-client
export PYTHONPATH="${PWD}/src:${PWD}/packages/openpi-client/src:${PWD}/third_party/libero:${PYTHONPATH:-}"
```

The distribution name is `skillnet`, while the Python import namespaces remain
`openpi` and `openpi_client` for pi0.5 checkpoint/runtime and client
compatibility.

For GPU training, install the JAX wheel that matches your CUDA setup in this
same environment.

Install the LIBERO evaluation dependencies in the environment that runs the
client:

```bash
# Reuses the currently active venv by default. Set SKILLNET_VENV_PATH to create
# a separate LIBERO environment.
bash install_libero.sh
source examples/libero/skillnet_env.sh
```

`install_libero.sh` installs the editable `skillnet` and `openpi-client`
packages, installs bundled LIBERO requirement files when they are available, and
writes `examples/libero/skillnet_env.sh` for the repo-local source paths. Set
`SKILLNET_SKIP_CORE_INSTALL=1` if the core packages are already installed.
Install any simulator-specific LIBERO/MuJoCo/Robosuite dependencies that your
machine still needs in the same environment. The script prints a warning if the
full `libero` simulator package is not importable yet. When LIBERO is
importable, it also verifies that `libero_skill` or `libero_skill_obj` is
registered. To fail early instead of warning, run setup with
`SKILLNET_REQUIRE_LIBERO=1`.

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
build them locally from the public RLDS sources and the skill-slice metadata,
then set `LEROBOT_HOME` so LeRobot can find the same `repo_id` layout. From this
source root, `../../docs/libero_data_processing.md` documents the conversion
workflow.

Exact training reproduction needs accessible copies of those derived LeRobot
datasets, or the `libero40_plan_sliced.json` and `libero90_plan_sliced.json`
metadata needed by the public converter. RLDS source frames alone do not contain
SkillNet's frame-level skill boundaries.

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
of up to four skill tokens, not a one-hot vector.

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
`--include-derived-datasets` only when the derived LeRobot dataset repos should
be visible to the current account.

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
LIBERO-90-to-LIBERO-Skill zero-shot setting. For LIBERO-40 evaluation, override
`CONFIG_NAME`, `TASK_SUITE`, and `SKILL_ANNOTATION_PATH` as shown below.

Evaluate the LIBERO-40 checkpoint over the four standard LIBERO suites:

```bash
for suite in libero_spatial libero_object libero_goal libero_10; do
  CONFIG_NAME=pi05_libero_moe_skill_4_40 \
  CKPT_DIR=checkpoints/pi05_libero_moe_skill_4_40/moe_add_balance_4_128_30000/29999 \
  TASK_SUITE="${suite}" \
  SKILL_ANNOTATION_PATH=examples/libero/annotations/instruct2plan_40.json \
  NUM_TRIALS=50 \
  VIDEO_OUT_PATH="data/libero/videos/skillnet_libero40_${suite}" \
  bash examples/libero/run_eval_libero_skill_moe.sh
done
```

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
by 8 skill-composition tasks.
Most LIBERO installs register the benchmark class under the key `libero_skill`;
the evaluation script accepts both `libero_skill` and `libero_skill_obj`.
After setup, `install_libero.sh` checks that at least one of those keys is
visible from `benchmark.get_benchmark_dict()`. If the check warns, source
`examples/libero/skillnet_env.sh` or register/copy the bundled
`libero_skill_obj` bddl/init/map files into the LIBERO installation used for
evaluation.

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
VIDEO_OUT_PATH=data/libero/videos/skill_moe_libero_skill_obj \
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
For LIBERO-Skill, the base rollout limit is 800 simulator steps; zero-shot mode
doubles it to 1600.

The 9 registered LIBERO-Skill tasks are:

1. `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`
2. `KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_on_the_plate`
3. `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_the_bottom_drawer_of_the_cabinet`
4. `KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_the_plate`
5. `KITCHEN_SCENE11_close_the_top_drawer_of_the_cabinet_and_close_the_microwave`
6. `KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl_and_open_the_top_drawer_of_the_cabinet`
7. `KITCHEN_SCENE12_put_the_black_bowl_on_the_plate_and_close_the_microwave`
8. `KITCHEN_SCENE15_close_the_drawer_of_the_cabinet_and_turn_off_the_stove`
9. `KITCHEN_SCENE13_put_the_black_bowl_on_the_plate_and_open_the_microwave`

Expected outputs:

- Rollout videos are written to `VIDEO_OUT_PATH`.
- Per-task success rates are printed by the LIBERO evaluator.
- The reported LIBERO-Skill number should be computed as the mean success rate
  over the 9 tasks above with the same `NUM_TRIALS` for every task.

## Release Scope

The release tree is pruned to the public SkillNet reproduction path. Internal
ablation, visualization, and real-robot launchers are intentionally excluded.
For reproducible LIBERO training and LIBERO-Skill evaluation, use the public
launch scripts in this document; they use repo-relative paths and require
explicit checkpoints.
