# RoboTwin Few-Shot Experiments

This document records the RoboTwin-2.0 few-shot transfer setting used in the
SkillNet paper and the public data-preparation, training, and evaluation
entrypoints. The public evaluator is split into a SkillNet policy server and a
RoboTwin environment-side adapter so simulator dependencies stay outside the
SkillNet package.

## Experiment Setting

RoboTwin-2.0 contains 50 tasks covering 12 manipulation skills. The SkillNet
few-shot experiment uses:

- 15 pretraining tasks.
- 15 held-out transfer tasks.
- 50 expert trajectories per pretraining task.
- 50 single-task transfer trajectories per held-out task.
- Transfer tasks are fine-tuned separately.

The key training parameters are:

| Phase | Batch | Steps | Peak LR | Optimizer | EMA | Action horizon | Experts | Top-k | Skill dim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrain | 32 | 20,000 | `2.5e-5` | AdamW, beta1=0.9, beta2=0.95 | 0.99 | 10 | 4 | 1 | 64 |
| Transfer | 32 | 1,000 | `2.5e-5` | AdamW, beta1=0.9, beta2=0.95 | 0.99 | 10 | 4 | 1 | 64 |

## Pretraining Tasks

The skill names in the two tables below are human-readable summaries. The
training and conversion code uses the flat integer skill ids in
`data_process/robotwin/robotwin_plan.json` as the source of truth.

| Task | Skills |
| --- | --- |
| `adjust_bottle` | pick |
| `beat_block_hammer` | pick, strike |
| `click_alarmclock` | press |
| `click_bell` | press |
| `grab_roller` | pick, pick |
| `handover_block` | pick, pass, place |
| `lift_pot` | pick, pick, lift |
| `move_can_pot` | pick, place |
| `move_playingcard_away` | pick, place |
| `open_microwave` | open |
| `place_burger_fries` | pick, pick, place, place |
| `place_object_basket` | pick, place, pick, place |
| `rotate_qrcode` | pick, rotate |
| `shake_bottle_horizontally` | pick, shake, place |
| `stack_blocks_two` | pick, place, pick, place |

## Transfer Tasks

| Task | Skills |
| --- | --- |
| `blocks_ranking_size` | pick, place, pick, place, pick, place |
| `hanging_mug` | pick, rotate, place, pick, hang |
| `move_pillbottle_pad` | pick, place |
| `open_laptop` | pick, lift |
| `place_a2b_left` | pick, place |
| `place_bread_basket` | pick, pick, place |
| `place_bread_skillet` | pick, place |
| `place_cans_plasticbox` | pick, place, pick, place |
| `place_fan` | pick, place |
| `press_stapler` | press |
| `scan_object` | pick, pick, scan |
| `shake_bottle` | pick, shake, place |
| `stack_blocks_three` | pick, place, pick, place, pick, place, pick, place, pick, place |
| `stack_bowls_two` | pick, place |
| `stamp_seal` | pick, stamp |

## Data Preparation

Download RoboTwin-2.0 50-demo zips:

```bash
python data_process/robotwin/download_robotwin_sources.py \
  --task-set paper \
  --output-dir ./robotwin_datasets
```

For networks that need a Hugging Face mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com \
python data_process/robotwin/download_robotwin_sources.py --task-set paper
```

If collecting demonstrations from a local RoboTwin installation:

```bash
ROBOTWIN_ROOT="<path-to-robotwin-checkout>" GPU_ID=0 \
bash data_process/robotwin/collect_train_data.sh pretrain

ROBOTWIN_ROOT="<path-to-robotwin-checkout>" GPU_ID=0 \
bash data_process/robotwin/collect_train_data.sh transfer
```

Convert the downloaded or collected demonstrations to LeRobot format:

```bash
python data_process/robotwin/convert_robotwin_to_lerobot.py \
  --source-dir ./robotwin_datasets \
  --task-set pretrain \
  --output-repo-id jsw19/robotwin_pretrain_v1

python data_process/robotwin/convert_robotwin_to_lerobot.py \
  --source-dir ./robotwin_datasets \
  --task-set transfer \
  --output-repo-id jsw19/robotwin_transfer_v1
```

The converter reads the public RoboTwin raw layout under
`data/episode*.hdf5`. The downloaded archives also include
`instructions/episode*.json`, but the released converter uses the canonical
task description from `robotwin_plan.json` so the language prompt and skill ids
stay aligned. Older `aligned_joints.h5` extracted episodes are supported via
`--source-format aligned`. Both raw RoboTwin episodes and this compatibility
path are normalized to the same left-arm-first 16-D state/action order described
below. Use `--dry-run` before a full conversion to check that the expected
episodes are visible.

For paper-style per-task fine-tuning, a single held-out task can be converted
separately:

```bash
python data_process/robotwin/convert_robotwin_to_lerobot.py \
  --source-dir ./robotwin_datasets \
  --tasks blocks_ranking_size \
  --output-repo-id jsw19/robotwin_blocks_ranking_size_v1
```

To reproduce the paper-style "one checkpoint per held-out task" protocol for
all 15 transfer tasks, create all matching per-task LeRobot datasets before
running the training loop:

```bash
for task in \
  blocks_ranking_size hanging_mug move_pillbottle_pad open_laptop \
  place_a2b_left place_bread_basket place_bread_skillet \
  place_cans_plasticbox place_fan press_stapler scan_object \
  shake_bottle stack_blocks_three stack_bowls_two stamp_seal; do
  python data_process/robotwin/convert_robotwin_to_lerobot.py \
    --source-dir ./robotwin_datasets \
    --tasks "${task}" \
    --output-repo-id "jsw19/robotwin_${task}_v1"
done
```

These per-task repo ids match the transfer launcher's default behavior when
`TRANSFER_TASK` is set.

The `jsw19/robotwin_*_v1` values are LeRobot `repo_id` names for local datasets
under `LEROBOT_HOME` or `HF_LEROBOT_HOME`. Set the same cache location before
conversion and training. RoboTwin derived LeRobot datasets are generated locally
in this release; they are not required to be publicly downloadable from Hub.

Build paper-task metadata from the released task plan and hierarchical skill
annotations:

```bash
python data_process/robotwin/build_robotwin_skill_metadata.py \
  --task-set paper \
  --output robotwin_skill_metadata.jsonl
```

The metadata records contain the task name, instruction, flat skill ids,
subtask plan, hierarchical skill tokens, and split. The current RoboTwin
training path consumes the flat `skills` sequence from `robotwin_plan.json`;
hierarchical tokens from `skill_anno_robotwin.json` are provided for analysis
and hierarchy-token experiments.

## LeRobot Schema

The RoboTwin SkillNet configs expect LeRobot-format datasets with these feature
keys:

| Column | Meaning |
| --- | --- |
| `head_color` | base camera RGB image |
| `hand_left_color` | left wrist RGB image |
| `hand_right_color` | right wrist RGB image |
| `state` | 16-D dual-arm proprioceptive state |
| `actions` | dual-arm action sequence, first 16 dims used |
| `task` | language instruction used as the model prompt |
| `skills` | list of flat skill ids for the instruction |

The converter writes state and action in
`[left_arm, left_gripper, right_arm, right_gripper]` order. The training config
then repacks these LeRobot features into the SkillNet model input fields.

`skills` is stored in the LeRobot dataset as a JSON-encoded string, for example
`"[1, 3, 5]"`, because the public LeRobot writer path expects a scalar column
for variable-length metadata. The SkillNet RoboTwin transform parses this JSON
string back into a variable-length integer sequence. It is not a one-hot vector:
during training it is truncated or padded to four entries, shifted by +1 so `0`
can be used as padding, and paired with `skill_mask`.

The paper describes 12 semantic manipulation skills. The public flat skill id
space contains 13 non-padding ids in `robotwin_plan.json`, and the released
RoboTwin configs set `skill_num=14` to reserve id `0` for padding.

Default dataset ids can be overridden with environment variables:

```bash
export SKILLNET_ROBOTWIN_PRETRAIN_REPO_ID=jsw19/robotwin_pretrain_v1
export SKILLNET_ROBOTWIN_TRANSFER_REPO_ID=jsw19/robotwin_transfer_v1
```

For per-task transfer datasets, the transfer launcher also supports:

```bash
cd "${SKILLNET_REPO_ROOT}/skill_moe/skillnet"
TRANSFER_TASK=blocks_ranking_size bash scripts/run_train_robotwin_transfer_moe_skill.sh
```

When `TRANSFER_TASK` is set and `SKILLNET_ROBOTWIN_TRANSFER_REPO_ID` is not set,
the launcher uses `jsw19/robotwin_${TRANSFER_TASK}_v1`.

## Training

Set `SKILLNET_REPO_ROOT` as described in `docs/quick_start.md`, then run from
the SkillNet source root:

Pretrain on the 15 RoboTwin pretraining tasks:

```bash
cd "${SKILLNET_REPO_ROOT}/skill_moe/skillnet"
bash scripts/run_train_robotwin_pretrain_moe_skill.sh
```

Fine-tune on a held-out transfer task:

```bash
cd "${SKILLNET_REPO_ROOT}/skill_moe/skillnet"
export SKILLNET_ROBOTWIN_TRANSFER_INIT_PARAMS=checkpoints/pi05_robotwin_moe_skill_pretrain/robotwin_moe_skill_pretrain/19999/params
TRANSFER_TASK=blocks_ranking_size \
bash scripts/run_train_robotwin_transfer_moe_skill.sh
```

The launchers compute normalization statistics automatically when
`assets/<config>/<repo_id>/norm_stats.json` is missing. They call:

```bash
scripts/compute_norm_stats_moe_skill.py
scripts/train_moe_skill.py
```

The released config names are:

| Phase | Config | Default dataset id |
| --- | --- | --- |
| Pretrain | `pi05_robotwin_moe_skill_pretrain` | `jsw19/robotwin_pretrain_v1` |
| Transfer | `pi05_robotwin_moe_skill_transfer` | `jsw19/robotwin_transfer_v1` |

`jsw19/robotwin_transfer_v1` is the aggregate transfer default for smoke runs
or local experiments. The paper-style few-shot setup fine-tunes one checkpoint
per held-out task; set `TRANSFER_TASK` so the launcher switches the dataset id
to `jsw19/robotwin_${TRANSFER_TASK}_v1` and keeps normalization assets aligned
with that task.

The pi0.5 base checkpoint is controlled by `SKILLNET_PI05_BASE_PARAMS`.
Transfer initialization is controlled by
`SKILLNET_ROBOTWIN_TRANSFER_INIT_PARAMS`. The transfer launcher requires this
variable by default so the paper few-shot path cannot silently fall back to the
pi0.5 base checkpoint. Set `ALLOW_PI05_TRANSFER_INIT=1` only for a debugging run
that intentionally starts transfer from `SKILLNET_PI05_BASE_PARAMS`.

Fine-tune all 15 transfer tasks with one checkpoint per task:

```bash
cd "${SKILLNET_REPO_ROOT}/skill_moe/skillnet"
export SKILLNET_ROBOTWIN_TRANSFER_INIT_PARAMS=checkpoints/pi05_robotwin_moe_skill_pretrain/robotwin_moe_skill_pretrain/19999/params

for task in \
  blocks_ranking_size hanging_mug move_pillbottle_pad open_laptop \
  place_a2b_left place_bread_basket place_bread_skillet \
  place_cans_plasticbox place_fan press_stapler scan_object \
  shake_bottle stack_blocks_three stack_bowls_two stamp_seal; do
  TRANSFER_TASK="${task}" \
  EXP_NAME="robotwin_moe_skill_transfer_${task}" \
  bash scripts/run_train_robotwin_transfer_moe_skill.sh
done
```

Checkpoint release status:

| Phase | Status |
| --- | --- |
| RoboTwin pretraining | Train locally with `pi05_robotwin_moe_skill_pretrain` |
| RoboTwin transfer | Train locally per task from the RoboTwin pretraining checkpoint |

The public repository includes configs and launchers for the paper protocol,
but RoboTwin checkpoint weights are not published in this release.

## Evaluation

The cleaned evaluator lives under:

```bash
skill_moe/skillnet/examples/robotwin/
```

Run it from the SkillNet source root and point `ROBOTWIN_ROOT` to a local
RoboTwin-2.0 checkout with simulator dependencies installed. The launcher starts
the SkillNet policy server, connects the RoboTwin environment adapter to it,
injects the same flat skill ids from
`${SKILLNET_REPO_ROOT}/data_process/robotwin/robotwin_plan.json`, and writes
per-task `episodes.jsonl` plus an aggregate `summary.json`.
The wrapper validates `ROBOTWIN_ROOT`, `SKILL_PLAN`, and `CKPT_DIR` before
starting evaluation, writes policy-server logs under
`data/robotwin/server_logs/`, and waits for the server port to accept
connections before launching the simulator adapter.

The evaluator expects a RoboTwin-2.0-style checkout that provides this public
API/tree contract:

- `collect_data.sh` for optional demonstration collection.
- `task_config/<TASK_CONFIG>.yml`, `_camera_config.yml`, and
  `_embodiment_config.yml`.
- `envs/__init__.py` exporting `CONFIGS_PATH`.
- importable task modules named `envs.<task>` with classes named `<task>`.
- importable `policy` and `description/utils` packages on `PYTHONPATH`.
- `test_render.Sapien_TEST` for the default render preflight, or pass
  `--skip-render-test` through the launcher after validating rendering
  separately.

Use the official RoboTwin-2.0 codebase or a fork that preserves this API. The
SkillNet repository does not vendor the simulator, assets, or SAPIEN/MuJoCo
runtime dependencies.

Public RoboTwin-2.0 entrypoints:

- Repository: <https://github.com/RoboTwin-Platform/RoboTwin>
- Documentation: <https://robotwin-platform.github.io/doc/index.html>

Evaluate a single fine-tuned transfer checkpoint:

```bash
cd "${SKILLNET_REPO_ROOT}/skill_moe/skillnet"

export ROBOTWIN_ROOT="<path-to-robotwin-checkout>"
export CKPT_DIR=checkpoints/pi05_robotwin_moe_skill_transfer/robotwin_moe_skill_transfer_blocks_ranking_size/999
export TRANSFER_TASK=blocks_ranking_size
export TASKS=blocks_ranking_size

bash examples/robotwin/run_eval_robotwin_moe_skill.sh
```

Evaluate a set of transfer tasks with an already running SkillNet policy
server:

```bash
cd "${SKILLNET_REPO_ROOT}/skill_moe/skillnet"
START_SERVER=0 \
ROBOTWIN_ROOT="<path-to-robotwin-checkout>" \
TASK_SET=transfer \
NUM_TRIALS=20 \
bash examples/robotwin/run_eval_robotwin_moe_skill.sh
```

The launcher accepts these common overrides:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CONFIG_NAME` | `pi05_robotwin_moe_skill_transfer` | SkillNet training config used to load the checkpoint |
| `CKPT_DIR` | unset | Checkpoint directory containing `params` and `assets` |
| `ROBOTWIN_ROOT` | unset | Local RoboTwin-2.0 checkout |
| `TASK_CONFIG` | `demo_clean` | RoboTwin `task_config/*.yml` stem |
| `TASKS` | unset | Space-separated task list, overrides `TASK_SET` |
| `TASK_SET` | `transfer` | `pretrain`, `transfer`, `paper`, or `all` |
| `NUM_TRIALS` | `20` | Trials per task |
| `SEED` | `0` | First simulator seed considered for each task |
| `HOST` | `127.0.0.1` | SkillNet policy-server host |
| `PORT` | `8098` | SkillNet policy-server port |
| `SERVER_GPU` | `0` | GPU id used when the wrapper starts a policy server |
| `START_SERVER` | `1` | Set to `0` to reuse an already running policy server |
| `SERVER_WAIT_SECONDS` | `60` | Maximum wait time for the server port to become reachable |
| `SERVER_LOG_PATH` | `data/robotwin/server_logs/<config>_<port>.log` | Policy-server log file |
| `ACTION_HORIZON` | `10` | Actions executed per policy-server call |
| `RESULT_DIR` | `data/robotwin/eval_results` | Output directory |

Additional Python evaluator flags can be passed after the launcher command, for
example `--skip-render-test` after validating rendering separately,
`--max-seed-attempts` to control expert-seed filtering, `--no-expert-seed-filter`
for debugging, or `--instruction-type task` to force the canonical plan
description prompt.

For per-task transfer checkpoints, set `TRANSFER_TASK`. If
`SKILLNET_ROBOTWIN_TRANSFER_REPO_ID` is not already set, the launcher maps it to
`jsw19/robotwin_${TRANSFER_TASK}_v1` so the checkpoint loads the same
normalization-stat asset id used during fine-tuning.

Evaluate all transfer tasks after the per-task checkpoints are available:

```bash
cd "${SKILLNET_REPO_ROOT}/skill_moe/skillnet"
export ROBOTWIN_ROOT="<path-to-robotwin-checkout>"

for task in \
  blocks_ranking_size hanging_mug move_pillbottle_pad open_laptop \
  place_a2b_left place_bread_basket place_bread_skillet \
  place_cans_plasticbox place_fan press_stapler scan_object \
  shake_bottle stack_blocks_three stack_bowls_two stamp_seal; do
  TRANSFER_TASK="${task}" \
  TASKS="${task}" \
  CKPT_DIR="checkpoints/pi05_robotwin_moe_skill_transfer/robotwin_moe_skill_transfer_${task}/999" \
  RESULT_DIR="data/robotwin/eval_results/${task}" \
  bash examples/robotwin/run_eval_robotwin_moe_skill.sh
done
```

Each run writes `episodes.jsonl` and `summary.json` under `RESULT_DIR`.
Aggregate the paper-style average by averaging the `success_rate` values across
the 15 transfer-task `summary.json` files.

The public adapter keeps the evaluation semantics used by the paper protocol:
expert seed filtering, RoboTwin environment rollout, instruction selection,
skill-plan injection, and success-rate reporting.

## Reported Results

The paper reports success rate (%) on 15 RoboTwin transfer tasks:

| Method | Average |
| --- | --- |
| pi0 | 22.1 |
| pi0.5 | 38.8 |
| Vanilla MoE | 40.5 |
| SkillNet | 44.7 |
| Ablation, motion code | 38.5 |
| Ablation, balance loss | 38.0 |

Per-task values are in Appendix D.1 of the paper. The main takeaway is that
SkillNet improves over pi0.5 by 5.9 points in this RoboTwin few-shot setting.

## Release Contents

- Public data download helper.
- Public data-collection wrapper.
- Raw RoboTwin zip or extracted episode to LeRobot conversion script.
- Paper-aligned task lists and experiment protocol.
- Paper task metadata builder with flat and hierarchical skill annotations.
- RoboTwin LeRobot policy transform.
- Pretraining and transfer training configs.
- Public pretraining and transfer training launch scripts.
- Public RoboTwin simulator adapter for SkillNet websocket policies.
- Per-task `episodes.jsonl` and aggregate `summary.json` success-rate outputs.
