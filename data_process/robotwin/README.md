# RoboTwin Few-Shot Data

This directory contains public helpers for the RoboTwin-2.0 few-shot transfer
experiments described in the SkillNet paper.

Files in this directory:

- `download_robotwin_sources.py`: download RoboTwin-2.0 50-demo archives.
- `convert_robotwin_to_lerobot.py`: convert downloaded or locally collected demonstrations to LeRobot format.
- `collect_train_data.sh`: collect demonstrations from a local RoboTwin checkout.
- `robotwin_plan.json`: released flat skill plan for RoboTwin tasks.
- `skill_anno_robotwin.json`: hierarchical SkillNet annotations for RoboTwin tasks.
- `build_robotwin_skill_metadata.py`: export task metadata as JSONL.

## Task Sets

The paper uses 15 pretraining tasks and 15 transfer tasks.

Pretraining tasks:

```text
adjust_bottle
beat_block_hammer
click_alarmclock
click_bell
grab_roller
handover_block
lift_pot
move_can_pot
move_playingcard_away
open_microwave
place_burger_fries
place_object_basket
rotate_qrcode
shake_bottle_horizontally
stack_blocks_two
```

Transfer tasks:

```text
blocks_ranking_size
hanging_mug
move_pillbottle_pad
open_laptop
place_a2b_left
place_bread_basket
place_bread_skillet
place_cans_plasticbox
place_fan
press_stapler
scan_object
shake_bottle
stack_blocks_three
stack_bowls_two
stamp_seal
```

## Download Released RoboTwin Data

Download the paper pretraining tasks:

```bash
python data_process/robotwin/download_robotwin_sources.py \
  --task-set pretrain \
  --output-dir ./robotwin_datasets
```

Download both pretraining and transfer tasks:

```bash
python data_process/robotwin/download_robotwin_sources.py \
  --task-set paper \
  --output-dir ./robotwin_datasets
```

If Hugging Face access is slow, use a mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com \
python data_process/robotwin/download_robotwin_sources.py --task-set paper
```

The downloader fetches files from `TianxingChen/RoboTwin2.0` using the filename
pattern:

```text
dataset/<task>/aloha-agilex_clean_50.zip
```

## Convert to LeRobot

Convert the 15 pretraining tasks:

```bash
python data_process/robotwin/convert_robotwin_to_lerobot.py \
  --source-dir ./robotwin_datasets \
  --task-set pretrain \
  --output-repo-id jsw19/robotwin_pretrain_v1
```

Convert the 15 held-out transfer tasks:

```bash
python data_process/robotwin/convert_robotwin_to_lerobot.py \
  --source-dir ./robotwin_datasets \
  --task-set transfer \
  --output-repo-id jsw19/robotwin_transfer_v1
```

Convert one transfer task into its own dataset:

```bash
python data_process/robotwin/convert_robotwin_to_lerobot.py \
  --source-dir ./robotwin_datasets \
  --tasks blocks_ranking_size \
  --output-repo-id jsw19/robotwin_blocks_ranking_size_v1
```

The `jsw19/robotwin_*_v1` values are local LeRobot `repo_id` names generated
from `SKILLNET_RELEASE_HF_NAMESPACE=jsw19` by default. Set
`SKILLNET_RELEASE_HF_NAMESPACE` before conversion to use another namespace, or
pass `--output-repo-id` for an exact repo id. Set `LEROBOT_HOME` or
`HF_LEROBOT_HOME` before conversion if you want a specific dataset cache
location, and keep the same cache when running SkillNet training. The derived
RoboTwin LeRobot datasets are generated locally in this release rather than
treated as public Hub download dependencies.

The converter accepts either downloaded zip files or already extracted
demonstration folders. It defaults to `--source-format auto`, which first looks
for the public RoboTwin raw HDF5 layout:

```text
dataset/<task>/aloha-agilex_clean_50.zip
dataset/<task>/aloha-agilex_clean_50/data/episode0.hdf5
dataset/<task>/aloha-agilex_clean_50/instructions/episode0.json
```

The canonical language prompt is taken from `robotwin_plan.json` instead of
sampling per-episode instructions, so the prompt and released skill ids stay
aligned.

The raw HDF5 reader uses these fields:

```text
/joint_action/left_arm
/joint_action/left_gripper
/joint_action/right_arm
/joint_action/right_gripper
/observation/head_camera/rgb
/observation/right_camera/rgb
/observation/left_camera/rgb
```

The 16-D state/action order is:

```text
[left_arm, left_gripper, right_arm, right_gripper]
```

Actions are the next-frame absolute joint state, matching the RoboTwin
processing path used for the few-shot experiments. Images are decoded as RGB and
resized to `640x480` by default. The script also retains a compatibility path
for older extracted episodes containing:

```text
aligned_joints.h5
camera/<frame>/head_color.jpg
camera/<frame>/hand_right_color.jpg
camera/<frame>/hand_left_color.jpg
```

Both raw episodes and this older `aligned_joints.h5` compatibility path are
normalized to the same left-arm-first state/action order above.

Use `--dry-run` to inspect available episodes without writing a dataset. Use
`--source-format raw` or `--source-format aligned` to force one layout. By
default, existing LeRobot outputs under `HF_LEROBOT_HOME` are preserved; pass
`--overwrite` only when intentionally rebuilding.

## Collect New Demonstrations

If you are generating demonstrations with a local RoboTwin installation, use:

```bash
ROBOTWIN_ROOT="<path-to-robotwin-checkout>" GPU_ID=0 \
bash data_process/robotwin/collect_train_data.sh pretrain

ROBOTWIN_ROOT="<path-to-robotwin-checkout>" GPU_ID=0 \
bash data_process/robotwin/collect_train_data.sh transfer
```

The script calls RoboTwin's `collect_data.sh` for each task:

```bash
bash collect_data.sh "<task with spaces>" demo_clean "${GPU_ID}"
```

## Build Skill Metadata

Generate JSONL metadata for the paper task split:

```bash
python data_process/robotwin/build_robotwin_skill_metadata.py \
  --task-set paper \
  --output robotwin_skill_metadata.jsonl
```

Each row contains:

- `task`: RoboTwin task id.
- `description`: language instruction.
- `skills`: flat skill-id sequence used by the current training configs.
- `plan`: subtask plan when available.
- `hierarchical_skill_tokens`: SkillNet hierarchy tokens from the annotation file.
- `split`: `pretrain` or `transfer`.

The converter writes a LeRobot dataset with `task` as the language prompt,
`skills` as the flat skill-id sequence, RGB observations under `head_color`,
`hand_left_color`, and `hand_right_color`, proprioception under `state`, and
actions under `actions`. The SkillNet training config repacks those LeRobot
features into the model input fields.

Implementation detail: `skills` is written as a JSON-encoded string column such
as `"[1, 3, 5]"` so variable-length skill metadata can live in a scalar LeRobot
field. The SkillNet RoboTwin transform parses the JSON string back into integer
ids, pads or truncates to four skills, shifts ids by +1 to reserve `0` for
padding, and emits `skill_mask`.

For optional data collection and simulator evaluation, `ROBOTWIN_ROOT` must
point to a RoboTwin-2.0 checkout that contains `collect_data.sh`,
`task_config/*.yml`, `_camera_config.yml`, `_embodiment_config.yml`, importable
`envs.<task>` modules, `envs.CONFIGS_PATH`, `policy`, `description/utils`, and
`test_render.Sapien_TEST`.
