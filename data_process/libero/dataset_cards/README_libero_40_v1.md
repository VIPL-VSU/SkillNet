---
pretty_name: SkillNet LIBERO-40 v1
task_categories:
- robotics
tags:
- skillnet
- libero
- lerobot
- robot-learning
---

# SkillNet LIBERO-40 v1

This dataset is the LeRobot-format LIBERO-40 training split used by the
SkillNet public release.

## Source

The source demonstrations are derived from the public RLDS release:

```text
openvla/modified_libero_rlds
```

SkillNet adds frame-level skill labels from the released instruction-to-plan
maps and skill-slice metadata described in the main repository.

## Dataset Statistics

| Field | Value |
| --- | ---: |
| Tasks | 40 |
| Episodes | 3,862 |
| Frames | 273,465 |
| Format | LeRobot |

Expected key features include:

```text
image
wrist_image
state
actions
class
all_classes
task_index
episode_index
frame_index
timestamp
```

## Verification

From the SkillNet repository root:

```bash
python scripts/verify_lerobot_dataset.py "$LEROBOT_HOME/jsw19/libero_40_v1" \
  --expected-tasks 40 \
  --expected-episodes 3862 \
  --expected-frames 273465 \
  --require-feature class \
  --require-feature all_classes \
  --strict-parquet \
  --require-column class \
  --require-column all_classes
```

## Citation

If you use this dataset, please cite SkillNet and the original LIBERO/OpenVLA
data sources.
