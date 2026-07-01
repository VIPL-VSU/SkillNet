---
pretty_name: SkillNet LIBERO-90 v1
task_categories:
- robotics
tags:
- skillnet
- libero
- lerobot
- robot-learning
---

# SkillNet LIBERO-90 v1

This dataset is the LeRobot-format LIBERO-90 training split used by the
SkillNet LIBERO-Skill zero-shot evaluation workflow.

## Source

The source demonstrations are derived from the public RLDS release:

```text
jesbu1/libero_90_openvla_processed
```

SkillNet adds frame-level skill labels from the released instruction-to-plan
maps and skill-slice metadata described in the main repository.

## Dataset Statistics

| Field | Value |
| --- | ---: |
| Tasks | 73 |
| Episodes | 7,874 |
| Frames | 574,571 |
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
python scripts/verify_lerobot_dataset.py "$LEROBOT_HOME/jsw19/libero_90_v1" \
  --expected-tasks 73 \
  --expected-episodes 7874 \
  --expected-frames 574571 \
  --require-feature class \
  --require-feature all_classes \
  --strict-parquet \
  --require-column class \
  --require-column all_classes
```

## Citation

If you use this dataset, please cite SkillNet and the original LIBERO/OpenVLA
data sources.
