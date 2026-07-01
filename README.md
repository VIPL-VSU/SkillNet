# SkillNet

[![SkillNet release checks](https://github.com/VIPL-VSU/SkillNet/actions/workflows/release-check.yml/badge.svg?branch=skillnet-public-release)](https://github.com/VIPL-VSU/SkillNet/actions/workflows/release-check.yml?query=branch%3Askillnet-public-release)

This repository contains the open-source release scaffold for SkillNet. The
current drop focuses on the SkillNet code used for LIBERO in-domain training,
LIBERO-Skill compositional out-of-domain evaluation, and RoboTwin few-shot
transfer.

## Release Contents

- `skill_moe/skillnet/`: SkillNet source tree for training and evaluation.
- `skill_moe/skillnet/examples/libero/`: LIBERO and LIBERO-Skill evaluation entrypoints.
- `skill_moe/skillnet/third_party/libero/libero/libero/bddl_files/libero_skill_obj/`: LIBERO-Skill task definitions.
- `skill_moe/skillnet/third_party/libero/libero/libero/bddl_files/libero_skill_obj/public_task_manifest.json`: public 9-task LIBERO-Skill manifest.
- `skill_moe/skillnet/third_party/libero/libero/libero/init_files/libero_skill_obj/`: LIBERO-Skill initial-state files.
- `data_process/skill_hierarchy/`: motion-code annotation, clustering, and tokenization utilities.
- `data_process/libero/`: LIBERO-40 and LIBERO-90 source-download and LeRobot conversion scripts.
- `data_process/robotwin/`: RoboTwin-2.0 few-shot task download, conversion, metadata, and collection helpers.
- `docs/quick_start.md`: installation, environment setup, checkpoint download, and smoke checks.
- `docs/release_status.md`: current release-readiness checklist and known external asset gates.
- `docs/skill_hierarchy.md`: SkillNet hierarchy construction and tokenization notes.
- `docs/libero_data_processing.md`: complete notes for reproducing the LIBERO LeRobot datasets used by the SkillNet training configs.
- `docs/training_and_evaluation.md`: public launch instructions for LIBERO-40/90 Skill-MoE training and LIBERO-Skill evaluation.
- `docs/robotwin_few_shot.md`: RoboTwin few-shot task lists, data setup, training, evaluation, and reported results.

The RoboTwin few-shot release includes paper-aligned task plans, public data
helpers, LeRobot conversion, skill metadata, training configs, training
launchers, and a simulator-side evaluation adapter for a local RoboTwin-2.0
checkout.

## Status

This is a source-code release scaffold with a lightweight editable Python
package. The runnable SkillNet source root is `skill_moe/skillnet`. A few
low-level package names are retained only so pi0.5-compatible checkpoints and
policy clients load without conversion; public commands, environment variables,
checkpoints, and documentation are organized as SkillNet workflows. See
`docs/release_status.md` for the current release-readiness checklist, including
the derived LeRobot dataset visibility gate for direct LIBERO training. The
GitHub Actions release gate in `.github/workflows/release-check.yml` runs the
same public checks on pushes and pull requests to the release branch.

## Experiments Covered

- LIBERO-40 in-domain Skill-MoE training and evaluation.
- LIBERO-90 Skill-MoE training and LIBERO-Skill zero-shot evaluation.
- RoboTwin-2.0 few-shot transfer protocol, data preparation, training launchers, and simulator-side evaluation adapter.

| Track | Config | Dataset | Asset status | Checkpoint |
| --- | --- | --- | --- | --- |
| LIBERO-40 | `pi05_libero_moe_skill_4_40` | `jsw19/libero_40_v1` | Derived LeRobot dataset; direct Hub visibility is checked separately | [`jsw19/SkillNet-LIBERO-40`](https://huggingface.co/jsw19/SkillNet-LIBERO-40) |
| LIBERO-90 to LIBERO-Skill | `pi05_libero_moe_skill_4_90` | `jsw19/libero_90_v1` | Derived LeRobot dataset; direct Hub visibility is checked separately | [`jsw19/SkillNet-LIBERO-90`](https://huggingface.co/jsw19/SkillNet-LIBERO-90) |
| RoboTwin pretraining | `pi05_robotwin_moe_skill_pretrain` | `jsw19/robotwin_pretrain_v1` | Generate locally from RoboTwin-2.0 sources | Train locally |
| RoboTwin per-task few-shot transfer | `pi05_robotwin_moe_skill_transfer` | `jsw19/robotwin_<task>_v1` | Generate one per-task dataset locally | Train locally from the RoboTwin pretraining checkpoint |

The dataset names are the canonical LeRobot `repo_id` values used by the
released configs. The default release namespace is controlled by
`SKILLNET_RELEASE_HF_NAMESPACE` and currently points at the published assets
listed above; mirror the assets under another Hub namespace and set that
variable to switch the training configs and public converters together. LIBERO
checkpoint weights are published; RoboTwin checkpoint weights are not part of
this release and should be trained from the included configs. Direct LIBERO
training requires either public access to the derived LeRobot datasets above or
a local rebuild from the public RLDS sources plus the included compact
slice-index metadata under `data_process/libero/slice_indices/`. The plain
`--hub-smoke` check verifies checkpoints and source datasets; add
`--include-libero-derived-datasets` only when those derived LIBERO datasets
should be publicly visible.

## 1. Quick Start

Clone SkillNet and keep the repository root as your documentation and data
processing root:

```bash
# On Windows, clone under a short, non-user-specific directory because
# LIBERO-Skill task filenames are long. The -c flag handles checkout; the
# local config keeps later git status/diff commands from hitting path limits.
git -c core.longpaths=true clone --branch skillnet-public-release --depth 1 https://github.com/VIPL-VSU/SkillNet.git SkillNet
cd SkillNet
git config core.longpaths true
```

Follow `docs/quick_start.md` for installation, smoke checks, checkpoint
download, normalization-stat computation, training launchers, and evaluation
commands. The runnable SkillNet source root used by training commands is
`skill_moe/skillnet`.

Before installing simulator stacks, you can run:

```bash
python scripts/check_public_release.py
```

To also verify the editable package metadata in a temporary Python 3.10
environment without downloading heavyweight runtime dependencies, run:

```bash
python scripts/check_public_release.py --install-smoke
```

If Python 3.10 is not discoverable as `3.10` on your machine, pass an explicit
interpreter:

```bash
python scripts/check_public_release.py --install-smoke --install-python /path/to/python3.10
```

To verify the public Hugging Face checkpoint and source-dataset links, run:

```bash
python scripts/check_public_release.py --hub-smoke
```

`--hub-smoke` checks public Hub visibility without using local Hub tokens.
Before announcing direct LIBERO training from Hub datasets, also run the
dataset-visibility check documented in `docs/release_status.md`.

## 2. Skill Hierarchy

SkillNet builds reusable skill tokens from short manipulation subtasks. The
public hierarchy tools cover released-tokenizer usage, optional motion-code
annotation, fixed-center weighted motion-code assignment, and final
tokenization into:

```text
[motion_cluster_id, verbnet_class_id, verb_id]
```

See `docs/skill_hierarchy.md` and `data_process/skill_hierarchy/`.

## 3. In-Domain Training and Evaluation

LIBERO data processing is documented in `docs/libero_data_processing.md`. The
released converters build `jsw19/libero_40_v1` and `jsw19/libero_90_v1` from the
public RLDS sources plus the included compact slice-index metadata.

LIBERO-40 in-domain training and evaluation launch commands are documented in
`docs/training_and_evaluation.md`.

## 4. LIBERO-90 Training for LIBERO-Skill Zero-Shot Evaluation

LIBERO-90 training and LIBERO-Skill zero-shot evaluation are documented in
`docs/training_and_evaluation.md`. The zero-shot benchmark does not use
LIBERO-Skill task trajectories for training. The released LIBERO-Skill
benchmark files are included under the `bddl_files/libero_skill_obj` and
`init_files/libero_skill_obj` directories inside
`skill_moe/skillnet/third_party/libero/libero/libero/`. The reported public
benchmark task order is recorded in
`bddl_files/libero_skill_obj/public_task_manifest.json`.

## 5. Few-Shot Transfer

RoboTwin-2.0 few-shot data download, LeRobot conversion, task lists, training
commands, evaluation adapter, and reported SkillNet results are documented in
`docs/robotwin_few_shot.md`.

## License

Unless otherwise noted, this repository is released under Apache-2.0. Third-party components retain their original licenses; see `NOTICE.md` and any license files under `skill_moe/skillnet/third_party/`.
