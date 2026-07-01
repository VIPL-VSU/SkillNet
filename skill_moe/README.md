# SkillNet

This directory contains the runnable SkillNet Skill-MoE source used for the
public release. The user-facing project structure is SkillNet; a few Python
package paths keep compatibility namespaces because the pi0.5
checkpoint/runtime format imports those modules directly. Public instructions
should refer to this directory as the SkillNet runtime root.

## What Is Included

- `skillnet/src/openpi/models/*moe*.py`: compatibility import path for the core Skill-MoE model and config variants.
- `skillnet/src/openpi/training/config_moe_skill.py`: compatibility import path for LIBERO, RoboCasa, RoboTwin, and Skill-MoE training configs.
- `skillnet/src/openpi/training/data_loader_skill.py` and `skillnet/src/openpi/transforms_skill.py`: compatibility import paths for skill/object annotation support.
- `skillnet/scripts/train_moe_skill.py`: Skill-MoE training launcher.
- `skillnet/scripts/serve_policy_moe_skill.py`: Skill-MoE policy server.
- `skillnet/examples/libero/`: LIBERO and LIBERO-Skill evaluation scripts.
- `skillnet/examples/robocasa/scripts/`: RoboCasa evaluation and data-conversion helpers.
- `skillnet/third_party/libero/libero/libero/bddl_files/libero_skill_obj/`: LIBERO-Skill benchmark task files.
- `skillnet/third_party/libero/libero/libero/bddl_files/libero_skill_obj/public_task_manifest.json`: public 9-task LIBERO-Skill manifest.
- `skillnet/third_party/libero/libero/libero/init_files/libero_skill_obj/`: LIBERO-Skill initial states.

## Not Included

- Checkpoints and pretrained weights.
- Training datasets and generated LeRobot datasets.
- Evaluation videos and logs.
- The full GR00T/RoboCasa dependency tree; install RoboCasa/GR00T dependencies separately if using those scripts.
- Internal ablation, visualization, and real-robot launch scripts.

## Using SkillNet

Use this directory as the SkillNet source root:

```bash
cd skill_moe/skillnet
uv pip install -e .
uv pip install -e packages/openpi-client
export PYTHONPATH="${PWD}/src:${PWD}/packages/openpi-client/src:${PWD}/third_party/libero:${PYTHONPATH:-}"
```

Then install any dataset-specific dependencies. For LIBERO evaluation, run
`bash install_libero.sh`, then source `examples/libero/skillnet_env.sh`.
The installer reuses the currently active venv by default; set
`SKILLNET_VENV_PATH=examples/libero/.venv` if you want a separate LIBERO
environment. It also attempts to register the bundled LIBERO-Skill benchmark in
an external LIBERO install by calling
`examples/libero/install_libero_skill_assets.py --install`; use that helper with
`--dry-run` to inspect changes first. Generated activation/env files are not
present in a fresh clone.
RoboCasa helpers additionally require a local RoboCasa/GR00T installation.

## Path Configuration

For LIBERO Skill-MoE training and LIBERO-Skill evaluation, prefer the public
launch scripts added under `skillnet/scripts/` and
`skillnet/examples/libero/`. They use repo-relative paths and explicit
checkpoint arguments. The original internal ablation, visualization, and
real-robot launch scripts are intentionally excluded from this release tree.

## Main Entry Points

- Skill hierarchy tokenization: `../../data_process/skill_hierarchy/skill_hierarchy_tokenizer.py`.
- LIBERO in-domain training: `scripts/train_moe_skill.py`, configs in `src/openpi/training/config_moe_skill.py`.
- RoboTwin few-shot training: `scripts/run_train_robotwin_pretrain_moe_skill.sh` and `scripts/run_train_robotwin_transfer_moe_skill.sh`.
- RoboTwin few-shot eval: `examples/robotwin/run_eval_robotwin_moe_skill.sh`.
- RoboCasa helpers: `examples/robocasa/scripts/robocasa_eval_skill.py` and related data-conversion scripts.
- LIBERO-Skill OOD eval: `examples/libero/main_skill_obj_test_moe_skill.py` plus `third_party/libero/libero/libero/bddl_files/libero_skill_obj/public_task_manifest.json` and the corresponding bddl/init folders.

See `../../docs/skill_hierarchy.md` for hierarchy construction and
`../../docs/training_and_evaluation.md` for the public LIBERO launch commands.
See `../../docs/robotwin_few_shot.md` for RoboTwin few-shot setup, training,
and evaluation.
