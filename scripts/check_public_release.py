"""Run lightweight checks for the public SkillNet release tree.

The checks avoid heavyweight simulator/model imports. They are intended for a
fresh clone before running the GPU or simulator workflows.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "NOTICE.md",
    "README.md",
    "SECURITY.md",
    "SUPPORT.md",
    ".github/workflows/release-check.yml",
    "docs/quick_start.md",
    "docs/github_repository_setup.md",
    "docs/release_status.md",
    "docs/skill_hierarchy.md",
    "docs/training_and_evaluation.md",
    "docs/libero_data_processing.md",
    "docs/robotwin_few_shot.md",
    "scripts/verify_lerobot_dataset.py",
    "scripts/publish_lerobot_dataset.py",
    "scripts/check_github_repository_metadata.py",
    "scripts/set_hf_dataset_visibility.py",
    "data_process/skill_hierarchy/README.md",
    "data_process/skill_hierarchy/tokenization_strategy.json",
    "data_process/skill_hierarchy/skill_graph_example.json",
    "data_process/skill_hierarchy/motion_code_clusters.json",
    "data_process/skill_hierarchy/motion_code_annotation_examples.jsonl",
    "data_process/libero/instruct2plan_40.json",
    "data_process/libero/instruct2plan_90.json",
    "data_process/libero/instruct2plan_obj_90.json",
    "data_process/libero/export_libero_skill_slices.py",
    "data_process/libero/slice_indices/libero40_slice_index.json",
    "data_process/libero/slice_indices/libero90_slice_index.json",
    "data_process/libero/dataset_cards/README_libero_40_v1.md",
    "data_process/libero/dataset_cards/README_libero_90_v1.md",
    "data_process/robotwin/robotwin_plan.json",
    "data_process/robotwin/skill_anno_robotwin.json",
    "data_process/robotwin/README.md",
    "skill_moe/skillnet/pyproject.toml",
    "skill_moe/skillnet/packages/openpi-client/pyproject.toml",
    "skill_moe/skillnet/packages/openpi-client/src/openpi_client/__init__.py",
    "skill_moe/skillnet/packages/openpi-client/src/openpi_client/websocket_client_policy.py",
    "skill_moe/skillnet/install_libero.sh",
    "skill_moe/skillnet/scripts/compute_norm_stats_moe_skill.py",
    "skill_moe/skillnet/scripts/train_moe_skill.py",
    "skill_moe/skillnet/scripts/serve_policy_moe_skill.py",
    "skill_moe/skillnet/scripts/run_train_libero40_moe_skill.sh",
    "skill_moe/skillnet/scripts/run_train_libero90_moe_skill.sh",
    "skill_moe/skillnet/examples/libero/annotations/instruct2plan_40.json",
    "skill_moe/skillnet/examples/libero/annotations/instruct2plan_obj_90.json",
    "skill_moe/skillnet/examples/libero/annotations/libero_skill_obj_annotations.json",
    "skill_moe/skillnet/examples/libero/install_libero_skill_assets.py",
    "skill_moe/skillnet/examples/libero/run_eval_libero_skill_moe.sh",
    "skill_moe/skillnet/examples/robotwin/run_eval_robotwin_moe_skill.sh",
    "skill_moe/skillnet/third_party/libero/libero/libero/bddl_files/libero_skill_obj/README.md",
    "skill_moe/skillnet/third_party/libero/libero/libero/bddl_files/libero_skill_obj/public_task_manifest.json",
    "skill_moe/skillnet/third_party/libero/scripts/README.md",
]

PYTHON_FILES = [
    "data_process/skill_hierarchy/skill_hierarchy_tokenizer.py",
    "data_process/libero/download_libero_sources.py",
    "data_process/libero/convert_libero_to_lerobot.py",
    "data_process/libero/export_libero_skill_slices.py",
    "data_process/robotwin/download_robotwin_sources.py",
    "data_process/robotwin/convert_robotwin_to_lerobot.py",
    "data_process/robotwin/build_robotwin_skill_metadata.py",
    "scripts/verify_lerobot_dataset.py",
    "scripts/publish_lerobot_dataset.py",
    "scripts/check_github_repository_metadata.py",
    "scripts/set_hf_dataset_visibility.py",
    "skill_moe/skillnet/examples/libero/install_libero_skill_assets.py",
    "skill_moe/skillnet/examples/robotwin/eval_robotwin_moe_skill.py",
]

HELP_COMMANDS = [
    ["data_process/skill_hierarchy/skill_hierarchy_tokenizer.py", "--help"],
    ["data_process/libero/download_libero_sources.py", "--help"],
    ["data_process/libero/export_libero_skill_slices.py", "--help"],
    ["scripts/verify_lerobot_dataset.py", "--help"],
    ["scripts/publish_lerobot_dataset.py", "--help"],
    ["scripts/check_github_repository_metadata.py", "--help"],
    ["scripts/set_hf_dataset_visibility.py", "--help"],
    ["skill_moe/skillnet/examples/libero/install_libero_skill_assets.py", "--help"],
    ["data_process/robotwin/download_robotwin_sources.py", "--help"],
    ["data_process/robotwin/convert_robotwin_to_lerobot.py", "--help"],
    ["data_process/robotwin/build_robotwin_skill_metadata.py", "--help"],
    ["skill_moe/skillnet/examples/robotwin/eval_robotwin_moe_skill.py", "--help"],
]

SHELL_FILES = [
    "data_process/robotwin/collect_train_data.sh",
    "skill_moe/skillnet/install_libero.sh",
    "skill_moe/skillnet/scripts/run_train_libero40_moe_skill.sh",
    "skill_moe/skillnet/scripts/run_train_libero90_moe_skill.sh",
    "skill_moe/skillnet/scripts/run_train_robotwin_pretrain_moe_skill.sh",
    "skill_moe/skillnet/scripts/run_train_robotwin_transfer_moe_skill.sh",
    "skill_moe/skillnet/examples/libero/run_eval_libero_skill_moe.sh",
    "skill_moe/skillnet/examples/robotwin/run_eval_robotwin_moe_skill.sh",
]

REMOTE_SHARE_PATTERN = re.compile("/share" + r"/project")
WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(r"\b[A-Za-z]:\\")


def literal_word_pattern(codes: tuple[int, ...]) -> re.Pattern[str]:
    return re.compile(r"\b" + re.escape("".join(chr(code) for code in codes)) + r"\b", re.IGNORECASE)


# Site-specific private words are assembled without spelling them out so this
# public checker can scan itself and the git history.
SITE_PRIVATE_WORD_PATTERNS = [
    literal_word_pattern(codes)
    for codes in (
        (120, 115, 119),
        (120, 105, 101, 115, 101),
        (99, 117, 105, 104, 117),
        (106, 105, 110, 103, 110, 101, 110, 103),
        (99, 97, 111, 109, 105, 110, 103, 121, 117),
        (110, 101, 105, 109, 111, 110, 103, 111, 108),
        (104, 101, 108, 105, 110, 103, 101, 101, 114),
    )
]

SENSITIVE_PATTERNS = [
    REMOTE_SHARE_PATTERN,
    re.compile("C:" + r"\\Users|C:" + "/Users"),
    WINDOWS_ABSOLUTE_PATH_PATTERN,
    re.compile(r"~[/\\]"),
    re.compile(r"\$HOME[/\\]"),
    re.compile(r"/home/[A-Za-z0-9_.-]+"),
    re.compile("/" + "root/"),
    re.compile(r"/mnt/[A-Za-z0-9_.-]+"),
    re.compile(r"10\.8\.36\."),
    re.compile(r"ssh\.platform"),
    re.compile(r"job-[0-9a-f-]{16,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"sk-proj-[A-Za-z0-9_-]{20,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"xox[baprs]-[0-9A-Za-z-]{20,}"),
    re.compile(r"wandb_[A-Za-z0-9]{20,}"),
    *SITE_PRIVATE_WORD_PATTERNS,
    re.compile("openpi" + r"[-_/ ]+" + "overlay", re.IGNORECASE),
    re.compile("openpi" + r"[-_/ ]+" + "backend", re.IGNORECASE),
]
HISTORY_SENSITIVE_PATTERNS = [
    pattern for pattern in SENSITIVE_PATTERNS if pattern.pattern not in {r"\b[A-Za-z]:\\", r"~[/\\]", r"\$HOME[/\\]"}
]

SCAN_SUFFIXES = {".cff", ".md", ".py", ".sh", ".json", ".jsonl", ".toml", ".yml", ".yaml"}

TOKENIZATION_STRATEGY_SHA256 = "13d025527a240738e20be3a5e2a208157d623250667db6fc8949830a1ffd8081"
MOTION_CODE_CENTERS = {
    "200200": "cluster_0",
    "200100": "cluster_1",
    "100200": "cluster_2",
    "200010": "cluster_3",
    "100100": "cluster_4",
    "201010": "cluster_5",
    "200001": "cluster_6",
    "201201": "cluster_7",
    "000001": "cluster_8",
    "220001": "cluster_9",
    "100001": "cluster_10",
    "200201": "cluster_11",
}
MOTION_CODE_WEIGHTS = [3.056, 1.539, 1.35, 2.844, 1.039, 2.632]

LIBERO_SKILL_TASKS = [
    (
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "put both the alphabet soup and the tomato sauce in the basket",
    ),
    (
        "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_on_the_plate",
        "open the top drawer of the cabinet and put the bowl on the plate",
    ),
    (
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_the_bottom_drawer_of_the_cabinet",
        "put the black bowl in the bottom drawer of the cabinet and close the bottom drawer of the cabinet",
    ),
    (
        "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_the_plate",
        "close the top drawer of the cabinet and put the black bowl on the plate",
    ),
    (
        "KITCHEN_SCENE11_close_the_top_drawer_of_the_cabinet_and_close_the_microwave",
        "close the top drawer of the cabinet and close the microwave",
    ),
    (
        "KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl_and_open_the_top_drawer_of_the_cabinet",
        "stack the middle black bowl on the back black bowl and open the top drawer of the cabinet",
    ),
    (
        "KITCHEN_SCENE12_put_the_black_bowl_on_the_plate_and_close_the_microwave",
        "put the black bowl on the plate and close the microwave",
    ),
    (
        "KITCHEN_SCENE15_close_the_drawer_of_the_cabinet_and_turn_off_the_stove",
        "close the drawer of the cabinet and turn off the stove",
    ),
    (
        "KITCHEN_SCENE13_put_the_black_bowl_on_the_plate_and_open_the_microwave",
        "put the black bowl on the plate and open the microwave",
    ),
]

ROBOTWIN_PRETRAIN_TASKS = [
    "adjust_bottle",
    "beat_block_hammer",
    "click_alarmclock",
    "click_bell",
    "grab_roller",
    "handover_block",
    "lift_pot",
    "move_can_pot",
    "move_playingcard_away",
    "open_microwave",
    "place_burger_fries",
    "place_object_basket",
    "rotate_qrcode",
    "shake_bottle_horizontally",
    "stack_blocks_two",
]

ROBOTWIN_TRANSFER_TASKS = [
    "blocks_ranking_size",
    "hanging_mug",
    "move_pillbottle_pad",
    "open_laptop",
    "place_a2b_left",
    "place_bread_basket",
    "place_bread_skillet",
    "place_cans_plasticbox",
    "place_fan",
    "press_stapler",
    "scan_object",
    "shake_bottle",
    "stack_blocks_three",
    "stack_bowls_two",
    "stamp_seal",
]

CONFIG_EXPECTATIONS = {
    "pi05_libero_moe_skill_4_40": [
        "repo_id=LIBERO40_REPO_ID",
        'action_expert_variant="gemma_300m_moe_4"',
        "batch_size=128",
        "peak_lr=5.0e-5",
        "decay_lr=5.0e-6",
        "ema_decay=0.999",
        "num_train_steps=30_000",
    ],
    "pi05_libero_moe_skill_4_90": [
        "repo_id=LIBERO90_REPO_ID",
        'action_expert_variant="gemma_300m_moe_4"',
        "batch_size=32",
        "peak_lr=2.5e-5",
        "decay_lr=2.5e-6",
        "ema_decay=0.99",
        "num_train_steps=20_000",
    ],
    "pi05_robotwin_moe_skill_pretrain": [
        "repo_id=ROBOTWIN_PRETRAIN_REPO_ID",
        'action_expert_variant="gemma_300m_moe_4"',
        "skill_num=14",
        "skill_embed_dim=64",
        "batch_size=32",
        "peak_lr=2.5e-5",
        "num_train_steps=20_000",
    ],
    "pi05_robotwin_moe_skill_transfer": [
        "repo_id=ROBOTWIN_TRANSFER_REPO_ID",
        "CheckpointWeightLoader_MoE(ROBOTWIN_TRANSFER_INIT_PARAMS)",
        'action_expert_variant="gemma_300m_moe_4"',
        "skill_num=14",
        "skill_embed_dim=64",
        "batch_size=32",
        "peak_lr=2.5e-5",
        "num_train_steps=1_000",
    ],
}

SCRIPT_EXPECTATIONS = [
    (
        "skill_moe/skillnet/scripts/serve_policy_moe_skill.py",
        [
            "EnvMode.LIBERO90",
            'config="pi05_libero_moe_skill_4_40"',
            'config="pi05_libero_moe_skill_4_90"',
            "docs/training_and_evaluation.md",
        ],
    ),
    (
        "skill_moe/skillnet/scripts/run_train_libero40_moe_skill.sh",
        [
            'CONFIG_NAME="${CONFIG_NAME:-pi05_libero_moe_skill_4_40}"',
            'SKILLNET_RELEASE_HF_NAMESPACE="${SKILLNET_RELEASE_HF_NAMESPACE:-jsw19}"',
            'LIBERO_REPO_ID="${SKILLNET_LIBERO40_REPO_ID:-${SKILLNET_RELEASE_HF_NAMESPACE}/libero_40_v1}"',
            'NORM_STATS_PATH="${NORM_STATS_PATH:-assets/${CONFIG_NAME}/${LIBERO_REPO_ID}/norm_stats.json}"',
            'elif [[ -n "${VIRTUAL_ENV:-}" ]]',
        ],
    ),
    (
        "skill_moe/skillnet/scripts/run_train_libero90_moe_skill.sh",
        [
            'CONFIG_NAME="${CONFIG_NAME:-pi05_libero_moe_skill_4_90}"',
            'SKILLNET_RELEASE_HF_NAMESPACE="${SKILLNET_RELEASE_HF_NAMESPACE:-jsw19}"',
            'LIBERO_REPO_ID="${SKILLNET_LIBERO90_REPO_ID:-${SKILLNET_RELEASE_HF_NAMESPACE}/libero_90_v1}"',
            'NORM_STATS_PATH="${NORM_STATS_PATH:-assets/${CONFIG_NAME}/${LIBERO_REPO_ID}/norm_stats.json}"',
            'elif [[ -n "${VIRTUAL_ENV:-}" ]]',
        ],
    ),
    (
        "skill_moe/skillnet/install_libero.sh",
        [
            'if python -c "import libero"',
            "benchmark.get_benchmark_dict()",
            "libero_skill_obj",
            "install_libero_skill_assets.py --install",
            "SKILLNET_SKIP_LIBERO_SKILL_ASSET_INSTALL",
            "SKILLNET_REQUIRE_LIBERO",
            'ACTIVATE_SCRIPT="${VENV_PATH}/bin/activate"',
            '${VENV_PATH}/Scripts/activate',
        ],
    ),
    (
        "skill_moe/skillnet/scripts/run_train_robotwin_pretrain_moe_skill.sh",
        [
            'CONFIG_NAME="${CONFIG_NAME:-pi05_robotwin_moe_skill_pretrain}"',
            'SKILLNET_RELEASE_HF_NAMESPACE="${SKILLNET_RELEASE_HF_NAMESPACE:-jsw19}"',
            'ROBOTWIN_REPO_ID="${SKILLNET_ROBOTWIN_PRETRAIN_REPO_ID:-${SKILLNET_RELEASE_HF_NAMESPACE}/robotwin_pretrain_v1}"',
            'elif [[ -n "${VIRTUAL_ENV:-}" ]]',
        ],
    ),
    (
        "skill_moe/skillnet/scripts/run_train_robotwin_transfer_moe_skill.sh",
        [
            'CONFIG_NAME="${CONFIG_NAME:-pi05_robotwin_moe_skill_transfer}"',
            'SKILLNET_RELEASE_HF_NAMESPACE="${SKILLNET_RELEASE_HF_NAMESPACE:-jsw19}"',
            'export SKILLNET_ROBOTWIN_TRANSFER_REPO_ID="${SKILLNET_RELEASE_HF_NAMESPACE}/robotwin_${TRANSFER_TASK}_v1"',
            "Error: SKILLNET_ROBOTWIN_TRANSFER_INIT_PARAMS is unset.",
            "ALLOW_PI05_TRANSFER_INIT",
        ],
    ),
    (
        "skill_moe/skillnet/examples/libero/run_eval_libero_skill_moe.sh",
        [
            'CONFIG_NAME="${CONFIG_NAME:-pi05_libero_moe_skill_4_90}"',
            'TASK_SUITE="${TASK_SUITE:-libero_skill_obj}"',
            'SKILL_ANNOTATION_PATH="${SKILL_ANNOTATION_PATH:-examples/libero/annotations/libero_skill_obj_annotations.json}"',
            'VIDEO_OUT_PATH="${VIDEO_OUT_PATH:-data/libero/videos/skillnet_libero_skill_obj}"',
            'SERVER_LOG_PATH="${SERVER_LOG_PATH:-data/libero/server_logs/${CONFIG_NAME}_${PORT}.log}"',
            'FAIL_FAST="${FAIL_FAST:-1}"',
            'ZERO_SHOT="${ZERO_SHOT:-1}"',
            "--no-zero-shot",
            "CKPT_DIR does not exist",
            "Policy server exited before becoming ready",
        ],
    ),
    (
        "skill_moe/skillnet/examples/robotwin/run_eval_robotwin_moe_skill.sh",
        [
            'CONFIG_NAME="${CONFIG_NAME:-pi05_robotwin_moe_skill_transfer}"',
            'TASK_SET="${TASK_SET:-transfer}"',
            'ACTION_HORIZON="${ACTION_HORIZON:-10}"',
            'SKILL_PLAN="${SKILL_PLAN:-${SKILLNET_REPO_ROOT}/data_process/robotwin/robotwin_plan.json}"',
            'SKILLNET_RELEASE_HF_NAMESPACE="${SKILLNET_RELEASE_HF_NAMESPACE:-jsw19}"',
            'export SKILLNET_ROBOTWIN_TRANSFER_REPO_ID="${SKILLNET_RELEASE_HF_NAMESPACE}/robotwin_${TRANSFER_TASK}_v1"',
            'SERVER_LOG_PATH="${SERVER_LOG_PATH:-data/robotwin/server_logs/${CONFIG_NAME}_${PORT}.log}"',
            "ROBOTWIN_ROOT does not exist",
            "SKILL_PLAN does not exist",
            "CKPT_DIR does not exist",
            "Policy server exited before becoming ready",
        ],
    ),
    (
        "data_process/libero/convert_libero_to_lerobot.py",
        [
            'RELEASE_HF_NAMESPACE = os.environ.get("SKILLNET_RELEASE_HF_NAMESPACE", "jsw19")',
            'os.environ.get("SKILLNET_LIBERO40_REPO_ID", release_repo_id("libero_40_v1"))',
            'os.environ.get("SKILLNET_LIBERO90_REPO_ID", release_repo_id("libero_90_v1"))',
        ],
    ),
    (
        "data_process/robotwin/convert_robotwin_to_lerobot.py",
        [
            'RELEASE_HF_NAMESPACE = os.environ.get("SKILLNET_RELEASE_HF_NAMESPACE", "jsw19")',
            'return release_repo_id(f"robotwin_{tasks[0]}_v1")',
            '"pretrain": release_repo_id("robotwin_pretrain_v1")',
        ],
    ),
    (
        "scripts/publish_lerobot_dataset.py",
        [
            "upload_large_folder",
            "--allow-existing-visibility",
            "--allow-missing-card",
            "--skip-hub-preflight",
            "--strict-parquet",
            "HUGGINGFACE_HUB_TOKEN",
        ],
    ),
    (
        "scripts/set_hf_dataset_visibility.py",
        [
            "api.update_repo_settings",
            "--skip-anonymous-check",
            "HUGGINGFACE_HUB_TOKEN",
            "Anonymous read check failed",
            "do not put tokens in command lines or docs",
        ],
    ),
]

MOE_EXPECTATIONS = [
    (
        "skill_moe/skillnet/src/openpi/models/gemma_moe_skill.py",
        [
            'if variant == "gemma_300m_moe_4":',
            "expert_num=4",
            "top_k: int = 1",
            "router_loss_scale: float = 0.01",
        ],
    ),
    (
        "skill_moe/skillnet/src/openpi/models/pi0_config_moe_skill.py",
        [
            'action_expert_variant: _gemma.Variant = "gemma_300m_moe_4"',
            "skill_num: int = 6",
            "skill_embed_dim: int = 64",
        ],
    ),
]

CONFIG_FORBIDDEN_SNIPPETS = [
    (
        "skill_moe/skillnet/src/openpi/training/config_moe_skill.py",
        [
            "LeRobotAlohaDataConfig",
            "LeRobotRobocasaSkillDataConfig",
            "RLDSDroidDataConfig",
            "LeRobotDROIDDataConfig",
            "openpi.policies.aloha_policy",
            "openpi.policies.droid_policy",
            "openpi.policies.robocasa_policy",
            "droid_rlds_dataset.DroidActionSpace",
        ],
    ),
]

WORKFLOW_EXPECTATIONS = [
    (
        ".github/workflows/release-check.yml",
        [
            "branches:",
            "skillnet-public-release",
            "fetch-depth: 0",
            "python-version: \"3.10\"",
            "python -m pip install --upgrade pip uv",
            "python scripts/check_public_release.py --skip-help --verbose",
            "python scripts/check_public_release.py --history-smoke --skip-help --verbose",
            "python scripts/check_public_release.py --require-bash --skip-help --verbose",
            "python scripts/check_public_release.py --hub-smoke --skip-help --verbose --hub-retries 3 --hub-timeout 30",
            "--install-smoke",
            "--install-python",
            "$(python -c 'import sys; print(sys.executable)')",
            "--skip-help",
            "--verbose",
        ],
    ),
]

DOC_EXPECTATIONS = [
    (
        "README.md",
        [
            "## 1. Quick Start",
            "actions/workflows/release-check.yml/badge.svg?branch=skillnet-public-release",
            "## 2. Skill Hierarchy",
            "## 3. In-Domain Training and Evaluation",
            "## 4. LIBERO-90 Training for LIBERO-Skill Zero-Shot Evaluation",
            "does not use",
            "LIBERO-Skill task trajectories for training",
            "## 5. Few-Shot Transfer",
            "docs/release_status.md",
            "docs/github_repository_setup.md",
            "## Citation",
            "CITATION.cff",
            "GitHub Actions release gate",
            ".github/workflows/release-check.yml",
            "git -c core.longpaths=true clone --branch skillnet-public-release --depth 1 https://github.com/VIPL-VSU/SkillNet.git SkillNet",
            "git config core.longpaths true",
            "RoboTwin checkpoint",
            "checkpoint weights are not part of",
            "included configs",
            "Direct LIBERO",
            "training requires either public access",
            "SKILLNET_RELEASE_HF_NAMESPACE",
            "--include-libero-derived-datasets",
            "Generate one per-task dataset locally",
            "public_task_manifest.json",
            "machine-readable citation metadata",
            "CONTRIBUTING.md",
            "SUPPORT.md",
            "SECURITY.md",
        ],
    ),
    (
        "CITATION.cff",
        [
            "cff-version: 1.2.0",
            "SkillNet: Hierarchical Skill Modeling for Compositional Generalization in Vision-Language Action Models",
            "family-names: Xie",
            "family-names: Zhang",
            "family-names: Tan",
            "family-names: Wang",
            "family-names: Chen",
            "Proceedings of the 43rd International Conference on Machine Learning",
            "PMLR",
            "year: 2026",
            "https://github.com/VIPL-VSU/SkillNet",
        ],
    ),
    (
        "CONTRIBUTING.md",
        [
            "Contributing",
            "docs/release_status.md",
            "Do not add API keys",
            "machine-local absolute paths",
            "python scripts/check_public_release.py --verbose",
            "--history-smoke",
            "--hub-smoke",
            "--require-bash",
            "New public files are included in `scripts/check_public_release.py`",
        ],
    ),
    (
        "SUPPORT.md",
        [
            "Support",
            "docs/quick_start.md",
            "docs/release_status.md",
            "docs/training_and_evaluation.md",
            "GitHub issue",
            "Do not include tokens",
            "private dataset paths",
            "check_public_release.py --verbose",
        ],
    ),
    (
        "SECURITY.md",
        [
            "Security Policy",
            "Reporting a Vulnerability",
            "GitHub private vulnerability reporting",
            "Do not include exploit details",
            "API keys",
            "Private dataset locations",
            "Machine-local absolute paths",
            "--history-smoke",
        ],
    ),
    (
        "skill_moe/README.md",
        [
            "SkillNet runtime root",
            "public LIBERO, LIBERO-Skill, RoboTwin",
            "Skill hierarchy tokenization",
            "LIBERO in-domain training",
            "RoboTwin few-shot training",
            "LIBERO-Skill OOD eval",
        ],
    ),
    (
        "docs/quick_start.md",
        [
            "git -c core.longpaths=true clone --branch skillnet-public-release --depth 1 https://github.com/VIPL-VSU/SkillNet.git SkillNet",
            "git config core.longpaths true",
            "Command blocks in this guide use Bash syntax",
            "On Windows, use WSL",
            ".venv/Scripts/activate",
            "Optional Google Cloud SDK",
            "Download released SkillNet checkpoints from the runnable SkillNet source root",
            "SKILLNET_RELEASE_HF_NAMESPACE",
            "SKILLNET_LIBERO40_REPO_ID",
            "no-deps install smoke",
            "python scripts/check_public_release.py --hub-smoke",
            "--install-python /path/to/python3.10",
            "gcloud storage cp -r gs://openpi-assets/checkpoints/pi05_base/params",
            "A plain `--hub-smoke` pass means",
            "intentionally does not use",
            "--hub-authenticated",
            "docs/release_status.md",
            "https://github.com/Lifelong-Robot-Learning/LIBERO",
            "https://github.com/RoboTwin-Platform/RoboTwin",
            'export LIBERO40_RLDS_DIR="${LIBERO40_RLDS_DIR:-$SKILLNET_LIBERO_DATA_ROOT/libero40_rlds}"',
            "--expected-episodes 3862",
            "--expected-episodes 7874",
            "SKILLNET_REQUIRE_LIBERO=1",
            "install_libero_skill_assets.py --install",
            "SKILLNET_SKIP_LIBERO_SKILL_ASSET_INSTALL",
            "LIBERO-Skill evaluation",
            "RoboTwin few-shot evaluation",
            "ALLOW_PI05_TRANSFER_INIT=1",
            "The `jsw19/robotwin_*_v1` names above are local LeRobot `repo_id` values",
            "slice_indices/libero40_slice_index.json",
            "Set `TRANSFER_TASK` for the paper-style per-task dataset",
            "--motion-code 200100",
        ],
    ),
    (
        "docs/release_status.md",
        [
            "## Ready",
            "Fresh clone gate",
            "GitHub Actions release gate",
            ".github/workflows/release-check.yml",
            "short-path fresh clone",
            "git config core.longpaths true",
            "Record the exact git SHA",
            "self-referential documentation",
            "## Pending External Assets",
            "jsw19/libero_40_v1",
            "jsw19/libero_90_v1",
            "RoboTwin derived dataset ids",
            "Final organization Hub namespace",
            "Current published assets remain under `jsw19/*`",
            "GitHub repository metadata",
            "docs/github_repository_setup.md",
            "scripts/check_github_repository_metadata.py",
            "SKILLNET_RELEASE_HF_NAMESPACE",
            "--include-libero-derived-datasets",
            "--expected-episodes 3862",
            "--expected-episodes 7874",
            "HF_TOKEN",
            "HF_XET_HIGH_PERFORMANCE",
            "HfApi.upload_large_folder",
            "data_process/libero/dataset_cards/",
            "--strict-parquet",
            "--require-bash",
            "no-deps editable metadata check",
            "--install-python /path/to/python3.10",
            "scripts/set_hf_dataset_visibility.py",
            "--dry-run",
            "unset HF_TOKEN",
        ],
    ),
    (
        "docs/github_repository_setup.md",
        [
            "GitHub Repository Setup",
            "Default branch: `skillnet-public-release`",
            "SkillNet: skill-hierarchy-conditioned MoE policies for LIBERO, LIBERO-Skill, and RoboTwin few-shot transfer.",
            "robot-learning",
            "mixture-of-experts",
            "python scripts/check_github_repository_metadata.py --verbose",
            "public GitHub API",
            "not part of the default CI gate",
            "history privacy",
            "clean",
            "release procedure",
        ],
    ),
    (
        "docs/skill_hierarchy.md",
        [
            "## Motion Code",
            "## Tokenization Strategy",
            "motion_code",
            "Allowed values",
            "Manual annotation protocol",
            "motion_code_clusters.json",
            "weighted distance",
            "flat integer skill-id",
            "hierarchy-token experiments",
            TOKENIZATION_STRATEGY_SHA256,
            "tokenization_strategy.json",
            "frozen vocabulary",
            "verb_map",
            "raises an error",
        ],
    ),
    (
        "data_process/skill_hierarchy/README.md",
        [
            "Skill Hierarchy Tokenization",
            "flat integer `skills`",
            "model input contract",
            "skill_graph_example.json",
            "motion_code_clusters.json",
            "motion_code_annotation_examples.jsonl",
            "OpenAI-compatible endpoint",
            TOKENIZATION_STRATEGY_SHA256,
            "example graph is intentionally small",
            "frozen vocabulary",
            "verb_map",
            "raises an error",
        ],
    ),
    (
        "docs/libero_data_processing.md",
        [
            "Release contract:",
            "RLDS source datasets alone are not enough",
            "scripts/verify_lerobot_dataset.py",
            "scripts/publish_lerobot_dataset.py",
            "export_libero_skill_slices.py",
            "--slice-index",
            "slice_indices/libero40_slice_index.json",
            "libero40_plan_sliced.json",
            "libero90_plan_sliced.json",
            "--include-libero-derived-datasets",
            "HF_XET_HIGH_PERFORMANCE",
            "upload_large_folder",
            "--public",
            "--private",
            "--strict-parquet",
            "README_libero_40_v1.md",
            "scripts/set_hf_dataset_visibility.py",
            "SKILLNET_RELEASE_HF_NAMESPACE",
            "SKILLNET_LIBERO40_REPO_ID",
        ],
    ),
    (
        "docs/training_and_evaluation.md",
        [
            "git -c core.longpaths=true clone --branch skillnet-public-release --depth 1 https://github.com/VIPL-VSU/SkillNet.git SkillNet",
            "git -C ../.. config core.longpaths true",
            "## LIBERO-40 Training",
            "## LIBERO-90 Training",
            "training stage used before LIBERO-Skill zero-shot evaluation",
            "LIBERO-Skill benchmark task trajectories are used",
            "## LIBERO-40 Evaluation",
            "## LIBERO-Skill Evaluation",
            "TASK_SUITE=libero_skill_obj",
            "public_task_manifest.json",
            "tasks_info.txt",
            "asset inventory",
            "ZERO_SHOT=0",
            "base rollout limit is 800 control steps",
            "10 initial stabilization steps",
            "install_libero_skill_assets.py --install",
            "--include-libero-derived-datasets",
            "FAIL_FAST=0",
            "data/libero/server_logs/",
            "benchmark.get_benchmark_dict()",
            "https://github.com/Lifelong-Robot-Learning/LIBERO",
            "RLDS source frames alone do not contain",
            "SKILLNET_RELEASE_HF_NAMESPACE",
            "SKILLNET_LIBERO40_REPO_ID",
            "class` and `all_classes",
            "skills` and `skill_mask",
            "--expected-episodes 3862",
            "--expected-episodes 7874",
            "skillnet_libero_skill_obj",
            "Expected outputs:",
        ],
    ),
    (
        "docs/robotwin_few_shot.md",
        [
            "## Data Preparation",
            "## Training",
            "## Evaluation",
            "RoboTwin-2.0",
            "https://github.com/RoboTwin-Platform/RoboTwin",
            "https://robotwin-platform.github.io/doc/index.html",
            "SKILLNET_ROBOTWIN_TRANSFER_INIT_PARAMS",
            "SKILLNET_RELEASE_HF_NAMESPACE",
            "Fine-tune all 15 transfer tasks",
            "create all matching per-task LeRobot datasets",
            'jsw19/robotwin_${task}_v1',
            "local datasets",
            "`LEROBOT_HOME` or `HF_LEROBOT_HOME`",
            "aggregate transfer default",
            "RoboTwin checkpoint weights are not published",
            "ALLOW_PI05_TRANSFER_INIT",
            "success_rate",
            "data/robotwin/server_logs/",
            "waits for the server port",
            "JSON-encoded string",
            "RoboTwin-2.0-style checkout",
            "envs/__init__.py",
            "CONFIGS_PATH",
            "test_render.Sapien_TEST",
            "--skip-render-test",
            "--max-seed-attempts",
            "START_SERVER",
            "SERVER_WAIT_SECONDS",
            "left-arm-first",
            "## Release Contents",
        ],
    ),
    (
        "data_process/robotwin/README.md",
        [
            "RoboTwin Few-Shot Data",
            "local LeRobot `repo_id` names",
            "SKILLNET_RELEASE_HF_NAMESPACE",
            "JSON-encoded string",
            "ROBOTWIN_ROOT",
            "collect_data.sh",
            "envs.CONFIGS_PATH",
            "test_render.Sapien_TEST",
            "left-arm-first",
        ],
    ),
]

DOC_FORBIDDEN_SNIPPETS = [
    (
        "docs/libero_data_processing.md",
        ["generate_data_40.py", "generate_data_90.py", "libero_90_obj", "historical research"],
    ),
    (
        "data_process/libero/README.md",
        ["generate_data_40.py", "generate_data_90.py", "libero_90_obj", "historical research"],
    ),
    (
        "docs/robotwin_few_shot.md",
        ["Migration Status", "Completed in this step", "cluster-specific conda", "tmux orchestration"],
    ),
    (
        "skill_moe/skillnet/third_party/libero/scripts/README.md",
        ["internal task-construction process"],
    ),
    (
        "skill_moe/skillnet/third_party/libero/scripts/create_libero_skill.py",
        ["internal task-construction process"],
    ),
]

PUBLIC_DOC_FILES = [
    "CONTRIBUTING.md",
    "NOTICE.md",
    "README.md",
    "SECURITY.md",
    "SUPPORT.md",
    "skill_moe/README.md",
    "docs/quick_start.md",
    "docs/github_repository_setup.md",
    "docs/release_status.md",
    "docs/skill_hierarchy.md",
    "docs/training_and_evaluation.md",
    "docs/libero_data_processing.md",
    "docs/robotwin_few_shot.md",
    "data_process/skill_hierarchy/README.md",
    "data_process/libero/README.md",
    "data_process/robotwin/README.md",
]

PUBLIC_DOC_FORBIDDEN_PATTERNS = [
    (re.compile(r"\bbackend\b", re.IGNORECASE), "backend"),
    (re.compile(r"\boverlay\b", re.IGNORECASE), "overlay"),
]

RELEASE_STATUS_FORBIDDEN_PATTERNS = [
    (re.compile(r"commit\s+`?[0-9a-f]{7,40}`?", re.IGNORECASE), "pinned commit hash"),
    (re.compile(r"\b\d+\s+commit(?:\(s\)|s)?\s+scanned\b", re.IGNORECASE), "pinned history scan count"),
]

OPENPI_PUBLIC_DOC_ALLOWED_FRAGMENTS = [
    "packages/openpi-client",
    "openpi-client",
    "gs://openpi-assets",
    "gcloud storage cp -r gs://openpi-assets",
    "import openpi.training.config_moe_skill",
    "pi0.5/openpi components",
    "openpi is licensed",
    "skillnet/src/openpi",
    "src/openpi/",
]

NON_RELEASE_PATH_PATTERNS = [
    re.compile(r"^skill_moe/skillnet/examples/robocasa/"),
    re.compile(r"^skill_moe/skillnet/src/openpi/training/misc/roboarena_config\.py$"),
    re.compile(r"^skill_moe/skillnet/src/openpi/training/droid_rlds_dataset\.py$"),
    re.compile(r"^skill_moe/skillnet/src/openpi/policies/droid_policy\.py$"),
    re.compile(r"^skill_moe/skillnet/src/openpi/policies/robocasa_policy\.py$"),
    re.compile(r"^skill_moe/skillnet/packages/openpi-client/src/openpi_client/websocket_client_policy_robocasa\.py$"),
]

PUBLIC_HUB_RESOURCES = [
    ("model", "jsw19/SkillNet-LIBERO-40"),
    ("model", "jsw19/SkillNet-LIBERO-90"),
    ("dataset", "openvla/modified_libero_rlds"),
    ("dataset", "jesbu1/libero_90_openvla_processed"),
    ("dataset", "TianxingChen/RoboTwin2.0"),
]

LIBERO_DERIVED_LEROBOT_HUB_RESOURCES = [
    ("dataset", "jsw19/libero_40_v1"),
    ("dataset", "jsw19/libero_90_v1"),
]

ROBOTWIN_DERIVED_LEROBOT_HUB_RESOURCES = [
    ("dataset", "jsw19/robotwin_pretrain_v1"),
    ("dataset", "jsw19/robotwin_transfer_v1"),
    ("dataset", "jsw19/robotwin_blocks_ranking_size_v1"),
    ("dataset", "jsw19/robotwin_paper_v1"),
    ("dataset", "jsw19/robotwin_all_v1"),
]

# Common derived LeRobot repo ids used in the public docs and default scripts.
DERIVED_LEROBOT_HUB_RESOURCES = LIBERO_DERIVED_LEROBOT_HUB_RESOURCES + ROBOTWIN_DERIVED_LEROBOT_HUB_RESOURCES


def tracked_files(suffix: str, fallback: list[str]) -> list[str]:
    """Return tracked files for a suffix when running from a git checkout."""
    result = subprocess.run(
        ["git", "ls-files", f"*{suffix}"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return fallback
    files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return files or fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-help", action="store_true", help="Skip CLI --help subprocess checks.")
    parser.add_argument(
        "--install-smoke",
        action="store_true",
        help="Create a temporary venv and run a --no-deps editable package metadata smoke.",
    )
    parser.add_argument("--install-python", default="3.10", help="Python version or executable for --install-smoke.")
    parser.add_argument(
        "--hub-smoke",
        action="store_true",
        help="Check that public Hugging Face model/source-dataset resources are reachable.",
    )
    parser.add_argument(
        "--history-smoke",
        action="store_true",
        help="Scan commits reachable from HEAD for private paths, tokens, and release-blocking names.",
    )
    parser.add_argument(
        "--require-bash",
        action="store_true",
        help="Fail when bash is unavailable for shell-script syntax checks.",
    )
    parser.add_argument(
        "--include-derived-datasets",
        action="store_true",
        help="With --hub-smoke, also check common derived LeRobot dataset repo ids used as local output names.",
    )
    parser.add_argument(
        "--include-libero-derived-datasets",
        action="store_true",
        help="With --hub-smoke, also check derived LIBERO LeRobot dataset repo ids.",
    )
    parser.add_argument(
        "--include-robotwin-derived-datasets",
        action="store_true",
        help="With --hub-smoke, also check common derived RoboTwin LeRobot dataset repo ids.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        help="Hugging Face endpoint for --hub-smoke. Defaults to HF_ENDPOINT or https://huggingface.co.",
    )
    parser.add_argument("--hub-timeout", type=float, default=20.0, help="Per-request timeout for --hub-smoke.")
    parser.add_argument("--hub-retries", type=int, default=3, help="Retry count for transient --hub-smoke failures.")
    parser.add_argument(
        "--hub-authenticated",
        action="store_true",
        help="Use a Hub token for --hub-smoke. Public release checks are anonymous by default.",
    )
    parser.add_argument("--hub-token-env", default="HF_TOKEN", help="Token environment variable for --hub-authenticated.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)
    print(f"[FAIL] {message}")


def ok(message: str, *, verbose: bool = True) -> None:
    if verbose:
        print(f"[ OK ] {message}")


def skip(message: str, *, verbose: bool = True) -> None:
    if verbose:
        print(f"[SKIP] {message}")


def check_required_files(errors: list[str], *, verbose: bool) -> None:
    for rel_path in REQUIRED_FILES:
        path = REPO_ROOT / rel_path
        if not path.exists():
            fail(f"missing required file: {rel_path}", errors)
        else:
            ok(f"found {rel_path}", verbose=verbose)


def check_json_files(errors: list[str], *, verbose: bool) -> None:
    json_paths = [
        "data_process/skill_hierarchy/tokenization_strategy.json",
        "data_process/skill_hierarchy/skill_graph_example.json",
        "data_process/skill_hierarchy/motion_code_clusters.json",
        "data_process/libero/instruct2plan_40.json",
        "data_process/libero/instruct2plan_90.json",
        "data_process/libero/instruct2plan_obj_90.json",
        "data_process/libero/slice_indices/libero40_slice_index.json",
        "data_process/libero/slice_indices/libero90_slice_index.json",
        "skill_moe/skillnet/examples/libero/annotations/instruct2plan_40.json",
        "skill_moe/skillnet/examples/libero/annotations/instruct2plan_obj_90.json",
        "skill_moe/skillnet/examples/libero/annotations/libero_skill_obj_annotations.json",
        "skill_moe/skillnet/third_party/libero/libero/libero/bddl_files/libero_skill_obj/public_task_manifest.json",
        "data_process/robotwin/robotwin_plan.json",
        "data_process/robotwin/skill_anno_robotwin.json",
    ]
    for rel_path in json_paths:
        path = REPO_ROOT / rel_path
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            fail(f"invalid JSON in {rel_path}: {exc}", errors)
        else:
            ok(f"valid JSON {rel_path}", verbose=verbose)


def check_jsonl_files(errors: list[str], *, verbose: bool) -> None:
    rel_path = "data_process/skill_hierarchy/motion_code_annotation_examples.jsonl"
    path = REPO_ROOT / rel_path
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception as exc:
        fail(f"invalid JSONL in {rel_path}: {exc}", errors)
        return
    if not rows:
        fail(f"empty JSONL file: {rel_path}", errors)
    else:
        ok(f"valid JSONL {rel_path} ({len(rows)} rows)", verbose=verbose)


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def load_json(rel_path: str) -> object:
    return json.loads((REPO_ROOT / rel_path).read_text(encoding="utf-8"))


def load_python_literal_assignment(rel_path: str, variable_name: str) -> object:
    source = (REPO_ROOT / rel_path).read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=rel_path)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == variable_name:
                return ast.literal_eval(node.value)
    raise ValueError(f"missing literal assignment {variable_name!r} in {rel_path}")


def parse_shell_array(rel_path: str, array_name: str) -> list[str]:
    text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    match = re.search(rf"{re.escape(array_name)}=\((.*?)\)", text, flags=re.S)
    if match is None:
        raise ValueError(f"missing shell array {array_name!r} in {rel_path}")
    return re.findall(r'"([^"]+)"', match.group(1))


def extract_call_block(text: str, call_name: str, anchor: str) -> str:
    anchor_index = text.find(anchor)
    if anchor_index < 0:
        raise ValueError(f"missing anchor {anchor!r}")
    call_index = text.rfind(call_name, 0, anchor_index)
    if call_index < 0:
        raise ValueError(f"missing call {call_name!r} before {anchor!r}")
    open_index = text.find("(", call_index)
    if open_index < 0:
        raise ValueError(f"missing '(' after {call_name!r}")

    depth = 0
    quote = ""
    escape = False
    for index in range(open_index, len(text)):
        char = text[index]
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = ""
            continue
        if char in ("'", '"'):
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[call_index : index + 1]
    raise ValueError(f"unterminated {call_name!r} block for {anchor!r}")


def check_config_and_script_contracts(errors: list[str], *, verbose: bool) -> None:
    error_count = len(errors)
    config_path = "skill_moe/skillnet/src/openpi/training/config_moe_skill.py"
    config_text = (REPO_ROOT / config_path).read_text(encoding="utf-8")
    compact_config = compact_text(config_text)

    global_expectations = [
        'PI05_BASE_PARAMS=os.environ.get("SKILLNET_PI05_BASE_PARAMS","gs://openpi-assets/checkpoints/pi05_base/params")',
        'RELEASE_HF_NAMESPACE=os.environ.get("SKILLNET_RELEASE_HF_NAMESPACE","jsw19")',
        'LIBERO40_REPO_ID=os.environ.get("SKILLNET_LIBERO40_REPO_ID",release_repo_id("libero_40_v1"))',
        'LIBERO90_REPO_ID=os.environ.get("SKILLNET_LIBERO90_REPO_ID",release_repo_id("libero_90_v1"))',
        'ROBOTWIN_PRETRAIN_REPO_ID=os.environ.get("SKILLNET_ROBOTWIN_PRETRAIN_REPO_ID",release_repo_id("robotwin_pretrain_v1"))',
        'ROBOTWIN_TRANSFER_REPO_ID=os.environ.get("SKILLNET_ROBOTWIN_TRANSFER_REPO_ID",release_repo_id("robotwin_transfer_v1"))',
        'ROBOTWIN_TRANSFER_INIT_PARAMS=os.environ.get("SKILLNET_ROBOTWIN_TRANSFER_INIT_PARAMS",PI05_BASE_PARAMS)',
    ]
    for snippet in global_expectations:
        if compact_text(snippet) not in compact_config:
            fail(f"config global contract missing in {config_path}: {snippet}", errors)

    public_names_match = re.search(r"_PUBLIC_CONFIG_NAMES\s*=\s*\((.*?)\)", config_text, re.S)
    if not public_names_match:
        fail(f"{config_path}: missing _PUBLIC_CONFIG_NAMES release surface", errors)
    else:
        public_config_names = re.findall(r'"([^"]+)"', public_names_match.group(1))
        expected_config_names = list(CONFIG_EXPECTATIONS)
        if public_config_names != expected_config_names:
            fail(
                f"{config_path}: public config surface mismatch: "
                f"expected {expected_config_names}, found {public_config_names}",
                errors,
            )
    if compact_text("if tuple(config.name for config in _CONFIGS) != _PUBLIC_CONFIG_NAMES:") not in compact_config:
        fail(f"{config_path}: public _CONFIGS order must be checked against _PUBLIC_CONFIG_NAMES", errors)

    for config_name, snippets in CONFIG_EXPECTATIONS.items():
        try:
            block = extract_call_block(config_text, "TrainConfig", f'name="{config_name}"')
        except ValueError as exc:
            fail(f"{config_path}: {exc}", errors)
            continue
        compact_block = compact_text(block)
        for snippet in snippets:
            if compact_text(snippet) not in compact_block:
                fail(f"{config_path}: {config_name} missing expected snippet: {snippet}", errors)

    for rel_path, snippets in SCRIPT_EXPECTATIONS:
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        compact_script = compact_text(text)
        for snippet in snippets:
            if compact_text(snippet) not in compact_script:
                fail(f"{rel_path}: missing expected snippet: {snippet}", errors)

    for rel_path, snippets in MOE_EXPECTATIONS:
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        compact_model = compact_text(text)
        for snippet in snippets:
            if compact_text(snippet) not in compact_model:
                fail(f"{rel_path}: missing expected MoE snippet: {snippet}", errors)

    for rel_path, snippets in CONFIG_FORBIDDEN_SNIPPETS:
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        for snippet in snippets:
            if snippet in text:
                fail(f"{rel_path}: public config surface contains non-release snippet: {snippet}", errors)

    if len(errors) == error_count:
        ok("public config/script/model contracts match documented release settings", verbose=verbose)


def check_workflow_contracts(errors: list[str], *, verbose: bool) -> None:
    error_count = len(errors)
    for rel_path, snippets in WORKFLOW_EXPECTATIONS:
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        for snippet in snippets:
            if snippet not in text:
                fail(f"{rel_path}: missing expected workflow snippet: {snippet}", errors)
    if len(errors) == error_count:
        ok("GitHub Actions release workflow contract matches documented gates", verbose=verbose)


def check_documentation_contracts(errors: list[str], *, verbose: bool) -> None:
    error_count = len(errors)
    for rel_path, snippets in DOC_EXPECTATIONS:
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        for snippet in snippets:
            if snippet not in text:
                fail(f"{rel_path}: missing expected documentation snippet: {snippet}", errors)
    for rel_path, snippets in DOC_FORBIDDEN_SNIPPETS:
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        for snippet in snippets:
            if snippet in text:
                fail(f"{rel_path}: public documentation contains legacy/internal snippet: {snippet}", errors)

    release_status_text = (REPO_ROOT / "docs/release_status.md").read_text(encoding="utf-8")
    for pattern, label in RELEASE_STATUS_FORBIDDEN_PATTERNS:
        if pattern.search(release_status_text):
            fail(f"docs/release_status.md should not pin a {label}; record exact SHAs in tags or CI logs", errors)

    allowed_openpi_fragments = [fragment.lower() for fragment in OPENPI_PUBLIC_DOC_ALLOWED_FRAGMENTS]
    for rel_path in PUBLIC_DOC_FILES:
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for pattern, label in PUBLIC_DOC_FORBIDDEN_PATTERNS:
                if pattern.search(line):
                    fail(
                        f"{rel_path}:{line_number}: public docs should not use '{label}' terminology",
                        errors,
                    )
            lower_line = line.lower()
            if "openpi" in lower_line and not any(fragment in lower_line for fragment in allowed_openpi_fragments):
                fail(
                    f"{rel_path}:{line_number}: public docs should describe workflows as SkillNet, "
                    f"not openpi: {line.strip()}",
                    errors,
                )

    skill_moe_readme = (REPO_ROOT / "skill_moe/README.md").read_text(encoding="utf-8")
    for excluded in ("RoboCasa", "robocasa", "GR00T"):
        if excluded in skill_moe_readme:
            fail(f"skill_moe/README.md should not advertise non-release helper surface: {excluded}", errors)
    if len(errors) == error_count:
        ok("public documentation covers the five release workflows", verbose=verbose)


def check_release_surface_paths(errors: list[str], *, verbose: bool) -> None:
    result = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        fail(f"could not list tracked files: {result.stderr.strip()}", errors)
        return

    matches: list[str] = []
    for rel_path in result.stdout.splitlines():
        rel_path = rel_path.strip()
        if not rel_path:
            continue
        for pattern in NON_RELEASE_PATH_PATTERNS:
            if pattern.search(rel_path):
                matches.append(rel_path)
                break

    if matches:
        fail("non-release entrypoints are tracked:\n" + "\n".join(matches), errors)
    else:
        ok("tracked release surface excludes non-release entrypoints", verbose=verbose)


def check_skill_hierarchy_contract(errors: list[str], *, verbose: bool) -> None:
    error_count = len(errors)
    strategy_path = REPO_ROOT / "data_process/skill_hierarchy/tokenization_strategy.json"
    strategy_digest = hashlib.sha256(strategy_path.read_bytes()).hexdigest()
    if strategy_digest != TOKENIZATION_STRATEGY_SHA256:
        fail(
            "Skill hierarchy tokenization_strategy.json checksum changed: "
            f"expected {TOKENIZATION_STRATEGY_SHA256}, found {strategy_digest}",
            errors,
        )

    strategy = load_json("data_process/skill_hierarchy/tokenization_strategy.json")
    if not isinstance(strategy, dict):
        fail("Skill hierarchy tokenization strategy must be a JSON object", errors)
        return
    expected_sizes = {"category_map": 13, "verbnet_map": 127, "verb_map": 267}
    for key, expected_size in expected_sizes.items():
        mapping = strategy.get(key)
        if not isinstance(mapping, dict):
            fail(f"Skill hierarchy strategy missing object map: {key}", errors)
        elif len(mapping) != expected_size:
            fail(f"Skill hierarchy {key} expected {expected_size} entries, found {len(mapping)}", errors)

    clusters = load_json("data_process/skill_hierarchy/motion_code_clusters.json")
    if not isinstance(clusters, dict):
        fail("Skill hierarchy motion_code_clusters.json must be a JSON object", errors)
        return
    if clusters.get("weights") != MOTION_CODE_WEIGHTS:
        fail("Skill hierarchy motion-code weights do not match the released contract", errors)
    centers = clusters.get("centers")
    if not isinstance(centers, list):
        fail("Skill hierarchy motion-code centers must be a list", errors)
    else:
        center_map = {
            str(item.get("motion_code")): str(item.get("cluster"))
            for item in centers
            if isinstance(item, dict) and "motion_code" in item and "cluster" in item
        }
        if center_map != MOTION_CODE_CENTERS:
            fail("Skill hierarchy motion-code centers do not match the released contract", errors)

    graph = load_json("data_process/skill_hierarchy/skill_graph_example.json")
    if not isinstance(graph, dict) or not graph:
        fail("Skill hierarchy skill_graph_example.json must be a non-empty JSON object", errors)
    else:
        for cluster_name, verbnet_map in graph.items():
            if not isinstance(cluster_name, str) or not cluster_name.startswith("cluster_"):
                fail(f"Skill hierarchy example graph has invalid cluster key: {cluster_name!r}", errors)
                break
            if not isinstance(verbnet_map, dict) or not verbnet_map:
                fail(f"Skill hierarchy example graph cluster must map to a non-empty object: {cluster_name}", errors)
                break
            for verbnet_class, verb_map in verbnet_map.items():
                if not isinstance(verbnet_class, str) or not isinstance(verb_map, dict) or not verb_map:
                    fail(
                        "Skill hierarchy example graph VerbNet level must be a non-empty object: "
                        f"{cluster_name}/{verbnet_class!r}",
                        errors,
                    )
                    break
                for verb, examples in verb_map.items():
                    if not isinstance(verb, str) or not isinstance(examples, list) or not examples:
                        fail(
                            "Skill hierarchy example graph verb level must contain examples: "
                            f"{cluster_name}/{verbnet_class}/{verb!r}",
                            errors,
                        )
                        break
                    first_example = examples[0]
                    if (
                        not isinstance(first_example, list)
                        or len(first_example) != 2
                        or not isinstance(first_example[0], str)
                        or not isinstance(first_example[1], list)
                    ):
                        fail(
                            "Skill hierarchy example graph examples must be [text, metadata] pairs: "
                            f"{cluster_name}/{verbnet_class}/{verb}",
                            errors,
                        )
                        break

    examples_path = REPO_ROOT / "data_process/skill_hierarchy/motion_code_annotation_examples.jsonl"
    rows = [json.loads(line) for line in examples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for index, row in enumerate(rows, start=1):
        if not isinstance(row.get("motion_code"), str):
            fail(f"Skill hierarchy annotation example row {index} is missing motion_code", errors)
        if not (row.get("subtask") or row.get("phrase") or row.get("plan_step")):
            fail(f"Skill hierarchy annotation example row {index} is missing subtask/phrase", errors)

    if len(errors) == error_count:
        ok("Skill hierarchy checksum, motion-code centers, and examples match the release contract", verbose=verbose)


def check_libero_skill_contract(errors: list[str], *, verbose: bool) -> None:
    error_count = len(errors)
    annotations = load_json("skill_moe/skillnet/examples/libero/annotations/libero_skill_obj_annotations.json")
    if not isinstance(annotations, dict):
        fail("LIBERO-Skill annotations must be a JSON object", errors)
        return
    if len(annotations) != len(LIBERO_SKILL_TASKS):
        fail(f"LIBERO-Skill annotations expected {len(LIBERO_SKILL_TASKS)} tasks, found {len(annotations)}", errors)

    task_names = [task_name for task_name, _ in LIBERO_SKILL_TASKS]
    manifest = load_json(
        "skill_moe/skillnet/third_party/libero/libero/libero/bddl_files/libero_skill_obj/public_task_manifest.json"
    )
    if not isinstance(manifest, dict):
        fail("LIBERO-Skill public task manifest must be a JSON object", errors)
    else:
        if manifest.get("benchmark_key") != "libero_skill_obj":
            fail("LIBERO-Skill public task manifest has the wrong benchmark_key", errors)
        if manifest.get("task_count") != len(task_names):
            fail("LIBERO-Skill public task manifest has the wrong task_count", errors)
        if manifest.get("tasks") != task_names:
            fail("LIBERO-Skill public task manifest does not match the release task order", errors)

    suite_map = load_python_literal_assignment(
        "skill_moe/skillnet/third_party/libero/libero/libero/benchmark/libero_suite_task_map.py",
        "libero_task_map",
    )
    registered_tasks = suite_map.get("libero_skill_obj") if isinstance(suite_map, dict) else None
    if registered_tasks != task_names:
        fail("libero_skill_obj registration does not match the public 9-task manifest", errors)

    bddl_dir = REPO_ROOT / "skill_moe/skillnet/third_party/libero/libero/libero/bddl_files/libero_skill_obj"
    init_dir = REPO_ROOT / "skill_moe/skillnet/third_party/libero/libero/libero/init_files/libero_skill_obj"
    bddl_names = sorted(path.stem for path in bddl_dir.glob("*.bddl"))
    tasks_info_path = bddl_dir / "tasks_info.txt"
    tasks_info_names = [Path(line.strip()).stem for line in tasks_info_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(tasks_info_names) != len(set(tasks_info_names)):
        fail(f"{tasks_info_path.relative_to(REPO_ROOT).as_posix()} contains duplicate task entries", errors)
    if sorted(tasks_info_names) != bddl_names:
        fail("LIBERO-Skill tasks_info.txt does not match the bundled bddl files", errors)
    readme_text = (bddl_dir / "README.md").read_text(encoding="utf-8")
    for snippet in ["public_task_manifest.json", "tasks_info.txt", "asset inventory", "not be used"]:
        if snippet not in readme_text:
            fail("LIBERO-Skill bddl README does not explain the public manifest versus auxiliary assets", errors)

    missing_init_for_bddl = sorted(set(bddl_names) - {path.stem for path in init_dir.glob("*.pruned_init")})
    if missing_init_for_bddl:
        fail("LIBERO-Skill bddl files missing pruned init files: " + ", ".join(missing_init_for_bddl), errors)

    for bddl_name, annotation_key in LIBERO_SKILL_TASKS:
        if not (bddl_dir / f"{bddl_name}.bddl").exists():
            fail(f"missing LIBERO-Skill bddl file: {bddl_name}.bddl", errors)
        if not (init_dir / f"{bddl_name}.pruned_init").exists():
            fail(f"missing LIBERO-Skill init file: {bddl_name}.pruned_init", errors)
        entry = annotations.get(annotation_key)
        if not isinstance(entry, dict):
            fail(f"missing LIBERO-Skill annotation: {annotation_key}", errors)
            continue
        plan = entry.get("plan")
        classes = entry.get("all_classes")
        objects = entry.get("objects")
        if not isinstance(plan, list) or not plan:
            fail(f"LIBERO-Skill annotation has empty plan: {annotation_key}", errors)
        if not isinstance(classes, list) or len(classes) != len(plan):
            fail(f"LIBERO-Skill all_classes length mismatch: {annotation_key}", errors)
        if not isinstance(objects, list) or len(objects) != len(plan):
            fail(f"LIBERO-Skill objects length mismatch: {annotation_key}", errors)
        if not all(isinstance(item, int) for item in classes or []):
            fail(f"LIBERO-Skill all_classes must be integers: {annotation_key}", errors)
        if not all(isinstance(item, int) for item in objects or []):
            fail(f"LIBERO-Skill objects must be integers: {annotation_key}", errors)

    if len(errors) == error_count:
        ok("LIBERO-Skill 9-task annotation/bddl/init contract is present", verbose=verbose)


def check_libero_annotation_contract(errors: list[str], *, verbose: bool) -> None:
    error_count = len(errors)
    libero40 = load_json("data_process/libero/instruct2plan_40.json")
    libero40_examples = load_json("skill_moe/skillnet/examples/libero/annotations/instruct2plan_40.json")
    libero90 = load_json("data_process/libero/instruct2plan_90.json")
    libero90_obj = load_json("data_process/libero/instruct2plan_obj_90.json")
    libero90_obj_examples = load_json("skill_moe/skillnet/examples/libero/annotations/instruct2plan_obj_90.json")

    if not isinstance(libero40, dict) or len(libero40) != 40:
        fail(f"LIBERO-40 instruct2plan expected 40 entries, found {len(libero40) if isinstance(libero40, dict) else 'non-object'}", errors)
    if libero40 != libero40_examples:
        fail("LIBERO-40 data_process and examples annotation copies differ", errors)
    if not isinstance(libero90, dict) or len(libero90) != 73:
        fail(f"LIBERO-90 instruct2plan expected 73 entries, found {len(libero90) if isinstance(libero90, dict) else 'non-object'}", errors)
    if not isinstance(libero90_obj, dict) or len(libero90_obj) != 73:
        fail(f"LIBERO-90 object annotation expected 73 entries, found {len(libero90_obj) if isinstance(libero90_obj, dict) else 'non-object'}", errors)
    if set(libero90) != set(libero90_obj):
        fail("LIBERO-90 instruct2plan and object annotation keys differ", errors)
    if libero90_obj != libero90_obj_examples:
        fail("LIBERO-90 data_process and examples object annotation copies differ", errors)

    for rel_path, data, require_objects in [
        ("data_process/libero/instruct2plan_40.json", libero40, False),
        ("data_process/libero/instruct2plan_90.json", libero90, False),
        ("data_process/libero/instruct2plan_obj_90.json", libero90_obj, True),
    ]:
        if not isinstance(data, dict):
            continue
        for instruction, entry in data.items():
            if not isinstance(entry, dict):
                fail(f"{rel_path}: annotation entry must be an object: {instruction}", errors)
                continue
            plan = entry.get("plan")
            classes = entry.get("all_classes")
            objects = entry.get("objects")
            plan_len = len(plan) if isinstance(plan, list) else None
            if plan_len is None or plan_len == 0:
                fail(f"{rel_path}: empty plan for {instruction}", errors)
            if not isinstance(classes, list) or plan_len is None or len(classes) != plan_len:
                fail(f"{rel_path}: all_classes length mismatch for {instruction}", errors)
            if require_objects and (not isinstance(objects, list) or plan_len is None or len(objects) != plan_len):
                fail(f"{rel_path}: objects length mismatch for {instruction}", errors)

    if len(errors) == error_count:
        ok("LIBERO in-domain annotation copies and schemas match", verbose=verbose)


def check_libero_slice_index_contract(errors: list[str], *, verbose: bool) -> None:
    error_count = len(errors)
    expectations = {
        "data_process/libero/slice_indices/libero40_slice_index.json": {
            "repo_id": "jsw19/libero_40_v1",
            "tasks": 40,
            "source_episodes": 1693,
            "lerobot_episodes": 3862,
            "frames": 273465,
        },
        "data_process/libero/slice_indices/libero90_slice_index.json": {
            "repo_id": "jsw19/libero_90_v1",
            "tasks": 73,
            "source_episodes": 4006,
            "lerobot_episodes": 7874,
            "frames": 574571,
        },
    }
    verification_doc_paths = [
        "docs/quick_start.md",
        "docs/training_and_evaluation.md",
        "docs/libero_data_processing.md",
        "docs/release_status.md",
    ]
    for rel_path, expected in expectations.items():
        payload = load_json(rel_path)
        if not isinstance(payload, dict):
            fail(f"LIBERO slice-index must be a JSON object: {rel_path}", errors)
            continue
        text = json.dumps(payload)
        if REMOTE_SHARE_PATTERN.search(text) or WINDOWS_ABSOLUTE_PATH_PATTERN.search(text):
            fail(f"LIBERO slice-index contains a local absolute path: {rel_path}", errors)
        if payload.get("format") != "skillnet_libero_slice_index_v1":
            fail(f"unexpected LIBERO slice-index format in {rel_path}: {payload.get('format')!r}", errors)
        if payload.get("repo_id") != expected["repo_id"]:
            fail(f"unexpected repo_id in {rel_path}: {payload.get('repo_id')!r}", errors)
        if payload.get("is_partial"):
            fail(f"LIBERO slice-index must not be partial: {rel_path}", errors)
        episodes = payload.get("episodes")
        if not isinstance(episodes, list):
            fail(f"LIBERO slice-index episodes must be a list: {rel_path}", errors)
            continue
        if payload.get("num_episodes") != len(episodes) or len(episodes) != expected["source_episodes"]:
            fail(
                f"LIBERO slice-index source episode count mismatch in {rel_path}: "
                f"{payload.get('num_episodes')} / {len(episodes)}",
                errors,
            )
        total_segments = 0
        total_frames = 0
        for expected_index, episode in enumerate(episodes):
            if episode.get("episode_index") != expected_index:
                fail(
                    f"LIBERO slice-index episode_index mismatch in {rel_path}: "
                    f"{episode.get('episode_index')} != {expected_index}",
                    errors,
                )
                break
            segments = episode.get("segments")
            if not isinstance(segments, list) or not segments:
                fail(f"LIBERO slice-index episode has no segments in {rel_path}: {expected_index}", errors)
                break
            cursor = 0
            classes = []
            for segment in segments:
                start = int(segment["start"])
                end = int(segment["end"])
                if start != cursor or end <= start:
                    fail(f"non-contiguous LIBERO slice segment in {rel_path}, episode {expected_index}: {segment}", errors)
                    break
                cursor = end
                classes.append(int(segment["class"]))
            if cursor != int(episode["length"]):
                fail(
                    f"LIBERO slice-index length mismatch in {rel_path}, episode {expected_index}: "
                    f"{cursor} != {episode['length']}",
                    errors,
                )
                break
            try:
                all_classes = ast.literal_eval(segments[0]["all_classes"])
            except Exception as exc:
                fail(f"invalid all_classes in {rel_path}, episode {expected_index}: {exc}", errors)
                break
            if [int(item) for item in all_classes] != classes:
                fail(f"LIBERO slice-index classes do not match all_classes in {rel_path}, episode {expected_index}", errors)
                break
            total_segments += len(segments)
            total_frames += int(episode["length"])
        if total_segments != expected["lerobot_episodes"]:
            fail(f"LIBERO slice-index segment count mismatch in {rel_path}: {total_segments}", errors)
        if total_frames != expected["frames"]:
            fail(f"LIBERO slice-index frame count mismatch in {rel_path}: {total_frames}", errors)
        for doc_rel_path in verification_doc_paths:
            doc_text = (REPO_ROOT / doc_rel_path).read_text(encoding="utf-8")
            for snippet in [
                f"$LEROBOT_HOME/{expected['repo_id']}",
                f"--expected-tasks {expected['tasks']}",
                f"--expected-episodes {expected['lerobot_episodes']}",
                f"--expected-frames {expected['frames']}",
                "--require-feature class",
                "--require-feature all_classes",
            ]:
                if snippet not in doc_text:
                    fail(
                        f"{doc_rel_path}: missing LIBERO dataset verification snippet for "
                        f"{expected['repo_id']}: {snippet}",
                        errors,
                    )
    if len(errors) == error_count:
        ok("LIBERO compact slice-index metadata and documented verification counts match", verbose=verbose)


def check_robotwin_contract(errors: list[str], *, verbose: bool) -> None:
    error_count = len(errors)
    robotwin_plan = load_json("data_process/robotwin/robotwin_plan.json")
    skill_annotations = load_json("data_process/robotwin/skill_anno_robotwin.json")
    if not isinstance(robotwin_plan, dict) or not isinstance(skill_annotations, dict):
        fail("RoboTwin plan and skill annotations must be JSON objects", errors)
        return

    builder_pretrain = load_python_literal_assignment("data_process/robotwin/build_robotwin_skill_metadata.py", "PRETRAIN_TASKS")
    builder_transfer = load_python_literal_assignment("data_process/robotwin/build_robotwin_skill_metadata.py", "TRANSFER_TASKS")
    download_pretrain = load_python_literal_assignment("data_process/robotwin/download_robotwin_sources.py", "PRETRAIN_TASKS")
    download_transfer = load_python_literal_assignment("data_process/robotwin/download_robotwin_sources.py", "TRANSFER_TASKS")
    eval_pretrain = load_python_literal_assignment(
        "skill_moe/skillnet/examples/robotwin/eval_robotwin_moe_skill.py", "PRETRAIN_TASKS"
    )
    eval_transfer = load_python_literal_assignment(
        "skill_moe/skillnet/examples/robotwin/eval_robotwin_moe_skill.py", "TRANSFER_TASKS"
    )
    collect_pretrain = [task.replace(" ", "_") for task in parse_shell_array("data_process/robotwin/collect_train_data.sh", "PRETRAIN_TASKS")]
    collect_transfer = [task.replace(" ", "_") for task in parse_shell_array("data_process/robotwin/collect_train_data.sh", "TRANSFER_TASKS")]

    for label, actual, expected in [
        ("metadata pretrain tasks", builder_pretrain, ROBOTWIN_PRETRAIN_TASKS),
        ("metadata transfer tasks", builder_transfer, ROBOTWIN_TRANSFER_TASKS),
        ("download pretrain tasks", download_pretrain, ROBOTWIN_PRETRAIN_TASKS),
        ("download transfer tasks", download_transfer, ROBOTWIN_TRANSFER_TASKS),
        ("eval pretrain tasks", eval_pretrain, ROBOTWIN_PRETRAIN_TASKS),
        ("eval transfer tasks", eval_transfer, ROBOTWIN_TRANSFER_TASKS),
        ("collect pretrain tasks", collect_pretrain, ROBOTWIN_PRETRAIN_TASKS),
        ("collect transfer tasks", collect_transfer, ROBOTWIN_TRANSFER_TASKS),
    ]:
        if actual != expected:
            fail(f"RoboTwin {label} do not match the public paper task manifest", errors)

    if len(robotwin_plan) != 50:
        fail(f"RoboTwin full plan expected 50 tasks, found {len(robotwin_plan)}", errors)

    description_to_tasks: dict[str, list[str]] = {}
    all_skill_ids = []
    longest_skill_sequence = 0
    for task, entry in robotwin_plan.items():
        if not isinstance(entry, dict):
            fail(f"RoboTwin task entry must be an object: {task}", errors)
            continue
        description = entry.get("description")
        skills = entry.get("skills")
        if not isinstance(description, str) or not description:
            fail(f"RoboTwin task missing description: {task}", errors)
        else:
            description_to_tasks.setdefault(description, []).append(task)
            if description not in skill_annotations:
                fail(f"RoboTwin task description missing hierarchy annotation: {task}", errors)
        if not isinstance(skills, list) or not skills or not all(isinstance(item, int) for item in skills):
            fail(f"RoboTwin task must contain integer skills: {task}", errors)
        else:
            all_skill_ids.extend(skills)
            longest_skill_sequence = max(longest_skill_sequence, len(skills))

    duplicate_descriptions = {
        description: sorted(tasks) for description, tasks in description_to_tasks.items() if len(tasks) > 1
    }
    expected_duplicate = {
        "pick up one bottle with one arm, and pick up another bottle with the other arm": [
            "pick_diverse_bottles",
            "pick_dual_bottles",
        ]
    }
    if duplicate_descriptions != expected_duplicate:
        fail(f"Unexpected RoboTwin duplicate descriptions: {duplicate_descriptions}", errors)
    if len(skill_annotations) != len(description_to_tasks):
        fail(
            f"RoboTwin hierarchy annotations should match unique task descriptions: "
            f"{len(skill_annotations)} annotations vs {len(description_to_tasks)} descriptions",
            errors,
        )

    expected_tasks = ROBOTWIN_PRETRAIN_TASKS + ROBOTWIN_TRANSFER_TASKS
    if len(expected_tasks) != 30:
        fail(f"RoboTwin paper task list should contain 30 tasks, found {len(expected_tasks)}", errors)

    max_skill_id = -1
    for task in expected_tasks:
        entry = robotwin_plan.get(task)
        if not isinstance(entry, dict):
            fail(f"missing RoboTwin paper task in robotwin_plan.json: {task}", errors)
            continue
        description = entry.get("description")
        skills = entry.get("skills")
        if not isinstance(description, str) or not description:
            fail(f"RoboTwin task missing description: {task}", errors)
        if not isinstance(skills, list) or not skills or not all(isinstance(item, int) for item in skills):
            fail(f"RoboTwin task must contain integer skills: {task}", errors)
        else:
            max_skill_id = max(max_skill_id, max(skills))

        annotation = skill_annotations.get(description)
        if not isinstance(annotation, dict):
            fail(f"missing RoboTwin hierarchy annotation for task description: {task}", errors)
            continue
        subtask_plan = annotation.get("plan")
        classes = annotation.get("all_classes")
        if not isinstance(subtask_plan, list) or not subtask_plan:
            fail(f"RoboTwin hierarchy annotation has empty plan: {task}", errors)
        if not isinstance(classes, list) or not classes or len(classes) % 3 != 0:
            fail(f"RoboTwin hierarchy all_classes must be nonempty triples: {task}", errors)
        if not all(isinstance(item, int) for item in classes or []):
            fail(f"RoboTwin hierarchy all_classes must be integers: {task}", errors)

    if max_skill_id != 12:
        fail(f"RoboTwin paper tasks expected max flat skill id 12, found {max_skill_id}", errors)
    if all_skill_ids and (min(all_skill_ids) != 0 or max(all_skill_ids) != 12):
        fail(f"RoboTwin full plan expected flat skill id range 0..12, found {min(all_skill_ids)}..{max(all_skill_ids)}", errors)
    if longest_skill_sequence != 6:
        fail(f"RoboTwin full plan expected longest flat skill sequence length 6, found {longest_skill_sequence}", errors)

    robotwin_pipeline_snippets = [
        (
            "data_process/robotwin/convert_robotwin_to_lerobot.py",
            [
                '"task": {"dtype": "string"',
                '"skills": {"dtype": "string"',
                '"task": task_desc',
                '"skills": skill_string',
                "[action_left, action_left_gripper, action_right, action_right_gripper]",
                "[state_left, state_left_gripper, state_right, state_right_gripper]",
            ],
        ),
        ("skill_moe/skillnet/src/openpi/training/data_loader_skill.py", ["SkillPromptFromLeRobotTask", "if data_config.prompt_from_task:"]),
        ("skill_moe/skillnet/src/openpi/policies/robotwin_policy.py", ['inputs["skills"] = padded', 'skill_ids[:num_skills] + 1']),
        (
            "skill_moe/skillnet/examples/robotwin/eval_robotwin_moe_skill.py",
            [
                "def robotwin_joint_vector",
                '("left_arm", "left_gripper", "right_arm", "right_gripper")',
                "np.random.default_rng(seed_start)",
            ],
        ),
    ]
    for rel_path, snippets in robotwin_pipeline_snippets:
        text = compact_text((REPO_ROOT / rel_path).read_text(encoding="utf-8"))
        for snippet in snippets:
            if compact_text(snippet) not in text:
                fail(f"{rel_path}: missing RoboTwin data/policy pipeline snippet: {snippet}", errors)

    if len(errors) == error_count:
        ok("RoboTwin full-plan schema and 30-task few-shot contract are present", verbose=verbose)


def check_python_compile(errors: list[str], *, verbose: bool) -> None:
    python_files = tracked_files(".py", PYTHON_FILES)
    failures = []
    for rel_path in python_files:
        path = REPO_ROOT / rel_path
        try:
            compile(path.read_bytes(), rel_path, "exec")
        except SyntaxError as exc:
            failures.append(f"{rel_path}:{exc.lineno}: {exc.msg}")
        except Exception as exc:
            failures.append(f"{rel_path}: {exc}")
    if failures:
        fail("python syntax check failed:\n" + "\n".join(failures), errors)
    else:
        ok(f"Python syntax ok for {len(python_files)} tracked files", verbose=verbose)


def check_help_commands(errors: list[str], *, verbose: bool) -> None:
    for command in HELP_COMMANDS:
        result = subprocess.run([sys.executable, *command], cwd=REPO_ROOT, text=True, capture_output=True)
        if result.returncode != 0:
            fail(f"help command failed: python {' '.join(command)}\n{result.stderr.strip()}", errors)
        else:
            ok(f"help works: python {' '.join(command)}", verbose=verbose)


def check_shell_syntax(errors: list[str], *, verbose: bool, require_bash: bool) -> None:
    bash = shutil.which("bash")
    if bash is None:
        message = "bash not found; skipping shell syntax checks"
        if require_bash:
            fail(message + " (--require-bash set)", errors)
        else:
            skip(message, verbose=verbose)
        return
    shell_files = tracked_files(".sh", SHELL_FILES)
    for rel_path in shell_files:
        result = subprocess.run([bash, "-n", rel_path], cwd=REPO_ROOT, text=True, capture_output=True)
        if result.returncode != 0:
            fail(f"shell syntax failed: {rel_path}\n{result.stderr.strip()}", errors)
        else:
            ok(f"shell syntax ok: {rel_path}", verbose=verbose)


def check_sensitive_patterns(errors: list[str], *, verbose: bool) -> None:
    matches = []
    for path in REPO_ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file() or path.suffix not in SCAN_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for pattern in SENSITIVE_PATTERNS:
                if pattern.search(line):
                    rel_path = path.relative_to(REPO_ROOT).as_posix()
                    matches.append(f"{rel_path}:{line_number}: {pattern.pattern}")
    if matches:
        fail("sensitive/private patterns found:\n" + "\n".join(matches), errors)
    else:
        ok("no private path/token patterns found", verbose=verbose)


def check_history_sensitive_patterns(errors: list[str], *, verbose: bool) -> None:
    result = subprocess.run(["git", "rev-list", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        fail("--history-smoke requires a git checkout with a valid HEAD", errors)
        return

    commits = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    matches: list[str] = []
    total_matches = 0
    max_reported = 80
    for commit in commits:
        tree_result = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", commit],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )
        if tree_result.returncode != 0:
            fail(f"git history scan could not list files for {commit[:12]}: {tree_result.stderr.strip()}", errors)
            return

        for rel_path in tree_result.stdout.splitlines():
            if not rel_path or Path(rel_path).suffix not in SCAN_SUFFIXES:
                continue
            blob_result = subprocess.run(
                ["git", "show", f"{commit}:{rel_path}"],
                cwd=REPO_ROOT,
                capture_output=True,
            )
            if blob_result.returncode != 0:
                continue
            text = blob_result.stdout.decode("utf-8", errors="ignore")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if rel_path == "scripts/check_public_release.py" and "re.compile" in line:
                    continue
                for pattern in HISTORY_SENSITIVE_PATTERNS:
                    if not pattern.search(line):
                        continue
                    total_matches += 1
                    if len(matches) < max_reported:
                        matches.append(f"{commit[:12]}:{rel_path}:{line_number}: {pattern.pattern}")
                    break

    if total_matches:
        suffix = ""
        if total_matches > len(matches):
            suffix = f"\n... {total_matches - len(matches)} additional history matches omitted"
        fail("sensitive/private patterns found in git history:\n" + "\n".join(matches) + suffix, errors)
    else:
        ok(f"git history reachable from HEAD is clean ({len(commits)} commit(s) scanned)", verbose=verbose)


def check_install_smoke(errors: list[str], *, python_spec: str, verbose: bool) -> None:
    uv = shutil.which("uv")
    if uv is None:
        fail("--install-smoke requires uv on PATH. Install prerequisites with: python -m pip install -U uv", errors)
        return

    with tempfile.TemporaryDirectory(prefix="skillnet-install-smoke-") as temp_dir:
        venv_dir = Path(temp_dir) / ".venv"
        python = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        commands = [
            [uv, "venv", "--python", python_spec, str(venv_dir)],
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                "--no-deps",
                "-e",
                str(REPO_ROOT / "skill_moe/skillnet"),
            ],
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                "--no-deps",
                "-e",
                str(REPO_ROOT / "skill_moe/skillnet/packages/openpi-client"),
            ],
        ]
        for command in commands:
            result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
            if result.returncode != 0:
                fail(
                    "install smoke command failed: "
                    + " ".join(command)
                    + "\n"
                    + (result.stderr.strip() or result.stdout.strip()),
                    errors,
                )
                return

        smoke_code = """
import importlib
import importlib.metadata as metadata
import importlib.util

assert metadata.version("skillnet")
assert metadata.version("openpi-client")
assert importlib.util.find_spec("openpi.training.config_moe_skill")
assert importlib.util.find_spec("openpi_client.websocket_client_policy")
client = importlib.import_module("openpi_client")
assert client.__version__ == "0.1.0"
"""
        result = subprocess.run([str(python), "-c", smoke_code], cwd=REPO_ROOT, text=True, capture_output=True)
        if result.returncode != 0:
            fail(f"install smoke import check failed:\n{result.stderr.strip() or result.stdout.strip()}", errors)
            return

    ok("temporary --no-deps editable install smoke passed", verbose=verbose)


def hub_api_url(endpoint: str, repo_type: str, repo_id: str) -> str:
    endpoint = endpoint.rstrip("/")
    if repo_type == "model":
        return f"{endpoint}/api/models/{repo_id}"
    if repo_type == "dataset":
        return f"{endpoint}/api/datasets/{repo_id}"
    raise ValueError(f"unsupported Hugging Face repo type: {repo_type}")


def fetch_hub_status(url: str, *, timeout: float, retries: int, token: str | None) -> tuple[int | None, str]:
    headers = {"User-Agent": "skillnet-release-check/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    attempts = max(1, retries)
    last_error = ""
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, ""
        except urllib.error.HTTPError as exc:
            if exc.code in {429, 500, 502, 503, 504} and attempt < attempts:
                time.sleep(min(2**attempt, 5))
                continue
            return exc.code, exc.reason or str(exc)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
            if attempt < attempts:
                time.sleep(min(2**attempt, 5))
                continue
    return None, last_error


def check_hub_resources(
    errors: list[str],
    *,
    endpoint: str,
    include_derived_datasets: bool,
    include_libero_derived_datasets: bool,
    include_robotwin_derived_datasets: bool,
    timeout: float,
    retries: int,
    token: str | None,
    verbose: bool,
) -> None:
    resources = list(PUBLIC_HUB_RESOURCES)
    if include_derived_datasets:
        resources.extend(DERIVED_LEROBOT_HUB_RESOURCES)
    else:
        if include_libero_derived_datasets:
            resources.extend(LIBERO_DERIVED_LEROBOT_HUB_RESOURCES)
        if include_robotwin_derived_datasets:
            resources.extend(ROBOTWIN_DERIVED_LEROBOT_HUB_RESOURCES)

    for repo_type, repo_id in resources:
        url = hub_api_url(endpoint, repo_type, repo_id)
        status, detail = fetch_hub_status(url, timeout=timeout, retries=retries, token=token)
        label = f"{repo_type} {repo_id}"
        if status == 200:
            ok(f"Hugging Face resource reachable: {label}", verbose=verbose)
            continue
        if status in {401, 403}:
            fail(f"Hugging Face resource is private or requires authentication: {label}", errors)
        elif status == 404:
            fail(f"Hugging Face resource not found: {label}", errors)
        elif status is None:
            fail(f"Hugging Face resource check failed for {label}: {detail}", errors)
        else:
            fail(f"Hugging Face resource check returned HTTP {status} for {label}: {detail}", errors)


def main() -> None:
    args = parse_args()
    errors: list[str] = []

    check_required_files(errors, verbose=args.verbose)
    check_json_files(errors, verbose=args.verbose)
    check_jsonl_files(errors, verbose=args.verbose)
    check_documentation_contracts(errors, verbose=args.verbose)
    check_release_surface_paths(errors, verbose=args.verbose)
    check_config_and_script_contracts(errors, verbose=args.verbose)
    check_workflow_contracts(errors, verbose=args.verbose)
    check_skill_hierarchy_contract(errors, verbose=args.verbose)
    check_libero_annotation_contract(errors, verbose=args.verbose)
    check_libero_slice_index_contract(errors, verbose=args.verbose)
    check_libero_skill_contract(errors, verbose=args.verbose)
    check_robotwin_contract(errors, verbose=args.verbose)
    check_python_compile(errors, verbose=args.verbose)
    if not args.skip_help:
        check_help_commands(errors, verbose=args.verbose)
    check_shell_syntax(errors, verbose=args.verbose, require_bash=args.require_bash)
    check_sensitive_patterns(errors, verbose=args.verbose)
    if args.history_smoke:
        check_history_sensitive_patterns(errors, verbose=args.verbose)
    if args.install_smoke:
        check_install_smoke(errors, python_spec=args.install_python, verbose=args.verbose)
    if args.hub_smoke:
        hub_token = None
        if args.hub_authenticated:
            hub_token = os.environ.get(args.hub_token_env)
            if hub_token is None and args.hub_token_env == "HF_TOKEN":
                hub_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
            if not hub_token:
                fail(f"--hub-authenticated requires token environment variable: {args.hub_token_env}", errors)
        check_hub_resources(
            errors,
            endpoint=args.hf_endpoint,
            include_derived_datasets=args.include_derived_datasets,
            include_libero_derived_datasets=args.include_libero_derived_datasets,
            include_robotwin_derived_datasets=args.include_robotwin_derived_datasets,
            timeout=args.hub_timeout,
            retries=args.hub_retries,
            token=hub_token,
            verbose=args.verbose,
        )

    if errors:
        raise SystemExit(1)
    print("SkillNet public release checks passed.")


if __name__ == "__main__":
    main()
