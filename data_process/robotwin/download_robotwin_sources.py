"""Download RoboTwin-2.0 50-demo zip files used by SkillNet few-shot experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


REPO_ID = "TianxingChen/RoboTwin2.0"
REPO_TYPE = "dataset"
DEFAULT_SUFFIX = "aloha-agilex_clean_50.zip"

PRETRAIN_TASKS = [
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

TRANSFER_TASKS = [
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

TASK_SETS = {
    "pretrain": PRETRAIN_TASKS,
    "transfer": TRANSFER_TASKS,
    "paper": PRETRAIN_TASKS + TRANSFER_TASKS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--repo-type", default=REPO_TYPE)
    parser.add_argument("--task-set", choices=sorted(TASK_SETS), default="pretrain")
    parser.add_argument("--tasks", nargs="*", default=None, help="Override --task-set with explicit task names.")
    parser.add_argument("--file-suffix", default=DEFAULT_SUFFIX)
    parser.add_argument("--output-dir", type=Path, default=Path("robotwin_datasets"))
    parser.add_argument("--extract", action="store_true", help="Extract each zip after downloading.")
    return parser.parse_args()


def maybe_extract(zip_path: Path, task_dir: Path) -> None:
    extract_dir = task_dir / zip_path.stem
    if extract_dir.exists():
        return
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(task_dir)


def main() -> None:
    args = parse_args()
    tasks = args.tasks if args.tasks else TASK_SETS[args.task_set]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Please install huggingface_hub before downloading RoboTwin data.") from exc

    for task in tasks:
        filename = f"dataset/{task}/{args.file_suffix}"
        task_dir = args.output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        path = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                filename=filename,
                repo_type=args.repo_type,
                local_dir=task_dir,
                local_dir_use_symlinks=False,
            )
        )
        print(f"[downloaded] {task}: {path}")
        if args.extract:
            maybe_extract(path, task_dir)
            print(f"[extracted] {task}: {task_dir / path.stem}")


if __name__ == "__main__":
    main()
