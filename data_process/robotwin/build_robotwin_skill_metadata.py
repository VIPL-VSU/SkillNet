"""Build public RoboTwin task metadata used by SkillNet few-shot experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
    "all": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("robotwin_plan.json"))
    parser.add_argument("--skill-annotation", type=Path, default=Path(__file__).with_name("skill_anno_robotwin.json"))
    parser.add_argument("--task-set", choices=sorted(TASK_SETS), default="paper")
    parser.add_argument("--output", type=Path, default=Path("robotwin_skill_metadata.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_data = json.loads(args.plan.read_text(encoding="utf-8"))
    annotation_data = json.loads(args.skill_annotation.read_text(encoding="utf-8"))

    task_names = TASK_SETS[args.task_set] or sorted(plan_data)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as f:
        for task_name in task_names:
            if task_name not in plan_data:
                raise KeyError(f"Task {task_name!r} is missing from {args.plan}")
            item = plan_data[task_name]
            description = item["description"]
            annotation = annotation_data.get(description, {})
            record = {
                "task": task_name,
                "description": description,
                "skills": item.get("skills", []),
                "plan": annotation.get("plan", item.get("plan", [])),
                "hierarchical_skill_tokens": annotation.get("all_classes", []),
                "split": "pretrain" if task_name in PRETRAIN_TASKS else "transfer",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(task_names)} tasks to {args.output}")


if __name__ == "__main__":
    main()
