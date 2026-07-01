"""Convert SkillNet LIBERO RLDS data into LeRobot format.

This public converter defaults to non-overwrite behavior: if the target
LeRobot dataset already exists, the script exits before creating or deleting
anything.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(os.environ.get("SKILLNET_LIBERO_DATA_ROOT", "data/libero")).expanduser()
RELEASE_HF_NAMESPACE = os.environ.get("SKILLNET_RELEASE_HF_NAMESPACE", "jsw19")


def release_repo_id(name: str) -> str:
    return f"{RELEASE_HF_NAMESPACE}/{name}"

VERB_TO_CLASS = {
    "pick": 0,
    "place": 1,
    "push": 2,
    "open": 3,
    "close": 3,
    "turn": 4,
}

PRESETS = {
    "libero40": {
        "repo_id": os.environ.get("SKILLNET_LIBERO40_REPO_ID", release_repo_id("libero_40_v1")),
        "raw_dataset_names": (
            "libero_10_no_noops",
            "libero_goal_no_noops",
            "libero_object_no_noops",
            "libero_spatial_no_noops",
        ),
        "data_dir": Path(os.environ.get("LIBERO40_RLDS_DIR", DEFAULT_DATA_ROOT / "libero40_rlds")).expanduser(),
        "plan_root": Path(os.environ.get("LIBERO40_PLAN_ROOT", DEFAULT_DATA_ROOT / "libero40_slices")).expanduser(),
        "plan_file": "libero40_plan_sliced.json",
        "instruction_map": SCRIPT_DIR / "instruct2plan_40.json",
        "include_objects": False,
        "sequence_state_action": False,
    },
    "libero90": {
        "repo_id": os.environ.get("SKILLNET_LIBERO90_REPO_ID", release_repo_id("libero_90_v1")),
        "raw_dataset_names": ("libero_90_openvla_processed",),
        "data_dir": Path(os.environ.get("LIBERO90_RLDS_DIR", DEFAULT_DATA_ROOT / "libero90_rlds")).expanduser(),
        "plan_root": Path(os.environ.get("LIBERO90_PLAN_ROOT", DEFAULT_DATA_ROOT / "libero90_slices")).expanduser(),
        "plan_file": "libero90_plan_sliced.json",
        "instruction_map": SCRIPT_DIR / "instruct2plan_90.json",
        "include_objects": False,
        "sequence_state_action": True,
    },
}


def parse_args(default_preset: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=PRESETS.keys(),
        default=default_preset,
        required=default_preset is None,
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--plan-root", type=Path, default=None)
    parser.add_argument(
        "--slice-index",
        type=Path,
        default=None,
        help=(
            "Compact SkillNet slice-index JSON exported from an existing "
            "LeRobot v1 dataset. This is an alternative to --plan-root."
        ),
    )
    parser.add_argument("--output-repo-id", default=None)
    parser.add_argument("--instruction-map", type=Path, default=None)
    parser.add_argument(
        "--include-objects",
        action="store_true",
        help="Include an objects feature. The instruction map must contain an objects field.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate an existing LeRobot output directory.",
    )
    return parser.parse_args()


def output_path_for(repo_id: str) -> Path:
    return HF_LEROBOT_HOME / repo_id


def create_dataset(repo_id: str, include_objects: bool, sequence_state_action: bool) -> LeRobotDataset:
    state_feature = {"dtype": "float32", "shape": (8,)}
    action_feature = {"dtype": "float32", "shape": (7,)}
    if sequence_state_action:
        state_feature["sequence"] = True
        action_feature["sequence"] = True

    features = {
        "image": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "state": state_feature,
        "actions": action_feature,
        "class": {"dtype": "int64", "shape": (1,)},
        "all_classes": {"dtype": "string", "shape": (1,)},
    }
    if include_objects:
        features["objects"] = {"dtype": "string", "shape": (1,)}

    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=10,
        features=features,
    )


def load_plan_index(plan_root: Path, plan_file: str) -> dict[str, dict]:
    plan_path = plan_root / plan_file
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing skill slice file: {plan_path}")
    with plan_path.open("r", encoding="utf-8") as f:
        sliced_items = json.load(f)
    return {item["image_root"]: item for item in sliced_items}


def load_slice_index(slice_index_path: Path) -> dict[int, dict]:
    if not slice_index_path.exists():
        raise FileNotFoundError(f"Missing slice-index file: {slice_index_path}")
    with slice_index_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("format") != "skillnet_libero_slice_index_v1":
        raise ValueError(
            f"Unsupported slice-index format in {slice_index_path}: "
            f"{payload.get('format')!r}"
        )
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"slice-index file must contain an episodes list: {slice_index_path}")
    return {int(item["episode_index"]): item for item in episodes}


def infer_class(plan_step: str) -> int:
    verb = plan_step.split(" ")[0].lower()
    if verb not in VERB_TO_CLASS:
        raise KeyError(f"Unsupported skill verb in plan step: {plan_step!r}")
    return VERB_TO_CLASS[verb]


def class_value_to_array(value) -> np.ndarray:
    if isinstance(value, list):
        values = value
    else:
        values = [value]
    if len(values) != 1:
        raise ValueError(
            "The released LIBERO v1 converter expects one flat skill class per frame; "
            f"got {values!r}."
        )
    return np.array([int(values[0])], dtype=np.int64)


def add_segment_frames(
    dataset: LeRobotDataset,
    steps: list[dict],
    segments: list[dict],
    instruction_map: dict,
    *,
    include_objects: bool,
) -> None:
    episode_length = len(steps)
    for segment in segments:
        start = int(segment["start"])
        end = int(segment["end"])
        if not (0 <= start < end <= episode_length):
            raise ValueError(f"Invalid segment {segment!r} for episode length {episode_length}")

        for step in steps[start:end]:
            instruction = step["language_instruction"].decode()
            instruction_entry = instruction_map.get(instruction, {})
            all_classes = segment.get("all_classes", instruction_entry.get("all_classes"))
            if all_classes is None:
                raise KeyError(f"Missing all_classes for instruction: {instruction!r}")
            frame = {
                "image": step["observation"]["image"],
                "wrist_image": step["observation"]["wrist_image"],
                "state": step["observation"]["state"],
                "actions": step["action"],
                "task": instruction,
                "class": class_value_to_array(segment["class"]),
                "all_classes": str(all_classes),
            }
            if include_objects:
                objects = segment.get("objects", instruction_entry.get("objects"))
                if objects is None:
                    raise KeyError(f"Missing objects for instruction: {instruction!r}")
                frame["objects"] = str(objects)
            dataset.add_frame(frame)
        dataset.save_episode()


def convert(args: argparse.Namespace) -> None:
    preset = PRESETS[args.preset]
    repo_id = args.output_repo_id or preset["repo_id"]
    data_dir = args.data_dir or preset["data_dir"]
    plan_root = args.plan_root or preset["plan_root"]
    instruction_map_path = args.instruction_map or preset["instruction_map"]

    output_path = output_path_for(repo_id)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output dataset already exists: {output_path}. "
            "Use --overwrite only when you intentionally want to recreate it."
        )
    if output_path.exists() and args.overwrite:
        import shutil

        shutil.rmtree(output_path)

    if not instruction_map_path.exists():
        raise FileNotFoundError(f"Missing instruction map: {instruction_map_path}")

    if args.slice_index and args.plan_root:
        raise ValueError("Pass either --slice-index or --plan-root, not both.")

    slice_by_episode = load_slice_index(args.slice_index) if args.slice_index else None
    plan_by_root = None if slice_by_episode is not None else load_plan_index(plan_root, preset["plan_file"])
    with instruction_map_path.open("r", encoding="utf-8") as f:
        instruction_map = json.load(f)

    include_objects = bool(args.include_objects or preset["include_objects"])

    dataset = create_dataset(
        repo_id=repo_id,
        include_objects=include_objects,
        sequence_state_action=preset["sequence_state_action"],
    )

    counter: defaultdict[str, int] = defaultdict(int)
    episode_index = 0
    for raw_dataset_name in preset["raw_dataset_names"]:
        raw_dataset = tfds.load(
            raw_dataset_name,
            data_dir=str(data_dir),
            split="train",
            shuffle_files=False,
        )
        for episode in tqdm(raw_dataset, desc=raw_dataset_name):
            file_path = episode["episode_metadata"]["file_path"].numpy().decode("utf-8")
            folder_name = os.path.splitext(os.path.basename(file_path))[0]
            counter[folder_name] += 1
            steps = list(episode["steps"].as_numpy_iterator())
            if slice_by_episode is not None:
                slice_item = slice_by_episode.get(episode_index)
                if slice_item is None:
                    raise KeyError(f"No slice-index entry for episode_index={episode_index}")
                if int(slice_item["length"]) != len(steps):
                    raise ValueError(
                        f"slice-index length mismatch for episode_index={episode_index}: "
                        f"{slice_item['length']} != {len(steps)}"
                    )
                add_segment_frames(
                    dataset,
                    steps,
                    slice_item["segments"],
                    instruction_map,
                    include_objects=include_objects,
                )
            else:
                image_root = str(plan_root / f"{folder_name}_{counter[folder_name]}")
                if image_root not in plan_by_root:
                    raise KeyError(f"No skill slice entry for {image_root}")

                plan_item = plan_by_root[image_root]
                seq = plan_item["seq"]
                intervals = plan_item["skill_slices"]
                keypoints = [seq[i - 1] for i in intervals]
                plans = plan_item["plan"]
                segments = []
                for i in range(1, len(keypoints)):
                    left = keypoints[i - 1] - 1
                    right = keypoints[i] - 1
                    if i == len(keypoints) - 1:
                        right += 1
                    segments.append(
                        {
                            "start": left,
                            "end": right,
                            "class": infer_class(plans[i - 1]),
                        }
                    )
                add_segment_frames(dataset, steps, segments, instruction_map, include_objects=include_objects)
            episode_index += 1


def main() -> None:
    convert(parse_args())


def main_with_preset(preset: str) -> None:
    convert(parse_args(default_preset=preset))


if __name__ == "__main__":
    main()
