"""Convert RoboTwin-2.0 demonstration archives into SkillNet LeRobot datasets."""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import os
import shutil
import zipfile
from collections.abc import Iterable
from pathlib import Path

from build_robotwin_skill_metadata import TASK_SETS


SCRIPT_DIR = Path(__file__).resolve().parent
RELEASE_HF_NAMESPACE = os.environ.get("SKILLNET_RELEASE_HF_NAMESPACE", "jsw19")
RAW_CAMERA_MAP = {
    "head_color": "head_camera",
    "hand_right_color": "right_camera",
    "hand_left_color": "left_camera",
}


def require_numpy():
    import numpy as np

    return np


def release_repo_id(name: str) -> str:
    return f"{RELEASE_HF_NAMESPACE}/{name}"


@dataclasses.dataclass(frozen=True)
class EpisodeSource:
    kind: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing downloaded RoboTwin zips or extracted task folders.",
    )
    parser.add_argument(
        "--task-set",
        choices=sorted(TASK_SETS),
        default="pretrain",
        help="Task split to convert when --tasks is not provided.",
    )
    parser.add_argument(
        "--source-format",
        choices=("auto", "raw", "aligned"),
        default="auto",
        help="Input episode format. Raw RoboTwin zips use data/episode*.hdf5; aligned episodes use aligned_joints.h5.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Explicit task ids to convert. Overrides --task-set.",
    )
    parser.add_argument("--plan", type=Path, default=SCRIPT_DIR / "robotwin_plan.json")
    parser.add_argument("--output-repo-id", default=None)
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=None,
        help="Where zip files are extracted. Defaults to <source-dir>/_extracted.",
    )
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    parser.add_argument("--max-frames-per-episode", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Inspect inputs without writing a LeRobot dataset.")
    parser.add_argument(
        "--keep-static",
        action="store_true",
        help="Keep consecutive frames with unchanged joint state. By default they are skipped.",
    )
    return parser.parse_args()


def default_repo_id(task_set: str, tasks: list[str] | None) -> str:
    if tasks and len(tasks) == 1:
        return release_repo_id(f"robotwin_{tasks[0]}_v1")
    return {
        "pretrain": release_repo_id("robotwin_pretrain_v1"),
        "transfer": release_repo_id("robotwin_transfer_v1"),
        "paper": release_repo_id("robotwin_paper_v1"),
        "all": release_repo_id("robotwin_all_v1"),
    }[task_set]


def select_tasks(args: argparse.Namespace, plan_data: dict) -> list[str]:
    if args.tasks:
        task_names = args.tasks
    else:
        task_names = TASK_SETS[args.task_set] or sorted(plan_data)
    missing = [task for task in task_names if task not in plan_data]
    if missing:
        raise KeyError(f"Tasks missing from {args.plan}: {missing}")
    return list(task_names)


def find_task_archives(source_dir: Path, task_name: str) -> list[Path]:
    candidates = [
        source_dir / "dataset" / task_name / "aloha-agilex_clean_50.zip",
        source_dir / task_name / "aloha-agilex_clean_50.zip",
        source_dir / f"{task_name}.zip",
    ]
    archives = [path for path in candidates if path.exists()]
    archives.extend(source_dir.glob(f"**/{task_name}/*clean*.zip"))
    archives.extend(source_dir.glob(f"**/{task_name}*.zip"))
    return sorted(set(path.resolve() for path in archives))


def extract_archive(archive: Path, extract_root: Path, task_name: str) -> Path:
    task_extract_dir = extract_root / task_name / archive.stem
    marker = task_extract_dir / ".complete"
    if marker.exists():
        return task_extract_dir
    task_extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(task_extract_dir)
    marker.write_text("ok\n", encoding="utf-8")
    return task_extract_dir


def find_extracted_task_roots(source_dir: Path, task_name: str) -> list[Path]:
    roots = [
        source_dir / "dataset" / task_name,
        source_dir / "dataset" / task_name / "aloha-agilex_clean_50",
        source_dir / task_name,
        source_dir / task_name / "aloha-agilex_clean_50",
    ]
    roots.extend(path.parent for path in source_dir.glob(f"**/{task_name}/aligned_joints.h5"))
    roots.extend(path.parent.parent for path in source_dir.glob(f"**/{task_name}/**/data/episode*.hdf5"))
    roots.extend(path for path in source_dir.glob(f"**/{task_name}") if path.is_dir())
    return sorted(set(path.resolve() for path in roots if path.exists()))


def find_aligned_episode_dirs(task_roots: Iterable[Path]) -> list[Path]:
    episode_dirs: set[Path] = set()
    for root in task_roots:
        if (root / "aligned_joints.h5").exists():
            episode_dirs.add(root.resolve())
        for h5_path in root.glob("**/aligned_joints.h5"):
            episode_dirs.add(h5_path.parent.resolve())
    return sorted(episode_dirs)


def find_raw_episode_files(task_roots: Iterable[Path]) -> list[Path]:
    episode_files: set[Path] = set()
    for root in task_roots:
        data_dir = root / "data"
        if data_dir.exists():
            episode_files.update(path.resolve() for path in data_dir.glob("episode*.hdf5"))
        episode_files.update(path.resolve() for path in root.glob("**/data/episode*.hdf5"))
    return sorted(episode_files)


def find_episode_sources(task_roots: Iterable[Path], source_format: str) -> list[EpisodeSource]:
    sources: list[EpisodeSource] = []
    roots = list(task_roots)
    if source_format in ("auto", "raw"):
        sources.extend(EpisodeSource("raw", path) for path in find_raw_episode_files(roots))
    if source_format in ("auto", "aligned"):
        sources.extend(EpisodeSource("aligned", path) for path in find_aligned_episode_dirs(roots))
    return sources


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_episode_metadata(episode_dir: Path) -> dict:
    for name in ("metadata.json", "episode.json", "label.json", "labels.json", "data.json"):
        data = load_json_if_exists(episode_dir / name)
        if isinstance(data, dict):
            return data
    return {}


def action_config_range(metadata: dict, fallback_length: int) -> tuple[int, int]:
    label_info = metadata.get("label_info", metadata)
    action_config = label_info.get("action_config") if isinstance(label_info, dict) else None
    if not action_config:
        return 0, fallback_length
    valid_actions = [item for item in action_config if not item.get("is_mistake", False)]
    if not valid_actions:
        return 0, fallback_length
    start = int(valid_actions[0].get("start_frame", 0))
    end = int(valid_actions[-1].get("end_frame", fallback_length - 1)) + 1
    start = max(0, min(start, fallback_length - 1))
    end = min(max(start + 1, end), fallback_length)
    return start, end


def as_column(value: np.ndarray) -> np.ndarray:
    np = require_numpy()
    value = np.asarray(value)
    if value.ndim == 1:
        return value[:, None]
    return value


def load_joint_arrays(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    np = require_numpy()
    with h5py.File(h5_path, "r") as f:
        action_left = np.asarray(f["action"]["joint"]["position"][:, :7], dtype=np.float32)
        action_right = np.asarray(f["action"]["joint"]["position"][:, 7:], dtype=np.float32)
        action_left_gripper = 1.0 - as_column(f["action"]["left_effector"]["position"][:]).astype(np.float32)
        action_right_gripper = 1.0 - as_column(f["action"]["right_effector"]["position"][:]).astype(np.float32)
        actions = np.concatenate(
            [action_left, action_left_gripper, action_right, action_right_gripper],
            axis=-1,
        )

        state_left = np.asarray(f["state"]["joint"]["position"][:, :7], dtype=np.float32)
        state_right = np.asarray(f["state"]["joint"]["position"][:, 7:], dtype=np.float32)
        state_left_gripper = 1.0 - (
            as_column(f["state"]["left_effector"]["position"][:]).astype(np.float32) - 35.0
        ) / (120.0 - 35.0)
        state_right_gripper = 1.0 - (
            as_column(f["state"]["right_effector"]["position"][:]).astype(np.float32) - 35.0
        ) / (120.0 - 35.0)
        states = np.concatenate(
            [state_left, state_left_gripper, state_right, state_right_gripper],
            axis=-1,
        )
    if states.shape[-1] != 16 or actions.shape[-1] != 16:
        raise ValueError(f"Expected 16-D state/action arrays, got {states.shape} and {actions.shape}")
    return states, actions


def h5_has(root, path: str) -> bool:
    return path in root


def first_column(value) -> np.ndarray:
    np = require_numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        return array[:, None]
    return array.reshape(array.shape[0], -1)[:, :1]


def raw_joint_matrix(root) -> np.ndarray:
    np = require_numpy()
    component_paths = (
        "/joint_action/left_arm",
        "/joint_action/left_gripper",
        "/joint_action/right_arm",
        "/joint_action/right_gripper",
    )
    vector_path = "/joint_action/vector"

    component_state = None
    if all(h5_has(root, path) for path in component_paths):
        left_arm = np.asarray(root["/joint_action/left_arm"][:], dtype=np.float32)
        left_gripper = first_column(root["/joint_action/left_gripper"][:])
        right_arm = np.asarray(root["/joint_action/right_arm"][:], dtype=np.float32)
        right_gripper = first_column(root["/joint_action/right_gripper"][:])
        component_state = np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=-1)
        if component_state.shape[-1] == 16:
            return component_state

    if h5_has(root, vector_path):
        vector_state = np.asarray(root[vector_path][:], dtype=np.float32)
        if vector_state.shape[-1] == 16:
            return vector_state

    if component_state is not None:
        raise ValueError(f"Expected 16-D raw joint state, got {component_state.shape}")
    raise KeyError("Missing RoboTwin joint_action fields")


def load_raw_joint_arrays(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(h5_path, "r") as f:
        state = raw_joint_matrix(f)
    if len(state) < 2:
        raise ValueError(f"Raw RoboTwin episode needs at least 2 frames: {h5_path}")
    return state[:-1].astype(np.float32), state[1:].astype(np.float32)


def decode_image_value(value, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    np = require_numpy()
    array = np.asarray(value)
    if array.ndim == 3:
        if array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
            array = np.moveaxis(array, 0, -1)
        if array.shape[-1] == 4:
            array = array[..., :3]
        image = Image.fromarray(array.astype(np.uint8), mode="RGB")
    else:
        if isinstance(value, np.void):
            data = bytes(value)
        elif isinstance(value, bytes | bytearray):
            data = bytes(value)
        else:
            data = array.tobytes()
        image = Image.open(io.BytesIO(data.rstrip(b"\0"))).convert("RGB")
    if image.size != size:
        image = image.resize(size)
    return np.asarray(image, dtype=np.uint8)


def read_raw_image_from_root(root, frame_index: int, view: str, size: tuple[int, int]) -> np.ndarray:
    camera_name = RAW_CAMERA_MAP[view]
    image_key = f"/observation/{camera_name}/rgb"
    if image_key not in root:
        raise KeyError(f"Missing raw RoboTwin image key {image_key}")
    return decode_image_value(root[image_key][frame_index], size)


def read_raw_image(h5_path: Path, frame_index: int, view: str, size: tuple[int, int]) -> np.ndarray:
    import h5py

    with h5py.File(h5_path, "r") as f:
        return read_raw_image_from_root(f, frame_index, view, size)


def image_path_for(camera_dir: Path, frame_index: int, view: str) -> Path:
    candidates = [
        camera_dir / str(frame_index) / f"{view}.jpg",
        camera_dir / str(frame_index) / f"{view}.png",
        camera_dir / view / f"{frame_index}.jpg",
        camera_dir / view / f"{frame_index}.png",
        camera_dir / view / f"{frame_index:06d}.jpg",
        camera_dir / view / f"{frame_index:06d}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing image for frame {frame_index}, view {view}, under {camera_dir}")


def read_image(camera_dir: Path, frame_index: int, view: str, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    np = require_numpy()
    image = Image.open(image_path_for(camera_dir, frame_index, view)).convert("RGB")
    if image.size != size:
        image = image.resize(size)
    return np.asarray(image, dtype=np.uint8)


def lerobot_output_path(repo_id: str) -> Path:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

    return HF_LEROBOT_HOME / repo_id


def create_dataset(repo_id: str, fps: int, image_shape: tuple[int, int, int]):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="aloha-agilex",
        fps=fps,
        features={
            "head_color": {"dtype": "image", "shape": image_shape, "names": ["height", "width", "channel"]},
            "hand_right_color": {"dtype": "image", "shape": image_shape, "names": ["height", "width", "channel"]},
            "hand_left_color": {"dtype": "image", "shape": image_shape, "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (16,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (16,), "names": ["actions"]},
            "task": {"dtype": "string", "shape": (1,), "names": ["task"]},
            "skills": {"dtype": "string", "shape": (1,), "names": ["skills"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )


def convert_aligned_episode(
    dataset,
    episode_dir: Path,
    task_desc: str,
    skill_string: str,
    args: argparse.Namespace,
) -> int:
    np = require_numpy()
    states, actions = load_joint_arrays(episode_dir / "aligned_joints.h5")
    metadata = load_episode_metadata(episode_dir)
    start, end = action_config_range(metadata, fallback_length=len(states))
    camera_dir = episode_dir / "camera"
    written = 0
    for frame_index in range(start + 1, end):
        if not args.keep_static and np.allclose(states[frame_index - 1], states[frame_index], rtol=1e-4, atol=1e-4):
            continue
        if args.max_frames_per_episode is not None and written >= args.max_frames_per_episode:
            break
        if dataset is not None:
            dataset.add_frame(
                {
                    "head_color": read_image(
                        camera_dir, frame_index, "head_color", (args.image_width, args.image_height)
                    ),
                    "hand_right_color": read_image(
                        camera_dir, frame_index, "hand_right_color", (args.image_width, args.image_height)
                    ),
                    "hand_left_color": read_image(
                        camera_dir, frame_index, "hand_left_color", (args.image_width, args.image_height)
                    ),
                    "state": states[frame_index].astype(np.float32),
                    "actions": actions[frame_index].astype(np.float32),
                    "task": task_desc,
                    "skills": skill_string,
                }
            )
        written += 1
    return written


def convert_raw_episode(
    dataset,
    h5_path: Path,
    task_desc: str,
    skill_string: str,
    args: argparse.Namespace,
) -> int:
    import h5py

    np = require_numpy()
    with h5py.File(h5_path, "r") as f:
        raw_state = raw_joint_matrix(f)
        if len(raw_state) < 2:
            raise ValueError(f"Raw RoboTwin episode needs at least 2 frames: {h5_path}")
        states = raw_state[:-1].astype(np.float32)
        actions = raw_state[1:].astype(np.float32)
        written = 0
        for frame_index in range(len(states)):
            if not args.keep_static and frame_index > 0 and np.allclose(
                states[frame_index - 1], states[frame_index], rtol=1e-4, atol=1e-4
            ):
                continue
            if args.max_frames_per_episode is not None and written >= args.max_frames_per_episode:
                break
            if dataset is not None:
                dataset.add_frame(
                    {
                        "head_color": read_raw_image_from_root(
                            f, frame_index, "head_color", (args.image_width, args.image_height)
                        ),
                        "hand_right_color": read_raw_image_from_root(
                            f, frame_index, "hand_right_color", (args.image_width, args.image_height)
                        ),
                        "hand_left_color": read_raw_image_from_root(
                            f, frame_index, "hand_left_color", (args.image_width, args.image_height)
                        ),
                        "state": states[frame_index],
                        "actions": actions[frame_index],
                        "task": task_desc,
                        "skills": skill_string,
                    }
                )
            written += 1
    return written


def convert(args: argparse.Namespace) -> None:
    plan_data = json.loads(args.plan.read_text(encoding="utf-8"))
    task_names = select_tasks(args, plan_data)
    repo_id = args.output_repo_id or default_repo_id(args.task_set, args.tasks)
    extract_dir = args.extract_dir or (args.source_dir / "_extracted")

    dataset = None
    if not args.dry_run:
        output_path = lerobot_output_path(repo_id)
        if output_path.exists() and args.overwrite:
            shutil.rmtree(output_path)
        elif output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_path}. Use --overwrite to recreate it.")
        dataset = create_dataset(
            repo_id,
            fps=args.fps,
            image_shape=(args.image_height, args.image_width, 3),
        )

    total_episodes = 0
    total_frames = 0
    for task_name in task_names:
        archives = find_task_archives(args.source_dir, task_name)
        task_roots = find_extracted_task_roots(args.source_dir, task_name)
        for archive in archives:
            task_roots.append(extract_archive(archive, extract_dir, task_name))
        episode_sources = find_episode_sources(task_roots, args.source_format)
        if args.max_episodes_per_task is not None:
            episode_sources = episode_sources[: args.max_episodes_per_task]
        if not episode_sources:
            print(f"[WARN] No episodes found for task {task_name}")
            continue

        task_item = plan_data[task_name]
        task_desc = task_item["description"]
        skill_string = json.dumps(task_item.get("skills", []))
        source_counts: dict[str, int] = {"raw": 0, "aligned": 0}
        for source in episode_sources:
            if source.kind == "raw":
                written = convert_raw_episode(dataset, source.path, task_desc, skill_string, args)
            elif source.kind == "aligned":
                written = convert_aligned_episode(dataset, source.path, task_desc, skill_string, args)
            else:
                raise ValueError(f"Unknown episode source kind: {source.kind}")
            if written:
                if dataset is not None:
                    dataset.save_episode()
                total_episodes += 1
                total_frames += written
            source_counts[source.kind] += 1
        print(
            f"[INFO] {task_name}: {len(episode_sources)} episodes inspected "
            f"(raw={source_counts['raw']}, aligned={source_counts['aligned']})"
        )

    print(
        f"Converted {total_episodes} episodes / {total_frames} frames "
        f"to {repo_id if not args.dry_run else '[dry-run only]'}"
    )


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
