"""Export compact SkillNet LIBERO slice metadata from a LeRobot v1 dataset.

The exporter is useful when the large derived LeRobot dataset exists locally
but you want to publish a small metadata file that lets others rebuild the same
frame-level skill labels from the public RLDS source order.
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import re
from pathlib import Path
from typing import Any


EPISODE_RE = re.compile(r"episode_(\d+)\.parquet$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path, help="Local LeRobot dataset root, for example $LEROBOT_HOME/jsw19/libero_40_v1.")
    parser.add_argument("--output", type=Path, required=True, help="Output slice-index JSON path.")
    parser.add_argument("--repo-id", default=None, help="Canonical LeRobot repo_id to store in the metadata.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation. Use 0 for compact output.")
    parser.add_argument(
        "--limit-episodes",
        type=int,
        default=None,
        help="Export only the first N reconstructed source episodes for smoke tests.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parquet_files_by_episode(dataset_root: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path_str in glob.glob(str(dataset_root / "data" / "chunk-*" / "episode_*.parquet")):
        path = Path(path_str)
        match = EPISODE_RE.search(path.name)
        if not match:
            continue
        files[int(match.group(1))] = path
    return files


def to_plain(value: Any) -> Any:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, tuple):
        value = list(value)
    return value


def stable_key(value: Any) -> str:
    return json.dumps(to_plain(value), sort_keys=True, ensure_ascii=True)


def normalize_class(value: Any) -> int | list[int]:
    value = to_plain(value)
    if isinstance(value, list):
        return [int(item) for item in value]
    return int(value)


def parse_class_sequence(value: Any) -> list[int]:
    value = to_plain(value)
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, list):
        raise ValueError(f"all_classes must decode to a list, got {value!r}")
    return [int(item) for item in value]


def read_segments(parquet_path: Path) -> tuple[int, list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    schema_names = set(pq.read_schema(parquet_path).names)
    columns = ["class", "all_classes"]
    if "objects" in schema_names:
        columns.append("objects")
    table = pq.read_table(parquet_path, columns=columns)
    rows = table.to_pylist()

    segments: list[dict[str, Any]] = []
    current_key: tuple[str, str, str | None] | None = None
    current_segment: dict[str, Any] | None = None

    for frame_offset, row in enumerate(rows):
        class_value = normalize_class(row["class"])
        all_classes = to_plain(row["all_classes"])
        objects = to_plain(row["objects"]) if "objects" in row else None
        key = (stable_key(class_value), stable_key(all_classes), stable_key(objects))
        if key != current_key:
            if current_segment is not None:
                current_segment["end"] = frame_offset
                segments.append(current_segment)
            current_segment = {
                "start": frame_offset,
                "class": class_value,
                "all_classes": all_classes,
            }
            if objects is not None:
                current_segment["objects"] = objects
            current_key = key

    if current_segment is not None:
        current_segment["end"] = len(rows)
        segments.append(current_segment)
    if not segments:
        raise ValueError(f"No segments exported for {parquet_path}")
    return len(rows), segments


def export_slice_index(dataset_root: Path, output: Path, repo_id: str | None, indent: int, limit_episodes: int | None) -> None:
    try:
        import pyarrow.parquet  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Please install pyarrow before exporting LIBERO slice metadata.") from exc

    dataset_root = dataset_root.expanduser().resolve()
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing LeRobot episodes metadata: {episodes_path}")

    episodes_meta = load_jsonl(episodes_path)
    parquet_by_episode = parquet_files_by_episode(dataset_root)
    exported_episodes: list[dict[str, Any]] = []
    current_episode: dict[str, Any] | None = None
    current_key: tuple[str, tuple[int, ...], str | None] | None = None
    current_cursor = 0

    def flush_current() -> None:
        nonlocal current_episode, current_key, current_cursor
        if current_episode is None:
            return
        expected = current_episode.pop("_expected_classes")
        if current_cursor != len(expected):
            raise ValueError(
                "Cannot flush incomplete reconstructed source episode: "
                f"task={current_episode.get('task')!r}, cursor={current_cursor}, expected={expected}"
            )
        current_episode["episode_index"] = len(exported_episodes)
        exported_episodes.append(current_episode)
        current_episode = None
        current_key = None
        current_cursor = 0

    for episode_meta in episodes_meta:
        if limit_episodes is not None and len(exported_episodes) >= limit_episodes:
            break
        episode_index = int(episode_meta["episode_index"])
        parquet_path = parquet_by_episode.get(episode_index)
        if parquet_path is None:
            raise FileNotFoundError(f"Missing parquet for episode_index={episode_index}")

        episode_length, segments = read_segments(parquet_path)
        if episode_length != int(episode_meta["length"]):
            raise ValueError(
                f"Episode length mismatch for {episode_index}: "
                f"{episode_length} rows != {episode_meta['length']} in metadata"
            )

        task = episode_meta["tasks"][0] if episode_meta.get("tasks") else None
        for segment in segments:
            class_value = normalize_class(segment["class"])
            if isinstance(class_value, list):
                if len(class_value) != 1:
                    raise ValueError(f"Expected one flat class id per derived episode, got {class_value!r}")
                class_value = class_value[0]
            expected_classes = parse_class_sequence(segment["all_classes"])
            key = (task, tuple(expected_classes), stable_key(segment.get("objects")))
            if current_episode is None:
                current_episode = {
                    "task": task,
                    "length": 0,
                    "segments": [],
                    "_expected_classes": expected_classes,
                }
                current_key = key
                current_cursor = 0
            elif key != current_key:
                flush_current()
                current_episode = {
                    "task": task,
                    "length": 0,
                    "segments": [],
                    "_expected_classes": expected_classes,
                }
                current_key = key

            if class_value != expected_classes[current_cursor]:
                if current_episode["segments"] and class_value == expected_classes[0]:
                    flush_current()
                    current_episode = {
                        "task": task,
                        "length": 0,
                        "segments": [],
                        "_expected_classes": expected_classes,
                    }
                    current_key = key
                else:
                    raise ValueError(
                        f"Class sequence mismatch at derived episode {episode_index}: "
                        f"got {class_value}, expected {expected_classes[current_cursor]} "
                        f"from {expected_classes}"
                    )

            start = current_episode["length"]
            end = start + int(segment["end"]) - int(segment["start"])
            output_segment = {
                "start": start,
                "end": end,
                "class": class_value,
                "all_classes": segment["all_classes"],
            }
            if "objects" in segment:
                output_segment["objects"] = segment["objects"]
            current_episode["segments"].append(output_segment)
            current_episode["length"] = end
            current_cursor += 1
            if current_cursor == len(expected_classes):
                flush_current()

    if limit_episodes is None and current_episode is not None:
        flush_current()

    payload = {
        "format": "skillnet_libero_slice_index_v1",
        "repo_id": repo_id or dataset_root.name,
        "num_episodes": len(exported_episodes),
        "is_partial": limit_episodes is not None,
        "episodes": exported_episodes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    json_kwargs = {} if indent == 0 else {"indent": indent}
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, **json_kwargs)
        f.write("\n")


def main() -> None:
    args = parse_args()
    export_slice_index(args.dataset_root, args.output, args.repo_id, args.indent, args.limit_episodes)


if __name__ == "__main__":
    main()
