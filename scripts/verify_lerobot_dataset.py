"""Verify lightweight LeRobot dataset metadata before training or Hub upload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def summarize_dataset(dataset_root: Path) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    meta_dir = dataset_root / "meta"
    info_path = meta_dir / "info.json"
    tasks_path = meta_dir / "tasks.jsonl"
    episodes_path = meta_dir / "episodes.jsonl"
    stats_path = meta_dir / "episodes_stats.jsonl"

    missing = [path for path in [info_path, tasks_path, episodes_path, stats_path] if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path.relative_to(dataset_root)) for path in missing)
        raise FileNotFoundError(f"missing LeRobot metadata files under {dataset_root}: {missing_text}")

    info = read_json(info_path)
    features = sorted((info.get("features") or {}).keys())
    parquet_files = sorted((dataset_root / "data").glob("**/*.parquet"))
    return {
        "dataset_root": str(dataset_root),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "tasks": count_jsonl(tasks_path),
        "episodes": count_jsonl(episodes_path),
        "episodes_stats": count_jsonl(stats_path),
        "parquet_files": len(parquet_files),
        "features": features,
    }


def validate_summary(
    summary: dict[str, Any],
    *,
    expected_tasks: int | None,
    expected_episodes: int | None,
    expected_frames: int | None,
    require_features: list[str],
) -> list[str]:
    errors: list[str] = []
    checks = [
        ("tasks", expected_tasks),
        ("episodes", expected_episodes),
        ("total_episodes", expected_episodes),
        ("total_frames", expected_frames),
    ]
    for key, expected in checks:
        if expected is not None and summary.get(key) != expected:
            errors.append(f"{key}: expected {expected}, found {summary.get(key)}")

    features = set(summary.get("features") or [])
    for feature in require_features:
        if feature not in features:
            errors.append(f"missing required feature: {feature}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path, help="Path to a local LeRobot dataset directory.")
    parser.add_argument("--expected-tasks", type=int)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--require-feature", action="append", default=[])
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_dataset(args.dataset_root)
    errors = validate_summary(
        summary,
        expected_tasks=args.expected_tasks,
        expected_episodes=args.expected_episodes,
        expected_frames=args.expected_frames,
        require_features=args.require_feature,
    )

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"dataset_root: {summary['dataset_root']}")
        print(f"tasks: {summary['tasks']}")
        print(f"episodes: {summary['episodes']}")
        print(f"total_frames: {summary['total_frames']}")
        print(f"fps: {summary['fps']}")
        print(f"parquet_files: {summary['parquet_files']}")
        print("features: " + ", ".join(summary["features"]))

    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        raise SystemExit(1)
    print("LeRobot dataset metadata checks passed.")


if __name__ == "__main__":
    main()
