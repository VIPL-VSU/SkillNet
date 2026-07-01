"""Verify lightweight LeRobot dataset metadata before training or Hub upload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


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
    all_files = [path for path in dataset_root.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in all_files)
    return {
        "dataset_root": str(dataset_root),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "tasks": count_jsonl(tasks_path),
        "episodes": count_jsonl(episodes_path),
        "episodes_stats": count_jsonl(stats_path),
        "parquet_files": len(parquet_files),
        "files": len(all_files),
        "total_bytes": total_bytes,
        "human_total_bytes": human_bytes(total_bytes),
        "features": features,
    }


def summarize_parquet(dataset_root: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Strict parquet checks require pyarrow. Install pyarrow or omit --strict-parquet.") from exc

    parquet_files = sorted((dataset_root / "data").glob("**/*.parquet"))
    total_rows = 0
    columns: set[str] = set()
    empty_files: list[str] = []
    for path in parquet_files:
        parquet = pq.ParquetFile(path)
        rows = parquet.metadata.num_rows
        total_rows += rows
        columns.update(parquet.schema.names)
        if rows == 0:
            empty_files.append(path.relative_to(dataset_root).as_posix())
    return {
        "parquet_rows": total_rows,
        "parquet_columns": sorted(columns),
        "empty_parquet_files": empty_files[:20],
        "empty_parquet_file_count": len(empty_files),
    }


def validate_summary(
    summary: dict[str, Any],
    *,
    expected_tasks: int | None,
    expected_episodes: int | None,
    expected_frames: int | None,
    require_features: list[str],
    strict_parquet: bool,
    require_columns: list[str],
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

    if summary.get("parquet_files") == 0:
        errors.append("parquet_files: expected at least 1, found 0")
    episodes = summary.get("episodes")
    episodes_stats = summary.get("episodes_stats")
    if episodes is not None and episodes_stats is not None and episodes_stats != episodes:
        errors.append(f"episodes_stats: expected {episodes}, found {episodes_stats}")

    if strict_parquet:
        parquet_rows = summary.get("parquet_rows")
        expected_rows = expected_frames if expected_frames is not None else summary.get("total_frames")
        if expected_rows is not None and parquet_rows != expected_rows:
            errors.append(f"parquet_rows: expected {expected_rows}, found {parquet_rows}")
        if summary.get("empty_parquet_file_count"):
            errors.append(f"empty parquet files found: {summary['empty_parquet_file_count']}")
        columns = set(summary.get("parquet_columns") or [])
        for column in require_columns:
            if column not in columns:
                errors.append(f"missing required parquet column: {column}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path, help="Path to a local LeRobot dataset directory.")
    parser.add_argument("--expected-tasks", type=int)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--require-feature", action="append", default=[])
    parser.add_argument(
        "--strict-parquet",
        action="store_true",
        help="Read parquet metadata with pyarrow and verify row counts against total_frames.",
    )
    parser.add_argument("--require-column", action="append", default=[], help="Required parquet column for --strict-parquet.")
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_dataset(args.dataset_root)
    if args.strict_parquet:
        summary.update(summarize_parquet(Path(summary["dataset_root"])))
    errors = validate_summary(
        summary,
        expected_tasks=args.expected_tasks,
        expected_episodes=args.expected_episodes,
        expected_frames=args.expected_frames,
        require_features=args.require_feature,
        strict_parquet=args.strict_parquet,
        require_columns=args.require_column,
    )

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"dataset_root: {summary['dataset_root']}")
        print(f"tasks: {summary['tasks']}")
        print(f"episodes: {summary['episodes']}")
        print(f"total_frames: {summary['total_frames']}")
        print(f"fps: {summary['fps']}")
        print(f"files: {summary['files']}")
        print(f"size: {summary['human_total_bytes']}")
        print(f"parquet_files: {summary['parquet_files']}")
        if args.strict_parquet:
            print(f"parquet_rows: {summary['parquet_rows']}")
            print("parquet_columns: " + ", ".join(summary["parquet_columns"]))
        print("features: " + ", ".join(summary["features"]))

    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        raise SystemExit(1)
    print("LeRobot dataset metadata checks passed.")


if __name__ == "__main__":
    main()
