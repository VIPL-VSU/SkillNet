"""Publish a local LeRobot dataset folder to Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from verify_lerobot_dataset import summarize_dataset, validate_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path, help="Path to a local LeRobot dataset directory.")
    parser.add_argument("--repo-id", required=True, help="Target Hugging Face dataset repo id.")
    parser.add_argument("--private", action="store_true", help="Create or keep the target dataset repo private.")
    parser.add_argument("--dry-run", action="store_true", help="Only verify the local dataset and print the upload plan.")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT"), help="Optional Hugging Face endpoint.")
    parser.add_argument("--token-env", default="HF_TOKEN", help="Environment variable that contains the Hub token.")
    parser.add_argument("--commit-message", default="Upload SkillNet LeRobot dataset")
    parser.add_argument("--expected-tasks", type=int)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--require-feature", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    summary = summarize_dataset(dataset_root)
    errors = validate_summary(
        summary,
        expected_tasks=args.expected_tasks,
        expected_episodes=args.expected_episodes,
        expected_frames=args.expected_frames,
        require_features=args.require_feature,
    )
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        raise SystemExit(1)

    print(f"dataset_root: {dataset_root}")
    print(f"repo_id: {args.repo_id}")
    print(f"tasks: {summary['tasks']}")
    print(f"episodes: {summary['episodes']}")
    print(f"total_frames: {summary['total_frames']}")
    print(f"private: {args.private}")
    if args.dry_run:
        print("Dry run only; no Hub writes were performed.")
        return

    token = os.environ.get(args.token_env)
    if not token:
        raise SystemExit(f"Missing Hugging Face token environment variable: {args.token_env}")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Please install huggingface_hub before publishing datasets.") from exc

    api = HfApi(endpoint=args.endpoint, token=token) if args.endpoint else HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(dataset_root),
        commit_message=args.commit_message,
        ignore_patterns=[".cache/*", "__pycache__/*"],
    )
    print(f"Uploaded {dataset_root} to dataset repo {args.repo_id}.")


if __name__ == "__main__":
    main()
