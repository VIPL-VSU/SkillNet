"""Publish a local LeRobot dataset folder to Hugging Face Hub.

This script intentionally uses Hugging Face Hub's large-folder uploader for
derived SkillNet datasets. The LIBERO v1 LeRobot exports are tens of GiB, so a
single ordinary folder upload is too fragile for the public release workflow.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from verify_lerobot_dataset import summarize_dataset, summarize_parquet, validate_summary


def visibility_label(private: bool) -> str:
    return "private" if private else "public"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path, help="Path to a local LeRobot dataset directory.")
    parser.add_argument("--repo-id", required=True, help="Target Hugging Face dataset repo id.")
    visibility = parser.add_mutually_exclusive_group(required=True)
    visibility.add_argument("--public", action="store_true", help="Create or verify a public dataset repo.")
    visibility.add_argument("--private", action="store_true", help="Create or verify a private dataset repo.")
    parser.add_argument(
        "--allow-existing-visibility",
        action="store_true",
        help="Continue if the target repo already exists with a different visibility.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only verify the local dataset and print the upload plan.")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT"), help="Optional Hugging Face endpoint.")
    parser.add_argument("--token-env", default="HF_TOKEN", help="Environment variable that contains the Hub token.")
    parser.add_argument(
        "--skip-hub-preflight",
        action="store_true",
        help="Skip Hub client/uploader/permission checks during a dry run. Local dataset checks still run.",
    )
    parser.add_argument("--num-workers", type=int, help="Worker count passed to upload_large_folder.")
    parser.add_argument("--print-report-every", type=int, default=60, help="Seconds between upload progress reports.")
    parser.add_argument("--revision", help="Optional Hub revision/branch.")
    parser.add_argument(
        "--allow-missing-card",
        action="store_true",
        help="Allow upload when the dataset root does not contain README.md.",
    )
    parser.add_argument("--ignore-pattern", action="append", default=[".cache/*", "__pycache__/*"])
    parser.add_argument("--expected-tasks", type=int)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--require-feature", action="append", default=[])
    parser.add_argument(
        "--strict-parquet",
        action="store_true",
        help="Also verify parquet row counts and required parquet columns before upload.",
    )
    parser.add_argument("--require-column", action="append", default=[], help="Required parquet column for --strict-parquet.")
    return parser.parse_args()


def load_hub_api(endpoint: str | None, token: str | None):
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Please install huggingface_hub before publishing datasets.") from exc

    return HfApi(endpoint=endpoint, token=token) if endpoint else HfApi(token=token)


def ensure_large_folder_uploader(api) -> None:
    if getattr(api, "upload_large_folder", None) is None:
        raise SystemExit("huggingface_hub.HfApi.upload_large_folder is required. Upgrade huggingface_hub and install hf_xet.")


def hub_token(token_env: str) -> str | None:
    token = os.environ.get(token_env)
    if token is None and token_env == "HF_TOKEN":
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    return token


def repo_not_found(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 404:
        return True
    text = str(exc).lower()
    return "404" in text or "repository not found" in text or "not found" in text


def get_dataset_repo_info(api, repo_id: str):
    try:
        return api.repo_info(repo_id=repo_id, repo_type="dataset")
    except Exception as exc:  # huggingface_hub versions expose different error classes.
        if repo_not_found(exc):
            return None
        raise SystemExit(f"Could not inspect dataset repo {repo_id}: {exc}") from exc


def check_repo_visibility(*, repo_info, requested_private: bool, allow_mismatch: bool, repo_id: str) -> None:
    if repo_info is None:
        print(f"hub_repo: {repo_id} does not exist yet")
        return

    existing_private = getattr(repo_info, "private", None)
    if existing_private is None:
        print(f"hub_repo: {repo_id} exists; visibility could not be read from this hub client")
        return

    existing_visibility = visibility_label(bool(existing_private))
    requested_visibility = visibility_label(requested_private)
    print(f"hub_repo: {repo_id} exists as {existing_visibility}")
    if bool(existing_private) == requested_private:
        return

    message = (
        f"Target dataset repo {repo_id} already exists as {existing_visibility}, "
        f"but this run requested {requested_visibility}."
    )
    if allow_mismatch:
        print(f"[WARN] {message}")
        return
    raise SystemExit(message + " Pass --allow-existing-visibility only after confirming this is intentional.")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    summary = summarize_dataset(dataset_root)
    if args.strict_parquet:
        summary.update(summarize_parquet(dataset_root))
    errors = validate_summary(
        summary,
        expected_tasks=args.expected_tasks,
        expected_episodes=args.expected_episodes,
        expected_frames=args.expected_frames,
        require_features=args.require_feature,
        strict_parquet=args.strict_parquet,
        require_columns=args.require_column,
    )
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        raise SystemExit(1)

    requested_private = args.private
    card_path = dataset_root / "README.md"
    if not card_path.exists():
        message = f"Dataset card missing: {card_path}"
        if args.allow_missing_card:
            print(f"[WARN] {message}")
        elif args.dry_run and requested_private:
            print(f"[WARN] {message}")
        else:
            raise SystemExit(message + " Add README.md or pass --allow-missing-card for a staged/internal upload.")

    print(f"dataset_root: {dataset_root}")
    print(f"repo_id: {args.repo_id}")
    print(f"tasks: {summary['tasks']}")
    print(f"episodes: {summary['episodes']}")
    print(f"total_frames: {summary['total_frames']}")
    print(f"files: {summary['files']}")
    print(f"size: {summary['human_total_bytes']}")
    print(f"visibility: {visibility_label(requested_private)}")
    if args.strict_parquet:
        print(f"parquet_rows: {summary['parquet_rows']}")

    token = hub_token(args.token_env)
    api = None
    repo_info = None
    if args.dry_run and args.skip_hub_preflight:
        print("Hub preflight skipped by --skip-hub-preflight.")
    elif not token:
        raise SystemExit(
            f"Missing {args.token_env} for Hub preflight. "
            "Set the token or pass --skip-hub-preflight for a local-only dry run."
        )
    elif token or not requested_private:
        try:
            api = load_hub_api(args.endpoint, token)
            ensure_large_folder_uploader(api)
            repo_info = get_dataset_repo_info(api, args.repo_id)
            check_repo_visibility(
                repo_info=repo_info,
                requested_private=requested_private,
                allow_mismatch=args.allow_existing_visibility,
                repo_id=args.repo_id,
            )
        except SystemExit as exc:
            raise

    if args.dry_run:
        print("Dry run only; no Hub writes were performed.")
        return

    if not token:
        raise SystemExit(f"Missing Hugging Face token environment variable: {args.token_env}")
    if api is None:
        api = load_hub_api(args.endpoint, token)
        repo_info = get_dataset_repo_info(api, args.repo_id)
        check_repo_visibility(
            repo_info=repo_info,
            requested_private=requested_private,
            allow_mismatch=args.allow_existing_visibility,
            repo_id=args.repo_id,
        )

    if repo_info is None:
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=requested_private, exist_ok=False)
        print(f"Created {visibility_label(requested_private)} dataset repo {args.repo_id}.")

    ensure_large_folder_uploader(api)
    upload_large_folder = api.upload_large_folder

    upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(dataset_root),
        revision=args.revision,
        private=requested_private,
        ignore_patterns=args.ignore_pattern,
        num_workers=args.num_workers,
        print_report_every=args.print_report_every,
    )
    print(f"Uploaded {dataset_root} to dataset repo {args.repo_id}.")


if __name__ == "__main__":
    main()
