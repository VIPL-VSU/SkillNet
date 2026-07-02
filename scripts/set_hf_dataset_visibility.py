"""Set Hugging Face dataset repository visibility without exposing tokens.

The script reads credentials from an environment variable and never accepts a
token on the command line. Use it when a derived LeRobot dataset has already
been uploaded but still needs to be made public for the release.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_ids", nargs="+", help="Hugging Face dataset repo ids, for example jsw19/libero_40_v1.")
    visibility = parser.add_mutually_exclusive_group(required=True)
    visibility.add_argument("--public", action="store_true", help="Set dataset repos to public.")
    visibility.add_argument("--private", action="store_true", help="Set dataset repos to private.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned visibility changes without writing.")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT"), help="Optional Hugging Face endpoint.")
    parser.add_argument("--token-env", default="HF_TOKEN", help="Environment variable containing a write-capable token.")
    parser.add_argument(
        "--skip-anonymous-check",
        action="store_true",
        help="Skip the anonymous read check after setting repos public.",
    )
    return parser.parse_args()


def token_from_env(token_env: str) -> str | None:
    token = os.environ.get(token_env)
    if token is None and token_env == "HF_TOKEN":
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    return token


def load_api(*, endpoint: str | None, token: str | None):
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Please install huggingface_hub before changing Hub dataset visibility.") from exc

    return HfApi(endpoint=endpoint, token=token) if endpoint else HfApi(token=token)


def visibility(info) -> str:
    private = getattr(info, "private", None)
    if private is None:
        return "unknown"
    return "private" if private else "public"


def inspect_repo(api, repo_id: str):
    try:
        return api.repo_info(repo_id=repo_id, repo_type="dataset")
    except Exception as exc:
        raise SystemExit(f"Could not inspect dataset repo {repo_id}: {exc}") from exc


def try_inspect_repo(api, repo_id: str):
    try:
        return api.repo_info(repo_id=repo_id, repo_type="dataset"), None
    except Exception as exc:  # pragma: no cover - exact exception class depends on huggingface_hub version
        return None, exc


def anonymous_public_check(repo_ids: Iterable[str], *, endpoint: str | None) -> None:
    anon_api = load_api(endpoint=endpoint, token=None)
    for repo_id in repo_ids:
        try:
            info = anon_api.repo_info(repo_id=repo_id, repo_type="dataset")
        except Exception as exc:
            raise SystemExit(f"Anonymous read check failed for dataset repo {repo_id}: {exc}") from exc
        if getattr(info, "private", None):
            raise SystemExit(f"Anonymous read check still reports private repo: {repo_id}")
        print(f"anonymous_read: {repo_id} ok")


def main() -> None:
    args = parse_args()
    requested_private = args.private
    requested_visibility = "private" if requested_private else "public"
    token = token_from_env(args.token_env)
    if not token and not args.dry_run:
        raise SystemExit(
            f"Missing {args.token_env}. Set a write-capable Hub token in the environment; "
            "do not put tokens in command lines or docs."
        )

    try:
        api = load_api(endpoint=args.endpoint, token=token)
        api_error = None
    except SystemExit as exc:
        if not args.dry_run:
            raise
        api = None
        api_error = exc
    for repo_id in args.repo_ids:
        if args.dry_run:
            before, inspect_error = (None, api_error) if api is None else try_inspect_repo(api, repo_id)
            if before is None:
                print(f"{repo_id}: current=unavailable target={requested_visibility}")
                print(f"{repo_id}: inspect_error={inspect_error}")
            else:
                print(f"{repo_id}: current={visibility(before)} target={requested_visibility}")
        else:
            before = inspect_repo(api, repo_id)
            print(f"{repo_id}: current={visibility(before)} target={requested_visibility}")
        if args.dry_run:
            continue
        try:
            api.update_repo_settings(repo_id=repo_id, repo_type="dataset", private=requested_private)
        except Exception as exc:
            raise SystemExit(f"Could not update dataset repo visibility for {repo_id}: {exc}") from exc
        after = inspect_repo(api, repo_id)
        print(f"{repo_id}: updated={visibility(after)}")

    if args.dry_run:
        print("Dry run only; no Hub writes were performed.")
        if not token:
            print(
                f"Set {args.token_env}"
                + (" or HUGGINGFACE_HUB_TOKEN" if args.token_env == "HF_TOKEN" else "")
                + " before applying the visibility change."
            )
        return

    if args.public and not args.skip_anonymous_check:
        anonymous_public_check(args.repo_ids, endpoint=args.endpoint)


if __name__ == "__main__":
    main()
