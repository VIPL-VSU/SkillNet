"""Set public GitHub repository metadata for the SkillNet release.

The script reads a write-capable GitHub token from GITHUB_TOKEN or GH_TOKEN. Do
not pass tokens on the command line. Use --dry-run first to inspect the planned
changes.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from typing import Any

from check_github_repository_metadata import DEFAULT_DESCRIPTION
from check_github_repository_metadata import DEFAULT_HOMEPAGE
from check_github_repository_metadata import DEFAULT_TOPICS
from check_github_repository_metadata import fetch_repo
from check_github_repository_metadata import github_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="VIPL-VSU/SkillNet", help="GitHub repository in owner/name form.")
    parser.add_argument("--api-url", default="https://api.github.com", help="GitHub API base URL.")
    parser.add_argument("--default-branch", default="skillnet-public-release")
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION)
    parser.add_argument("--homepage", default=DEFAULT_HOMEPAGE)
    parser.add_argument("--topic", action="append", default=None, help="Topic to set. Repeat to override defaults.")
    parser.add_argument("--github-token-env", default="GITHUB_TOKEN")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def github_request(
    *,
    api_url: str,
    repo: str,
    path: str = "",
    method: str,
    token: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    url = f"{api_url.rstrip('/')}/repos/{repo}{path}"
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "skillnet-repository-metadata-set/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"GitHub API returned HTTP {exc.code} for {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"GitHub API request failed for {url}: {exc}") from exc


def planned_changes(repo_state: dict[str, Any], desired: dict[str, Any]) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    for key, value in desired.items():
        current = repo_state.get(key)
        if key == "homepage":
            current = (current or "").rstrip("/")
            compare_value = str(value).rstrip("/")
        else:
            compare_value = value
        if current != compare_value:
            changes[key] = {"current": repo_state.get(key), "desired": value}
    return changes


def main() -> None:
    args = parse_args()
    topics = list(args.topic or DEFAULT_TOPICS)
    token = github_token(args.github_token_env)
    repo_state = fetch_repo(args.api_url, args.repo, args.timeout, token=token)

    repo_payload = {
        "description": args.description,
        "homepage": args.homepage,
        "default_branch": args.default_branch,
    }
    topic_payload = {"names": topics}
    changes = planned_changes(repo_state, repo_payload)
    missing_topics = sorted(set(topics) - set(repo_state.get("topics") or []))

    if args.dry_run:
        print(json.dumps({"repo": args.repo, "metadata_changes": changes, "missing_topics": missing_topics}, indent=2))
        return

    if not token:
        env_hint = args.github_token_env
        fallback = " or GH_TOKEN" if env_hint == "GITHUB_TOKEN" else ""
        raise SystemExit(f"Missing {env_hint}{fallback}. Set a write-capable GitHub token in the environment.")

    if changes:
        github_request(
            api_url=args.api_url,
            repo=args.repo,
            method="PATCH",
            token=token,
            payload=repo_payload,
            timeout=args.timeout,
        )
        if args.verbose:
            print(f"Updated repository metadata fields: {sorted(changes)}")
    elif args.verbose:
        print("Repository metadata fields already match.")

    if missing_topics:
        github_request(
            api_url=args.api_url,
            repo=args.repo,
            path="/topics",
            method="PUT",
            token=token,
            payload=topic_payload,
            timeout=args.timeout,
        )
        if args.verbose:
            print(f"Updated repository topics: {topics}")
    elif args.verbose:
        print("Repository topics already include the release topic set.")

    print("SkillNet GitHub repository metadata update finished.")


if __name__ == "__main__":
    main()
