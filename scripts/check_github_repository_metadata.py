"""Check public GitHub repository metadata for the SkillNet release.

This script is intentionally read-only and uses GitHub's public REST API. By
default it runs anonymously; set GITHUB_TOKEN or GH_TOKEN when the public API
rate limit is exhausted. It is meant for final release setup checks that cannot
be represented inside the git tree itself, such as the repository description,
topics, and default branch.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


DEFAULT_DESCRIPTION = (
    "SkillNet: skill-hierarchy-conditioned MoE policies for LIBERO, "
    "LIBERO-Skill, and RoboTwin few-shot transfer."
)
DEFAULT_TOPICS = (
    "skillnet",
    "robot-learning",
    "imitation-learning",
    "libero",
    "robotwin",
    "mixture-of-experts",
)
DEFAULT_HOMEPAGE = "https://xsw1208.github.io/skillnet-website/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="VIPL-VSU/SkillNet", help="GitHub repository in owner/name form.")
    parser.add_argument(
        "--api-url",
        default="https://api.github.com",
        help="GitHub API base URL. Defaults to https://api.github.com.",
    )
    parser.add_argument(
        "--expected-default-branch",
        default="skillnet-public-release",
        help="Expected default branch for public release browsing.",
    )
    parser.add_argument(
        "--expected-description",
        default=DEFAULT_DESCRIPTION,
        help="Exact expected public repository description.",
    )
    parser.add_argument(
        "--expected-homepage",
        default=DEFAULT_HOMEPAGE,
        help="Exact expected public repository homepage URL.",
    )
    parser.add_argument(
        "--expected-topic",
        action="append",
        default=None,
        help="Expected GitHub topic. Repeat to override the default topic list.",
    )
    parser.add_argument(
        "--allow-description",
        action="append",
        default=[],
        help="Additional allowed repository description. Repeat as needed.",
    )
    parser.add_argument(
        "--reject-description",
        action="append",
        default=["coming soon", ""],
        help="Description value that should fail the release metadata check.",
    )
    parser.add_argument(
        "--github-token-env",
        default="GITHUB_TOKEN",
        help="Optional environment variable containing a GitHub token for higher API rate limits.",
    )
    parser.add_argument("--timeout", type=float, default=20.0, help="GitHub API request timeout.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def github_token(token_env: str) -> str | None:
    import os

    token = os.environ.get(token_env)
    if token is None and token_env == "GITHUB_TOKEN":
        token = os.environ.get("GH_TOKEN")
    return token


def fetch_repo(api_url: str, repo: str, timeout: float, token: str | None) -> dict:
    url = f"{api_url.rstrip('/')}/repos/{repo}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "skillnet-repository-metadata-check/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code in {403, 429} and token is None:
            raise SystemExit(
                f"GitHub API returned HTTP {exc.code} for {url}: {detail}\n"
                "If this is a rate-limit response, set GITHUB_TOKEN or GH_TOKEN in the environment "
                "and rerun without putting the token on the command line."
            ) from exc
        raise SystemExit(f"GitHub API returned HTTP {exc.code} for {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"GitHub API request failed for {url}: {exc}") from exc


def ok(message: str, *, verbose: bool) -> None:
    if verbose:
        print(f"[ OK ] {message}")


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)
    print(f"[FAIL] {message}")


def main() -> None:
    args = parse_args()
    token = github_token(args.github_token_env)
    repo = fetch_repo(args.api_url, args.repo, args.timeout, token)
    errors: list[str] = []
    expected_topics = set(args.expected_topic or DEFAULT_TOPICS)

    if repo.get("private") is False:
        ok(f"{args.repo} is public", verbose=args.verbose)
    else:
        fail(f"{args.repo} is not public", errors)

    default_branch = repo.get("default_branch")
    if default_branch == args.expected_default_branch:
        ok(f"default branch is {default_branch}", verbose=args.verbose)
    else:
        fail(
            f"default branch is {default_branch!r}; expected {args.expected_default_branch!r}",
            errors,
        )

    description = repo.get("description") or ""
    rejected_descriptions = {value.strip().lower() for value in args.reject_description}
    allowed_descriptions = {args.expected_description, *args.allow_description}
    if description.strip().lower() in rejected_descriptions:
        fail(f"repository description is still a placeholder: {description!r}", errors)
    elif description not in allowed_descriptions:
        fail(
            f"repository description is {description!r}; expected one of {sorted(allowed_descriptions)!r}",
            errors,
        )
    else:
        ok("repository description matches the release metadata contract", verbose=args.verbose)

    homepage = repo.get("homepage") or ""
    if homepage.rstrip("/") != args.expected_homepage.rstrip("/"):
        fail(
            f"repository homepage is {homepage!r}; expected {args.expected_homepage!r}",
            errors,
        )
    else:
        ok("repository homepage matches the release metadata contract", verbose=args.verbose)

    topics = set(repo.get("topics") or [])
    missing_topics = sorted(expected_topics - topics)
    if missing_topics:
        fail(f"missing GitHub topics: {missing_topics}", errors)
    else:
        ok("repository topics include the release metadata contract", verbose=args.verbose)

    if errors:
        raise SystemExit(1)
    print("SkillNet GitHub repository metadata checks passed.")


if __name__ == "__main__":
    main()
