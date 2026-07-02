# GitHub Repository Setup

This page tracks public GitHub settings that live outside the git tree. Keep
tokens out of command lines, scripts, and documentation; the checks below use
the public GitHub API anonymously by default and can optionally read a token
from `GITHUB_TOKEN` or `GH_TOKEN` when the public API rate limit is exhausted.

## Required Release Settings

Before announcing the release, the GitHub repository should have:

- Visibility: public.
- Default branch: `skillnet-public-release`, unless `main` is intentionally
  updated to the same release tree without reintroducing private or legacy
  history.
- Description: `SkillNet: skill-hierarchy-conditioned MoE policies for LIBERO, LIBERO-Skill, and RoboTwin few-shot transfer.`
- Homepage: `https://xsw1208.github.io/skillnet-website/`
- Topics: `skillnet`, `robot-learning`, `imitation-learning`, `libero`,
  `robotwin`, `mixture-of-experts`.
- The release-check badge visible at the top of `README.md`.

The repository should not present placeholder metadata such as `coming soon`
after the release branch is public.

## Read-Only Check

Run this from the repository root:

```bash
python scripts/check_github_repository_metadata.py --verbose
```

The same check is also available through the main release checker:

```bash
python scripts/check_public_release.py --github-metadata-smoke --skip-help --verbose
```

If the public API is rate-limited, set a read-only token in the environment and
rerun the same command. Do not put the token value in shell history:

```bash
read -rsp "GITHUB_TOKEN: " GITHUB_TOKEN
export GITHUB_TOKEN
echo
python scripts/check_github_repository_metadata.py --verbose
unset GITHUB_TOKEN
```

This check verifies the public repository metadata through GitHub's REST API.
It intentionally is not part of the default CI gate because repository settings
may require owner or admin permissions outside the source tree.
Before announcing the final external release state, use
`python scripts/check_public_release.py --external-release-smoke --skip-help --verbose`
to verify both GitHub metadata and public LIBERO derived-dataset visibility.

## Apply Settings

If you have repository admin permission, inspect the planned changes first:

```bash
python scripts/set_github_repository_metadata.py --dry-run --verbose
```

Then set a write-capable token through the environment and apply the release
metadata. Do not put the token value in command lines, scripts, docs, or git
history:

```bash
read -rsp "GITHUB_TOKEN: " GITHUB_TOKEN
export GITHUB_TOKEN
echo

python scripts/set_github_repository_metadata.py --verbose

unset GITHUB_TOKEN
python scripts/check_github_repository_metadata.py --verbose
```

The setter updates the repository description, homepage URL, default branch,
and topics. It reads tokens from `GITHUB_TOKEN` or `GH_TOKEN`; it never needs a
token value on the command line.

## Default Branch Note

The release branch keeps a clean public history checked by
`scripts/check_public_release.py --history-smoke`. If `main` has legacy commits
that are not ancestors of the release branch, do not merge them into the release
branch just to change the default view; doing so can make the history privacy
gate scan old, non-release content. Prefer switching the GitHub default branch
to `skillnet-public-release`, or update `main` only through a deliberate clean
release procedure.
