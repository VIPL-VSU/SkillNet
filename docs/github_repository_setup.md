# GitHub Repository Setup

This page tracks public GitHub settings that live outside the git tree. Keep
tokens out of command lines, scripts, and documentation; the checks below use
the public GitHub API and do not require credentials.

## Required Release Settings

Before announcing the release, the GitHub repository should have:

- Visibility: public.
- Default branch: `skillnet-public-release`, unless `main` is intentionally
  updated to the same release tree without reintroducing private or legacy
  history.
- Description: `SkillNet: skill-hierarchy-conditioned MoE policies for LIBERO, LIBERO-Skill, and RoboTwin few-shot transfer.`
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

This check verifies the public repository metadata through GitHub's REST API.
It intentionally is not part of the default CI gate because repository settings
may require owner or admin permissions outside the source tree.

## Default Branch Note

The release branch keeps a clean public history checked by
`scripts/check_public_release.py --history-smoke`. If `main` has legacy commits
that are not ancestors of the release branch, do not merge them into the release
branch just to change the default view; doing so can make the history privacy
gate scan old, non-release content. Prefer switching the GitHub default branch
to `skillnet-public-release`, or update `main` only through a deliberate clean
release procedure.
