# Contributing

Thank you for your interest in improving SkillNet. This release is organized
around reproducible LIBERO, LIBERO-Skill, skill-hierarchy, and RoboTwin
few-shot workflows.

## Before Opening a Change

- Read `README.md`, `docs/quick_start.md`, and `docs/release_status.md` to
  understand the current release scope and known external asset gates.
- Keep public workflows under the SkillNet naming used in the documentation.
- Do not add API keys, tokens, credentials, private dataset paths, machine-local absolute paths, or cluster-specific launch details.
- Do not commit generated datasets, checkpoints, simulator caches, or other
  large binary artifacts. Use documented Hugging Face or local data-generation
  flows instead.
- Keep changes focused on one workflow or documentation surface at a time.

## Local Checks

Run the lightweight public release check before proposing changes:

```bash
python scripts/check_public_release.py --verbose
```

For release-facing changes, also run:

```bash
python -m pip install -U pyyaml
python scripts/check_public_release.py --history-smoke --skip-help --verbose
python scripts/check_public_release.py --yaml-smoke --skip-help --verbose
python scripts/check_public_release.py --hub-smoke --skip-help --verbose --hub-retries 3 --hub-timeout 30
```

On Linux or WSL, run the shell syntax gate:

```bash
python scripts/check_public_release.py --require-bash --skip-help --verbose
```

If the change touches package metadata or import paths, run the no-deps install
smoke documented in `docs/quick_start.md`.

## Documentation Changes

Documentation should be runnable from a fresh clone and should use environment
variables for user-specific locations. If a workflow depends on external
simulators, checkpoint visibility, or derived datasets, state that dependency
explicitly and point to the matching verification command.

## Pull Request Checklist

- The relevant release checks pass locally.
- New public files are included in `scripts/check_public_release.py` when they
  are part of the release contract.
- Public docs do not use private paths, placeholders, or legacy naming.
- Large artifacts are referenced through documented download or generation
  steps instead of being committed.
