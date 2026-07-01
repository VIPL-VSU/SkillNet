# Support

SkillNet support is organized around the public documentation and GitHub issue
tracker.

## Getting Help

1. Start with `README.md` and `docs/quick_start.md`.
2. Check `docs/release_status.md` for known external asset gates, including
   derived LIBERO dataset visibility and repository-level GitHub settings.
3. For data processing questions, use `docs/libero_data_processing.md`,
   `docs/skill_hierarchy.md`, or `docs/robotwin_few_shot.md`.
4. For training and evaluation questions, use
   `docs/training_and_evaluation.md`.

If the documentation does not answer your question, open a GitHub issue with:

- The workflow you are trying to run.
- The command you ran.
- The relevant error message or log excerpt.
- Your operating system, Python version, and simulator version when applicable.
- Whether `python scripts/check_public_release.py --verbose` passes.

Do not include tokens, credentials, private dataset paths, machine-local
absolute paths, or unpublished checkpoint locations in issues.

## Scope

This repository provides source code, public data-processing utilities,
configuration files, and documentation for the SkillNet release. External
simulators, base checkpoints, and derived datasets may have their own access or
installation requirements; those gates are tracked in `docs/release_status.md`.
