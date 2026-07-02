## Summary

- 

## Scope

- [ ] Quick start / install
- [ ] Skill hierarchy
- [ ] LIBERO data processing
- [ ] LIBERO training or evaluation
- [ ] LIBERO-Skill zero-shot evaluation
- [ ] RoboTwin few-shot transfer
- [ ] Release checks, docs, or repository metadata

## Checks

- [ ] `python scripts/check_public_release.py --verbose`
- [ ] `python scripts/check_public_release.py --history-smoke --skip-help --verbose`
- [ ] `python scripts/check_public_release.py --hub-smoke --skip-help --verbose --hub-retries 3 --hub-timeout 30`
- [ ] `python scripts/check_public_release.py --require-bash --skip-help --verbose` on Linux or WSL, if shell scripts changed

## Sensitive Information

- [ ] This change does not add API keys, tokens, credentials, private dataset paths, machine-local absolute paths, cluster-specific account details, or unpublished checkpoint locations.
- [ ] Large artifacts are referenced through documented download or generation steps rather than committed to git.
