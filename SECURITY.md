# Security Policy

## Reporting a Vulnerability

Please do not open a public issue containing credentials, private paths, or
other sensitive details.

If GitHub private vulnerability reporting is enabled for this repository, use
the repository security advisory flow. If private reporting is not available,
open a minimal public issue asking maintainers to establish a private contact
channel. Do not include exploit details until a private channel is available.

## Sensitive Information

Never include the following in issues, pull requests, logs, scripts, or
documentation:

- API keys, tokens, cookies, or credentials.
- Private dataset locations or unpublished asset URLs.
- Machine-local absolute paths or cluster-specific account details.
- Checkpoints or artifacts that are not intended for public release.

The release checker scans the public tree and reachable git history for common
private-path and token patterns:

```bash
python scripts/check_public_release.py --history-smoke --skip-help --verbose
```

Run this check before publishing release-facing changes.
