# SkillNet Release Status

This page records the public-release readiness state for the
`skillnet-public-release` branch. It is intentionally conservative: do not claim
that a workflow is externally runnable until the listed evidence passes from a
fresh checkout.

## Ready

| Area | Status | Evidence |
| --- | --- | --- |
| Clean source history | Ready | `python scripts/check_public_release.py --history-smoke --skip-help` passes on the release branch. |
| Quick start and install smoke | Ready | `python scripts/check_public_release.py` and `--install-smoke` pass in a fresh Python 3.10 environment. |
| Skill hierarchy | Ready | Tokenization strategy, motion-code centers, annotation examples, and tokenizer CLI are included. |
| LIBERO checkpoints | Ready | `jsw19/SkillNet-LIBERO-40` and `jsw19/SkillNet-LIBERO-90` are reachable with `--hub-smoke`. |
| LIBERO-Skill benchmark files | Ready | The 9-task bddl/init/annotation contract is checked by `check_public_release.py`. |
| RoboTwin few-shot source release | Ready | Data helpers, metadata, train launchers, evaluation adapter, and reported protocol are included. |

## Pending External Assets

| Asset | Status | Why It Matters |
| --- | --- | --- |
| `jsw19/libero_40_v1` | Pending public dataset visibility | Direct LIBERO-40 training with the released config needs this LeRobot dataset or an equivalent local copy under `LEROBOT_HOME`. |
| `jsw19/libero_90_v1` | Pending public dataset visibility | Direct LIBERO-90 training for LIBERO-Skill needs this LeRobot dataset or an equivalent local copy under `LEROBOT_HOME`. |
| RoboTwin LeRobot datasets | Local-generation path documented | The release provides download/conversion scripts; checkpoint weights and derived task datasets are not published in this release. |

Before claiming that external users can train LIBERO directly from Hub datasets,
run:

```bash
python scripts/check_public_release.py --hub-smoke --include-libero-derived-datasets
```

Before claiming all common derived LeRobot dataset ids are public, run:

```bash
python scripts/check_public_release.py --hub-smoke --include-derived-datasets
```

The derived LIBERO datasets can be verified or published with:

```bash
python scripts/verify_lerobot_dataset.py "$LEROBOT_HOME/jsw19/libero_40_v1" \
  --expected-tasks 40 \
  --expected-episodes 3862 \
  --expected-frames 273465 \
  --require-feature class \
  --require-feature all_classes \
  --strict-parquet \
  --require-column class \
  --require-column all_classes

python scripts/verify_lerobot_dataset.py "$LEROBOT_HOME/jsw19/libero_90_v1" \
  --expected-tasks 73 \
  --expected-episodes 7874 \
  --expected-frames 574571 \
  --require-feature class \
  --require-feature all_classes \
  --strict-parquet \
  --require-column class \
  --require-column all_classes
```

Use `scripts/publish_lerobot_dataset.py` with `--public --dry-run` first for
the release repos, or `--private --dry-run` for a separate staging repo. The
publisher checks the target repo visibility, refuses accidental
public/private mismatches unless `--allow-existing-visibility` is passed, and
uses `HfApi.upload_large_folder` for the real large-directory transfer. Add a
dataset card `README.md` at the dataset root before a public upload. The script
reads credentials from `HF_TOKEN`; do not put tokens in command lines, scripts,
docs, or git history.

For large LIBERO dataset uploads, install `huggingface_hub` with `hf_xet` and
set `HF_XET_HIGH_PERFORMANCE=1` in the upload shell.
Dataset-card templates are provided under `data_process/libero/dataset_cards/`;
copy the matching template to `README.md` at the dataset root before the public
dry run.

## Verification Checklist

Run these before tagging or announcing a release:

```bash
python scripts/check_public_release.py --verbose
python scripts/check_public_release.py --history-smoke --skip-help --verbose
python scripts/check_public_release.py --hub-smoke --skip-help --verbose
```

On Linux, also run:

```bash
python scripts/check_public_release.py --install-smoke --skip-help
```

For Windows users, clone under a short, non-user-specific directory and keep
`core.longpaths=true`, because bundled LIBERO-Skill filenames are long.
