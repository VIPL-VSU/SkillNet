# SkillNet Release Status

This page records the public-release readiness state for the
`skillnet-public-release` branch. It is intentionally conservative: do not claim
that a workflow is externally runnable until the listed evidence passes from a
fresh checkout.

## Ready

| Area | Status | Evidence |
| --- | --- | --- |
| Fresh clone gate | Ready | A short-path fresh clone of `skillnet-public-release` passes `python scripts/check_public_release.py --verbose`, `--yaml-smoke --skip-help --verbose`, `--history-smoke --skip-help --verbose`, and `--hub-smoke --skip-help --verbose`. |
| GitHub Actions release gate | Ready | `.github/workflows/release-check.yml` runs the static release check, YAML syntax check, history privacy scan, Bash syntax check, public Hub reachability check, and no-deps editable install smoke on pushes and pull requests for `skillnet-public-release`. |
| Clean source history | Ready | `python scripts/check_public_release.py --history-smoke --skip-help` passes on the release branch. |
| Quick start and package smoke | Ready | `python scripts/check_public_release.py` passes in a fresh checkout; the lightweight `--install-smoke` passes after the documented Python 3.10 and `uv` prerequisites are installed. Simulator/runtime dependency installs are checked separately on Linux. |
| Skill hierarchy | Ready | Tokenization strategy, motion-code centers, annotation examples, and tokenizer CLI are included. |
| LIBERO compact slice indices | Ready | `data_process/libero/slice_indices/libero40_slice_index.json` and `libero90_slice_index.json` reproduce the v1 frame labels from public RLDS source order without local absolute paths. |
| LIBERO checkpoints | Ready | `jsw19/SkillNet-LIBERO-40` and `jsw19/SkillNet-LIBERO-90` are reachable with `--hub-smoke`. |
| LIBERO-Skill benchmark files | Ready | The 9-task manifest plus bddl/init/annotation contract are checked by `check_public_release.py`. |
| RoboTwin few-shot source release | Ready | Data helpers, metadata, train launchers, evaluation adapter, and reported protocol are included. |

## Last Verified Source Snapshot

The source tree is verified from short-path shallow clones of
`skillnet-public-release` and by the GitHub Actions release gate. Use HTTPS or
SSH according to your network and GitHub credentials. Record the
exact git SHA and timestamp in the release tag, GitHub release, or CI artifact
when cutting a public release; this status page lists the reproducible gates
rather than pinning a self-referential documentation commit.

Passed gates:

```bash
python -m pip install -U pyyaml
python scripts/check_public_release.py --verbose
python scripts/check_public_release.py --yaml-smoke --skip-help --verbose
python scripts/check_public_release.py --history-smoke --skip-help --verbose  # scan depth follows the checkout history
python scripts/check_public_release.py --hub-smoke --skip-help --verbose --hub-retries 3 --hub-timeout 30
python scripts/check_public_release.py --require-bash --skip-help --verbose
python scripts/check_public_release.py --install-smoke --install-python /path/to/python --skip-help --verbose
```

The anonymous derived-dataset gate still fails for
`jsw19/libero_40_v1` and `jsw19/libero_90_v1`, so direct LIBERO training from
those Hub dataset ids remains pending until their visibility is changed or the
datasets are mirrored under another public namespace. The stricter
`--include-derived-datasets` gate also fails for RoboTwin derived dataset ids;
this is expected for the current source-code release because RoboTwin LeRobot
datasets are generated locally rather than published as release assets.

## Pending External Assets

| Asset | Status | Why It Matters |
| --- | --- | --- |
| `jsw19/libero_40_v1` | Pending public dataset visibility | Direct Hub loading needs this LeRobot dataset to be public; users can also rebuild the same `repo_id` locally from public RLDS sources plus the included slice-index metadata. |
| `jsw19/libero_90_v1` | Pending public dataset visibility | Direct Hub loading needs this LeRobot dataset to be public; users can also rebuild the same `repo_id` locally from public RLDS sources plus the included slice-index metadata. |
| Final organization Hub namespace | Pending optional mirror | Current published assets remain under `jsw19/*`; runtime configs and converters support `SKILLNET_RELEASE_HF_NAMESPACE` so the release can move to an organization namespace after mirroring. |
| GitHub repository metadata | Pending repository setting | `docs/github_repository_setup.md` records the public default branch, description, homepage, and topic settings that must be updated outside the git tree; the current public API state still reports default branch `main`, description `coming soon`, no homepage, and no release topics. Verify the final state with `scripts/check_github_repository_metadata.py`. |
| RoboTwin LeRobot datasets | Local-generation path documented | The release provides download/conversion scripts; checkpoint weights and derived task datasets are not published in this release. |
| pi0.5 base checkpoint | External dependency | Training configs default to the public pi0.5 base checkpoint; mirror it locally and set `SKILLNET_PI05_BASE_PARAMS` if the default GCS asset is not reachable. |
| LIBERO/RoboTwin simulators | External dependency | Evaluation requires working simulator installs outside this repository. `install_libero.sh` runs the repo-local LIBERO-Skill registration helper when possible; use `SKILLNET_REQUIRE_LIBERO=1` to make LIBERO registration failures fatal during setup. |

Before claiming that external users can train LIBERO directly from Hub datasets,
run:

```bash
python scripts/check_public_release.py --hub-smoke --include-libero-derived-datasets
```

Before claiming all common derived LeRobot dataset ids are public, run:

```bash
python scripts/check_public_release.py --hub-smoke --include-derived-datasets
```

Before announcing the GitHub repository landing page, verify repository-level
metadata that cannot be stored in this source tree:

```bash
python scripts/check_github_repository_metadata.py --verbose
```

If the check fails and you have repository admin permission, use the safe setter
with a token from the environment:

```bash
python scripts/set_github_repository_metadata.py --dry-run --verbose

read -rsp "GITHUB_TOKEN: " GITHUB_TOKEN
export GITHUB_TOKEN
echo
python scripts/set_github_repository_metadata.py --verbose
unset GITHUB_TOKEN

python scripts/check_github_repository_metadata.py --verbose
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
reads credentials from `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`; do not put tokens
in command lines, scripts, docs, or git history. Use `--skip-hub-preflight`
only for local-only validation when no Hub read/write check should run.

For large LIBERO dataset uploads, install `huggingface_hub` with `hf_xet` and
set `HF_XET_HIGH_PERFORMANCE=1` in the upload shell.
Dataset-card templates are provided under `data_process/libero/dataset_cards/`;
copy the matching template to `README.md` at the dataset root before the public
dry run.

If the derived LIBERO datasets are already uploaded but remain private, switch
their visibility with an authenticated token that has write access, then rerun
the anonymous `--include-libero-derived-datasets` gate:

```bash
python -m pip install -U huggingface_hub
python scripts/set_hf_dataset_visibility.py \
  --public \
  --dry-run \
  jsw19/libero_40_v1 \
  jsw19/libero_90_v1

read -rsp "HF_TOKEN: " HF_TOKEN
export HF_TOKEN
echo

python scripts/set_hf_dataset_visibility.py \
  --public \
  jsw19/libero_40_v1 \
  jsw19/libero_90_v1

unset HF_TOKEN
python scripts/check_public_release.py --hub-smoke --include-libero-derived-datasets
```

The visibility helper allows `--dry-run` without a token, and even without
`huggingface_hub` installed, so release maintainers can inspect the intended
target repos before placing credentials in the shell. Real visibility changes
still require `huggingface_hub` and a write-capable `HF_TOKEN` or
`HUGGINGFACE_HUB_TOKEN`.

## Verification Checklist

Run these before tagging or announcing a release:

```bash
# Run from a clean short-path clone of the release branch.
python scripts/check_public_release.py --verbose
python scripts/check_public_release.py --yaml-smoke --skip-help --verbose
python scripts/check_public_release.py --history-smoke --skip-help --verbose
python scripts/check_public_release.py --hub-smoke --skip-help --verbose
```

On Linux, also run:

```bash
python -m pip install -U uv
python -m pip install -U pyyaml
python scripts/check_public_release.py --yaml-smoke --skip-help
python scripts/check_public_release.py --require-bash --skip-help
python scripts/check_public_release.py --install-smoke --skip-help
# If `uv venv --python 3.10` cannot resolve Python 3.10, pass the interpreter:
python scripts/check_public_release.py --install-smoke --install-python /path/to/python3.10 --skip-help
python skill_moe/skillnet/examples/libero/install_libero_skill_assets.py --help
SKILLNET_REQUIRE_LIBERO=1 bash skill_moe/skillnet/install_libero.sh
```

`--install-smoke` is a no-deps editable metadata check. It verifies package
metadata and import paths without resolving the full CUDA/JAX/simulator stack;
run the documented runtime install separately on the target training machine.

For Windows users, clone under a short, non-user-specific directory and run
`git config core.longpaths true` inside the clone after checkout. The
`git -c core.longpaths=true clone ...` flag handles the initial checkout, while
the local config keeps later `git status`, `git diff`, and release checks from
hitting path-length limits on bundled LIBERO-Skill filenames.

Use a clean clone or `git archive` when packaging source artifacts for external
sharing, not a manually zipped working directory. Local ignored caches such as
`__pycache__` and `*.pyc` can contain machine-specific paths even when they are
not tracked by git.
