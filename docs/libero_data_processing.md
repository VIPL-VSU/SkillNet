# LIBERO Data Processing

This note documents how the released SkillNet LIBERO training datasets are
derived from the public RLDS sources and the compact slice-index metadata
included in this repository.

## Public Entrypoints

The maintained public entrypoints in `data_process/libero/` build the released
LIBERO LeRobot data format:

```text
convert_libero_40_to_lerobot.py
convert_libero_90_to_lerobot.py
convert_libero_to_lerobot.py
instruct2plan_40.json
instruct2plan_90.json
instruct2plan_obj_90.json
```

The released LIBERO-40 wrapper converts the four LIBERO-40 TFDS builders into
`jsw19/libero_40_v1`. The released LIBERO-90 wrapper converts the LIBERO-90
TFDS builder into `jsw19/libero_90_v1`.

The scripts also include experimental helpers for keypoint extraction and skill
slicing. Those helpers depend on environment-specific assets and model/API
services, so the release keeps only the deterministic RLDS-to-LeRobot
conversion path.

The task keys in `instruct2plan_40.json` match the 40 tasks in the existing
`jsw19/libero_40` metadata exactly. The task keys in `instruct2plan_90.json`
match the 73 tasks in the existing `jsw19/libero_90` metadata exactly.
`instruct2plan_obj_90.json` has the same 73 task keys and skill classes as
`instruct2plan_90.json`, with an extra `objects` field.

## Source Data

The released training configs use these canonical LeRobot `repo_id` values:

```text
jsw19/libero_40_v1
jsw19/libero_90_v1
```

If those derived dataset repos are visible to your Hugging Face account, LeRobot
can load them directly. If they are private or unavailable in your environment,
rebuild them with the conversion commands below and keep the same repo_id layout
under `LEROBOT_HOME`.
The public converters and training configs derive these defaults from
`SKILLNET_RELEASE_HF_NAMESPACE=jsw19`; set that variable to an organization
namespace after mirroring the assets, or set `SKILLNET_LIBERO40_REPO_ID` and
`SKILLNET_LIBERO90_REPO_ID` for per-dataset overrides.

Release contract:

- The public repository includes source downloaders, deterministic converters,
  instruction-to-plan maps, and schema checks.
- Exact SkillNet training reproduction requires either accessible derived
  LeRobot datasets with the repo ids above, or a local rebuild using the
  included compact slice-index metadata files listed below.
- The RLDS source datasets alone are not enough to reconstruct the v1 training
  frames exactly, because frame-level skill boundaries are separate metadata.
- If neither the derived LeRobot datasets nor equivalent frame-boundary metadata
  are available in your environment, you can still inspect configs, run release
  smoke checks, and evaluate the published checkpoints after simulator setup,
  but you cannot exactly reproduce the LIBERO training datasets from RLDS alone.

The conversion commands below rebuild compatible datasets from RLDS sources
using the included slice-index metadata.

Published v1 LeRobot dataset sanity-check counts:

| Dataset | Tasks | Episodes | Frames |
| --- | --- | --- | --- |
| `jsw19/libero_40_v1` | 40 | 3,862 | 273,465 |
| `jsw19/libero_90_v1` | 73 | 7,874 | 574,571 |

The compact slice-index JSON uses `num_episodes` for source RLDS episodes
before skill slicing: 1,693 for LIBERO-40 and 4,006 for LIBERO-90. During
conversion, each skill segment is saved as one LeRobot episode, so the final
LeRobot episode counts are 3,862 and 7,874.

## Dataset Hub Publishing

If you maintain the derived LeRobot datasets locally, verify the metadata before
training or publishing. Use `--strict-parquet` for public release validation so
the parquet row count is checked against `total_frames`.

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

To publish the datasets to Hugging Face Hub, install the large-folder upload
stack, set `HF_TOKEN` in the shell environment, and run a dry run first:

```bash
python -m pip install "huggingface_hub>=1.0" hf_xet pyarrow
export HF_XET_HIGH_PERFORMANCE=1
```

Copy the matching dataset card template into the dataset root before the public
dry run:

```bash
cp data_process/libero/dataset_cards/README_libero_40_v1.md \
  "$LEROBOT_HOME/jsw19/libero_40_v1/README.md"

cp data_process/libero/dataset_cards/README_libero_90_v1.md \
  "$LEROBOT_HOME/jsw19/libero_90_v1/README.md"
```

The dry run validates local metadata, checks parquet row counts, prints the
planned size, verifies the large-folder uploader, checks whether the target Hub
repo already exists, and never writes to the Hub:

```bash
python scripts/publish_lerobot_dataset.py "$LEROBOT_HOME/jsw19/libero_40_v1" \
  --repo-id jsw19/libero_40_v1 \
  --public \
  --expected-tasks 40 \
  --expected-episodes 3862 \
  --expected-frames 273465 \
  --require-feature class \
  --require-feature all_classes \
  --strict-parquet \
  --require-column class \
  --require-column all_classes \
  --dry-run

python scripts/publish_lerobot_dataset.py "$LEROBOT_HOME/jsw19/libero_90_v1" \
  --repo-id jsw19/libero_90_v1 \
  --public \
  --expected-tasks 73 \
  --expected-episodes 7874 \
  --expected-frames 574571 \
  --require-feature class \
  --require-feature all_classes \
  --strict-parquet \
  --require-column class \
  --require-column all_classes \
  --dry-run
```

For the public SkillNet release, the target repos should be public. For a
staging upload, use `--private` and a separate repo id. Do not point a private
staging run at the public repo id unless you intentionally pass
`--allow-existing-visibility`.

The publisher refuses a public dry run or real public upload without
`README.md` at the dataset root unless `--allow-missing-card` is passed for an
internal staging run. The script reads the token from `HF_TOKEN` or
`HUGGINGFACE_HUB_TOKEN`; do not put tokens into command lines, scripts, docs, or
repository files.

After dry-run validation, start the real upload in a long-running shell or
session manager by removing only `--dry-run`. The uploader uses
`HfApi.upload_large_folder`, which is resumable and better suited to the 33G and
65G derived LIBERO datasets than a single ordinary folder upload.

If the upload already completed but either dataset repo is still private, switch
only the repo visibility with the token-safe helper:

```bash
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
```

The dry run is credential-free and dependency-light: if `huggingface_hub` is
not installed, it still prints the intended target visibility and explains that
repo inspection is unavailable. The real visibility change requires
`huggingface_hub` and a write-capable `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` in
the environment.

After publishing, run:

```bash
python scripts/check_public_release.py --hub-smoke --include-libero-derived-datasets
```

The strict Hub check should pass for `jsw19/libero_40_v1` and
`jsw19/libero_90_v1` before claiming that external users can train directly
from the released configs.

LIBERO-40 uses:

```text
https://huggingface.co/datasets/openvla/modified_libero_rlds
```

Expected TFDS builder directories after download:

```text
libero_10_no_noops
libero_goal_no_noops
libero_object_no_noops
libero_spatial_no_noops
```

LIBERO-90 uses:

```text
https://huggingface.co/datasets/jesbu1/libero_90_openvla_processed
```

Expected TFDS builder directory after download:

```text
libero_90_openvla_processed
```

## Skill-Slice Inputs

The v1 datasets are not a direct frame-for-frame copy of the RLDS data. Each
episode is segmented by skill boundaries and each frame receives:

```text
class
all_classes
objects  # LIBERO-90 only
```

The converter expects these precomputed files:

```text
$LIBERO40_PLAN_ROOT/libero40_plan_sliced.json
$LIBERO90_PLAN_ROOT/libero90_plan_sliced.json
```

Each file is a JSON mapping from task/instruction keys to episode-level
skill-slice records. The converter reads the records to assign `class`,
`all_classes`, and optional `objects` to every frame before writing the LeRobot
dataset. A compatible export must preserve the same task keys as
`instruct2plan_40.json` or `instruct2plan_90.json` and provide monotonically
ordered skill segments for every episode.

In some source trees, these files are stored directly under:

```text
$PLAN_ROOT/libero40_plan_sliced.json
$PLAN_ROOT/libero90_plan_sliced.json
```

When using that layout, pass the directory explicitly:

```bash
python data_process/libero/convert_libero_40_to_lerobot.py \
  --data-dir "$LIBERO40_RLDS_DIR" \
  --plan-root "$PLAN_ROOT"

python data_process/libero/convert_libero_90_to_lerobot.py \
  --data-dir "$LIBERO90_RLDS_DIR" \
  --plan-root "$PLAN_ROOT"
```

Alternatively, copy or symlink the files into the default filtered-data
directories if you need to run legacy conversion scripts unchanged.

These two files are metadata inputs, not generated by the RLDS downloader. If
they are absent, use the already released LeRobot datasets above or provide an
equivalent skill-slice export generated with the same schema. The public
converter intentionally fails fast rather than silently producing an
unsegmented dataset.

As an alternative to the legacy `*_plan_sliced.json` files, SkillNet includes
compact slice-index metadata exported from the v1 LeRobot datasets:

```text
data_process/libero/slice_indices/libero40_slice_index.json
data_process/libero/slice_indices/libero90_slice_index.json
```

This is the preferred small metadata format for public release because it
records reconstructed source episode ids, frame ranges, `class`, and
`all_classes` labels without storing local absolute paths. You can regenerate it
from an existing LeRobot v1 dataset if needed:

```bash
mkdir -p data/libero/slice_indices

python data_process/libero/export_libero_skill_slices.py \
  "$LEROBOT_HOME/jsw19/libero_40_v1" \
  --repo-id jsw19/libero_40_v1 \
  --output data/libero/slice_indices/libero40_slice_index.json

python data_process/libero/export_libero_skill_slices.py \
  "$LEROBOT_HOME/jsw19/libero_90_v1" \
  --repo-id jsw19/libero_90_v1 \
  --output data/libero/slice_indices/libero90_slice_index.json

python data_process/libero/convert_libero_40_to_lerobot.py \
  --data-dir "$LIBERO40_RLDS_DIR" \
  --slice-index data/libero/slice_indices/libero40_slice_index.json \
  --output-repo-id jsw19/libero_40_v1

python data_process/libero/convert_libero_90_to_lerobot.py \
  --data-dir "$LIBERO90_RLDS_DIR" \
  --slice-index data/libero/slice_indices/libero90_slice_index.json \
  --output-repo-id jsw19/libero_90_v1
```

## Released Scripts

The cleaned scripts live at:

```text
data_process/libero/download_libero_sources.py
data_process/libero/convert_libero_40_to_lerobot.py
data_process/libero/convert_libero_90_to_lerobot.py
data_process/libero/convert_libero_to_lerobot.py
data_process/libero/export_libero_skill_slices.py
data_process/libero/instruct2plan_40.json
data_process/libero/instruct2plan_90.json
data_process/libero/instruct2plan_obj_90.json
```

They keep the original feature schema and skill-class mapping:

```text
pick  -> 0
place -> 1
push  -> 2
open/close -> 3
turn  -> 4
```

Unlike one-off conversion scripts, the released converter does not delete existing
outputs unless `--overwrite` is passed.

## Reproduction Commands

On the project machine, activate the environment that contains `tensorflow_datasets`,
`lerobot`, `numpy`, and `tqdm`, then download sources:

```bash
export http_proxy=http://<proxy-host>:<proxy-port>
export https_proxy=http://<proxy-host>:<proxy-port>
export SKILLNET_LIBERO_DATA_ROOT="${SKILLNET_LIBERO_DATA_ROOT:-data/libero}"
export LIBERO40_RLDS_DIR="${LIBERO40_RLDS_DIR:-$SKILLNET_LIBERO_DATA_ROOT/libero40_rlds}"
export LIBERO90_RLDS_DIR="${LIBERO90_RLDS_DIR:-$SKILLNET_LIBERO_DATA_ROOT/libero90_rlds}"

python data_process/libero/download_libero_sources.py --dataset all
```

If Hugging Face is not reachable, use the mirror:

```bash
python data_process/libero/download_libero_sources.py \
  --dataset all \
  --hf-endpoint https://hf-mirror.com
```

From the repository root, you can also check only the public checkpoint and
source-dataset Hub links:

```bash
python scripts/check_public_release.py --hub-smoke
```

Use `--include-libero-derived-datasets` only when the derived LIBERO LeRobot
dataset repos are expected to be accessible from the current account. A 401/403
result for those derived datasets means you should rebuild them locally or
request access.

Convert LIBERO-40:

```bash
python data_process/libero/convert_libero_40_to_lerobot.py \
  --data-dir "$LIBERO40_RLDS_DIR" \
  --slice-index data_process/libero/slice_indices/libero40_slice_index.json \
  --output-repo-id jsw19/libero_40_v1
```

Convert LIBERO-90:

```bash
python data_process/libero/convert_libero_90_to_lerobot.py \
  --data-dir "$LIBERO90_RLDS_DIR" \
  --slice-index data_process/libero/slice_indices/libero90_slice_index.json \
  --output-repo-id jsw19/libero_90_v1
```

For object-aware LIBERO-90 variants, use the object map explicitly:

```bash
python data_process/libero/convert_libero_90_to_lerobot.py \
  --data-dir "$LIBERO90_RLDS_DIR" \
  --slice-index data_process/libero/slice_indices/libero90_slice_index.json \
  --output-repo-id jsw19/libero_90_v1 \
  --instruction-map data_process/libero/instruct2plan_obj_90.json \
  --include-objects
```

Run conversion from a long-running shell or session manager if your source
download and parquet conversion will outlive an interactive terminal.

## Current Reproduction Status

Before conversion, confirm that the raw RLDS source directories and the two
included slice-index JSON files are present. If you use legacy
`*_plan_sliced.json` metadata instead, pass the directory with `--plan-root` or
link the files into the default locations before conversion. If you have neither
slice-index files nor legacy frame-boundary metadata, use accessible copies of
`jsw19/libero_40_v1` and `jsw19/libero_90_v1` directly for training and
evaluation, or recreate equivalent metadata before conversion.
