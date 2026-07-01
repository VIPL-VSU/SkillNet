# LIBERO Data Processing

This folder contains the open-source data processing entrypoints used to build
the SkillNet LIBERO LeRobot datasets.

## Source Datasets

- LIBERO-40 source RLDS data: `openvla/modified_libero_rlds`
- LIBERO-90 source RLDS data: `jesbu1/libero_90_openvla_processed`

The released configs use `jsw19/libero_40_v1` and `jsw19/libero_90_v1` as the
canonical LeRobot `repo_id` values. If those derived datasets are visible to
your Hugging Face account, LeRobot can load them directly. If they are private
or unavailable in your environment, rebuild compatible local datasets from RLDS
sources with the included compact slice-index metadata, and keep the same
repo_id layout under `LEROBOT_HOME`.

Published v1 sanity-check counts are 40 tasks / 3,862 episodes for
`libero_40_v1` and 73 tasks / 7,874 episodes for `libero_90_v1`.

This folder contains maintained public wrappers for the released data format:
`convert_libero_40_to_lerobot.py`, `convert_libero_90_to_lerobot.py`, and
`convert_libero_to_lerobot.py`.

The LIBERO-40 wrapper outputs `jsw19/libero_40_v1`; the LIBERO-90 wrapper
outputs `jsw19/libero_90_v1`.

## Inputs

The conversion expects two kinds of input:

1. The downloaded TFDS/RLDS source data.
2. Frame-boundary metadata generated with the same schema as the SkillNet v1
   datasets.

The public default metadata is the compact slice-index included in this
directory:

```text
slice_indices/libero40_slice_index.json
slice_indices/libero90_slice_index.json
```

Pass those files with `--slice-index` in the conversion commands below.

Legacy `*_plan_sliced.json` files are still supported for custom or historical
rebuilds. If omitted, the scripts default to `data/libero/...` under the current
working directory:

```text
$LIBERO40_RLDS_DIR
$LIBERO90_RLDS_DIR
$LIBERO40_PLAN_ROOT/libero40_plan_sliced.json
$LIBERO90_PLAN_ROOT/libero90_plan_sliced.json
```

Some checkouts keep the two skill-slice files directly under one shared plan
directory:

```text
$PLAN_ROOT/libero40_plan_sliced.json
$PLAN_ROOT/libero90_plan_sliced.json
```

In that case, either pass `--plan-root "$PLAN_ROOT"` to the converter or
copy/symlink the files into the corresponding plan-root directories.

The two `*_plan_sliced.json` files are metadata inputs and are not produced by
`download_libero_sources.py`.

The included slice-index files let the public converter rebuild the v1 frame
labels from the RLDS source episode order without local absolute paths. If you
need to regenerate them from an existing LeRobot v1 dataset, run:

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
```

The slice-index format stores reconstructed source episode order, frame ranges,
and the released `class` / `all_classes` labels without recording local
absolute paths. It can be used instead of `--plan-root` when converting from the
public RLDS sources.

The small instruction maps required by the converter are included here:

```text
instruct2plan_40.json
instruct2plan_90.json
instruct2plan_obj_90.json
```

`instruct2plan_40.json` and `instruct2plan_90.json` are the default maps used to
build `libero_40_v1` and `libero_90_v1`. `instruct2plan_obj_90.json` has the
same LIBERO-90 task keys and skill classes, plus object ids for object-aware or
positional variants.

## Download

Set proxies if required by your machine:

```bash
export http_proxy=http://<proxy-host>:<proxy-port>
export https_proxy=http://<proxy-host>:<proxy-port>
```

Use the Hugging Face mirror only when the default endpoint is unavailable:

```bash
python data_process/libero/download_libero_sources.py \
  --dataset all \
  --hf-endpoint https://hf-mirror.com
```

The downloader checks for the expected local TFDS directories and skips existing
data unless `--force` is passed.

## Convert

Build LIBERO-40:

```bash
python data_process/libero/convert_libero_40_to_lerobot.py \
  --data-dir "$LIBERO40_RLDS_DIR" \
  --slice-index data_process/libero/slice_indices/libero40_slice_index.json \
  --output-repo-id jsw19/libero_40_v1
```

If you are using legacy `*_plan_sliced.json` files instead, pass `--plan-root`:

```bash
python data_process/libero/convert_libero_40_to_lerobot.py \
  --data-dir "$LIBERO40_RLDS_DIR" \
  --plan-root "$LIBERO40_PLAN_ROOT" \
  --output-repo-id jsw19/libero_40_v1
```

Build LIBERO-90:

```bash
python data_process/libero/convert_libero_90_to_lerobot.py \
  --data-dir "$LIBERO90_RLDS_DIR" \
  --slice-index data_process/libero/slice_indices/libero90_slice_index.json \
  --output-repo-id jsw19/libero_90_v1
```

The same `--plan-root` legacy alternative is available for LIBERO-90.

For object-aware LIBERO-90 variants, use the object map explicitly:

```bash
python data_process/libero/convert_libero_90_to_lerobot.py \
  --data-dir "$LIBERO90_RLDS_DIR" \
  --plan-root "$LIBERO90_PLAN_ROOT" \
  --output-repo-id jsw19/libero_90_v1 \
  --instruction-map data_process/libero/instruct2plan_obj_90.json \
  --include-objects
```

By default the converter refuses to overwrite an existing LeRobot dataset under
`HF_LEROBOT_HOME`. Pass `--overwrite` only when intentionally rebuilding.
