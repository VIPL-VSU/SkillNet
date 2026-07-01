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
sources when the additional skill-slice metadata is available, and keep the
same repo_id layout under `LEROBOT_HOME`.

Published v1 sanity-check counts are 40 tasks / 3,862 episodes for
`libero_40_v1` and 73 tasks / 7,874 episodes for `libero_90_v1`.

Earlier research script names were `generate_data_40.py` and
`generate_data_90.py`; those files are not shipped as public entrypoints. This
folder contains maintained public wrappers for the same data format:
`convert_libero_40_to_lerobot.py`, `convert_libero_90_to_lerobot.py`, and
`convert_libero_to_lerobot.py`.

The LIBERO-40 wrapper outputs `jsw19/libero_40_v1`. Older LIBERO-90 research
scripts used the output name `jsw19/libero_90_obj`; the training configs expect
`jsw19/libero_90_v1`, so the released script uses `jsw19/libero_90_v1`.

## Inputs

The conversion expects two kinds of input:

1. The downloaded TFDS/RLDS source data.
2. Skill-slice metadata generated with the same schema as the SkillNet v1
   datasets.

Set these paths for your machine. If omitted, the scripts default to
`data/libero/...` under the current working directory:

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
`download_libero_sources.py`. If you do not have them, use accessible copies of
the derived LeRobot datasets instead of rebuilding from RLDS.

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
  --plan-root "$LIBERO40_PLAN_ROOT" \
  --output-repo-id jsw19/libero_40_v1
```

Build LIBERO-90:

```bash
python data_process/libero/convert_libero_90_to_lerobot.py \
  --data-dir "$LIBERO90_RLDS_DIR" \
  --plan-root "$LIBERO90_PLAN_ROOT" \
  --output-repo-id jsw19/libero_90_v1
```

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
