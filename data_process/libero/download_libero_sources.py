"""Download the RLDS sources used to build SkillNet LIBERO LeRobot datasets.

The script only downloads a source dataset when its expected TFDS builder
directory is missing. Existing local directories are left in place.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_DATA_ROOT = Path(os.environ.get("SKILLNET_LIBERO_DATA_ROOT", "data/libero")).expanduser()

DATASETS = {
    "libero40": {
        "repo_id": "openvla/modified_libero_rlds",
        "local_dir": Path(os.environ.get("LIBERO40_RLDS_DIR", DEFAULT_DATA_ROOT / "libero40_rlds")).expanduser(),
        "expected_dirs": (
            "libero_10_no_noops",
            "libero_goal_no_noops",
            "libero_object_no_noops",
            "libero_spatial_no_noops",
        ),
    },
    "libero90": {
        "repo_id": "jesbu1/libero_90_openvla_processed",
        "local_dir": Path(os.environ.get("LIBERO90_RLDS_DIR", DEFAULT_DATA_ROOT / "libero90_rlds")).expanduser(),
        "expected_dirs": ("libero_90_openvla_processed",),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("libero40", "libero90", "all"),
        default="all",
        help="Which source dataset to download.",
    )
    parser.add_argument(
        "--libero40-dir",
        type=Path,
        default=DATASETS["libero40"]["local_dir"],
        help="Local directory for openvla/modified_libero_rlds.",
    )
    parser.add_argument(
        "--libero90-dir",
        type=Path,
        default=DATASETS["libero90"]["local_dir"],
        help="Local directory for jesbu1/libero_90_openvla_processed.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=None,
        help="Optional endpoint, for example https://hf-mirror.com.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Call snapshot_download even when expected local directories exist.",
    )
    return parser.parse_args()


def expected_dirs_exist(local_dir: Path, expected_dirs: tuple[str, ...]) -> bool:
    return all((local_dir / name).exists() for name in expected_dirs)


def download_one(name: str, local_dir: Path, force: bool) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Please install huggingface_hub before downloading LIBERO source data.") from exc

    config = DATASETS[name]
    expected_dirs = config["expected_dirs"]
    if expected_dirs_exist(local_dir, expected_dirs) and not force:
        print(f"[skip] {name}: found expected directories under {local_dir}")
        return

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {config['repo_id']} -> {local_dir}")
    snapshot_download(
        repo_id=config["repo_id"],
        repo_type="dataset",
        local_dir=str(local_dir),
    )


def main() -> None:
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    selected = ("libero40", "libero90") if args.dataset == "all" else (args.dataset,)
    local_dirs = {
        "libero40": args.libero40_dir,
        "libero90": args.libero90_dir,
    }
    for name in selected:
        download_one(name, local_dirs[name], force=args.force)


if __name__ == "__main__":
    main()
