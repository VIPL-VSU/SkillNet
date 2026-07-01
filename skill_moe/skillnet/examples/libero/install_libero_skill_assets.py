"""Install the bundled LIBERO-Skill assets into an existing LIBERO package.

The SkillNet repository includes the LIBERO-Skill task files, initial states,
and task-order metadata needed by the public zero-shot evaluation. Some users
will run evaluation against an external LIBERO install rather than the bundled
third-party tree. This helper copies missing assets into that install and, when
possible, registers the benchmark keys used by the evaluator.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path


SKILLNET_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_LIBERO_ROOT = SKILLNET_ROOT / "third_party" / "libero" / "libero" / "libero"
LIBERO_SKILL_MANIFEST = BUNDLED_LIBERO_ROOT / "bddl_files" / "libero_skill_obj" / "public_task_manifest.json"


def load_libero_skill_tasks() -> list[str]:
    manifest = json.loads(LIBERO_SKILL_MANIFEST.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not all(isinstance(task, str) for task in tasks):
        raise ValueError(f"Invalid LIBERO-Skill manifest: {LIBERO_SKILL_MANIFEST}")
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy and register SkillNet's bundled LIBERO-Skill benchmark assets."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--install", action="store_true", help="Apply changes to the detected LIBERO package.")
    mode.add_argument("--dry-run", action="store_true", help="Print planned changes without writing files.")
    parser.add_argument(
        "--libero-root",
        type=Path,
        help=(
            "Path to the inner LIBERO package directory that contains bddl_files, "
            "init_files, and benchmark. If omitted, the helper searches Python's import path."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing LIBERO-Skill asset files. Source files are still backed up before text patches.",
    )
    return parser.parse_args()


def find_libero_root() -> Path | None:
    spec = importlib.util.find_spec("libero")
    if spec is None or spec.submodule_search_locations is None:
        return None

    candidates: list[Path] = []
    for location in spec.submodule_search_locations:
        package_dir = Path(location).resolve()
        candidates.extend([package_dir, package_dir / "libero"])

    for candidate in candidates:
        if (candidate / "bddl_files").is_dir() and (candidate / "benchmark").is_dir():
            return candidate
    return None


def copy_tree_missing(src: Path, dst: Path, *, dry_run: bool, force: bool) -> tuple[int, int]:
    copied = 0
    skipped = 0
    for src_file in src.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src)
        dst_file = dst / rel
        if dst_file.exists() and not force:
            skipped += 1
            continue
        copied += 1
        if dry_run:
            continue
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
    return copied, skipped


def backup_once(path: Path, *, dry_run: bool) -> None:
    backup = path.with_suffix(path.suffix + ".skillnet.bak")
    if backup.exists() or dry_run:
        return
    shutil.copy2(path, backup)


def write_text_if_changed(path: Path, text: str, *, dry_run: bool) -> bool:
    old_text = path.read_text(encoding="utf-8")
    if old_text == text:
        return False
    backup_once(path, dry_run=dry_run)
    if not dry_run:
        path.write_text(text, encoding="utf-8")
    return True


def task_map_patch() -> str:
    entries = ",\n".join(f'    "{task}"' for task in load_libero_skill_tasks())
    return (
        "\n\n# SkillNet LIBERO-Skill public benchmark tasks.\n"
        'libero_task_map["libero_skill_obj"] = [\n'
        f"{entries},\n"
        "]\n"
    )


def patch_task_map(libero_root: Path, *, dry_run: bool) -> bool:
    task_map_file = libero_root / "benchmark" / "libero_suite_task_map.py"
    if not task_map_file.exists():
        print(f"Warning: task map file not found: {task_map_file}")
        return False

    text = task_map_file.read_text(encoding="utf-8")
    if "libero_skill_obj" in text:
        print("LIBERO-Skill task map entry already present.")
        return False
    changed = write_text_if_changed(task_map_file, text.rstrip() + task_map_patch(), dry_run=dry_run)
    print(f"{'Would patch' if dry_run else 'Patched'} {task_map_file}")
    return changed


def benchmark_registration_patch(text: str) -> str | None:
    if "LIBERO_SKILL" in text and "LIBERO_SKILL_OBJ" in text:
        return None
    if "register_benchmark" not in text or "Benchmark" not in text:
        return None

    patched = text
    if "libero_skill_obj" not in patched and "task_maps = {}" in patched:
        patched = patched.replace(
            "task_maps = {}",
            'if "libero_skill_obj" not in libero_suites:\n'
            '    libero_suites.append("libero_skill_obj")\n\n'
            "task_maps = {}",
            1,
        )

    class_patch = """

# SkillNet LIBERO-Skill public benchmark registration.
@register_benchmark
class LIBERO_SKILL(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_skill_obj"
        self._make_benchmark()


@register_benchmark
class LIBERO_SKILL_OBJ(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_skill_obj"
        self._make_benchmark()
"""
    return patched.rstrip() + class_patch


def patch_benchmark_init(libero_root: Path, *, dry_run: bool) -> bool:
    init_file = libero_root / "benchmark" / "__init__.py"
    if not init_file.exists():
        print(f"Warning: benchmark registration file not found: {init_file}")
        return False

    text = init_file.read_text(encoding="utf-8")
    patched = benchmark_registration_patch(text)
    if patched is None:
        print("LIBERO benchmark registration already patched or unsupported.")
        return False
    changed = write_text_if_changed(init_file, patched, dry_run=dry_run)
    print(f"{'Would patch' if dry_run else 'Patched'} {init_file}")
    return changed


def main() -> None:
    args = parse_args()
    dry_run = not args.install
    if args.dry_run:
        dry_run = True

    libero_root = args.libero_root.resolve() if args.libero_root else find_libero_root()
    if libero_root is None:
        raise SystemExit(
            "Could not locate LIBERO on Python's import path. Pass --libero-root or install LIBERO first."
        )
    if not (libero_root / "bddl_files").is_dir() or not (libero_root / "init_files").is_dir():
        raise SystemExit(f"Not an inner LIBERO package root: {libero_root}")

    source_bddl = BUNDLED_LIBERO_ROOT / "bddl_files" / "libero_skill_obj"
    source_init = BUNDLED_LIBERO_ROOT / "init_files" / "libero_skill_obj"
    if not source_bddl.is_dir() or not source_init.is_dir():
        raise SystemExit("Bundled LIBERO-Skill bddl/init assets are missing from this SkillNet checkout.")

    print(f"LIBERO root: {libero_root}")
    print(f"Mode: {'dry-run' if dry_run else 'install'}")

    bddl_copied, bddl_skipped = copy_tree_missing(
        source_bddl,
        libero_root / "bddl_files" / "libero_skill_obj",
        dry_run=dry_run,
        force=args.force,
    )
    init_copied, init_skipped = copy_tree_missing(
        source_init,
        libero_root / "init_files" / "libero_skill_obj",
        dry_run=dry_run,
        force=args.force,
    )

    print(f"Bddl assets: {bddl_copied} {'would be copied' if dry_run else 'copied'}, {bddl_skipped} skipped.")
    print(f"Init files: {init_copied} {'would be copied' if dry_run else 'copied'}, {init_skipped} skipped.")
    patch_task_map(libero_root, dry_run=dry_run)
    patch_benchmark_init(libero_root, dry_run=dry_run)

    if dry_run:
        print("Dry run complete. Re-run with --install to apply these changes.")
    else:
        print("LIBERO-Skill assets are installed. Re-run install_libero.sh to verify registration.")


if __name__ == "__main__":
    main()
