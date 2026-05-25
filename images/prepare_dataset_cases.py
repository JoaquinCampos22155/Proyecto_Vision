from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover - script dependency guard
    Image = None


DATASET_ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIEW_SEQUENCE = [
    "detail_collar",
    "front_top_half",
    "main_back_flat",
    "main_front_flat",
    "main_front_flat",
    "partial_view",
]
CASE_TYPE = "same_product_normal"
EXPECTED_STATUS = "consistent"
DEFAULT_NOTES = (
    "All six images are expected to belong to the same product. This case is used to verify that the system "
    "approves normal same-product listings. View labels were prepared with an MVP helper and should be reviewed "
    "manually for final dataset quality."
)


@dataclass(frozen=True)
class PlannedImage:
    source: Path
    target: Path
    view_type: str
    product_group: str = "A"
    expected_flag: bool = False


def image_files(case_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in case_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )


def numbered_name(index: int, view_type: str, extension: str, seen: dict[str, int]) -> str:
    count = seen.get(view_type, 0) + 1
    seen[view_type] = count
    suffix = f"_{count}" if count > 1 else ""
    return f"{index:02d}_{view_type}{suffix}{extension.lower()}"


def estimate_capture_context(paths: list[Path]) -> str:
    if Image is None:
        return "mixed"

    landscape = 0
    portrait = 0
    for path in paths:
        try:
            with Image.open(path) as image:
                width, height = image.size
        except OSError:
            continue
        if width >= height:
            landscape += 1
        else:
            portrait += 1

    if landscape >= 4:
        return "flat_lay"
    if portrait >= 4:
        return "hanger"
    return "mixed"


def estimate_product_category(paths: list[Path]) -> str:
    # These first cases are clothing tops in the current dataset. Keep this conservative so a human can refine later.
    return "unknown"


def estimate_description(paths: list[Path]) -> str:
    return "same product normal visual validation case"


def plan_case(case_number: int, dataset_root: Path) -> tuple[list[PlannedImage], dict | None, str | None]:
    case_dir = dataset_root / str(case_number)
    if not case_dir.exists():
        return [], None, "folder_missing"

    files = image_files(case_dir)
    if len(files) != 6:
        return [], None, f"expected_6_images_found_{len(files)}"

    seen: dict[str, int] = {}
    planned_images: list[PlannedImage] = []
    for index, (source, view_type) in enumerate(zip(files, VIEW_SEQUENCE)):
        target = case_dir / numbered_name(index, view_type, source.suffix, seen)
        planned_images.append(PlannedImage(source=source, target=target, view_type=view_type))

    labels = {
        "case_id": f"case_{case_number:04d}",
        "case_type": CASE_TYPE,
        "expected_status": EXPECTED_STATUS,
        "capture_context": estimate_capture_context(files),
        "product_category": estimate_product_category(files),
        "main_product_description": estimate_description(files),
        "expected_flagged_images": [],
        "difficulty": "easy",
        "notes": DEFAULT_NOTES,
        "images": [
            {
                "filename": planned.target.name,
                "view_type": planned.view_type,
                "product_group": planned.product_group,
                "expected_flag": planned.expected_flag,
            }
            for planned in planned_images
        ],
    }
    return planned_images, labels, None


def backup_case(case_dir: Path, backup_root: Path) -> Path:
    destination = backup_root / case_dir.name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(case_dir, destination)
    return destination


def apply_plan(planned_images: list[PlannedImage], labels: dict, backup_root: Path | None) -> list[dict]:
    operations = []
    if not planned_images:
        return operations

    case_dir = planned_images[0].source.parent
    if backup_root is not None:
        backup_path = backup_case(case_dir, backup_root)
        operations.append({"type": "backup", "path": str(backup_path)})

    temp_paths = []
    for planned in planned_images:
        if planned.source == planned.target:
            temp_paths.append((planned, planned.source))
            continue
        temp_path = planned.source.with_name(f".__tmp__{planned.source.name}")
        if temp_path.exists():
            raise RuntimeError(f"Temporary path already exists: {temp_path}")
        planned.source.rename(temp_path)
        temp_paths.append((planned, temp_path))
        operations.append({"type": "rename_to_temp", "from": str(planned.source), "to": str(temp_path)})

    for planned, temp_path in temp_paths:
        if temp_path == planned.target:
            continue
        if planned.target.exists():
            raise RuntimeError(f"Target path already exists: {planned.target}")
        temp_path.rename(planned.target)
        operations.append({"type": "rename", "from": str(planned.source), "to": str(planned.target)})

    labels_path = case_dir / "labels.json"
    labels_path.write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    operations.append({"type": "write_labels", "path": str(labels_path)})
    return operations


def run(dataset_root: Path, start: int, end: int, dry_run: bool, backup: bool) -> dict:
    backup_root = dataset_root / "_backup_before_prepare" / datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "dataset_root": str(dataset_root),
        "dry_run": dry_run,
        "processed": [],
        "skipped": [],
        "renames": [],
        "labels_created": [],
        "errors": [],
    }

    for case_number in range(start, end + 1):
        if case_number == 1:
            report["skipped"].append({"case": case_number, "reason": "reference_case_not_modified"})
            continue

        planned_images, labels, skip_reason = plan_case(case_number, dataset_root)
        if skip_reason:
            report["skipped"].append({"case": case_number, "reason": skip_reason})
            continue

        report["processed"].append(case_number)
        report["renames"].extend(
            {
                "case": case_number,
                "from": str(planned.source.relative_to(dataset_root)),
                "to": str(planned.target.relative_to(dataset_root)),
                "view_type": planned.view_type,
            }
            for planned in planned_images
        )
        report["labels_created"].append(str((dataset_root / str(case_number) / "labels.json").relative_to(dataset_root)))

        if dry_run:
            continue

        try:
            apply_plan(planned_images, labels, backup_root if backup else None)
        except Exception as exc:  # pragma: no cover - operational safety report
            report["errors"].append({"case": case_number, "error": str(exc)})

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare visual validation dataset cases 2-30.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--start", type=int, default=2)
    parser.add_argument("--end", type=int, default=30)
    parser.add_argument("--apply", action="store_true", help="Apply renames and labels. Default is dry-run.")
    parser.add_argument("--no-backup", action="store_true", help="Do not copy folders before applying.")
    args = parser.parse_args()

    report = run(
        dataset_root=args.dataset_root.resolve(),
        start=args.start,
        end=args.end,
        dry_run=not args.apply,
        backup=not args.no_backup,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
