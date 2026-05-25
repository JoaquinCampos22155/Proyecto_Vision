from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VALID_STATUSES = {"consistent", "needs_review", "inconsistent"}
REQUIRED_LABEL_FIELDS = {
    "case_id",
    "case_type",
    "expected_status",
    "images",
    "image_count",
    "is_complete_case",
}


@dataclass(frozen=True)
class DatasetIssue:
    case_folder: str
    case_id: str | None
    case_type: str | None
    severity: str
    issue_type: str
    message: str
    filename: str | None = None

    def as_row(self) -> dict[str, str]:
        return {
            "case_folder": self.case_folder,
            "case_id": self.case_id or "",
            "case_type": self.case_type or "",
            "severity": self.severity,
            "issue_type": self.issue_type,
            "message": self.message,
            "filename": self.filename or "",
        }


@dataclass(frozen=True)
class DatasetCase:
    case_folder: str
    case_path: Path
    labels_path: Path
    labels: dict[str, Any]

    @property
    def case_id(self) -> str:
        return str(self.labels["case_id"])

    @property
    def case_type(self) -> str:
        return str(self.labels["case_type"])


def numeric_case_dirs(dataset_dir: Path) -> list[Path]:
    if not dataset_dir.exists():
        return []
    return sorted(
        [path for path in dataset_dir.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )


def image_files(case_dir: Path) -> list[Path]:
    return sorted(
        [path for path in case_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name.lower(),
    )


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON: {exc}"
    except OSError as exc:
        return None, f"Could not read file: {exc}"


def audit_case(case_dir: Path, dataset_dir: Path) -> tuple[dict[str, Any] | None, list[DatasetIssue], bool]:
    issues: list[DatasetIssue] = []
    relative_case = str(case_dir.relative_to(dataset_dir))
    real_images = image_files(case_dir)
    labels_path = case_dir / "labels.json"

    if not real_images:
        issues.append(
            DatasetIssue(
                case_folder=relative_case,
                case_id=None,
                case_type=None,
                severity="info",
                issue_type="empty_folder_skipped",
                message="Folder has no image files and was skipped.",
            )
        )
        return None, issues, True

    if not labels_path.exists():
        issues.append(
            DatasetIssue(
                case_folder=relative_case,
                case_id=None,
                case_type=None,
                severity="error",
                issue_type="missing_labels",
                message="Folder has images but no labels.json.",
            )
        )
        return None, issues, False

    labels, error = load_json(labels_path)
    if labels is None:
        issues.append(
            DatasetIssue(
                case_folder=relative_case,
                case_id=None,
                case_type=None,
                severity="error",
                issue_type="invalid_json",
                message=error or "labels.json is not valid JSON.",
                filename="labels.json",
            )
        )
        return None, issues, False

    case_id = str(labels.get("case_id", ""))
    case_type = str(labels.get("case_type", ""))
    for field in sorted(REQUIRED_LABEL_FIELDS):
        if field not in labels:
            issues.append(
                DatasetIssue(
                    case_folder=relative_case,
                    case_id=case_id,
                    case_type=case_type,
                    severity="error",
                    issue_type="missing_required_field",
                    message=f"labels.json is missing required field '{field}'.",
                    filename="labels.json",
                )
            )

    expected_status = labels.get("expected_status")
    if expected_status not in VALID_STATUSES:
        issues.append(
            DatasetIssue(
                case_folder=relative_case,
                case_id=case_id,
                case_type=case_type,
                severity="error",
                issue_type="invalid_expected_status",
                message=f"expected_status must be one of {sorted(VALID_STATUSES)}.",
                filename="labels.json",
            )
        )

    acceptable_status = labels.get("acceptable_status")
    if acceptable_status is not None:
        if not isinstance(acceptable_status, list) or any(status not in VALID_STATUSES for status in acceptable_status):
            issues.append(
                DatasetIssue(
                    case_folder=relative_case,
                    case_id=case_id,
                    case_type=case_type,
                    severity="error",
                    issue_type="invalid_acceptable_status",
                    message=f"acceptable_status must contain only {sorted(VALID_STATUSES)}.",
                    filename="labels.json",
                )
            )

    label_images = labels.get("images", [])
    if not isinstance(label_images, list):
        issues.append(
            DatasetIssue(
                case_folder=relative_case,
                case_id=case_id,
                case_type=case_type,
                severity="error",
                issue_type="invalid_images_field",
                message="images must be an array.",
                filename="labels.json",
            )
        )
        label_images = []

    real_filenames = {path.name for path in real_images}
    listed_filenames = []
    for index, item in enumerate(label_images):
        if not isinstance(item, dict):
            issues.append(
                DatasetIssue(
                    case_folder=relative_case,
                    case_id=case_id,
                    case_type=case_type,
                    severity="error",
                    issue_type="invalid_image_item",
                    message=f"images[{index}] must be an object.",
                    filename="labels.json",
                )
            )
            continue
        filename = item.get("filename")
        if not filename:
            issues.append(
                DatasetIssue(
                    case_folder=relative_case,
                    case_id=case_id,
                    case_type=case_type,
                    severity="error",
                    issue_type="missing_image_filename",
                    message=f"images[{index}] is missing filename.",
                    filename="labels.json",
                )
            )
            continue
        listed_filenames.append(str(filename))
        if filename not in real_filenames:
            issues.append(
                DatasetIssue(
                    case_folder=relative_case,
                    case_id=case_id,
                    case_type=case_type,
                    severity="error",
                    issue_type="missing_referenced_image",
                    message="Filename listed in labels.json does not exist in the case folder.",
                    filename=str(filename),
                )
            )

    for filename in sorted(real_filenames - set(listed_filenames)):
        issues.append(
            DatasetIssue(
                case_folder=relative_case,
                case_id=case_id,
                case_type=case_type,
                severity="error",
                issue_type="unlisted_image_file",
                message="Image file exists in folder but is not listed in labels.json.",
                filename=filename,
            )
        )

    if labels.get("image_count") != len(real_images):
        issues.append(
            DatasetIssue(
                case_folder=relative_case,
                case_id=case_id,
                case_type=case_type,
                severity="error",
                issue_type="image_count_mismatch",
                message=f"image_count={labels.get('image_count')} but folder contains {len(real_images)} images.",
                filename="labels.json",
            )
        )

    expected_complete = len(real_images) == 6
    if labels.get("is_complete_case") is not expected_complete:
        issues.append(
            DatasetIssue(
                case_folder=relative_case,
                case_id=case_id,
                case_type=case_type,
                severity="error",
                issue_type="is_complete_case_mismatch",
                message=f"is_complete_case should be {expected_complete} for {len(real_images)} images.",
                filename="labels.json",
            )
        )

    flagged_indexes = labels.get("expected_flagged_images", [])
    if flagged_indexes is not None:
        if not isinstance(flagged_indexes, list):
            issues.append(
                DatasetIssue(
                    case_folder=relative_case,
                    case_id=case_id,
                    case_type=case_type,
                    severity="error",
                    issue_type="invalid_expected_flagged_images",
                    message="expected_flagged_images must be an array.",
                    filename="labels.json",
                )
            )
        else:
            for value in flagged_indexes:
                if not isinstance(value, int) or value < 0 or value >= len(label_images):
                    issues.append(
                        DatasetIssue(
                            case_folder=relative_case,
                            case_id=case_id,
                            case_type=case_type,
                            severity="error",
                            issue_type="invalid_flagged_index",
                            message=f"Flagged index {value!r} is outside images array.",
                            filename="labels.json",
                        )
                    )

    flagged_filenames = labels.get("expected_flagged_filenames", [])
    if flagged_filenames is not None:
        if not isinstance(flagged_filenames, list):
            issues.append(
                DatasetIssue(
                    case_folder=relative_case,
                    case_id=case_id,
                    case_type=case_type,
                    severity="error",
                    issue_type="invalid_expected_flagged_filenames",
                    message="expected_flagged_filenames must be an array.",
                    filename="labels.json",
                )
            )
        else:
            for filename in flagged_filenames:
                if filename not in real_filenames:
                    issues.append(
                        DatasetIssue(
                            case_folder=relative_case,
                            case_id=case_id,
                            case_type=case_type,
                            severity="error",
                            issue_type="missing_flagged_filename",
                            message="expected_flagged_filenames references a file that does not exist.",
                            filename=str(filename),
                        )
                    )

    return labels, issues, False


def audit_dataset(dataset_dir: Path) -> tuple[dict[str, Any], list[DatasetIssue]]:
    case_dirs = numeric_case_dirs(dataset_dir)
    issues: list[DatasetIssue] = []
    summary: dict[str, Any] = {
        "total_folders_found": len(case_dirs),
        "folders_with_images": 0,
        "empty_folders_skipped": 0,
        "folders_with_labels": 0,
        "folders_missing_labels": 0,
        "valid_cases": 0,
        "cases_with_issues": 0,
        "issues_by_type": {},
    }

    for case_dir in case_dirs:
        has_images = bool(image_files(case_dir))
        if has_images:
            summary["folders_with_images"] += 1
        if (case_dir / "labels.json").exists():
            summary["folders_with_labels"] += 1
        labels, case_issues, is_empty = audit_case(case_dir, dataset_dir)
        issues.extend(case_issues)
        if is_empty:
            summary["empty_folders_skipped"] += 1
            continue
        if has_images and not (case_dir / "labels.json").exists():
            summary["folders_missing_labels"] += 1
        has_error_or_warning = any(issue.severity in {"warning", "error"} for issue in case_issues)
        if has_error_or_warning:
            summary["cases_with_issues"] += 1
        elif labels is not None:
            summary["valid_cases"] += 1

    issue_counts: dict[str, int] = {}
    for issue in issues:
        issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1
    summary["issues_by_type"] = issue_counts
    return summary, issues


def load_valid_cases(dataset_dir: Path) -> list[DatasetCase]:
    cases: list[DatasetCase] = []
    for case_dir in numeric_case_dirs(dataset_dir):
        labels, issues, is_empty = audit_case(case_dir, dataset_dir)
        if is_empty or labels is None:
            continue
        if any(issue.severity == "error" for issue in issues):
            continue
        cases.append(
            DatasetCase(
                case_folder=case_dir.name,
                case_path=case_dir,
                labels_path=case_dir / "labels.json",
                labels=labels,
            )
        )
    return cases


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
