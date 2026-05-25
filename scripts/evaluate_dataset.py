from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import sys
import uuid
from pathlib import Path
from typing import Any
from urllib import request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.dataset_utils import DatasetCase, ensure_output_dir, load_valid_cases, write_csv, write_json
    from scripts.evaluation_utils import (
        acceptable_match,
        build_confusion_matrix,
        exact_match,
        extract_predicted_flagged_details,
        flagged_exact_match,
        flagged_partial_match,
        score_value,
    )
    from scripts.regression_utils import compare_to_baseline, load_baseline, scan_product_code_for_leakage
except ModuleNotFoundError:
    from dataset_utils import DatasetCase, ensure_output_dir, load_valid_cases, write_csv, write_json
    from evaluation_utils import (
        acceptable_match,
        build_confusion_matrix,
        exact_match,
        extract_predicted_flagged_details,
        flagged_exact_match,
        flagged_partial_match,
        score_value,
    )
    from regression_utils import compare_to_baseline, load_baseline, scan_product_code_for_leakage


RESULT_COLUMNS = [
    "case_id",
    "case_folder",
    "case_type",
    "split",
    "image_count",
    "is_complete_case",
    "expected_status",
    "acceptable_status",
    "predicted_status",
    "exact_match",
    "acceptable_match",
    "expected_flagged_filenames",
    "predicted_flagged_filenames",
    "flagged_exact_match",
    "flagged_partial_match",
    "raw_consistency_score",
    "main_view_score",
    "detail_support_score",
    "color_consistency_score",
    "robust_consistency_score",
    "notes",
    "error",
]

ERROR_COLUMNS = [
    "case_id",
    "case_folder",
    "case_type",
    "expected_status",
    "acceptable_status",
    "predicted_status",
    "expected_flagged_filenames",
    "predicted_flagged_filenames",
    "predicted_flagged_issue_types",
    "predicted_flagged_severities",
    "predicted_flagged_reasons",
    "raw_consistency_score",
    "robust_consistency_score",
    "color_consistency_score",
    "error_type",
    "review_note",
]


class LocalUploadFile:
    def __init__(self, path: Path):
        self.path = path
        self.filename = path.name
        self.content_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"

    async def read(self) -> bytes:
        return self.path.read_bytes()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate visual validation dataset against the current pipeline.")
    parser.add_argument("--dataset-dir", default="images", help="Dataset directory with numeric case folders.")
    parser.add_argument("--output-dir", default="reports", help="Directory where evaluation reports are written.")
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "calibration", "validation", "test", "real_world"],
        help="Dataset split to evaluate. Uses reports/dataset_splits.json when present.",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="Optional FastAPI base URL used by --mode http or auto fallback.",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "direct", "http"],
        help="Evaluation mode. auto tries direct first and falls back to HTTP when --api-url is present.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name. Reports are written under output-dir/runs/run-name.",
    )
    parser.add_argument(
        "--baseline-file",
        default="docs/evaluation_baselines/baseline_v0_3_post_fix.json",
        help="Optional frozen baseline JSON used to produce regression and overfitting reports.",
    )
    return parser.parse_args()


def as_jsonable_response(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    raise TypeError(f"Unsupported response object: {type(response)!r}")


async def validate_direct(product_id: str | None, image_paths: list[Path]) -> dict[str, Any]:
    from app.config import Settings
    from app.services.validation_service import ProductImageValidationService

    service = ProductImageValidationService(Settings(min_images=1))
    uploads = [LocalUploadFile(path) for path in image_paths]
    response = await service.validate(product_id=product_id, uploads=uploads)
    return as_jsonable_response(response)


def smoke_test_direct_import() -> tuple[bool, str | None]:
    try:
        from app.config import Settings  # noqa: F401
        from app.services.validation_service import ProductImageValidationService  # noqa: F401
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def encode_multipart(product_id: str | None, image_paths: list[Path]) -> tuple[bytes, str]:
    boundary = f"----dataset-eval-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    def add_field(name: str, value: str) -> None:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )

    def add_file(name: str, path: Path) -> None:
        content_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{path.name}"\r\n'
                    f"Content-Type: {content_type}\r\n\r\n"
                ).encode("utf-8"),
                path.read_bytes(),
                b"\r\n",
            ]
        )

    if product_id:
        add_field("product_id", product_id)
    for image_path in image_paths:
        add_file("images", image_path)
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def validate_http(api_url: str, product_id: str | None, image_paths: list[Path]) -> dict[str, Any]:
    body, content_type = encode_multipart(product_id, image_paths)
    endpoint = api_url.rstrip("/") + "/validate-product-images"
    req = request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": content_type, "Content-Length": str(len(body))},
        method="POST",
    )
    with request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def split_lookup(output_dir: Path, split_name: str) -> tuple[dict[str, str], set[str] | None]:
    split_path = output_dir / "dataset_splits.json"
    if split_name == "all" or not split_path.exists():
        return {}, None
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    case_to_split: dict[str, str] = {}
    for split, case_ids in payload.get("splits", {}).items():
        for case_id in case_ids:
            case_to_split[case_id] = split
    return case_to_split, set(payload.get("splits", {}).get(split_name, []))


def expected_flagged_filenames(labels: dict[str, Any], ordered_filenames: list[str]) -> list[str]:
    explicit = labels.get("expected_flagged_filenames")
    if isinstance(explicit, list):
        return [str(filename) for filename in explicit]

    filenames: list[str] = []
    for index in labels.get("expected_flagged_images", []) or []:
        if isinstance(index, int) and 0 <= index < len(ordered_filenames):
            filenames.append(ordered_filenames[index])
    return filenames


def json_for_csv(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def score_average(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return round(sum(1 for row in rows if row.get(key) is True) / len(rows), 6)


def build_summary(rows: list[dict[str, Any]], total_cases_found: int) -> dict[str, Any]:
    successful_rows = [row for row in rows if not row.get("error") and row.get("predicted_status")]
    confusion = build_confusion_matrix()
    by_type_rows: dict[str, list[dict[str, Any]]] = {}

    for row in successful_rows:
        expected = row.get("expected_status")
        predicted = row.get("predicted_status")
        if expected in confusion and predicted in confusion[expected]:
            confusion[expected][predicted] += 1
        by_type_rows.setdefault(str(row.get("case_type") or "unknown"), []).append(row)

    by_case_type: dict[str, dict[str, Any]] = {}
    for case_type, case_rows in sorted(by_type_rows.items()):
        by_case_type[case_type] = {
            "count": len(case_rows),
            "exact_match_accuracy": score_average(case_rows, "exact_match"),
            "acceptable_match_accuracy": score_average(case_rows, "acceptable_match"),
            "flagged_exact_match_accuracy": score_average(case_rows, "flagged_exact_match"),
            "flagged_partial_match_accuracy": score_average(case_rows, "flagged_partial_match"),
        }

    worst_case_types = sorted(
        [
            {
                "case_type": case_type,
                "acceptable_match_accuracy": metrics["acceptable_match_accuracy"],
                "count": metrics["count"],
            }
            for case_type, metrics in by_case_type.items()
        ],
        key=lambda item: (item["acceptable_match_accuracy"], -item["count"]),
    )[:5]

    notes = [
        "Accuracies are calculated only over cases that were successfully evaluated.",
        "acceptable_match uses acceptable_status when present; otherwise it falls back to expected_status.",
    ]
    if not successful_rows:
        notes.append("No cases were successfully evaluated; accuracy values are null.")

    return {
        "evaluation_failed": False,
        "total_cases_found": total_cases_found,
        "total_cases_attempted": len(rows),
        "total_cases_successfully_evaluated": len(successful_rows),
        "runtime_error_count": len(rows) - len(successful_rows),
        "total_cases_evaluated": len(successful_rows),
        "exact_match_accuracy": score_average(successful_rows, "exact_match"),
        "acceptable_match_accuracy": score_average(successful_rows, "acceptable_match"),
        "flagged_exact_match_accuracy": score_average(successful_rows, "flagged_exact_match"),
        "flagged_partial_match_accuracy": score_average(successful_rows, "flagged_partial_match"),
        "by_case_type": by_case_type,
        "confusion_matrix": confusion,
        "worst_case_types": worst_case_types,
        "notes": notes,
    }


def failed_summary(total_cases_found: int, failure_reason: str) -> dict[str, Any]:
    return {
        "evaluation_failed": True,
        "failure_reason": failure_reason,
        "total_cases_found": total_cases_found,
        "total_cases_attempted": 0,
        "total_cases_successfully_evaluated": 0,
        "runtime_error_count": 0,
        "total_cases_evaluated": 0,
        "exact_match_accuracy": None,
        "acceptable_match_accuracy": None,
        "flagged_exact_match_accuracy": None,
        "flagged_partial_match_accuracy": None,
        "by_case_type": {},
        "confusion_matrix": build_confusion_matrix(),
        "worst_case_types": [],
        "notes": [
            "Evaluation aborted before processing cases.",
            "No accuracy was calculated because no cases were successfully evaluated.",
        ],
    }


def error_type(row: dict[str, Any]) -> str | None:
    status_bad = row.get("acceptable_match") is False
    flagged_bad = row.get("flagged_exact_match") is False
    if status_bad and flagged_bad:
        return "both"
    if status_bad:
        return "status_mismatch"
    if flagged_bad:
        return "flagged_mismatch"
    return None


def ordered_image_paths(case: DatasetCase, ordered_filenames: list[str]) -> list[Path]:
    return [case.case_path / filename for filename in ordered_filenames]


async def evaluate_case(
    case: DatasetCase,
    split_name: str,
    mode: str,
    api_url: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = case.labels
    ordered_filenames = [str(item.get("filename")) for item in labels.get("images", []) if item.get("filename")]
    expected_status = str(labels.get("expected_status"))
    acceptable_statuses = labels.get("acceptable_status") or [expected_status]
    expected_flagged = expected_flagged_filenames(labels, ordered_filenames)
    image_paths = ordered_image_paths(case, ordered_filenames)

    response: dict[str, Any] = {}
    error = ""
    try:
        product_id = labels.get("case_id")
        if mode == "http":
            if not api_url:
                raise ValueError("--api-url is required when --mode http is used.")
            response = await asyncio.to_thread(validate_http, api_url, product_id, image_paths)
        else:
            response = await validate_direct(product_id, image_paths)
    except Exception as exc:  # Continue evaluating remaining cases.
        error = f"{type(exc).__name__}: {exc}"

    predicted_status = response.get("status") if response else None
    predicted_flagged_details = (
        extract_predicted_flagged_details(response, ordered_filenames) if response else
        {"filenames": [], "issue_types": [], "severities": [], "reasons": []}
    )
    predicted_flagged = predicted_flagged_details["filenames"]

    exact = exact_match(expected_status, predicted_status)
    acceptable = acceptable_match(expected_status, acceptable_statuses, predicted_status)
    flagged_exact = flagged_exact_match(expected_flagged, predicted_flagged)
    flagged_partial = flagged_partial_match(expected_flagged, predicted_flagged)
    if error:
        exact = False
        acceptable = False
        flagged_exact = False if expected_flagged else flagged_exact
        flagged_partial = False if expected_flagged else flagged_partial

    row = {
        "case_id": case.case_id,
        "case_folder": case.case_path.as_posix(),
        "case_type": case.case_type,
        "split": split_name,
        "image_count": labels.get("image_count"),
        "is_complete_case": labels.get("is_complete_case"),
        "expected_status": expected_status,
        "acceptable_status": acceptable_statuses,
        "predicted_status": predicted_status,
        "exact_match": exact,
        "acceptable_match": acceptable,
        "expected_flagged_filenames": expected_flagged,
        "predicted_flagged_filenames": predicted_flagged,
        "predicted_flagged_issue_types": predicted_flagged_details["issue_types"],
        "predicted_flagged_severities": predicted_flagged_details["severities"],
        "predicted_flagged_reasons": predicted_flagged_details["reasons"],
        "flagged_exact_match": flagged_exact,
        "flagged_partial_match": flagged_partial,
        "raw_consistency_score": score_value(response, "raw_consistency_score") if response else None,
        "main_view_score": score_value(response, "main_view_score") if response else None,
        "detail_support_score": score_value(response, "detail_support_score") if response else None,
        "color_consistency_score": score_value(response, "color_consistency_score") if response else None,
        "robust_consistency_score": score_value(response, "robust_consistency_score") if response else None,
        "notes": labels.get("notes", ""),
        "error": error,
    }

    json_case = {
        "case_id": case.case_id,
        "case_folder": case.case_path.as_posix(),
        "case_type": case.case_type,
        "expected_status": expected_status,
        "acceptable_status": acceptable_statuses,
        "predicted_status": predicted_status,
        "exact_match": exact,
        "acceptable_match": acceptable,
        "expected_flagged_filenames": expected_flagged,
        "predicted_flagged_filenames": predicted_flagged,
        "system_response": response,
        "error": error,
    }
    return row, json_case


async def run() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = ensure_output_dir(Path(args.output_dir))
    split_source_dir = output_dir
    if args.run_name:
        output_dir = ensure_output_dir(output_dir / "runs" / args.run_name)
    error_cases_dir = ensure_output_dir(output_dir / "error_cases")
    for old_error in error_cases_dir.glob("*.json"):
        old_error.unlink()

    case_to_split, selected_case_ids = split_lookup(split_source_dir, args.split)
    cases = load_valid_cases(dataset_dir)
    if selected_case_ids is not None:
        cases = [case for case in cases if case.case_id in selected_case_ids]

    effective_mode = args.mode
    if args.mode == "http":
        if not args.api_url:
            reason = "ValueError: --api-url is required when --mode http is used."
            print(f"Evaluation aborted: {reason}")
            summary = failed_summary(len(cases), reason)
            write_json(output_dir / "evaluation_summary.json", summary)
            write_regression_reports(output_dir, summary, Path(args.baseline_file))
            write_csv(output_dir / "evaluation_results.csv", [], RESULT_COLUMNS)
            write_json(output_dir / "evaluation_results.json", {"cases": []})
            write_csv(output_dir / "evaluation_errors.csv", [], ERROR_COLUMNS)
            return
    else:
        import_ok, import_error = smoke_test_direct_import()
        if import_ok:
            effective_mode = "direct"
        elif args.mode == "auto" and args.api_url:
            effective_mode = "http"
            print(f"Direct import failed ({import_error}); falling back to HTTP mode.")
        else:
            reason = import_error or "Unknown import error"
            print(
                "Evaluation aborted: could not import app modules. "
                "Make sure you are running from the project root or that PYTHONPATH includes the project root."
            )
            summary = failed_summary(len(cases), reason)
            write_json(output_dir / "evaluation_summary.json", summary)
            write_regression_reports(output_dir, summary, Path(args.baseline_file))
            write_csv(output_dir / "evaluation_results.csv", [], RESULT_COLUMNS)
            write_json(output_dir / "evaluation_results.json", {"cases": []})
            write_csv(output_dir / "evaluation_errors.csv", [], ERROR_COLUMNS)
            return

    result_rows: list[dict[str, Any]] = []
    json_cases: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for case in sorted(cases, key=lambda item: item.case_id):
        actual_split = case_to_split.get(case.case_id, "all")
        row, json_case = await evaluate_case(case, actual_split, effective_mode, args.api_url)
        result_rows.append(row)
        json_cases.append(json_case)

        kind = error_type(row)
        if kind:
            error_row = {
                "case_id": row["case_id"],
                "case_folder": row["case_folder"],
                "case_type": row["case_type"],
                "expected_status": row["expected_status"],
                "acceptable_status": row["acceptable_status"],
                "predicted_status": row["predicted_status"],
                "expected_flagged_filenames": row["expected_flagged_filenames"],
                "predicted_flagged_filenames": row["predicted_flagged_filenames"],
                "predicted_flagged_issue_types": row["predicted_flagged_issue_types"],
                "predicted_flagged_severities": row["predicted_flagged_severities"],
                "predicted_flagged_reasons": row["predicted_flagged_reasons"],
                "raw_consistency_score": row["raw_consistency_score"],
                "robust_consistency_score": row["robust_consistency_score"],
                "color_consistency_score": row["color_consistency_score"],
                "error_type": kind,
                "review_note": row["error"] or "Review expected vs predicted output.",
            }
            error_rows.append(error_row)
            write_json(
                error_cases_dir / f"{case.case_id}.json",
                {
                    "labels": case.labels,
                    "system_response": json_case.get("system_response", {}),
                    "comparison": {
                        "exact_match": row["exact_match"],
                        "acceptable_match": row["acceptable_match"],
                        "flagged_exact_match": row["flagged_exact_match"],
                        "flagged_partial_match": row["flagged_partial_match"],
                        "error_type": kind,
                    },
                    "expected_flagged_filenames": row["expected_flagged_filenames"],
                    "predicted_flagged_filenames": row["predicted_flagged_filenames"],
                    "error": row["error"],
                },
            )

    csv_rows = [
        {key: json_for_csv(row.get(key)) for key in RESULT_COLUMNS}
        for row in result_rows
    ]
    csv_error_rows = [
        {key: json_for_csv(row.get(key)) for key in ERROR_COLUMNS}
        for row in error_rows
    ]

    write_csv(output_dir / "evaluation_results.csv", csv_rows, RESULT_COLUMNS)
    write_json(output_dir / "evaluation_results.json", {"cases": json_cases})
    summary = build_summary(result_rows, len(cases))
    write_json(output_dir / "evaluation_summary.json", summary)
    write_regression_reports(output_dir, summary, Path(args.baseline_file))
    write_csv(output_dir / "evaluation_errors.csv", csv_error_rows, ERROR_COLUMNS)

    print(f"Evaluated {len(result_rows)} cases from: {dataset_dir} using {effective_mode} mode")
    print(f"Results: {output_dir / 'evaluation_results.csv'}")
    print(f"Summary: {output_dir / 'evaluation_summary.json'}")
    print(f"Errors: {output_dir / 'evaluation_errors.csv'}")


def write_regression_reports(output_dir: Path, summary: dict[str, Any], baseline_file: Path) -> None:
    baseline_payload = load_baseline(baseline_file)
    regression_report = compare_to_baseline(summary, baseline_payload)
    leakage_warnings = scan_product_code_for_leakage(PROJECT_ROOT)
    overfitting_warnings = list(regression_report.get("warnings", []))
    overfitting_warnings.extend(leakage_warnings)
    if regression_report.get("baseline_found") and not overfitting_warnings:
        overfitting_warnings.append("No protected-category regression warnings were detected against the frozen baseline.")

    regression_report["leakage_warnings"] = leakage_warnings
    write_json(output_dir / "regression_report.json", regression_report)
    write_json(
        output_dir / "no_overfitting_report.json",
        {
            "baseline_file": baseline_file.as_posix(),
            "validation_metrics": {
                "exact_match_accuracy": summary.get("exact_match_accuracy"),
                "acceptable_match_accuracy": summary.get("acceptable_match_accuracy"),
                "flagged_exact_match_accuracy": summary.get("flagged_exact_match_accuracy"),
                "flagged_partial_match_accuracy": summary.get("flagged_partial_match_accuracy"),
                "runtime_error_count": summary.get("runtime_error_count"),
            },
            "test_metrics": None,
            "delta_vs_baseline": regression_report.get("metric_deltas"),
            "protected_category_regression_warnings": regression_report.get("warnings", []),
            "possible_overfitting_warnings": overfitting_warnings,
            "notes": [
                "This report compares the current run to the frozen validation baseline.",
                "If a separate test run exists, compare it manually before using validation gains as evidence of production readiness.",
            ],
        },
    )


if __name__ == "__main__":
    asyncio.run(run())
