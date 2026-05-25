from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROTECTED_CATEGORIES = [
    "same_product_normal",
    "same_product_closeups",
    "same_garment_type_different_color",
    "invalid_images",
    "one_wrong_image",
]

METRIC_KEYS = [
    "exact_match_accuracy",
    "acceptable_match_accuracy",
    "flagged_exact_match_accuracy",
    "flagged_partial_match_accuracy",
    "runtime_error_count",
]

LEAKAGE_PATTERNS = [
    "wrong_product",
    "05_wrong_product",
    "black_front",
    "black_back",
    "case_0107",
    "case_0095",
    "case_0096",
]


def load_baseline(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_delta(current: dict[str, Any], baseline: dict[str, Any], key: str) -> dict[str, Any]:
    current_value = current.get(key)
    baseline_value = baseline.get(key)
    delta = None
    if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
        delta = round(float(current_value) - float(baseline_value), 6)
    return {
        "baseline": baseline_value,
        "current": current_value,
        "delta": delta,
    }


def compare_to_baseline(summary: dict[str, Any], baseline_payload: dict[str, Any] | None) -> dict[str, Any]:
    if baseline_payload is None:
        return {
            "baseline_found": False,
            "warnings": ["No baseline file was found; regression deltas were not calculated."],
        }

    baseline_metrics = baseline_payload.get("metrics", {})
    baseline_by_type = baseline_payload.get("by_case_type", {})
    current_by_type = summary.get("by_case_type", {})

    metric_deltas = {
        key: _metric_delta(summary, baseline_metrics, key)
        for key in METRIC_KEYS
    }

    by_case_type: dict[str, Any] = {}
    for case_type, baseline_metrics_for_type in sorted(baseline_by_type.items()):
        current_metrics_for_type = current_by_type.get(case_type, {})
        by_case_type[case_type] = {
            key: _metric_delta(current_metrics_for_type, baseline_metrics_for_type, key)
            for key in METRIC_KEYS
            if key != "runtime_error_count"
        }
        by_case_type[case_type]["count"] = {
            "baseline": baseline_metrics_for_type.get("count"),
            "current": current_metrics_for_type.get("count"),
        }

    warnings = regression_warnings(summary, baseline_payload, by_case_type)
    return {
        "baseline_found": True,
        "baseline_id": baseline_payload.get("baseline_id"),
        "source_run": baseline_payload.get("source_run"),
        "metric_deltas": metric_deltas,
        "by_case_type": by_case_type,
        "confusion_matrix": summary.get("confusion_matrix", {}),
        "warnings": warnings,
    }


def scan_product_code_for_leakage(project_root: Path) -> list[str]:
    warnings: list[str] = []
    search_roots = [project_root / "app"]
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in LEAKAGE_PATTERNS:
                if pattern in text:
                    warnings.append(
                        f"Potential dataset leakage pattern '{pattern}' found in product code: {path.relative_to(project_root).as_posix()}"
                    )
    return warnings


def regression_warnings(
    summary: dict[str, Any],
    baseline_payload: dict[str, Any],
    by_case_type_delta: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    runtime_errors = summary.get("runtime_error_count")
    if runtime_errors != 0:
        warnings.append(f"runtime_error_count is {runtime_errors}; guardrail requires 0.")

    current_by_type = summary.get("by_case_type", {})
    guardrails = baseline_payload.get("guardrails", {})

    normal_acceptable = current_by_type.get("same_product_normal", {}).get("acceptable_match_accuracy")
    if isinstance(normal_acceptable, (int, float)) and normal_acceptable < guardrails.get("same_product_normal_acceptable_min", 0.95):
        warnings.append("same_product_normal acceptable accuracy regressed below guardrail.")

    closeup_acceptable = current_by_type.get("same_product_closeups", {}).get("acceptable_match_accuracy")
    if isinstance(closeup_acceptable, (int, float)) and closeup_acceptable < guardrails.get("same_product_closeups_acceptable_min", 0.95):
        warnings.append("same_product_closeups acceptable accuracy regressed below guardrail.")

    wrong_partial = current_by_type.get("one_wrong_image", {}).get("flagged_partial_match_accuracy")
    if isinstance(wrong_partial, (int, float)) and wrong_partial < guardrails.get("one_wrong_image_flagged_partial_min", 0.8):
        warnings.append("one_wrong_image flagged_partial accuracy dropped below guardrail.")

    for case_type in PROTECTED_CATEGORIES:
        metrics = by_case_type_delta.get(case_type, {})
        acceptable_delta = metrics.get("acceptable_match_accuracy", {}).get("delta")
        if isinstance(acceptable_delta, (int, float)) and acceptable_delta < -0.02:
            warnings.append(f"{case_type} acceptable accuracy dropped by {acceptable_delta}.")

    exact_delta = _delta_for(summary, baseline_payload.get("metrics", {}), "exact_match_accuracy")
    acceptable_delta = _delta_for(summary, baseline_payload.get("metrics", {}), "acceptable_match_accuracy")
    flagged_delta = _delta_for(summary, baseline_payload.get("metrics", {}), "flagged_exact_match_accuracy")
    if exact_delta is not None and flagged_delta is not None and exact_delta > 0 and flagged_delta < 0:
        warnings.append("Exact match improved while flagged_exact accuracy dropped; review possible overfitting.")
    if exact_delta is not None and acceptable_delta is not None and exact_delta != 0 and abs(acceptable_delta) < 0.000001:
        warnings.append("Exact match changed while acceptable match stayed stable; review separately, not as total failure.")

    return warnings


def _delta_for(current: dict[str, Any], baseline: dict[str, Any], key: str) -> float | None:
    current_value = current.get(key)
    baseline_value = baseline.get(key)
    if not isinstance(current_value, (int, float)) or not isinstance(baseline_value, (int, float)):
        return None
    return round(float(current_value) - float(baseline_value), 6)
