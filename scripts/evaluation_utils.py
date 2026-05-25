from __future__ import annotations

from typing import Any


VALID_STATUSES = ("consistent", "needs_review", "inconsistent")


def exact_match(expected_status: str, predicted_status: str | None) -> bool:
    return predicted_status == expected_status


def acceptable_match(expected_status: str, acceptable_status: list[str] | None, predicted_status: str | None) -> bool:
    accepted = acceptable_status or [expected_status]
    return predicted_status in accepted


def flagged_exact_match(expected: list[str], predicted: list[str]) -> bool:
    return set(expected) == set(predicted)


def flagged_partial_match(expected: list[str], predicted: list[str]) -> bool:
    if not expected:
        return not predicted
    return bool(set(expected) & set(predicted))


def extract_predicted_flagged_filenames(response: dict[str, Any], ordered_filenames: list[str]) -> list[str]:
    filenames: list[str] = []
    seen: set[str] = set()
    items = response.get("flagged_images")
    if items is None:
        items = response.get("suspicious_images", [])

    if not isinstance(items, list):
        return []

    for item in items:
        if not isinstance(item, dict):
            continue
        filename = item.get("filename") or item.get("original_filename")
        if filename is None and isinstance(item.get("image_index"), int):
            index = item["image_index"]
            if 0 <= index < len(ordered_filenames):
                filename = ordered_filenames[index]
        if filename is not None and filename not in seen:
            seen.add(str(filename))
            filenames.append(str(filename))
    return filenames


def extract_predicted_flagged_details(
    response: dict[str, Any],
    ordered_filenames: list[str],
) -> dict[str, list[str]]:
    details = {
        "filenames": [],
        "issue_types": [],
        "severities": [],
        "reasons": [],
    }
    items = response.get("flagged_images")
    if items is None:
        items = response.get("suspicious_images", [])
    if not isinstance(items, list):
        return details

    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        filename = item.get("filename") or item.get("original_filename")
        if filename is None and isinstance(item.get("image_index"), int):
            index = item["image_index"]
            if 0 <= index < len(ordered_filenames):
                filename = ordered_filenames[index]
        if filename is None:
            continue
        filename = str(filename)
        if filename in seen:
            continue
        seen.add(filename)
        details["filenames"].append(filename)
        details["issue_types"].append(str(item.get("issue_type") or "unknown"))
        details["severities"].append(str(item.get("severity") or "unknown"))
        details["reasons"].append(str(item.get("reason") or ""))
    return details


def score_value(response: dict[str, Any], score_name: str) -> float | None:
    scores = response.get("scores")
    if isinstance(scores, dict) and score_name in scores:
        return scores.get(score_name)
    return response.get(score_name)


def build_confusion_matrix() -> dict[str, dict[str, int]]:
    return {expected: {predicted: 0 for predicted in VALID_STATUSES} for expected in VALID_STATUSES}
