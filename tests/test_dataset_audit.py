import json

from scripts.dataset_utils import audit_case, audit_dataset


def write_labels(case_dir, payload):
    (case_dir / "labels.json").write_text(json.dumps(payload), encoding="utf-8")


def valid_labels(filename="main_front_flat.jpeg"):
    return {
        "case_id": "case_0001",
        "case_type": "same_product_normal",
        "expected_status": "consistent",
        "image_count": 1,
        "is_complete_case": False,
        "images": [
            {
                "filename": filename,
                "view_type": "main_front_flat",
                "product_group": "A",
                "expected_flag": False,
            }
        ],
    }


def test_valid_labels_pass_dataset_audit(tmp_path):
    case_dir = tmp_path / "1"
    case_dir.mkdir()
    (case_dir / "main_front_flat.jpeg").write_bytes(b"not-used-by-audit")
    write_labels(case_dir, valid_labels())

    summary, issues = audit_dataset(tmp_path)

    assert summary["valid_cases"] == 1
    assert not [issue for issue in issues if issue.severity == "error"]


def test_missing_referenced_image_generates_issue(tmp_path):
    case_dir = tmp_path / "1"
    case_dir.mkdir()
    (case_dir / "main_front_flat.jpeg").write_bytes(b"not-used-by-audit")
    write_labels(case_dir, valid_labels(filename="missing.jpeg"))

    _labels, issues, is_empty = audit_case(case_dir, tmp_path)

    assert is_empty is False
    assert any(issue.issue_type == "missing_referenced_image" for issue in issues)
