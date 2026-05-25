from __future__ import annotations

import argparse
from pathlib import Path

from dataset_utils import audit_dataset, ensure_output_dir, write_csv, write_json


ISSUE_COLUMNS = [
    "case_folder",
    "case_id",
    "case_type",
    "severity",
    "issue_type",
    "message",
    "filename",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit visual validation dataset labels.")
    parser.add_argument("--dataset-dir", default="images", help="Dataset directory with numeric case folders.")
    parser.add_argument("--output-dir", default="reports", help="Directory where audit reports are written.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = ensure_output_dir(Path(args.output_dir))

    summary, issues = audit_dataset(dataset_dir)
    write_json(output_dir / "dataset_audit_summary.json", summary)
    write_csv(
        output_dir / "dataset_audit_issues.csv",
        [issue.as_row() for issue in issues],
        ISSUE_COLUMNS,
    )

    print(f"Audited dataset: {dataset_dir}")
    print(f"Summary: {output_dir / 'dataset_audit_summary.json'}")
    print(f"Issues: {output_dir / 'dataset_audit_issues.csv'}")
    print(f"Valid cases: {summary['valid_cases']}")
    print(f"Cases with issues: {summary['cases_with_issues']}")


if __name__ == "__main__":
    main()
