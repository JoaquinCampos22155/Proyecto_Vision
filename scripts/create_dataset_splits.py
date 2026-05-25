from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from dataset_utils import DatasetCase, ensure_output_dir, load_valid_cases, write_csv, write_json


SPLIT_RATIOS = {
    "calibration": 0.60,
    "validation": 0.25,
    "test": 0.15,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified dataset splits by case_type.")
    parser.add_argument("--dataset-dir", default="images", help="Dataset directory with numeric case folders.")
    parser.add_argument("--output-dir", default="reports", help="Directory where split reports are written.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for deterministic splits.")
    return parser.parse_args()


def allocate_counts(total: int) -> dict[str, int]:
    raw = {split: total * ratio for split, ratio in SPLIT_RATIOS.items()}
    counts = {split: int(value) for split, value in raw.items()}
    remaining = total - sum(counts.values())
    remainders = sorted(
        ((split, raw[split] - counts[split]) for split in SPLIT_RATIOS),
        key=lambda item: item[1],
        reverse=True,
    )
    for index in range(remaining):
        counts[remainders[index % len(remainders)][0]] += 1
    return counts


def group_cases_by_type(cases: Iterable[DatasetCase]) -> dict[str, list[DatasetCase]]:
    grouped: dict[str, list[DatasetCase]] = defaultdict(list)
    for case in cases:
        grouped[case.case_type].append(case)
    return dict(grouped)


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = ensure_output_dir(Path(args.output_dir))

    rng = random.Random(args.seed)
    cases = load_valid_cases(dataset_dir)
    grouped = group_cases_by_type(cases)

    split_cases: dict[str, list[str]] = {split: [] for split in SPLIT_RATIOS}
    case_paths: dict[str, str] = {}
    case_types: dict[str, str] = {}
    summary_rows: list[dict[str, object]] = []

    for case_type in sorted(grouped):
        case_group = sorted(grouped[case_type], key=lambda item: item.case_id)
        rng.shuffle(case_group)
        counts = allocate_counts(len(case_group))
        start = 0
        for split in SPLIT_RATIOS:
            stop = start + counts[split]
            selected = case_group[start:stop]
            start = stop
            for case in selected:
                split_cases[split].append(case.case_id)
                case_paths[case.case_id] = case.case_path.as_posix()
                case_types[case.case_id] = case.case_type
            summary_rows.append({"split": split, "case_type": case_type, "count": len(selected)})

    for split in split_cases:
        split_cases[split].sort()

    payload = {
        "seed": args.seed,
        "splits": split_cases,
        "case_paths": dict(sorted(case_paths.items())),
        "case_types": dict(sorted(case_types.items())),
    }
    write_json(output_dir / "dataset_splits.json", payload)
    write_csv(output_dir / "dataset_split_summary.csv", summary_rows, ["split", "case_type", "count"])

    print(f"Created dataset splits from: {dataset_dir}")
    print(f"Splits: {output_dir / 'dataset_splits.json'}")
    print(f"Summary: {output_dir / 'dataset_split_summary.csv'}")
    for split, case_ids in split_cases.items():
        print(f"{split}: {len(case_ids)} cases")


if __name__ == "__main__":
    main()
