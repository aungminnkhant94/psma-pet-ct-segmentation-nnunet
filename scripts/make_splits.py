#!/usr/bin/env python3
"""Create deterministic, patient-grouped cross-validation splits for nnU-Net."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


LESION_SUFFIX = re.compile(r"__lesion_\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--group-lesions-by-patient",
        action="store_true",
        help="Keep every __lesion_NNN crop from one patient in the same fold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("At least two folds are required.")
    cases = sorted(path.name.removesuffix(".nii.gz") for path in args.labels_dir.glob("*.nii.gz"))
    if len(cases) < args.folds:
        raise RuntimeError("There are fewer cases than requested folds.")

    groups: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        group = LESION_SUFFIX.sub("", case) if args.group_lesions_by_patient else case
        groups[group].append(case)
    group_names = sorted(groups)
    random.Random(args.seed).shuffle(group_names)
    bins: list[list[str]] = [[] for _ in range(args.folds)]
    for group in sorted(group_names, key=lambda name: len(groups[name]), reverse=True):
        target = min(range(args.folds), key=lambda index: len(bins[index]))
        bins[target].extend(groups[group])

    all_cases = set(cases)
    splits = []
    for validation in bins:
        validation_set = set(validation)
        splits.append(
            {
                "train": sorted(all_cases - validation_set),
                "val": sorted(validation_set),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "cases": len(cases),
                "groups": len(groups),
                "fold_sizes": [len(split["val"]) for split in splits],
                "seed": args.seed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
