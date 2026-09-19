#!/usr/bin/env python3
"""Evaluate NIfTI predictions with Dice, IoU, precision, recall, and HD95."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from psma_nnunet.metrics import CaseMetrics, evaluate_case, finite_summary
from psma_nnunet.roi import assert_same_geometry


FIELDS = [
    "case",
    "dice",
    "iou",
    "precision",
    "recall",
    "hd95_mm",
    "pred_voxels",
    "gt_voxels",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_pair(prediction_path: Path, reference_path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    prediction = sitk.ReadImage(str(prediction_path))
    reference = sitk.ReadImage(str(reference_path))
    assert_same_geometry([prediction, reference])
    pred_array = sitk.GetArrayFromImage(prediction)
    ref_array = sitk.GetArrayFromImage(reference)
    spacing_zyx = tuple(reversed(reference.GetSpacing()))
    return pred_array, ref_array, spacing_zyx


def metric_summary(rows: list[CaseMetrics]) -> dict[str, object]:
    return {
        "cases": len(rows),
        "metrics": {
            name: finite_summary(getattr(row, name) for row in rows)
            for name in ("dice", "iou", "precision", "recall", "hd95_mm")
        },
        "infinite_hd95_cases": [row.case for row in rows if not np.isfinite(row.hd95_mm)],
    }


def main() -> None:
    args = parse_args()
    prediction_names = {path.name for path in args.predictions.glob("*.nii.gz")}
    reference_names = {path.name for path in args.ground_truth.glob("*.nii.gz")}
    missing_predictions = sorted(reference_names - prediction_names)
    missing_references = sorted(prediction_names - reference_names)
    if args.strict and (missing_predictions or missing_references):
        raise RuntimeError(
            f"Unmatched files: {len(missing_predictions)} missing predictions, "
            f"{len(missing_references)} missing references."
        )

    rows: list[CaseMetrics] = []
    for name in sorted(prediction_names & reference_names):
        pred, ref, spacing = load_pair(args.predictions / name, args.ground_truth / name)
        rows.append(evaluate_case(name, pred, ref, spacing))
    if not rows:
        raise RuntimeError("No matched NIfTI files were found.")

    csv_path = args.output_prefix.with_name(args.output_prefix.name + "_per_case.csv")
    json_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(row.to_dict() for row in rows)
    summary = metric_summary(rows)
    summary.update(
        missing_predictions=missing_predictions,
        missing_references=missing_references,
    )
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
