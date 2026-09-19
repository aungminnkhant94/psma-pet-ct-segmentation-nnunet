#!/usr/bin/env python3
"""Aggregate one or more per-case metric CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from psma_nnunet.metrics import finite_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries: dict[str, object] = {}
    for path in args.csv:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        metrics: dict[str, object] = {}
        for name in ("dice", "iou", "precision", "recall", "hd95_mm", "hd95"):
            values = [float(row[name]) for row in rows if row.get(name) not in (None, "")]
            if values:
                metrics[name] = finite_summary(value for value in values if math.isfinite(value))
        if "iou" not in metrics and all(row.get("dice") not in (None, "") for row in rows):
            metrics["iou_derived_from_dice"] = finite_summary(
                (float(row["dice"]) / (2 - float(row["dice"]))) for row in rows
            )
        summaries[path.name] = {"cases": len(rows), "metrics": metrics}
    output = json.dumps(summaries, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
