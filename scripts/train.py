#!/usr/bin/env python3
"""Run one or more nnU-Net v2 folds, optionally from pretrained weights."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def parse_folds(value: str) -> list[str]:
    folds = [part.strip() for part in value.split(",") if part.strip()]
    if not folds or any(fold not in {"0", "1", "2", "3", "4", "all"} for fold in folds):
        raise argparse.ArgumentTypeError("Use a comma-separated subset of 0,1,2,3,4 or all.")
    return folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--folds", type=parse_folds, default=["0", "1", "2", "3", "4"])
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--pretrained-template", type=str)
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--preprocessed", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    environment = os.environ.copy()
    environment.update(
        nnUNet_raw=str(args.raw.resolve()),
        nnUNet_preprocessed=str(args.preprocessed.resolve()),
        nnUNet_results=str(args.results.resolve()),
    )
    for fold in args.folds:
        command = [
            "nnUNetv2_train",
            args.dataset,
            args.configuration,
            fold,
            "-tr",
            args.trainer,
            "-p",
            args.plans,
        ]
        if args.pretrained_template:
            weights = Path(args.pretrained_template.format(fold=fold)).expanduser().resolve()
            if not weights.is_file():
                raise FileNotFoundError(weights)
            command.extend(["-pretrained_weights", str(weights)])
        if args.continue_training:
            command.append("--c")
        subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
