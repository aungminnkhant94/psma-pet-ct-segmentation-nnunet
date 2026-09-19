#!/usr/bin/env python3
"""Run nnU-Net v2 inference for a single fold or a fold ensemble."""

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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--folds", type=parse_folds, default=["0", "1", "2", "3", "4"])
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    parser.add_argument("--save-probabilities", action="store_true")
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
    command = [
        "nnUNetv2_predict",
        "-i",
        str(args.input.resolve()),
        "-o",
        str(args.output.resolve()),
        "-d",
        args.dataset,
        "-c",
        args.configuration,
        "-f",
        *args.folds,
        "-tr",
        args.trainer,
        "-p",
        args.plans,
        "-chk",
        args.checkpoint,
    ]
    if args.save_probabilities:
        command.append("--save_probabilities")
    subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
