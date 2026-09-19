#!/usr/bin/env python3
"""Run nnU-Net v2 fingerprinting, planning, and preprocessing."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset ID or name, for example 297.")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--planner", help="Optional experiment planner class.")
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
        "nnUNetv2_plan_and_preprocess",
        "-d",
        args.dataset,
        "-c",
        args.configuration,
        "--verify_dataset_integrity",
        "-np",
        str(args.processes),
    ]
    if args.planner:
        command.extend(["-pl", args.planner])
    subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
