#!/usr/bin/env python3
"""Convert the recovered train/test layout into an nnU-Net v2 raw dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a source tree containing train/img, train/seg and optional "
            "test/img, test/seg directories into nnU-Net v2 format."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channel-0", choices=("PET", "CT"), default="PET")
    parser.add_argument("--channel-1", choices=("PET", "CT", "NONE"), default="CT")
    parser.add_argument("--channel-0-source-index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--channel-1-source-index", type=int, choices=(0, 1), default=1)
    parser.add_argument("--copy", action="store_true", help="Copy instead of hard-linking.")
    return parser.parse_args()


def case_ids(segmentation_dir: Path) -> list[str]:
    return sorted(path.name.removesuffix(".nii.gz") for path in segmentation_dir.glob("*.nii.gz"))


def place(source: Path, destination: Path, copy_files: bool) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if copy_files:
        shutil.copy2(source, destination)
    else:
        try:
            destination.hardlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, destination)


def convert_split(
    source_root: Path,
    output_root: Path,
    source_split: str,
    image_target: str,
    label_target: str,
    copy_files: bool,
    source_indices: tuple[int, ...],
) -> int:
    image_dir = source_root / source_split / "img"
    label_dir = source_root / source_split / "seg"
    if not label_dir.is_dir():
        return 0
    identifiers = case_ids(label_dir)
    for identifier in identifiers:
        for channel, source_channel in enumerate(source_indices):
            place(
                image_dir / f"{identifier}_{source_channel:04d}.nii.gz",
                output_root / image_target / f"{identifier}_{channel:04d}.nii.gz",
                copy_files,
            )
        place(
            label_dir / f"{identifier}.nii.gz",
            output_root / label_target / f"{identifier}.nii.gz",
            copy_files,
        )
    return len(identifiers)


def main() -> None:
    args = parse_args()
    if args.channel_1 != "NONE" and args.channel_0 == args.channel_1:
        raise ValueError("The two channels must represent different modalities.")
    if args.channel_1 != "NONE" and args.channel_0_source_index == args.channel_1_source_index:
        raise ValueError("Each output channel must use a different source channel.")
    source_indices = (
        (args.channel_0_source_index,)
        if args.channel_1 == "NONE"
        else (args.channel_0_source_index, args.channel_1_source_index)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    training_count = convert_split(
        args.source,
        args.output,
        "train",
        "imagesTr",
        "labelsTr",
        args.copy,
        source_indices,
    )
    test_count = convert_split(
        args.source,
        args.output,
        "test",
        "imagesTs",
        "labelsTs",
        args.copy,
        source_indices,
    )
    if training_count == 0:
        raise RuntimeError("No training segmentations were found.")

    channel_names = {"0": args.channel_0}
    if args.channel_1 != "NONE":
        channel_names["1"] = args.channel_1
    dataset = {
        "channel_names": channel_names,
        "labels": {"background": 0, "tumor": 1},
        "numTraining": training_count,
        "file_ending": ".nii.gz",
    }
    (args.output / "dataset.json").write_text(
        json.dumps(dataset, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"training_cases": training_count, "test_cases": test_count}, indent=2))


if __name__ == "__main__":
    main()
