#!/usr/bin/env python3
"""Build union-ROI or lesion-ROI nnU-Net datasets from whole-body cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from psma_nnunet.roi import (
    assert_same_geometry,
    bounding_box,
    crop_and_pad,
    image_from_crop,
)


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = tuple(int(item) for item in value.split(","))
    if len(parts) != 3 or any(item <= 0 for item in parts):
        raise argparse.ArgumentTypeError("Shape must be three positive z,y,x integers.")
    return parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("union", "lesion"), required=True)
    parser.add_argument("--margin", type=int, required=True)
    parser.add_argument("--minimum-shape", type=parse_shape, default=(96, 96, 96))
    parser.add_argument("--minimum-lesion-voxels", type=int, default=20)
    parser.add_argument("--channel-0", choices=("PET", "CT"), default="PET")
    parser.add_argument("--channel-1", choices=("PET", "CT"), default="CT")
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def load_case(source: Path, split: str, case: str) -> tuple[list[sitk.Image], sitk.Image]:
    images = [
        sitk.ReadImage(str(source / split / "img" / f"{case}_{channel:04d}.nii.gz"))
        for channel in (0, 1)
    ]
    segmentation = sitk.ReadImage(str(source / split / "seg" / f"{case}.nii.gz"))
    assert_same_geometry([*images, segmentation])
    return images, segmentation


def lesion_masks(mask: np.ndarray, minimum_voxels: int) -> tuple[list[np.ndarray], int]:
    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    components = sitk.ConnectedComponent(image)
    array = sitk.GetArrayFromImage(components)
    labels, counts = np.unique(array[array > 0], return_counts=True)
    accepted = [array == label for label, count in zip(labels, counts) if count >= minimum_voxels]
    skipped = int(sum(count < minimum_voxels for count in counts))
    return accepted, skipped


def write_crop(
    images: list[sitk.Image],
    segmentation: sitk.Image,
    mask: np.ndarray,
    destination: tuple[Path, Path],
    case: str,
    margin: int,
    minimum_shape: tuple[int, int, int],
) -> None:
    arrays = [sitk.GetArrayFromImage(image) for image in images]
    box = bounding_box(mask, margin)
    if box is None:
        raise ValueError(f"Mask is empty for {case}.")
    cropped: list[np.ndarray] = []
    padding: tuple[int, int, int] | None = None
    for array in [*arrays, mask.astype(np.uint8)]:
        result, before = crop_and_pad(array, box, minimum_shape)
        if padding is not None and before != padding:
            raise RuntimeError("Modalities produced inconsistent padding.")
        padding = before
        cropped.append(result)
    if padding is None:
        raise RuntimeError("No arrays were cropped.")

    image_dir, label_dir = destination
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for channel, (array, reference) in enumerate(zip(cropped[:2], images)):
        output = image_from_crop(array, reference, box, padding)
        sitk.WriteImage(output, str(image_dir / f"{case}_{channel:04d}.nii.gz"))
    label = image_from_crop(cropped[2].astype(np.uint8), segmentation, box, padding)
    sitk.WriteImage(label, str(label_dir / f"{case}.nii.gz"))


def source_cases(source: Path, split: str) -> list[str]:
    directory = source / split / "seg"
    if not directory.is_dir():
        return []
    return sorted(path.name.removesuffix(".nii.gz") for path in directory.glob("*.nii.gz"))


def main() -> None:
    args = parse_args()
    if args.margin < 0 or args.minimum_lesion_voxels < 1:
        raise ValueError("Margin must be non-negative and lesion size must be positive.")
    if args.channel_0 == args.channel_1:
        raise ValueError("The two channels must represent different modalities.")

    summary: dict[str, object] = {
        "mode": args.mode,
        "margin_voxels": args.margin,
        "minimum_shape_zyx": list(args.minimum_shape),
        "minimum_lesion_voxels": args.minimum_lesion_voxels if args.mode == "lesion" else None,
        "training_cases": 0,
        "test_cases": 0,
        "empty_masks_skipped": 0,
        "small_lesions_skipped": 0,
    }

    for split, image_name, label_name, counter in (
        ("train", "imagesTr", "labelsTr", "training_cases"),
        ("test", "imagesTs", "labelsTs", "test_cases"),
    ):
        for patient in source_cases(args.source, split):
            images, segmentation = load_case(args.source, split, patient)
            whole_mask = sitk.GetArrayFromImage(segmentation) > 0
            if not whole_mask.any():
                summary["empty_masks_skipped"] = int(summary["empty_masks_skipped"]) + 1
                continue
            if args.mode == "union":
                masks = [whole_mask]
            else:
                masks, skipped = lesion_masks(whole_mask, args.minimum_lesion_voxels)
                summary["small_lesions_skipped"] = int(summary["small_lesions_skipped"]) + skipped
            for index, mask in enumerate(masks):
                case = patient if args.mode == "union" else f"{patient}__lesion_{index:03d}"
                write_crop(
                    images,
                    segmentation,
                    mask,
                    (args.output / image_name, args.output / label_name),
                    case,
                    args.margin,
                    args.minimum_shape,
                )
                summary[counter] = int(summary[counter]) + 1

    if int(summary["training_cases"]) == 0:
        raise RuntimeError("No training crops were created.")
    dataset = {
        "channel_names": {"0": args.channel_0, "1": args.channel_1},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": int(summary["training_cases"]),
        "file_ending": ".nii.gz",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dataset.json").write_text(
        json.dumps(dataset, indent=2) + "\n", encoding="utf-8"
    )
    summary_path = args.summary or args.output / "build_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
