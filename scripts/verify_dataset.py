#!/usr/bin/env python3
"""Check nnU-Net raw filenames, case counts, and image geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import SimpleITK as sitk

from psma_nnunet.roi import assert_same_geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    return parser.parse_args()


def main() -> None:
    root = parse_args().dataset
    metadata = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    channels = sorted(int(channel) for channel in metadata["channel_names"])
    labels = sorted((root / "labelsTr").glob("*.nii.gz"))
    if len(labels) != int(metadata["numTraining"]):
        raise RuntimeError("dataset.json numTraining does not match labelsTr.")
    for label_path in labels:
        case = label_path.name.removesuffix(".nii.gz")
        image_paths = [root / "imagesTr" / f"{case}_{channel:04d}.nii.gz" for channel in channels]
        missing = [str(path) for path in image_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing modalities: " + ", ".join(missing))
        images = [sitk.ReadImage(str(path)) for path in image_paths]
        assert_same_geometry([*images, sitk.ReadImage(str(label_path))])
    print(json.dumps({"dataset": root.name, "training_cases": len(labels), "status": "ok"}, indent=2))


if __name__ == "__main__":
    main()
