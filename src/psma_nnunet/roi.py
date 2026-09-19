"""Geometry-safe ROI extraction for SimpleITK volumes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import SimpleITK as sitk


@dataclass(frozen=True)
class CropBox:
    z0: int
    z1: int
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return (
            slice(self.z0, self.z1),
            slice(self.y0, self.y1),
            slice(self.x0, self.x1),
        )


def bounding_box(
    mask: np.ndarray,
    margin: int,
    image_shape: tuple[int, int, int] | None = None,
) -> CropBox | None:
    coordinates = np.argwhere(np.asarray(mask) > 0)
    if coordinates.size == 0:
        return None
    shape = image_shape or tuple(int(x) for x in mask.shape)
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0) + 1
    starts = np.maximum(minimum - margin, 0)
    stops = np.minimum(maximum + margin, np.asarray(shape))
    return CropBox(
        int(starts[0]),
        int(stops[0]),
        int(starts[1]),
        int(stops[1]),
        int(starts[2]),
        int(stops[2]),
    )


def crop_and_pad(
    array: np.ndarray,
    box: CropBox,
    minimum_shape: tuple[int, int, int],
    pad_value: float = 0,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    cropped = np.asarray(array)[box.slices]
    missing = np.maximum(np.asarray(minimum_shape) - np.asarray(cropped.shape), 0)
    before = tuple(int(x) for x in missing // 2)
    after = tuple(int(x) for x in missing - np.asarray(before))
    padded = np.pad(
        cropped,
        tuple((before[i], after[i]) for i in range(3)),
        mode="constant",
        constant_values=pad_value,
    )
    return padded, before


def image_from_crop(
    array: np.ndarray,
    reference: "sitk.Image",
    box: CropBox,
    padding_before_zyx: tuple[int, int, int],
) -> "sitk.Image":
    import SimpleITK as sitk

    output = sitk.GetImageFromArray(array)
    output.SetSpacing(reference.GetSpacing())
    output.SetDirection(reference.GetDirection())

    z_pad, y_pad, x_pad = padding_before_zyx
    index_xyz = (
        box.x0 - x_pad,
        box.y0 - y_pad,
        box.z0 - z_pad,
    )
    output.SetOrigin(reference.TransformIndexToPhysicalPoint(index_xyz))
    return output


def assert_same_geometry(images: list["sitk.Image"], tolerance: float = 1e-5) -> None:
    if not images:
        raise ValueError("At least one image is required.")
    reference = images[0]
    for image in images[1:]:
        if image.GetSize() != reference.GetSize():
            raise ValueError("Input images have different voxel dimensions.")
        for left, right in (
            (image.GetSpacing(), reference.GetSpacing()),
            (image.GetOrigin(), reference.GetOrigin()),
            (image.GetDirection(), reference.GetDirection()),
        ):
            if not np.allclose(left, right, atol=tolerance, rtol=0):
                raise ValueError("Input images do not share physical geometry.")
