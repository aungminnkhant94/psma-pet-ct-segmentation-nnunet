"""Binary segmentation metrics used by the project."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage


def _binary(array: np.ndarray) -> np.ndarray:
    return np.asarray(array) > 0


def dice_score(prediction: np.ndarray, reference: np.ndarray) -> float:
    pred = _binary(prediction)
    ref = _binary(reference)
    denominator = int(pred.sum()) + int(ref.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(pred, ref).sum() / denominator)


def iou_score(prediction: np.ndarray, reference: np.ndarray) -> float:
    pred = _binary(prediction)
    ref = _binary(reference)
    union = int(np.logical_or(pred, ref).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, ref).sum() / union)


def precision_score(prediction: np.ndarray, reference: np.ndarray) -> float:
    pred = _binary(prediction)
    ref = _binary(reference)
    predicted = int(pred.sum())
    if predicted == 0:
        return 1.0 if not ref.any() else 0.0
    return float(np.logical_and(pred, ref).sum() / predicted)


def recall_score(prediction: np.ndarray, reference: np.ndarray) -> float:
    pred = _binary(prediction)
    ref = _binary(reference)
    positives = int(ref.sum())
    if positives == 0:
        return 1.0 if not pred.any() else 0.0
    return float(np.logical_and(pred, ref).sum() / positives)


def surface_distances(
    prediction: np.ndarray,
    reference: np.ndarray,
    spacing_zyx: Iterable[float],
) -> np.ndarray:
    pred = _binary(prediction)
    ref = _binary(reference)
    if not pred.any() and not ref.any():
        return np.array([0.0], dtype=np.float64)
    if not pred.any() or not ref.any():
        return np.array([np.inf], dtype=np.float64)

    connectivity = ndimage.generate_binary_structure(pred.ndim, 1)
    pred_surface = np.logical_xor(
        pred,
        ndimage.binary_erosion(pred, structure=connectivity, border_value=0),
    )
    ref_surface = np.logical_xor(
        ref,
        ndimage.binary_erosion(ref, structure=connectivity, border_value=0),
    )
    ref_distance = ndimage.distance_transform_edt(
        ~ref_surface, sampling=tuple(spacing_zyx)
    )
    pred_distance = ndimage.distance_transform_edt(
        ~pred_surface, sampling=tuple(spacing_zyx)
    )
    return np.concatenate(
        (ref_distance[pred_surface], pred_distance[ref_surface])
    ).astype(np.float64)


def hd95_score(
    prediction: np.ndarray,
    reference: np.ndarray,
    spacing_zyx: Iterable[float],
) -> float:
    distances = surface_distances(prediction, reference, spacing_zyx)
    if np.isinf(distances).any():
        return float("inf")
    return float(np.percentile(distances, 95))


@dataclass(frozen=True)
class CaseMetrics:
    case: str
    dice: float
    iou: float
    precision: float
    recall: float
    hd95_mm: float
    pred_voxels: int
    gt_voxels: int

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def evaluate_case(
    case: str,
    prediction: np.ndarray,
    reference: np.ndarray,
    spacing_zyx: Iterable[float],
) -> CaseMetrics:
    pred = _binary(prediction)
    ref = _binary(reference)
    return CaseMetrics(
        case=case,
        dice=dice_score(pred, ref),
        iou=iou_score(pred, ref),
        precision=precision_score(pred, ref),
        recall=recall_score(pred, ref),
        hd95_mm=hd95_score(pred, ref, spacing_zyx),
        pred_voxels=int(pred.sum()),
        gt_voxels=int(ref.sum()),
    )


def finite_summary(values: Iterable[float]) -> dict[str, int | float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }
