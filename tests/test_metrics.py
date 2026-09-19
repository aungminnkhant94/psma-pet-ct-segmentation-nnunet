import numpy as np

from psma_nnunet.metrics import dice_score, hd95_score, iou_score, precision_score, recall_score


def test_identical_masks_are_perfect() -> None:
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[1:4, 1:4, 1:4] = 1
    assert dice_score(mask, mask) == 1.0
    assert iou_score(mask, mask) == 1.0
    assert precision_score(mask, mask) == 1.0
    assert recall_score(mask, mask) == 1.0
    assert hd95_score(mask, mask, (1.0, 1.0, 1.0)) == 0.0


def test_disjoint_single_voxels() -> None:
    left = np.zeros((3, 3, 3), dtype=np.uint8)
    right = np.zeros_like(left)
    left[0, 0, 0] = 1
    right[0, 0, 1] = 1
    assert dice_score(left, right) == 0.0
    assert iou_score(left, right) == 0.0
    assert hd95_score(left, right, (1.0, 1.0, 1.0)) == 1.0


def test_empty_mask_conventions() -> None:
    empty = np.zeros((2, 2, 2), dtype=np.uint8)
    positive = empty.copy()
    positive[0, 0, 0] = 1
    assert dice_score(empty, empty) == 1.0
    assert iou_score(empty, empty) == 1.0
    assert hd95_score(empty, empty, (1.0, 1.0, 1.0)) == 0.0
    assert dice_score(empty, positive) == 0.0
    assert np.isinf(hd95_score(empty, positive, (1.0, 1.0, 1.0)))
