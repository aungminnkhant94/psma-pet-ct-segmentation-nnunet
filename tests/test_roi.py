import numpy as np

from psma_nnunet.roi import CropBox, bounding_box, crop_and_pad


def test_bounding_box_clips_to_image() -> None:
    mask = np.zeros((10, 11, 12), dtype=np.uint8)
    mask[0:2, 5:7, 10:12] = 1
    assert bounding_box(mask, margin=3) == CropBox(0, 5, 2, 10, 7, 12)


def test_crop_and_pad_reports_padding_before() -> None:
    array = np.ones((4, 6, 8), dtype=np.float32)
    padded, before = crop_and_pad(array, CropBox(0, 4, 0, 6, 0, 8), (8, 8, 8))
    assert padded.shape == (8, 8, 8)
    assert before == (2, 1, 0)
    assert padded.sum() == array.sum()
