# Recovered results

These tables distinguish held-out testing from nnU-Net validation. IoU in the
held-out tables was calculated per case as `Dice / (2 - Dice)` and then averaged;
it was not calculated from the aggregate mean Dice.

## Held-out and external tests

| Dataset / experiment | Model | n | Mean Dice | Median Dice | Mean IoU | Mean HD95 (mm) | Median HD95 (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Public whole-body (Dataset297) | Fold 2 | 100 | 0.6142 | 0.6851 | 0.4755 | 73.29 | 25.85 |
| Public whole-body (Dataset297) | 5-fold ensemble | 100 | 0.6218 | 0.6963 | 0.4848 | 69.02 | 25.43 |
| Public union ROI (Dataset300) | Fold 0 | 100 | 0.6518 | 0.7162 | 0.5118 | 33.21 | 15.91 |
| Private whole-body baseline (Dataset298) | Fold 0 | 8 | 0.6214 | 0.6794 | 0.4656 | 31.93 | 23.03 |
| Private final model, as stated in final report | 5-fold ensemble | 8 | 0.6465 | 0.6925 | 0.4891 | 41.09 | 39.54 |
| Private lesion ROI (Dataset313) | Fold 0 | 106 lesions | 0.6070 | 0.7085 | 0.4929 | 24.98 | 7.07 |
| Public-to-private transfer (Dataset314) | 5-fold ensemble | 8 | 0.6419 | 0.6749 | 0.4834 | 32.03 | 27.24 |
| Private PET-only (Dataset399) | Fold 0 | 8 | 0.6161 | 0.6595 | 0.4555 | not computed | not computed |
| Private residual encoder (Dataset398) | Fold 0 | 8 | 0.5715 | 0.6073 | 0.4118 | not computed | not computed |

The final-report-only row is retained because it was part of the submitted work,
but its original per-case file was not present in the recovered server export.
Every other row above is backed by a recovered CSV or JSON artifact.

## Transfer model: same held-out cases by fold

| Model | Mean Dice | Median Dice | Mean IoU | Mean HD95 (mm) | Median HD95 (mm) |
|---|---:|---:|---:|---:|---:|
| Fold 0 | 0.6147 | 0.6491 | 0.4547 | 38.28 | 37.40 |
| Fold 1 | 0.6473 | 0.6790 | 0.4888 | 31.61 | 27.72 |
| Fold 2 | 0.6341 | 0.6150 | 0.4700 | 29.01 | 20.57 |
| Fold 3 | 0.6411 | 0.6844 | 0.4851 | 30.13 | 25.97 |
| Fold 4 | 0.6418 | 0.6825 | 0.4823 | 28.81 | 18.97 |
| 5-fold ensemble | 0.6419 | 0.6749 | 0.4834 | 32.03 | 27.24 |

## Important audit findings

- The discarded `Dataset314` pre-fix fold-0 output used the wrong channel order
  and produced near-zero Dice. It is not included in the result tables.
- The historical lesion-ROI validation split distributed lesions rather than
  patients, so lesions from the same patient appeared in train and validation.
  Its validation score is labeled in `nnunet_validation_metrics.csv`; the final
  code now creates patient-grouped splits. The separate held-out lesion test used
  distinct test patients and remains reported above.
- The recovered Dataset297 fold-0 validation summary contains 91 records although
  the split file defines 54 validation cases. It is marked stale and excluded
  from comparisons. External-test metrics are unaffected.
- HD95 was not saved for the PET-only and residual-encoder held-out experiments,
  so those cells are intentionally blank rather than reconstructed.
- No held-out metric artifact was recovered for the public lesion-full
  Dataset299 or private union-ROI Dataset312 experiments, so no result is claimed
  for either dataset.
