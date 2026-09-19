# File manifest

| File | Purpose |
|---|---|
| `README.md` | Project overview, installation, result highlights, and safety statement |
| `LICENSE` | MIT license for repository code |
| `requirements.txt` | Exact Python dependency versions and recovered CUDA PyTorch install command |
| `pyproject.toml` | Installable package metadata and test configuration |
| `.gitignore` | Prevents scans, checkpoints, predictions, logs, and private metadata from entering Git |
| `.github/workflows/ci.yml` | Lightweight syntax and unit-test checks on pushes and pull requests |
| `configs/experiments.yaml` | Consolidated experiment IDs, channel orders, ROI settings, and plan choices |
| `configs/datasets/*.json` | Final nnU-Net `dataset.json` definitions for all nine PSMA datasets |
| `src/psma_nnunet/metrics.py` | Dice, IoU, precision, recall, surface distance, HD95, and aggregation |
| `src/psma_nnunet/roi.py` | ROI boxes, padding, geometry validation, and correct physical origins |
| `scripts/prepare_dataset.py` | Whole-body nnU-Net dataset conversion and PET/CT channel reordering |
| `scripts/build_roi_dataset.py` | Union-ROI and connected-component lesion-ROI generation |
| `scripts/make_splits.py` | Deterministic cross-validation with optional patient grouping |
| `scripts/verify_dataset.py` | Filename, count, and physical-geometry checks |
| `scripts/preprocess.py` | nnU-Net fingerprint/planning/preprocessing wrapper |
| `scripts/train.py` | Multi-fold training and pretrained checkpoint wrapper |
| `scripts/predict.py` | Fold or multi-fold ensemble inference wrapper |
| `scripts/evaluate.py` | Complete per-case and aggregate segmentation evaluation |
| `scripts/summarize_csv.py` | Re-aggregation of saved per-case metric CSVs |
| `results/heldout_metrics.csv` | Identifier-free held-out/external test results |
| `results/transfer_metrics_by_fold.csv` | Dataset314 fold 0-4 and ensemble held-out results |
| `results/nnunet_validation_metrics.csv` | Recovered nnU-Net validation metrics and audit statuses |
| `results/RESULTS.md` | Human-readable results, provenance, and qualifications |
| `docs/REPRODUCIBILITY.md` | Exact setup, preprocessing, training, inference, and evaluation commands |
| `docs/RECOVERY_AUDIT.md` | Included, corrected, and excluded recovered material |
| `docs/PUBLISHING.md` | Exact Git initialization and GitHub CLI publication commands |
| `tests/test_metrics.py` | Unit tests for overlap and empty-mask conventions |
| `tests/test_roi.py` | Unit tests for ROI bounds and symmetric padding |
