# PSMA PET/CT tumour segmentation with nnU-Net v2

This repository is the cleaned, reproducible version of the final-year project
on automated tumour segmentation in PSMA PET/CT. It consolidates the recovered
GPU-server scripts, nnU-Net dataset definitions, training setup, evaluation
code, and aggregate results while excluding scans, checkpoints, server paths,
and private case identifiers.

The main experiments compare whole-body segmentation, union-of-lesions ROI
cropping, lesion-centred cropping, PET-only input, a residual-encoder preset,
and transfer learning from the public Dataset297 model to the private 5 mm
Dataset314 data.

## Repository contents

- `scripts/prepare_dataset.py` converts the recovered train/test layout to
  nnU-Net raw format and can reorder PET/CT channels.
- `scripts/build_roi_dataset.py` creates union-ROI or lesion-ROI datasets while
  preserving correct physical coordinates after padding.
- `scripts/make_splits.py` creates deterministic five-fold splits and can group
  lesion crops by patient.
- `scripts/preprocess.py`, `train.py`, and `predict.py` wrap the exact nnU-Net v2
  workflow used by the project.
- `scripts/evaluate.py` computes per-case Dice, IoU, precision, recall, and HD95.
- `configs/` records every recovered PSMA dataset definition and experiment.
- `results/` contains aggregate, identifier-free recovered metrics.
- `docs/REPRODUCIBILITY.md` contains end-to-end commands.
- `docs/RECOVERY_AUDIT.md` explains corrections and excluded files.

## Quick start

The recovered runtime was Python 3.10, PyTorch 2.5.1, CUDA 12.1, and nnU-Net v2.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==25.0.1
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -e .
```

Set the three paths required by nnU-Net:

```bash
export PROJECT_ROOT="$PWD"
export nnUNet_raw="$PROJECT_ROOT/workspace/nnUNet_raw"
export nnUNet_preprocessed="$PROJECT_ROOT/workspace/nnUNet_preprocessed"
export nnUNet_results="$PROJECT_ROOT/workspace/nnUNet_results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
```

Then follow [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). The raw medical
images are intentionally not distributed. The `.gitignore` prevents NIfTI
files, checkpoints, private metadata, predictions, and nnU-Net work directories
from being committed.

## Key recovered results

| Experiment | Model | Mean Dice | Mean IoU | Mean HD95 (mm) |
|---|---:|---:|---:|---:|
| Public whole-body external test | 5-fold ensemble | 0.6218 | 0.4848 | 69.02 |
| Public union-ROI external test | Fold 0 | 0.6518 | 0.5118 | 33.21 |
| Private whole-body held-out test | Fold 0 | 0.6214 | 0.4656 | 31.93 |
| Public-to-private transfer held-out test | 5-fold ensemble | 0.6419 | 0.4834 | 32.03 |
| Private lesion-ROI held-out lesion test | Fold 0 | 0.6070 | 0.4929 | 24.98 |

See [results/RESULTS.md](results/RESULTS.md) for every recovered fold and the
audit qualifications. This is research code and is not a medical device.

## Citation

If this code is used in research, cite nnU-Net:

> Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a
> self-configuring method for deep learning-based biomedical image segmentation.
> Nature Methods. 2021;18:203-211.

## License

The project code is released under the MIT License. nnU-Net and the datasets
retain their respective licenses and terms of use.
