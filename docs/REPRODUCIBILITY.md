# Reproducibility guide

Run every command from the repository root after completing the environment
setup in the README. The examples use the same dataset identifiers and channel
orders recovered from the server.

## 1. Expected source layout

The conversion and ROI scripts expect this input structure:

```text
data/public_psma18f_split/
├── train/img/CASE_0000.nii.gz
├── train/img/CASE_0001.nii.gz
├── train/seg/CASE.nii.gz
├── test/img/CASE_0000.nii.gz
├── test/img/CASE_0001.nii.gz
└── test/seg/CASE.nii.gz

data/private_psma5mm_split/
└── same structure
```

For the public source, channel 0 is PET and channel 1 is CT. For the original
private source, channel 0 is CT and channel 1 is PET.

## 2. Whole-body datasets

Create the public Dataset297 and private baseline Dataset298:

```bash
python scripts/prepare_dataset.py \
  --source data/public_psma18f_split \
  --output "$nnUNet_raw/Dataset297_PSMA18F" \
  --channel-0 PET --channel-1 CT

python scripts/prepare_dataset.py \
  --source data/private_psma5mm_split \
  --output "$nnUNet_raw/Dataset298_PSMA5mm" \
  --channel-0 CT --channel-1 PET

python scripts/verify_dataset.py "$nnUNet_raw/Dataset297_PSMA18F"
python scripts/verify_dataset.py "$nnUNet_raw/Dataset298_PSMA5mm"
```

Dataset314 uses the same private images but swaps the files so its input order
matches the public pretrained model: PET in channel 0 and CT in channel 1.

```bash
python scripts/prepare_dataset.py \
  --source data/private_psma5mm_split \
  --output "$nnUNet_raw/Dataset314_PSMA5mm_FROM_D297_WHOLEPATIENT" \
  --channel-0 PET --channel-1 CT \
  --channel-0-source-index 1 --channel-1-source-index 0

python scripts/verify_dataset.py \
  "$nnUNet_raw/Dataset314_PSMA5mm_FROM_D297_WHOLEPATIENT"
```

## 3. ROI and ablation datasets

Public union ROI, using the recovered 24-voxel margin and 96³ minimum crop:

```bash
python scripts/build_roi_dataset.py \
  --source data/public_psma18f_split \
  --output "$nnUNet_raw/Dataset300_PSMA18F_UNIONROI" \
  --mode union --margin 24 --minimum-shape 96,96,96 \
  --channel-0 PET --channel-1 CT
```

Public lesion ROI, using the recovered 12-voxel margin and 50-voxel filter:

```bash
python scripts/build_roi_dataset.py \
  --source data/public_psma18f_split \
  --output "$nnUNet_raw/Dataset299_PSMA18F_LESION_FULL" \
  --mode lesion --margin 12 --minimum-shape 96,96,96 \
  --minimum-lesion-voxels 50 --channel-0 PET --channel-1 CT
```

Private union ROI and lesion ROI:

```bash
python scripts/build_roi_dataset.py \
  --source data/private_psma5mm_split \
  --output "$nnUNet_raw/Dataset312_PSMA5mm_UNIONROI" \
  --mode union --margin 24 --minimum-shape 96,96,96 \
  --channel-0 CT --channel-1 PET

python scripts/build_roi_dataset.py \
  --source data/private_psma5mm_split \
  --output "$nnUNet_raw/Dataset313_PSMA5mm_LESIONROI" \
  --mode lesion --margin 24 --minimum-shape 96,96,96 \
  --minimum-lesion-voxels 20 --channel-0 CT --channel-1 PET
```

PET-only Dataset399:

```bash
python scripts/prepare_dataset.py \
  --source data/private_psma5mm_split \
  --output "$nnUNet_raw/Dataset399_PSMA5mm_PETONLY" \
  --channel-0 PET --channel-1 NONE --channel-0-source-index 1
```

Dataset398 uses the same two-channel raw files as Dataset298 under a new dataset
name so that the residual-encoder plans remain separate:

```bash
python scripts/prepare_dataset.py \
  --source data/private_psma5mm_split \
  --output "$nnUNet_raw/Dataset398_PSMA5mm_RESENC" \
  --channel-0 CT --channel-1 PET
```

## 4. Planning and preprocessing

```bash
for dataset in 297 298 299 300 312 313 314 399; do
  python scripts/preprocess.py "$dataset" \
    --raw "$nnUNet_raw" \
    --preprocessed "$nnUNet_preprocessed" \
    --results "$nnUNet_results" \
    --configuration 3d_fullres --processes 8
done

python scripts/preprocess.py 398 \
  --raw "$nnUNet_raw" \
  --preprocessed "$nnUNet_preprocessed" \
  --results "$nnUNet_results" \
  --configuration 3d_fullres --processes 8 \
  --planner nnUNetPlannerResEncL
```

Create deterministic splits after preprocessing and before training:

```bash
for dataset in \
  Dataset297_PSMA18F \
  Dataset298_PSMA5mm \
  Dataset300_PSMA18F_UNIONROI \
  Dataset312_PSMA5mm_UNIONROI \
  Dataset314_PSMA5mm_FROM_D297_WHOLEPATIENT \
  Dataset398_PSMA5mm_RESENC \
  Dataset399_PSMA5mm_PETONLY; do
  python scripts/make_splits.py \
    --labels-dir "$nnUNet_raw/$dataset/labelsTr" \
    --output "$nnUNet_preprocessed/$dataset/splits_final.json" \
    --folds 5 --seed 12345
done

python scripts/make_splits.py \
  --labels-dir "$nnUNet_raw/Dataset313_PSMA5mm_LESIONROI/labelsTr" \
  --output "$nnUNet_preprocessed/Dataset313_PSMA5mm_LESIONROI/splits_final.json" \
  --folds 5 --seed 12345 --group-lesions-by-patient

python scripts/make_splits.py \
  --labels-dir "$nnUNet_raw/Dataset299_PSMA18F_LESION_FULL/labelsTr" \
  --output "$nnUNet_preprocessed/Dataset299_PSMA18F_LESION_FULL/splits_final.json" \
  --folds 5 --seed 12345 --group-lesions-by-patient
```

## 5. Training

Train the public whole-body source model on all five folds:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py 297 \
  --folds 0,1,2,3,4 --configuration 3d_fullres \
  --trainer nnUNetTrainer --plans nnUNetPlans \
  --raw "$nnUNet_raw" --preprocessed "$nnUNet_preprocessed" \
  --results "$nnUNet_results"
```

Fine-tune Dataset314 from the matching Dataset297 fold checkpoints:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py 314 \
  --folds 0,1,2,3,4 --configuration 3d_fullres \
  --trainer nnUNetTrainer --plans nnUNetPlans \
  --pretrained-template "$nnUNet_results/Dataset297_PSMA18F/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{fold}/checkpoint_final.pth" \
  --raw "$nnUNet_raw" --preprocessed "$nnUNet_preprocessed" \
  --results "$nnUNet_results"
```

Train the recovered single-fold baselines:

```bash
for dataset in 298 300 312 313 399; do
  CUDA_VISIBLE_DEVICES=0 python scripts/train.py "$dataset" \
    --folds 0 --configuration 3d_fullres \
    --trainer nnUNetTrainer --plans nnUNetPlans \
    --raw "$nnUNet_raw" --preprocessed "$nnUNet_preprocessed" \
    --results "$nnUNet_results"
done

CUDA_VISIBLE_DEVICES=0 python scripts/train.py 398 \
  --folds 0 --configuration 3d_fullres \
  --trainer nnUNetTrainer --plans nnUNetResEncUNetLPlans \
  --raw "$nnUNet_raw" --preprocessed "$nnUNet_preprocessed" \
  --results "$nnUNet_results"
```

## 6. Inference and evaluation

Five-fold transfer ensemble on the private held-out cases:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/predict.py 314 \
  --input "$nnUNet_raw/Dataset314_PSMA5mm_FROM_D297_WHOLEPATIENT/imagesTs" \
  --output predictions/dataset314_ensemble \
  --folds 0,1,2,3,4 --configuration 3d_fullres \
  --trainer nnUNetTrainer --plans nnUNetPlans \
  --checkpoint checkpoint_final.pth \
  --raw "$nnUNet_raw" --preprocessed "$nnUNet_preprocessed" \
  --results "$nnUNet_results"

python scripts/evaluate.py \
  --predictions predictions/dataset314_ensemble \
  --ground-truth "$nnUNet_raw/Dataset314_PSMA5mm_FROM_D297_WHOLEPATIENT/labelsTs" \
  --output-prefix outputs/dataset314_ensemble
```

Public union-ROI fold 0:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/predict.py 300 \
  --input "$nnUNet_raw/Dataset300_PSMA18F_UNIONROI/imagesTs" \
  --output predictions/dataset300_fold0 \
  --folds 0 --configuration 3d_fullres \
  --trainer nnUNetTrainer --plans nnUNetPlans \
  --checkpoint checkpoint_final.pth \
  --raw "$nnUNet_raw" --preprocessed "$nnUNet_preprocessed" \
  --results "$nnUNet_results"

python scripts/evaluate.py \
  --predictions predictions/dataset300_fold0 \
  --ground-truth "$nnUNet_raw/Dataset300_PSMA18F_UNIONROI/labelsTs" \
  --output-prefix outputs/dataset300_fold0
```

The evaluator writes `<prefix>_per_case.csv` and `<prefix>_summary.json`. Empty
prediction versus non-empty ground truth produces Dice/IoU 0 and infinite HD95;
the JSON separately lists those infinite-HD95 cases.
