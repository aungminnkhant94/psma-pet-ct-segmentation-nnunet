# Recovery and cleaning audit

The source archive was exported from the project workspace inside the GPU
container. It contained PSMA experiments, FDG experiments, unrelated coursework,
cryptocurrency automation, logs, a partial virtual environment, and generated
nnU-Net artifacts. This repository contains only the PSMA segmentation project.

## Included and rewritten

- Dataset JSON metadata for PSMA datasets 297, 298, 299, 300, 312, 313, 314,
  398, and 399.
- Whole-body, union-ROI, lesion-ROI, PET-only, residual-encoder, and transfer
  learning workflows.
- Evaluation logic for Dice and HD95, extended consistently to IoU, precision,
  and recall.
- Aggregate metrics derived from recovered CSV and JSON outputs.
- Transfer-learning commands confirmed by the five training logs, each of which
  loaded the matching Dataset297 fold checkpoint.

PyTorch 2.5.1+cu121 and cuDNN 9.1 were recorded in nnU-Net debug files. The exact
nnU-Net package version was not printed. Version 2.6.4 is pinned because it was
the released version before the first recovered March 2026 experiments and its
CLI and generated plans match the archive. This is an evidence-based environment
reconstruction, not a recovered `pip freeze`.

## Corrected in the final implementation

1. The recovered ROI scripts used absolute server paths. All paths are now CLI
   arguments.
2. Symmetric padding was not subtracted when setting a cropped image's physical
   origin. The final `image_from_crop` implementation uses
   `TransformIndexToPhysicalPoint` with the padded start index.
3. Lesion crops historically received ordinary case-level splits. Because the
   case names represent lesions, this placed lesions from the same patient in
   both train and validation. The final split generator groups by patient.
4. Dataset314 initially inherited the private baseline's CT/PET file order while
   declaring PET/CT. The invalid pre-fix fold-0 metric file is excluded. The final
   conversion command physically swaps the input channels.
5. The two recovered evaluation scripts differed in empty-mask handling and
   output fields. They are replaced by one documented evaluator.

## Excluded

- All NIfTI data, checkpoints, predictions, private patient/case lists, and
  split files containing private identifiers.
- The erroneous pre-fix Dataset314 fold-0 results.
- FDG datasets 501, 510, and 511, which are outside the PSMA project reported here.
- `rollo-tennis-lab`, `rollostake`, `nlp_assignment1`, cryptocurrency loop files,
  and the experimental autoresearch harness.
- Legacy SPECT/CT registration code with a hard-coded patient path.
- Hostnames, IP addresses, container IDs, and absolute server paths.

The public repository therefore reproduces the method without publishing the
private data mapping. Exact historic private fold membership remains with the
original secured dataset; clean reruns use deterministic, privacy-safe split
generation.
