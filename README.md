# Bladder ConvNeXt Pathology

Reproducible patient-isolated ConvNeXt pipeline for tumor/non-tumor patch
classification in bladder H&E images.

## Current evidence status

The supplied metadata contain 7,067 patches from 26 patients: 18 train,
4 validation and 4 sealed independent-test patients. Patient IDs do not
overlap across these sets. The supplied package does **not** contain accessible
images, a model checkpoint, or frozen test probabilities. Consequently this
repository does not claim an independent-test AUC yet. Annotation-derived
`tumor_ratio` is never used as a model prediction.

## Included analyses

1. Frozen-model evaluation on the original independent test set.
2. Patch-level ROC AUC with patient-cluster percentile bootstrap 95% CI.
3. Publication-ready ROC and precision-recall curves.
4. Confusion matrix, sensitivity, specificity, PPV, NPV, accuracy and F1.
5. Grad-CAM and ranked TP/TN/FP/FN candidates for pathology review.
6. Patient-isolated 5-fold cross-validation on the 22 development patients;
   the 4 original test patients remain sealed.

## Data layout

Place source images outside Git and update `image_path` in the CSV files, or
pass `--image-root` to replace the directory component while retaining each
CSV basename.

```text
data/metadata/bladder_train.csv
data/metadata/bladder_val.csv
data/metadata/bladder_test.csv
data/images/<source image files>
```

Each source image is cropped at `(x, y)` using `patch_size` from the config.
Verify this coordinate convention against the original preprocessing code
before training.

## Installation

```bash
python -m venv .venv
.venv/Scripts/activate              # Windows
pip install -r requirements.txt
```

## Train and freeze the final model

```bash
python -m src.train --config configs/baseline.yaml
```

The threshold is selected on validation only and stored in the checkpoint.
Never optimize a threshold on the independent test labels.

## Independent test evaluation

```bash
python -m src.evaluate \
  --config configs/baseline.yaml \
  --checkpoint runs/baseline/best.pt \
  --csv data/metadata/bladder_test.csv \
  --outdir results/independent_test
```

This writes predictions, metrics, patient-cluster bootstrap CI, ROC/PR,
confusion matrix and TP/TN/FP/FN candidate tables.

## Patient-level 5-fold cross-validation

```bash
python -m src.cross_validate --config configs/baseline.yaml
```

Every fold trains a new model. The script produces out-of-fold predictions
only for the 22 development patients and summarizes fold variability.

## Grad-CAM

```bash
python -m src.gradcam \
  --config configs/baseline.yaml \
  --checkpoint runs/baseline/best.pt \
  --predictions results/independent_test/test_predictions.csv \
  --outdir results/gradcam
```

Representative cases must be approved by a pathologist. The script generates
candidates; it does not automatically assert that a heatmap is clinically
meaningful.

## Reproducibility rules

- Split by patient, never by patch.
- Keep the original four test patients sealed until model and threshold freeze.
- Report the number of patients alongside patch counts.
- Bootstrap patients as clusters; do not treat correlated patches as patients.
- Record all rejected single-class bootstrap replicates.
- Cross-validation does not replace the final independent test evaluation.

## Missing inputs required for real results

- Source images or pre-extracted patches corresponding to all CSV rows.
- Confirmation of patch size and coordinate convention.
- Original checkpoint/model definition if reproducing the previously reported
  validation AUC rather than retraining this baseline.
- Exact normalization and color preprocessing used by the original project.

