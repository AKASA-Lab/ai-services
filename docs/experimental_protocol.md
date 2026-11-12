# Experimental protocol

## Dataset and audit

The benchmark dataset is the [Kaggle Lettuce NPK dataset](https://www.kaggle.com/datasets/baronn/lettuce-npk-dataset). Counts reported in prior descriptions are unverified until the exact downloaded copy is audited. No dataset is included in this repository.

Run `python -m lettuce_benchmark.cli.audit_dataset data/raw --output-dir outputs/audit` after extraction. The audit lists image files, validates decodability, dimensions, labels inferred only from recognized class-directory names, exact SHA-256 duplicates, and optionally near-duplicate candidates. Review all unclassified and invalid files. Generate the manifest only after reviewing the audit. It records `image_id`, dataset-root-relative `image_path`, and `label`; `plant_id` and `source_group` are optional and must come from real source metadata.

Before splitting, decide and document a policy for exact/near duplicates. The tools report pairs but do not delete, relabel, or silently deduplicate images. If keeping duplicate images, use an appropriate group column to prevent leakage when reliable group IDs exist.

## Four-fold repeated protocol

Shared defaults are `SEEDS = [42, 123, 2026]` and `N_SPLITS = 4`. For each seed, stratify the original image manifest into four outer folds. Each run holds one outer fold out as test data and uses the other three folds as development data. Stratify a separate 80/20 train/validation split within development data. The test fold is not used for training, checkpoint selection, early stopping, or tuning.

The resulting output contains 12 fold CSVs. Each file has columns:

```text
image_id,image_path,label,partition,seed,fold
```

Each image is in exactly one partition in each fold run, each held-out fold is test exactly once per seed, and train/validation/test image IDs are pairwise disjoint within a run. Augment only training examples after split assignment; never split augmented copies.

When `plant_id` or `source_group` is available and reliable, pass its column using `--group-column` so no group crosses outer partitions or the inner train/validation boundary. The split generator does not create group IDs. If no such metadata is available, the generated splits are image-level and do not establish plant-independent or acquisition-independent generalization.

Small class/group counts can make grouped stratification impossible or less balanced. Treat a clear generator error as a protocol constraint to resolve with the team; do not silently fall back to image-level splitting.

## Training and model selection

All five approaches use the same source manifest and split CSVs. Members must not create independent splits. Use validation data for selecting checkpoints and hyperparameters. Compute class weights, fit scalers, feature selection, calibration, or resampling from the training partition only. Do not tune on test results.

CNN and YOLO implementations may use model-specific input preprocessing, but must preserve split membership. The classical model must time feature extraction independently from classifier prediction. Document any deviations from the approach configurations.

## Predictions and metrics

Export one row per held-out image and fold using:

```text
image_id,seed,fold,true_label,predicted_label,confidence
```

Use the exact class labels `FN`, `-N`, `-P`, `-K`. Confidence is the score/probability for the predicted class; state whether it is calibrated.

Required metrics are accuracy, balanced accuracy, macro precision/recall/F1, weighted F1, per-class precision/recall/F1, and confusion matrix. Macro F1 is primary. Concatenate the four held-out fold predictions for a seed and compute metrics once over that seed's out-of-fold set. Then report mean and sample standard deviation over the three seed-level values. Do not pool all repeated-seed predictions as independent observations. The evaluator checks each image has one held-out prediction per seed and writes CSV, JSON, and confusion-matrix plots.

## Reproducibility and efficiency

Set Python, NumPy, and PyTorch seeds before training; seed CUDA when available. Save the exact YAML configuration, model name, seed, fold, hardware, and software versions per run.

Report parameter count (where applicable), checkpoint size, training duration, batch-size-one inference latency after warm-up, and peak GPU memory (when available). Synchronize CUDA before and after timed GPU work. State the device. Timing is descriptive and should not be compared across dissimilar hardware without qualification.

Do not report results until they come from saved prediction files and the common evaluator. No expected accuracy is guaranteed.
