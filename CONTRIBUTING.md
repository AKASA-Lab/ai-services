# Contributing

## Setup

Use Python 3.10 or newer. In a local environment or Google Colab, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

PyTorch wheels depend on the selected CPU/CUDA runtime. If the default package index does not provide the desired CUDA build, install the matching PyTorch and torchvision build using the official PyTorch installation selector, then install the remaining requirements.

## Add the dataset locally

Download the Kaggle dataset using your own Kaggle credentials and extract it under `data/raw/`. Do not commit credentials, the archive, or extracted images. The audit tool recognizes image files whose path contains a class directory named `FN`, `-N`, `-P`, or `-K`; it reports other images as unclassified instead of guessing labels.

```bash
python -m lettuce_benchmark.cli.audit_dataset data/raw --output-dir outputs/audit
python -m lettuce_benchmark.cli.build_manifest data/raw --output data/processed/metadata.csv
```

The audit output includes discovered/valid counts, dimensions, duplicate groups, and unclassified/corrupt files. Review it before creating splits. `metadata.csv` contains valid, classifiable images only. Image paths are relative to the supplied dataset root. Optional metadata can be added as `plant_id` or `source_group` columns only when supported by the source data.

To preserve real plant/acquisition metadata from a sidecar, pass a CSV with a unique `image_id` or `image_path` key and one or both optional columns:

```bash
python -m lettuce_benchmark.cli.build_manifest data/raw \
  --metadata-sidecar path/to/source_metadata.csv \
  --output data/processed/metadata.csv
```

Unknown keys, duplicate sidecar keys, or absent group values are not guessed or repaired; review the source metadata before grouped splitting.

Exact duplicates are found by SHA-256. Near-duplicate scanning is optional and uses perceptual image hashes; review candidate pairs rather than treating them as proof of duplicates.

## Generate the shared splits

After review, generate all seeds and folds once:

```bash
python -m lettuce_benchmark.cli.generate_splits \
  data/processed/metadata.csv --output-dir data/splits
```

To use group-aware splitting, specify a real metadata column with reliable values:

```bash
python -m lettuce_benchmark.cli.generate_splits \
  data/processed/metadata.csv --output-dir data/splits --group-column plant_id
```

`source_group` may be used instead when it is the reliable leakage boundary. Group-aware stratification can be infeasible for small or highly imbalanced groups; the command fails clearly rather than silently switching to image-level splitting. Use `--config configs/common.yaml` to override shared seeds and fold settings in a coordinated change.

Each `seed_<seed>/fold_<fold>.csv` has one row for every source image, with exactly one of `train`, `validation`, or `test`. Validate a split set before training:

```bash
python -m lettuce_benchmark.cli.validate_splits data/splits
```

Do not modify/regenerate split assignments separately for each model.

## Implement your assigned approach

Each `models/<approach>/` directory contains its ownership notes and a starter configuration. Implement the training and prediction export for your assignment there; these starter directories intentionally do not contain complete training pipelines. Each run should:

1. Load a shared fold CSV and train only from its `train` rows.
2. Select checkpoints/configuration using `validation` rows only.
3. Evaluate the held-out `test` rows once per planned run.
4. Export the common prediction CSV and run metadata.
5. Record training time, checkpoint size, inference latency, and peak GPU memory when available. For SVM, time feature extraction and classifier prediction separately.

Set all relevant random seeds using `lettuce_benchmark.reproducibility.set_seed`. Save a YAML configuration snapshot alongside run outputs.

## Evaluate and report

When all four held-out folds are available for a seed, aggregate the model's out-of-fold predictions:

```bash
python -m lettuce_benchmark.cli.evaluate_predictions \
  outputs/<model>/predictions.csv --output-dir outputs/<model>/evaluation
```

The evaluator checks fold coverage, computes metrics per seed, writes seed and across-seed machine-readable results, and creates confusion-matrix plots. Compare the mean and standard deviation of the three seed-level scores; do not treat repeated predictions of the same image as independent samples. Macro F1 is the primary comparison metric.

## Tests

Run the shared unit tests:

```bash
python -m pytest
```

Do not add generated dataset files or checkpoints to test fixtures. Tests use small synthetic metadata and image samples created in temporary directories.
