# Classical computer vision and machine learning

This package contains the reusable classical pipeline: RGB/HSV/LAB color
statistics, normalized RGB/HSV histograms, averaged multi-angle GLCM features,
and uniform LBP histograms. Every image uses the shared
`lettuce_benchmark.preprocessing.watermark_safe_crop` before resizing.

Run from the repository root:

```bash
python -m models.classical_svm.pipeline --image-root data/raw
```

In Colab, set the repository root on `sys.path`, make the raw image folders
available under `data/raw/{FN,-N,-P,-K}` (or pass another `--image-root`), then
run the same module command or call `models.classical_svm.pipeline.run_experiments`.
The shared manifests in `data/splits/` are read but never modified.

Experiments E1–E4 use a class-balanced RBF SVM and E5 uses a Random Forest.
Each candidate uses a `StandardScaler` pipeline fitted on the training rows.
Validation macro-F1 selects the parameters; the selected pipeline is then
refitted on train plus validation before one test evaluation. The predefined
SVM search covers `C={0.1,1,10}` and `gamma={"scale",0.01,0.001}`. The Random
Forest search covers 100/300 trees, `max_depth=None/20`, and
`min_samples_leaf=1/2` (`max_features="sqrt"`). The SVM confidence value is an
uncalibrated softmax-normalized decision score; the Random Forest value is the
predicted-class probability.

The command caches all 128 features once per unique image. Results are written
under `results/classical_ml/`: combined common-schema `predictions.csv`,
fold- and seed-level `metrics.csv`, per-fold and per-seed OOF confusion matrix
CSVs, selected parameters for every run in `experiment_configurations.json`,
and feature-extraction/training/prediction/model-size measurements in
`efficiency.csv`.
