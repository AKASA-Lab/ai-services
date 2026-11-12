# Lettuce NPK Deficiency Classification Benchmark

A reproducible five-approach benchmark for classifying nutritional conditions in lettuce images using PyTorch and Google Colab.

> **Project duration:** 7 days  
> **Team size:** 5 members  
> **Dataset:** [Lettuce NPK Dataset – Kaggle](https://www.kaggle.com/datasets/baronn/lettuce-npk-dataset)  
> **Primary metric:** Macro F1-score

---

## 1. Project objective

The objective is to compare five machine-learning approaches on a small, imbalanced, four-class lettuce nutrient-condition image dataset.

The project has two distinct tracks:

1. **Public benchmark:** train and evaluate models using the Kaggle lettuce dataset.
2. **Target-domain research:** develop and evaluate our own fenugreek aeroponic nutrient-deficiency dataset.

The public lettuce benchmark is preliminary model-development work. Results on lettuce must **not** be presented as proof that the models can detect nutrient deficiencies in aeroponically grown fenugreek.

### Research questions

- How do deep-learning and classical computer-vision approaches compare on this four-class dataset?
- How does class imbalance affect minority-class performance?
- What is the trade-off between classification performance and inference efficiency?
- Which visual regions influence model predictions?

---

## 2. Dataset and labels

The dataset is commonly reported as containing 208 images across four classes:

| Label | Meaning | Reported image count |
|---|---|---:|
| `FN` | Fully nutritional / healthy | 12 |
| `-N` | Nitrogen deficiency | 58 |
| `-P` | Phosphorus deficiency | 66 |
| `-K` | Potassium deficiency | 72 |
| **Total** | | **208** |

These counts are reported in research publications using this dataset. They are **not a substitute for auditing the downloaded Kaggle archive**. The team must verify the actual number of files, labels, folder structure, duplicates, and image integrity before training.

### What this classification task is

This is a **four-class, single-label classification task**:

- Healthy / fully nutritional (`FN`)
- Nitrogen deficiency (`-N`)
- Phosphorus deficiency (`-P`)
- Potassium deficiency (`-K`)

It is not an eight-combination NPK experiment and it is not a multi-label task. The labels represent the four conditions above; do not assume the dataset contains combined nutrient-deficiency classes.

### Dataset handling

- Download the dataset from Kaggle and place the original files under `data/raw/`.
- Do not commit the dataset or large model checkpoints to GitHub.
- Preserve the original files. Do not overwrite or silently relabel them.
- Create a metadata manifest with at least `image_id`, `image_path`, and `label`.
- Record missing plant IDs or acquisition metadata as unavailable; do not invent them.

The shared audit, manifest, and split tools are documented in [CONTRIBUTING.md](./CONTRIBUTING.md). For example:

```bash
python -m pip install -r requirements.txt
python -m lettuce_benchmark.cli.audit_dataset data/raw --output-dir outputs/audit
python -m lettuce_benchmark.cli.build_manifest data/raw --output data/processed/metadata.csv
python -m lettuce_benchmark.cli.generate_splits data/processed/metadata.csv --output-dir data/splits
python -m lettuce_benchmark.cli.validate_splits data/splits
```

These commands have not been run against the Kaggle archive in this repository. Reported counts below are literature/context figures only, not a verified local audit.

---

## 3. Team division

Each person owns one approach. All members must use the **same split manifests and evaluation protocol**.

| Member | Assigned approach | Main responsibility |
|---|---|---|
| **Person 1** | MobileNetV2 + integration | Dataset audit, shared split generation, baseline model, integration and final comparison |
| **Person 2** | EfficientNet-B0 | Transfer learning, training, validation and evaluation |
| **Person 3** | ResNet50 | Transfer learning, training, validation and evaluation |
| **Person 4** | YOLOv8n-cls | Classification training, validation and evaluation |
| **Person 5** | Color/texture features + SVM | Handcrafted feature extraction, scaling, classifier training and evaluation |

Replace generic member labels with actual names and GitHub usernames.

### Person 1 — MobileNetV2 and integration coordinator

- Audit the downloaded dataset: class counts, dimensions, corrupt files, duplicates and available metadata.
- Create the common metadata file and split manifests for all members.
- Implement MobileNetV2 using pretrained weights and a four-class classification head.
- Train and evaluate using every shared split.
- Coordinate the common prediction format and final result consolidation.
- Generate Grad-CAM visualizations if time permits.

**Deliverables:** dataset audit, shared split CSVs, MobileNetV2 notebook/code, metrics, predictions and confusion matrices.

### Person 2 — EfficientNet-B0

- Implement pretrained EfficientNet-B0 with a four-class output layer.
- Use the shared split manifests.
- Train, validate and evaluate across all folds and seeds.
- Record training time, inference latency and model size.

**Deliverables:** notebook/code, configuration, checkpoint, metrics, predictions and confusion matrices.

### Person 3 — ResNet50

- Implement pretrained ResNet50 with a four-class output layer.
- Use the shared split manifests.
- Train, validate and evaluate across all folds and seeds.
- Record training time, inference latency and model size.

**Deliverables:** notebook/code, configuration, checkpoint, metrics, predictions and confusion matrices.

### Person 4 — YOLOv8n-cls

- Prepare the dataset in the directory format expected by the Ultralytics classification interface.
- Preserve the exact train/validation/test membership in the shared manifests.
- Record the Ultralytics version and training arguments.
- Export predictions in the common CSV format.
- Measure inference latency and model size.

**Deliverables:** notebook/code, configuration, weights, metrics, predictions and confusion matrices.

> YOLOv8n-cls is a classification baseline. It is not a reproduction of a custom YOLO-NPK research architecture unless that architecture is separately implemented.

### Person 5 — Classical computer vision + SVM

Extract handcrafted features from each image.

Minimum feature groups:

- RGB channel mean and standard deviation.
- HSV channel mean and standard deviation.
- CIELAB channel mean and standard deviation.
- GLCM texture features: contrast, homogeneity, energy and correlation.
- LBP histogram features.

Minimum classifiers:

- RBF SVM with class weighting.
- Random Forest with class weighting, if time permits.

Fit scaling, feature selection and any resampling **only on the training partition**.

**Deliverables:** feature extraction notebook/code, configuration, trained classifier, metrics, predictions and confusion matrices.

---

## 4. Shared dataset split and evaluation protocol

### 4.1 Primary split strategy

Use **repeated stratified 4-fold cross-validation** with three fixed random seeds:

```python
SEEDS = [42, 123, 2026]
N_SPLITS = 4
```

This produces 12 train/test runs per model (4 folds × 3 seeds).

For each seed:

1. Divide the original dataset into four stratified folds.
2. In each run, use three folds as the development partition and the remaining fold as the held-out test partition.
3. Rotate the held-out fold so that every image is used as test data once for that seed.
4. Repeat the process for all three seeds.

Approximate size per run:

| Partition | Approximate number of images |
|---|---:|
| Development (three folds) | 156 |
| Held-out test (one fold) | 52 |
| **Total** | **208** |

The exact counts may differ slightly because stratification assigns integer numbers of images.

### 4.2 Validation split inside development data

For every run, split the 156-image development partition into:

- **Training:** approximately 80% of development data (~125 images).
- **Validation:** approximately 20% of development data (~31 images).
- **Test:** the held-out 52 images.

The validation set is used for checkpoint selection, early stopping and model-selection decisions. The test fold must remain untouched until final evaluation.

Use stratification for the train/validation split where possible. With only 12 reported `FN` images in the entire dataset, each validation subset may contain only a few healthy examples. Class-wise validation metrics will therefore be unstable.

**Do not use the test fold for hyperparameter tuning, early stopping, threshold selection or repeated experimentation.**

### 4.3 Approximate class distribution

The reported class counts imply the following approximate distribution:

| Class | Total | Approx. test images per fold | Approx. development images |
|---|---:|---:|---:|
| `FN` | 12 | 3 | 9 |
| `-N` | 58 | 14–15 | 43–44 |
| `-P` | 66 | 16–17 | 49–50 |
| `-K` | 72 | 18 | 54 |
| **Total** | **208** | **~52** | **~156** |

These are planning estimates. Use the actual generated split manifests as the source of truth.

### 4.4 Group leakage and duplicate checks

If plant IDs or acquisition-session groups are available, images from the same plant or group must not appear in both development and test partitions. Use a group-aware stratified strategy where feasible.

If reliable group metadata is unavailable:

- Do not invent plant IDs.
- Check for exact and near-duplicate images before splitting.
- Keep duplicate/near-duplicate images together or remove redundant copies after documenting the decision.
- State clearly that the evaluation is image-level and that plant-independent generalization has not been established.

### 4.5 Split before augmentation

Mandatory order:

1. Audit original images and identify duplicates.
2. Generate train/test split manifests from original images.
3. Create the validation subset from the development partition.
4. Apply augmentation only to training images.
5. Do not augment validation or test images.

Never create augmented copies first and randomly split those copies afterward.

### 4.6 Shared split files

Person 1 generates the split manifests once. All members download/use the same files. No member should independently regenerate the splits.

Recommended structure:

```text
data/
├── raw/                         # Original Kaggle files; not committed
├── processed/
│   └── metadata.csv
└── splits/
    ├── seed_42/
    │   ├── fold_1.csv
    │   ├── fold_2.csv
    │   ├── fold_3.csv
    │   └── fold_4.csv
    ├── seed_123/
    │   ├── fold_1.csv
    │   ├── fold_2.csv
    │   ├── fold_3.csv
    │   └── fold_4.csv
    └── seed_2026/
        ├── fold_1.csv
        ├── fold_2.csv
        ├── fold_3.csv
        └── fold_4.csv
```

Each fold CSV should identify the partition membership explicitly. Recommended columns:

```csv
image_id,image_path,label,partition,seed,fold
img001,data/raw/FN/image001.jpg,FN,test,42,1
img002,data/raw/-N/image002.jpg,-N,train,42,1
img003,data/raw/-P/image003.jpg,-P,validation,42,1
```

The examples above illustrate the format only. Generate actual rows from the audited archive.

---

## 5. Common preprocessing and training expectations

### 5.1 CNN preprocessing

Starting common settings for MobileNetV2, EfficientNet-B0 and ResNet50:

| Setting | Starting value |
|---|---|
| Color space | RGB |
| Input size | 224 × 224 |
| Batch size | 16 |
| Maximum epochs | 15 |
| Optimizer | AdamW |
| Backbone learning rate | `1e-4` |
| Classifier learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Number of classes | 4 |

Use the normalization expected by the selected pretrained weights. Record any architecture-specific differences.

Suggested training approach:

- Initially freeze the pretrained backbone and train the classification head.
- Then unfreeze and fine-tune, if appropriate.
- Use validation data for checkpoint selection.
- Save the best checkpoint based on a validation metric, preferably macro F1 or validation loss with macro F1 also reported.
- Record the exact configuration and random seed.

These are starting configurations, not guaranteed optimal settings. Keep the training budget reasonably comparable and document deviations.

### 5.2 Augmentation

Use mild training-only augmentation, such as:

- Horizontal flip, if appropriate for the image orientation.
- Small rotations.
- Mild translation or crop.

Avoid aggressive hue, saturation or color transformations because leaf color is an important visual signal for nutrient deficiency.

YOLOv8n-cls and classical ML may use architecture-appropriate preprocessing, but must preserve identical split membership.

### 5.3 Class imbalance

The `FN` class is severely underrepresented. Consider class-weighted loss as the primary mitigation.

Optional comparisons, if time permits:

- Standard cross-entropy.
- Weighted cross-entropy.
- Focal loss.
- Training-only oversampling.

Compute class weights from the current training partition only. Any resampling must be restricted to training data. Do not apply SMOTE to raw images.

---

## 6. Evaluation metrics

### Primary metric

**Macro F1-score** is the primary comparison metric because it gives equal importance to all four classes.

### Required metrics for every model

- Accuracy.
- Balanced accuracy.
- Macro precision.
- Macro recall.
- Macro F1-score.
- Weighted F1-score.
- Per-class precision, recall and F1-score.
- Confusion matrix.

### Correct result aggregation

For each seed:

1. Collect predictions for all four held-out test folds.
2. Combine the out-of-fold predictions so each original image has one held-out prediction for that seed.
3. Calculate metrics over the complete out-of-fold prediction set.
4. Repeat for all three seeds.
5. Report the mean and standard deviation of the three seed-level metric values.

Do not treat the 12 folds as 12 independent datasets. Do not pool the three repetitions as if they were 624 unique images.

### Common prediction CSV

Every member must export predictions in this format:

```csv
image_id,seed,fold,true_label,predicted_label,confidence
img001,42,1,FN,-N,0.72
img002,42,1,-N,-N,0.91
img003,42,1,-P,-P,0.86
```

Use exactly this class spelling and order:

```python
CLASS_NAMES = ["FN", "-N", "-P", "-K"]
```

The integration coordinator will recompute metrics from the saved predictions.

---

## 7. Efficiency measurements

Every member must report:

| Measurement | Requirement |
|---|---|
| Accuracy | Required |
| Macro F1 | Required |
| Balanced accuracy | Required |
| Per-class recall | Required |
| Parameter count | Deep-learning models |
| Model/checkpoint size | Required |
| Inference latency | Required |
| Training time | Required |
| Peak GPU memory | If available |

Measure inference latency at batch size 1 after warm-up, using the same hardware where possible. State the device and timing method. Synchronize CUDA around GPU timing.

For the classical ML approach, report feature extraction time separately from classifier prediction time.

---

## 8. Expected results and interpretation

There is **no required accuracy target** and no model is guaranteed to achieve a particular score. The goal is a fair, reproducible comparison, not to force a high result.

Use these macro-F1 bands only as rough interpretation guidelines:

| Macro F1 | Interpretation |
|---|---|
| Below 0.50 | Weak performance; inspect class-wise errors and pipeline |
| 0.50–0.69 | Limited preliminary performance |
| 0.70–0.79 | Useful preliminary baseline |
| 0.80–0.89 | Strong preliminary result; verify carefully |
| 0.90 or above | Unusually high for this small dataset; investigate leakage, duplicates and split integrity |

These are not scientific benchmarks or guaranteed outcomes. Always report per-class recall and confusion matrices. A high overall score can hide poor detection of the minority `FN` class.

### Required final comparison

Create one comparison table with:

- Model name.
- Mean macro F1 ± standard deviation across the three seeds.
- Mean accuracy ± standard deviation.
- Mean balanced accuracy.
- Per-class recall.
- Model size.
- Inference latency.
- Training time.

Discuss the strongest model, the weakest model, class-specific failure patterns, efficiency trade-offs and limitations.

---

## 9. Optional extensions (only after the primary benchmark)

Prioritize the five-model benchmark first. If time remains:

### Explainability

- Generate Grad-CAM for selected CNN models.
- Generate LIME explanations for selected examples.
- Inspect correct and incorrect predictions from all classes.
- Check whether explanations focus on leaf tissue or irrelevant background regions.

Explanations are diagnostic aids, not proof that a model has learned biologically valid symptoms.

### Edge deployment

Select one or two lightweight models and investigate quantization.

Measure actual changes in:

- Model size.
- Inference latency.
- Macro F1.
- Per-class recall.

Do not assume quantization improves performance; measure it.

---

## 10. Seven-day execution plan

| Day | Team milestone |
|---|---|
| **Day 1** | Download and audit dataset; check duplicates; create shared metadata and split manifests; set up Colab and repository |
| **Day 2** | All five approaches implemented; run a small smoke test |
| **Day 3** | First complete training/evaluation runs; fix implementation issues |
| **Day 4** | Complete remaining folds and seeds; monitor reproducibility |
| **Day 5** | Finish experiments and export predictions, metrics and checkpoints |
| **Day 6** | Consolidate results; generate comparison tables, confusion matrices and error analysis |
| **Day 7** | Final report, presentation, README cleanup and reproducibility checks |

**Execution rule:** All members should begin implementing their assigned approach in parallel on Day 1 using a small sample. Do not make the other four members wait for Person 1 to finish every dataset task. However, final benchmark runs must use Person 1's approved shared split manifests.

---

## 11. Required submission structure

Each member submits:

```text
results/
└── member_X/
    ├── metrics.csv
    ├── predictions.csv
    ├── config.json
    ├── confusion_matrix.png
    ├── training_curves.png
    ├── error_analysis.md
    └── checkpoints/
```

The final repository should contain:

- Dataset audit and verified class counts.
- Shared split manifests.
- Five model implementations.
- Environment/dependency file and configuration files.
- Per-model metrics and predictions.
- Consolidated comparison table.
- Confusion matrices and error analysis.
- Final report documenting limitations and reproducibility steps.

---

## 12. Suggested repository structure

```text
lettuce-npk-benchmark/
├── README.md
├── PROJECT_GUIDELINES.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── dataset.yaml
│   ├── training.yaml
│   └── evaluation.yaml
├── data/
│   ├── raw/                  # Not committed
│   ├── processed/
│   └── splits/
├── notebooks/
│   ├── 00_dataset_audit.ipynb
│   ├── 01_mobilenetv2.ipynb
│   ├── 02_efficientnet_b0.ipynb
│   ├── 03_resnet50.ipynb
│   ├── 04_yolov8n_cls.ipynb
│   ├── 05_classical_ml.ipynb
│   └── 06_comparative_analysis.ipynb
├── src/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   └── explainability/
├── results/
│   ├── member_1/
│   ├── member_2/
│   ├── member_3/
│   ├── member_4/
│   ├── member_5/
│   └── comparison/
└── docs/
    ├── experimental_protocol.md
    ├── literature_review.md
    └── final_report.md
```

---

## 13. Limitations and responsible claims

- The dataset is small and imbalanced, especially the healthy `FN` class.
- Published class counts must be verified against the actual downloaded archive.
- Image-level splitting cannot establish plant-independent generalization when plant identity metadata is unavailable.
- Results may be sensitive to image duplicates, backgrounds, lighting and acquisition conditions.
- A model trained on lettuce may not generalize to fenugreek.
- Hydroponic lettuce results do not validate performance in an aeroponic cultivation environment.
- This benchmark evaluates four labelled image classes; it does not establish reliable detection of arbitrary nutrient concentrations or combined deficiencies.

The final report must distinguish benchmark findings from claims about the team's fenugreek aeroponic system.

---

## 14. Team checklist

- [ ] Dataset downloaded and audited.
- [ ] Actual class counts and folder structure verified.
- [ ] Corrupt, duplicate and near-duplicate images checked.
- [ ] Shared metadata manifest created.
- [ ] Shared repeated stratified 4-fold split manifests generated for all three seeds.
- [ ] Validation subsets created from development data only.
- [ ] All five approaches implemented.
- [ ] All models evaluated using identical test membership.
- [ ] Predictions exported in the common format.
- [ ] Macro F1, balanced accuracy and per-class metrics reported.
- [ ] Confusion matrices and error analysis completed.
- [ ] Model size, latency and training time recorded.
- [ ] Final comparison and limitations documented.
