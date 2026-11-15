# Lettuce NPK Deficiency Classification Benchmark

A reproducible five-approach benchmark for classifying nutritional conditions in lettuce images using computer vision and deep learning.

| Project detail | Description |
|---|---|
=| Dataset | [Lettuce NPK Dataset – Kaggle](https://www.kaggle.com/datasets/baronn/lettuce-npk-dataset) |
| Task | Four-class, single-label image classification |
| Primary metric | Macro F1-score |
| Frameworks | PyTorch, scikit-learn, Ultralytics |
| Evaluation | Repeated stratified 4-fold evaluation across 3 seeds |

## 1. Project overview

This project benchmarks five machine-learning approaches for classifying nutrient conditions in lettuce images. It investigates the performance of classical computer vision and deep-learning models on a small, imbalanced dataset.

The project has two distinct research tracks:

1. **Public benchmark:** Evaluate classification approaches using the Kaggle lettuce NPK dataset.
2. **Target-domain research:** Develop and evaluate an independent nutrient-deficiency dataset using aeroponically grown fenugreek.

The lettuce benchmark is preliminary model-development work. Results on lettuce must not be interpreted as proof that the models can detect nutrient deficiencies in aeroponically grown fenugreek.

### Research questions

- How do classical computer-vision and deep-learning approaches compare on this four-class dataset?
- How does class imbalance affect minority-class performance?
- What trade-offs exist between classification performance and inference efficiency?
- Which visual regions influence model predictions?

## 2. Dataset

The dataset contains four nutritional-condition classes:

| Label | Meaning | Original archive count | Images used |
|---|---|---:|---:|
| FN | Fully nutritional / healthy | 12 | 12 |
| -N | Nitrogen deficiency | 58 | 57 |
| -P | Phosphorus deficiency | 66 | 66 |
| -K | Potassium deficiency | 72 | 68 |
| **Total** | | **208** | **203** |

The original archive was audited before experimentation. Five exact duplicate copies were excluded from the metadata manifest, leaving 203 unique images for the benchmark. The original files were preserved.

### Classification task

This is a four-class, single-label classification task:

- `FN`: Fully nutritional / healthy
- `-N`: Nitrogen deficiency
- `-P`: Phosphorus deficiency
- `-K`: Potassium deficiency

It is not an eight-combination NPK experiment or a multi-label classification task.

### Dataset audit and preprocessing

The dataset audit identified an important image artifact: a DALL·E colour bar appearing in 82 of the 203 images. Its presence is correlated with class labels:

- All 12 FN images contain the colour bar.
- 48 of the 66 -P images contain it.
- 14% of -N images contain it.
- 21% of -K images contain it.

This creates a potential shortcut-learning risk: a model could exploit image-generation or presentation artifacts instead of relying entirely on plant symptoms.

To reduce this risk, the shared preprocessing pipeline applies a fixed **960 × 960 safe crop** before model-specific resizing or feature extraction.

The crop removes the colour bar, but it does not establish that all other visual artifacts have been eliminated. This remains an important limitation when interpreting the results.

The audit and duplicate policy are documented in [`docs/dataset_audit.md`](docs/dataset_audit.md).

### Dataset limitations

- Only 203 unique images are available.
- The healthy FN class contains just 12 images.
- Plant identity and acquisition-session metadata are unavailable or insufficient for establishing plant-independent generalization.
- The dataset may contain background, lighting, image-style, and acquisition artifacts.
- Results should be interpreted as performance on this particular image benchmark.

## 3. Experimental protocol

### 3.1 Repeated stratified cross-validation

All approaches use the same shared split manifests.

```python
SEEDS = [42, 123, 2026]
N_SPLITS = 4
```

For each seed, the dataset is divided into four stratified folds. Each experiment uses three folds as development data and one fold as held-out test data. The held-out fold rotates so every image is evaluated once per seed.

Within each development partition, a separate validation subset is used for model selection and checkpoint selection.

The resulting evaluation consists of:

- 3 random seeds.
- 4 folds per seed.
- 12 held-out test runs per approach.
- One out-of-fold prediction for every image per seed.

The exact partition membership is defined by the shared CSV manifests in `data/splits/`. These manifests are the source of truth; individual approaches must not regenerate their own splits.

### 3.2 Validation and test isolation

The validation partition is used for model selection, hyperparameter selection, and checkpoint selection.

The held-out test partition is reserved for final evaluation. It must not be used for tuning, early stopping, or repeated experimentation.

For each seed, predictions from all four held-out folds are combined to produce one out-of-fold prediction set. Metrics are calculated over that complete set.

The reported mean and standard deviation are calculated across the three seed-level results. The three seeds reuse the same underlying dataset, so this standard deviation describes variability across split repetitions; it is not a confidence interval based on independent datasets.

### 3.3 Common preprocessing

All approaches apply the shared 960 × 960 watermark-safe crop before their model-specific processing.

CNN approaches use architecture-appropriate input resizing and normalization. The classical approach extracts handcrafted features from the cropped images.

Training augmentation, where used, is restricted to training data.

## 4. Approaches

Five approaches were assigned across the team.

| Member | Approach | Method |
|---|---|---|
| Person 1 | MobileNetV2 | ImageNet-pretrained CNN |
| Person 2 (Arya Pramesh) | EfficientNet-B0 | ImageNet-pretrained CNN |
| Person 3 | ResNet50 | ImageNet-pretrained CNN |
| Person 4 | YOLOv8n-cls | Ultralytics image classification |
| Person 5 | Classical CV + SVM | Handcrafted colour and texture features |

### 4.1 MobileNetV2

MobileNetV2 is used as a lightweight pretrained CNN baseline with a four-class classification head.

It provides a reference point for evaluating a relatively compact deep-learning architecture.

### 4.2 EfficientNet-B0

EfficientNet-B0 was assigned as a transfer-learning approach. Its final benchmark results are pending and will be added after the complete evaluation and result verification.

### 4.3 ResNet50

ResNet50 uses ImageNet-1K pretrained weights and a reinitialized four-class classification head.

The implementation uses a warm-up stage with the backbone and BatchNorm layers frozen, followed by full fine-tuning. Weighted cross-entropy is used to account for class imbalance.

Checkpoint selection is based on validation macro F1, with validation loss used as a tie-breaker.

### 4.4 YOLOv8n-cls

The project uses the stock Ultralytics YOLOv8n classification architecture with ImageNet-pretrained weights and a four-class classification head.

This is an image-classification baseline, not an object-detection model and not a reproduction of a custom YOLO-NPK research architecture.

The training pipeline preserves the shared split membership and selects checkpoints using validation macro F1.

The reported benchmark results are available below. Additional saved-checkpoint verification and shortcut-learning analysis are being completed separately.

### 4.5 Classical computer vision and SVM

The classical approach extracts a 128-dimensional handcrafted feature vector from each cropped image.

The feature groups are:

| Feature group | Features |
|---|---|
| Colour statistics | RGB, HSV and CIELAB channel means and standard deviations |
| Colour histograms | RGB and HSV histograms with 16 bins per channel |
| GLCM texture | Contrast, homogeneity, energy and correlation |
| LBP texture | Uniform Local Binary Pattern histogram |

Five experiments were evaluated:

| Experiment | Feature set | Classifier |
|---|---|---|
| E1 | Colour statistics | RBF SVM |
| E2 | Colour histograms | RBF SVM |
| E3 | GLCM + LBP | RBF SVM |
| E4 | All 128 features | RBF SVM |
| E5 | All 128 features | Random Forest |

Scaling is performed within the training pipeline. Hyperparameters are selected using validation macro F1, and the selected pipeline is refitted on the combined training and validation data before test evaluation.

The SVM decision scores are not calibrated probabilities. Any normalized SVM scores should therefore be interpreted as relative confidence scores, not true class probabilities.

## 5. Evaluation metrics

Macro F1-score is the primary comparison metric because it gives equal importance to all four classes, including the underrepresented healthy class.

The benchmark records:

- Accuracy
- Balanced accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Weighted F1-score
- Per-class precision, recall and F1-score
- Confusion matrices

Efficiency measurements include model size, inference latency and training time where available. Classical feature-extraction time is recorded separately from classifier prediction time.

## 6. Benchmark results

The following results are mean ± standard deviation across the three seed-level out-of-fold evaluations.

### 6.1 Overall model comparison

| Approach | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| YOLOv8n-cls | 0.936 ± 0.013 | 0.948 ± 0.010 | **0.936 ± 0.016** |
| Classical CV + SVM (E4) | 0.913 ± 0.008 | 0.923 ± 0.014 | **0.926 ± 0.010** |
| MobileNetV2 | 0.888 ± 0.049 | 0.899 ± 0.058 | 0.879 ± 0.067 |
| ResNet50 | 0.870 ± 0.041 | 0.866 ± 0.044 | 0.863 ± 0.045 |
| EfficientNet-B0 | Pending | Pending | Pending |

**Important:** YOLOv8n-cls currently has the highest reported macro F1. Its result should be treated as provisional until saved-checkpoint verification and the additional shortcut-learning investigation are complete.

### 6.2 Classical CV experiment comparison

| Experiment | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| E1 – Colour statistics + SVM | 0.8768 ± 0.0226 | 0.8994 ± 0.0180 | 0.8764 ± 0.0270 |
| E2 – Colour histograms + SVM | 0.8834 ± 0.0186 | 0.8884 ± 0.0288 | 0.8917 ± 0.0318 |
| E3 – GLCM + LBP + SVM | 0.7356 ± 0.0103 | 0.7396 ± 0.0170 | 0.7093 ± 0.0195 |
| E4 – All features + RBF SVM | **0.9130 ± 0.0075** | **0.9228 ± 0.0139** | **0.9263 ± 0.0103** |
| E5 – All features + Random Forest | 0.8834 ± 0.0103 | 0.9054 ± 0.0073 | 0.8961 ± 0.0252 |

E4, combining all 128 features with an RBF SVM, was the strongest classical experiment.

The colour-statistics and histogram experiments performed substantially better than the texture-only experiment. Combining colour and texture features improved macro F1 over the colour-histogram-only experiment.

These results suggest that colour information carries substantial discriminative signal in this dataset. They do not establish that any particular feature group captures biologically meaningful nutrient-deficiency symptoms.

### 6.3 Per-class performance

The following are the reported mean per-class F1-scores for the strongest classical experiment, E4.

| Class | F1-score |
|---|---:|
| FN | 0.9855 |
| -N | 0.8724 |
| -P | 0.9726 |
| -K | 0.8748 |

The corresponding recall values are:

| Class | Recall |
|---|---:|
| FN | 0.9722 |
| -N | 0.8421 |
| -P | 0.9848 |
| -K | 0.8922 |

For ResNet50, the reported mean per-class F1-scores are:

| Class | F1-score |
|---|---:|
| FN | 0.838 |
| -N | 0.848 |
| -P | 0.907 |
| -K | 0.861 |

YOLOv8n-cls reported out-of-fold recalls of 1.000, 0.906, 0.929 and 0.956 for FN, -N, -P and -K, respectively. Given the very small FN sample and the image-artifact concern, the perfect FN recall should be investigated rather than interpreted as conclusive evidence of reliable healthy-plant recognition.

### 6.4 Error analysis

Across the five classical experiments, 3,045 held-out predictions were generated: 203 images evaluated across three seeds and five experiments.

The pooled errors show recurring confusion between nitrogen and potassium deficiency:

| True class | Predicted class | Errors |
|---|---|---:|
| -N | -K | 124 |
| -K | -N | 91 |
| -K | -P | 65 |
| -P | -K | 63 |

These four confusion pairs account for approximately 79.6% of the pooled errors.

For the strongest classical experiment, E4, the most frequent errors were also between -N and -K.

This suggests that distinguishing nitrogen and potassium deficiency is a recurring challenge in the benchmark and deserves closer examination.

### 6.5 Efficiency

Efficiency measurements depend on the hardware and software environment used by each member. They should not be directly ranked unless the models are rebenchmarked under a common setup.

Available measurements include:

| Approach | Model size | Inference latency | Notes |
|---|---:|---:|---|
| Classical CV + SVM (E4) | 132.2 kB | 0.0407 ms/image | Classifier prediction latency; feature extraction is separate |
| ResNet50 | 94.39 MB | 5.66 ms/image | RTX 4060; batch size 1 |
| YOLOv8n-cls | 2.98 MB FP16 | 3.1 ms/image | Apple M3 CPU; batch size 1 |
| MobileNetV2 | To be consolidated | To be consolidated | |
| EfficientNet-B0 | Pending | Pending | |

The classical E4 pipeline required approximately 9.245 seconds to extract features for the 203 images. The reported classifier prediction latency does not include this feature-extraction cost.

The ResNet50 and YOLOv8n-cls latency figures were obtained on different hardware, so they are not a controlled comparison.

## 7. Explainability and shortcut-learning investigation

Model performance alone does not establish that a classifier is using meaningful plant features.

The dataset's class-correlated colour-bar artifact makes visual inspection particularly important.

### ResNet50

Grad-CAM was implemented using the final ResNet50 convolutional stage. The analysis examined whether activation was concentrated on plant regions or potentially irrelevant background regions.

The initial analysis showed that attention patterns differed between classes. In particular, the FN class had lower average plant-region attention than the other classes in the examined seed.

This is a reason for further investigation, not proof that the model is relying on the colour bar. The plant-region mask is also an approximation and may include background or container regions.

### YOLOv8n-cls

Additional tooling has been developed for:

- Grad-CAM visualization.
- Saved-checkpoint verification.
- Shortcut-learning analysis.

The real-checkpoint analysis and controlled checks are still being completed. Until those results are available, no claim should be made that YOLOv8n-cls is free of shortcut learning.

### Interpretation

Explainability visualizations are diagnostic aids. They do not prove that a model has learned biologically valid nutrient-deficiency symptoms.

Further analysis should examine correct and incorrect predictions, compare attention across classes, and test whether predictions change under controlled background or image-style modifications.

## 8. Key findings

1. **YOLOv8n-cls has the highest currently reported macro F1.** Its score is provisional pending additional saved-checkpoint and shortcut-learning checks.
2. **The classical CV + SVM approach is competitive.** The E4 experiment achieved a macro F1 of 0.9263 using handcrafted colour and texture features.
3. **Colour features carry substantial signal.** The colour-statistics and histogram experiments outperformed the texture-only experiment.
4. **Nitrogen and potassium deficiency are frequently confused.** This pattern appears in the pooled classical-model errors.
5. **High benchmark scores require careful interpretation.** The small dataset, severe class imbalance and class-correlated image artifacts limit the strength of generalization claims.
6. **The benchmark does not establish transfer to fenugreek.** The target-domain aeroponic dataset must be collected and evaluated independently.

## 9. Limitations

The following limitations apply to the benchmark:

- **Small dataset:** Only 203 unique images are used.
- **Class imbalance:** The FN class contains just 12 images.
- **Image-level evaluation:** Plant-independent generalization has not been established because reliable plant identity metadata is unavailable.
- **Potential shortcut learning:** Image-generation and presentation artifacts are correlated with class labels.
- **Limited acquisition diversity:** Background, lighting and image style may influence predictions.
- **Repeated split dependence:** The three seeds reuse the same underlying images; their standard deviations are not independent-sample confidence intervals.
- **Architecture-specific implementations:** Training details and hardware differ between approaches.
- **Uncalibrated confidence:** SVM decision scores are not calibrated probabilities.
- **Incomplete model comparison:** EfficientNet-B0 results remain pending.
- **Target-domain gap:** Lettuce benchmark results do not establish performance on fenugreek or in an aeroponic environment.

## 10. Future work

The public benchmark is a preliminary step toward the larger aeroponic nutrient-deficiency detection project.

Planned work includes:

1. Complete the EfficientNet-B0 experiments and consolidate the five-model comparison.
2. Complete YOLOv8n-cls saved-checkpoint verification and shortcut-learning analysis.
3. Further inspect Grad-CAM outputs and misclassified examples.
4. Collect an independent image dataset of aeroponically grown fenugreek under controlled nutrient treatments.
5. Train and evaluate models on the target-domain dataset.
6. Investigate transfer learning from the public lettuce dataset to fenugreek, without assuming direct generalization.
7. Explore lightweight inference and quantization for possible edge deployment.

## 11. Repository structure

```text
lettuce-npk-benchmark/
├── README.md
├── CONTRIBUTING.md
├── requirements.txt
├── data/
│   ├── raw/                    # Original Kaggle files; not committed
│   ├── processed/
│   │   ├── metadata.csv
│   │   ├── audit.csv
│   │   └── excluded_exact_duplicates.csv
│   └── splits/
│       ├── seed_42/
│       ├── seed_123/
│       └── seed_2026/
├── docs/
│   └── dataset_audit.md
├── models/
│   ├── classical_svm/
│   └── resnet50/
├── notebooks/
├── results/
│   ├── classical_ml/
│   ├── member_1/
│   ├── member_3/
│   ├── member_4/
│   └── comparison/
└── tests/
```

The exact contents of individual model directories may evolve as implementations are integrated.

## 12. Reproducibility

To reproduce the benchmark:

1. Obtain the original Kaggle dataset.
2. Follow the audit and preprocessing procedure documented in `docs/dataset_audit.md`.
3. Use the existing metadata manifest and shared split CSVs.
4. Preserve the fixed class mapping:

```python
CLASS_NAMES = ["FN", "-N", "-P", "-K"]
```

5. Apply the shared watermark-safe crop before model-specific preprocessing.
6. Use validation data for model selection and keep held-out test data untouched.
7. Export predictions using the common schema.
8. Recompute seed-level out-of-fold metrics from saved predictions.

The raw dataset and large model checkpoints should not be committed to GitHub.

## 13. Team contributions

| Member | Contribution |
|---|---|
| Person 1 | Dataset audit, shared metadata and split generation, MobileNetV2 baseline, integration and result consolidation |
| Person 2 (Arya Pramesh) | EfficientNet-B0 implementation and evaluation |
| Person 3 | ResNet50 implementation, evaluation and Grad-CAM analysis |
| Person 4 | YOLOv8n-cls implementation, evaluation and additional verification/shortcut-analysis tooling |
| Person 5 | Classical computer-vision feature extraction, SVM and Random Forest experiments |

Replace the generic member labels with the contributors' names and GitHub usernames before publication.

---

**Project scope statement:** This repository presents a comparative benchmark of machine-learning approaches on a public lettuce nutrient-condition dataset. It is preliminary research supporting a separate effort on nutrient-deficiency detection in aeroponically grown fenugreek. The benchmark results should not be interpreted as validation of the target-domain system.