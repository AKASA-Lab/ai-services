# Within-1024×1024 Group Analysis & Shortcut Investigation (Person 4)

## 1. Overview and Motivation

In the lettuce nutrient deficiency benchmark, all 12 healthy (FN) images have original dimensions of 1024×1024 pixels, whereas deficiency classes (-N, -P, -K) contain a mix of 1024×1024 images and non-1024 images (ranging from 1015×999 to 1024×1008). This raised a concern from code review that the YOLOv8n-cls classifier might exploit acquisition resolution or resampling artifacts as a shortcut to predict FN.

To assess whether the model associates the 1024×1024 acquisition group with the FN label, we evaluated held-out test predictions across all 3 seeds (42, 123, 2026) stratified by original image dimensions. This analysis uses only the published `predictions.csv` and dataset `audit.csv` without requiring checkpoints.

---

## 2. Dataset Composition: Unique Images vs. Predictions

Across 4 folds and 3 random seeds, each image is evaluated once per seed in its held-out test fold, yielding exactly 3 test predictions per unique image (609 total predictions across 203 unique images):

| Group | Class | Unique Images | Total Predictions (3 Seeds) |
|---|---|---|---|
| **1024×1024** | FN | 12 | 36 |
| **1024×1024** | -N | 8 | 24 |
| **1024×1024** | -P | 48 | 144 |
| **1024×1024** | -K | 14 | 42 |
| *Subtotal (1024×1024)* | *All* | *82* | *246* |
| **Non-1024** | FN | 0 | 0 |
| **Non-1024** | -N | 49 | 147 |
| **Non-1024** | -P | 18 | 54 |
| **Non-1024** | -K | 54 | 162 |
| *Subtotal (Non-1024)* | *All* | *121* | *363* |
| **Overall Benchmark** | **Total** | **203** | **609** |

---

## 3. Confusion Matrices and Per-Class Recall

### Within-1024×1024 Group (246 Predictions)

| True \ Pred | Pred FN | Pred -N | Pred -P | Pred -K | Recall |
|---|---|---|---|---|---|
| **True FN** | 36 | 0 | 0 | 0 | **1.0000** (36/36) |
| **True -N** | 2 | 18 | 1 | 3 | **0.7500** (18/24) |
| **True -P** | 0 | 0 | 131 | 13 | **0.9097** (131/144) |
| **True -K** | 0 | 0 | 3 | 39 | **0.9286** (39/42) |

### Non-1024 Group (363 Predictions)

| True \ Pred | Pred FN | Pred -N | Pred -P | Pred -K | Recall |
|---|---|---|---|---|---|
| **True FN** | 0 | 0 | 0 | 0 | **N/A** (0 images) |
| **True -N** | 3 | 137 | 1 | 6 | **0.9320** (137/147) |
| **True -P** | 0 | 1 | 53 | 0 | **0.9815** (53/54) |
| **True -K** | 0 | 6 | 0 | 156 | **0.9630** (156/162) |

### Recall Comparison

- **FN**: 1.0000 in 1024×1024 group (36/36). No FN images exist in the non-1024 group.
- **-N**: 0.7500 in 1024×1024 group vs. 0.9320 in non-1024 group.
- **-P**: 0.9097 in 1024×1024 group vs. 0.9815 in non-1024 group.
- **-K**: 0.9286 in 1024×1024 group vs. 0.9630 in non-1024 group.

---

## 4. False FN Predictions and Within-Class (-N) Comparison

Across all 609 predictions in the benchmark, exactly 5 predictions predicted FN when the true label was not FN. All 5 false FN predictions are true -N samples:

| Image ID | Image Path | True Label | Pred | Seed | Fold | Conf | Size | 1024 Group? |
|---|---|---|---|---|---|---|---|---|
| `2e67b5f5a20579bb` | `-N/n_25.png` | -N | FN | 123 | 1 | 0.4632 | 1024×1024 | Yes |
| `5b925f6cf678bdc0` | `-N/n_24.png` | -N | FN | 123 | 1 | 0.3867 | 1024×1024 | Yes |
| `2c144644b44474b8` | `-N/n_39.png` | -N | FN | 2026 | 3 | 0.8728 | 1023×1005 | No |
| `64a09d14e1ec27c1` | `-N/n_38.png` | -N | FN | 2026 | 3 | 0.6788 | 1024×1003 | No |
| `7f69e9a092095462` | `-N/n_37.png` | -N | FN | 2026 | 3 | 0.7836 | 1023×1005 | No |

### Overall vs. Within-Class (-N) Comparison

The apparent overall parity between groups (2 / 210 = 0.95% vs. 3 / 363 = 0.83%) is heavily dominated by -P and -K, which never produce a single false-FN prediction (0 / 198 and 0 / 204 predictions). 

When restricted to the only class that exhibits confusion with FN (-N):
- **Predictions level**:
  - 1024×1024 -N: **2 / 24 (8.33%)**
  - Non-1024 -N: **3 / 147 (2.04%)**
  - Fisher exact test: **p = 0.1449** (odds ratio = 4.36)
- **Unique images level**:
  - 1024×1024 -N: **2 / 8 unique images (25.00%)**
  - Non-1024 -N: **3 / 49 unique images (6.12%)**
  - Fisher exact test: **p = 0.1398** (odds ratio = 5.11)

While the higher rate in the 1024 group (8.3% vs. 2.0% at prediction level; 25.0% vs. 6.1% at image level) does not reach statistical significance (p ≈ 0.14) due to small sample counts, **this failure to reject the null is not evidence of equivalence**. The elevated false-FN rate among 1024×1024 -N images leaves open the possibility that 1024-specific features increase susceptibility to false-FN predictions.

### Clustering of Errors

Crucially, all 5 false-FN predictions originate from exactly two isolated evaluation runs:
1. **Cluster 1**: Seed 123, Fold 1 — 2 errors (`n_24.png` and `n_25.png`), both with low confidence (< 0.50).
2. **Cluster 2**: Seed 2026, Fold 3 — 3 errors (`n_37.png`, `n_38.png`, `n_39.png`), with moderate-to-high confidence (0.68–0.87).

Because adjacent images in each run share a single fold model and identical test-set partitioning, these 5 errors represent approximately **two independent model/fold failure events** rather than five independent data points.

---

## 5. Caveats and Limitations

1. **Tiny FN sample size**: There are only 12 unique FN images in the entire dataset (averaging ~3 images per fold in 4-fold cross-validation). All statistical claims regarding FN recall rest on this small sample, severely constraining statistical power.
2. **Observational confounding**: While 1024×1024 -P and -K images show zero false-FN predictions, the 4-fold elevated rate of false-FNs within -N (8.3% vs 2.0%) means we cannot rule out size-, background-, or style-correlated shortcut learning.
3. **Pending experimental verification**: Complete confirmation of feature localization requires Grad-CAM spatial heatmaps, border-band attention mass, and the neutral-gray background ablation control (silencing background pixels while preserving leaf tissue). These analyses require the trained fold checkpoints and raw image cache, and will be executed in Phase B once checkpoints are provided.
