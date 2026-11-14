# Within-1024×1024 Group Analysis & Shortcut Investigation (Person 4)

## 1. Overview and Motivation

In the lettuce nutrient deficiency benchmark, all 12 healthy (FN) images have original dimensions of 1024×1024 pixels, whereas deficiency classes (-N, -P, -K) contain a mix of 1024×1024 images and non-1024 images (ranging from 1015×999 to 1024×1008). This raised a concern from code review that the YOLOv8n-cls classifier might exploit acquisition resolution or resampling artifacts as a shortcut to predict FN.

To assess whether the model simply associates the 1024×1024 acquisition group with the FN label, we evaluated held-out test predictions across all 3 seeds (42, 123, 2026) stratified by original image dimensions. This analysis uses only the published `predictions.csv` and dataset `audit.csv` without requiring checkpoints.

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

## 4. False FN Predictions (Non-FN Predicted as FN)

Across all 609 predictions in the benchmark, exactly 5 predictions predicted FN when the true label was not FN. All 5 false FN predictions are true -N samples:

| Image ID | Image Path | True Label | Pred | Seed | Fold | Conf | Size | 1024 Group? |
|---|---|---|---|---|---|---|---|---|
| `2e67b5f5a20579bb` | `-N/n_25.png` | -N | FN | 123 | 1 | 0.4632 | 1024×1024 | Yes |
| `5b925f6cf678bdc0` | `-N/n_24.png` | -N | FN | 123 | 1 | 0.3867 | 1024×1024 | Yes |
| `2c144644b44474b8` | `-N/n_39.png` | -N | FN | 2026 | 3 | 0.8728 | 1023×1005 | No |
| `64a09d14e1ec27c1` | `-N/n_38.png` | -N | FN | 2026 | 3 | 0.6788 | 1024×1003 | No |
| `7f69e9a092095462` | `-N/n_37.png` | -N | FN | 2026 | 3 | 0.7836 | 1023×1005 | No |

### Rate of False FN by Group

- **1024×1024 group**: 2 / 210 non-FN predictions = **0.9524%** (< 1%).
  - Both false positives occurred on distinct physical images (`-N/n_25.png` and `-N/n_24.png`) in Seed 123 Fold 1 with low confidence (< 0.50).
- **Non-1024 group**: 3 / 363 non-1024 predictions = **0.8264%** (< 1%).
  - All three occurred in Seed 2026 Fold 3 (`-N/n_39.png`, `-N/n_38.png`, `-N/n_37.png`).

The false FN rate is virtually identical between 1024×1024 (0.95%) and non-1024 images (0.83%). The model does not demonstrate a bias toward predicting FN simply because an image originates from the 1024×1024 acquisition group.

---

## 5. Caveats and Limitations

1. **Tiny FN sample size**: There are only 12 unique FN images in the entire dataset (averaging ~3 images per fold in 4-fold cross-validation). All statistical claims regarding FN recall rest on this small sample, limiting the statistical power of subgroup comparisons.
2. **Dimension vs. style confounding**: While this analysis demonstrates that the network does not simply associate 1024×1024 dimensions or aspect ratio with the FN class, it cannot rule out other FN-specific shortcut cues (e.g. distinctive background illumination, camera color temperature, or tray surfaces unique to the FN photos).
3. **Pending experimental verification**: Complete confirmation of feature localization requires Grad-CAM spatial heatmaps, border-band attention mass, and the neutral-gray background ablation control (silencing background pixels while preserving leaf tissue). These analyses require the trained fold checkpoints and raw image cache, and will be executed in Phase B once checkpoints are provided.
