# EfficientNet-B0 — results and error analysis (Arya Pramesh / Person 2)

All numbers are recomputed from `predictions.csv` with the shared evaluator
(`lettuce_benchmark.evaluation`). For each seed, the four held-out folds are combined into one
out-of-fold set of 203 images, and the metrics are then averaged over the 3 seeds
(mean ± sample std).

## Headline

| Metric | Mean ± std (3 seeds) | Baseline (MobileNetV2) |
|---|---|---|
| **Macro F1 (primary)** | **0.928 ± 0.026** | 0.879 ± 0.067 |
| Accuracy | 0.926 ± 0.018 | 0.888 ± 0.049 |
| Balanced accuracy | 0.936 ± 0.024 | 0.899 ± 0.058 |
| Macro precision / recall | 0.921 / 0.936 | 0.867 / 0.899 |
| Weighted F1 | 0.926 ± 0.018 | 0.889 ± 0.047 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| FN (Healthy) | 0.901 ± 0.086 | 0.972 ± 0.048 | 0.934 ± 0.061 | 12 |
| -N (Nitrogen) | 0.919 ± 0.020 | 0.930 ± 0.018 | 0.924 ± 0.018 | 57 |
| -P (Phosphorus) | 0.964 ± 0.023 | 0.944 ± 0.017 | 0.954 ± 0.020 | 66 |
| -K (Potassium) | 0.901 ± 0.009 | 0.897 ± 0.015 | 0.899 ± 0.012 | 68 |

| Seed | Macro F1 | Accuracy | Balanced Acc | FN Recall |
|---|---|---|---|---|
| 42 | 0.953 | 0.941 | 0.953 | 1.000 |
| 123 | 0.930 | 0.931 | 0.945 | 1.000 |
| 2026 | 0.901 | 0.906 | 0.909 | 0.917 |

**Efficiency (CPU, 12 threads; time.perf_counter at batch size 1, 224×224):**
- Total parameters: 4,012,672 (1,280 features into 4-class classifier)
- Checkpoint size: 16.35 MB per run
- Inference latency: 42.6 ± 12.4 ms per image (batch size 1, CPU)
- Training duration: 216.2 s (~3.6 min) mean per run; 43.2 min total for all 12 runs
- Peak GPU memory: not applicable (CPU execution)

---

## What goes wrong

1. **-K ↔ -N and -K ↔ -P confusion dominates.** Potassium deficiency (`-K`) manifests as marginal scorching and necrotic spotting on older leaves, which overlaps visually with severe Nitrogen chlorosis (`-N`) and dark anthocyanin purpling in Phosphorus deficiency (`-P`). For example:
   - `-K/k_33` and `-K/k_35` have pale yellow-green leaf bases and were predicted as `-N` across all 3 seeds.
   - `-K/k_60` has dark purple-red outer foliage and was consistently called `-P` in all seeds.
   - `-N/n_22` displays necrotic lower foliage and was called `-K` in all seeds.
2. **Substantially fewer total errors than MobileNetV2.** Across all 3 seeds, only 27 unique images were misclassified at least once (vs 46 in MobileNetV2), and only 6 images were misclassified across all three seeds (`misclassified_images.csv`).
3. **High minority class sensitivity (FN).** Unlike MobileNetV2 where false alarms dropped FN precision to 0.757, EfficientNet-B0 maintained a mean FN recall of **0.972** (35/36 held-out FN evaluations correct across seeds) and a mean FN precision of **0.901**.
4. **Low confidence on errors.** Confident predictions average > 0.85 probability, while misclassified predictions exhibit a much lower average confidence (0.63).

---

## Seed stability and convergence

EfficientNet-B0 exhibited markedly superior cross-seed stability compared to MobileNetV2:
- Macro F1 standard deviation across seeds was **0.026** (vs 0.067 for MobileNetV2).
- The lowest seed score was **0.901** (Seed 2026), compared to MobileNetV2's lowest score of 0.801.
- In Seed 2026, validation checkpoint selection picked Epoch 4 (Fold 1), Epoch 10 (Fold 2), Epoch 12 (Fold 3), and Epoch 15 (Fold 4). Even on early-stopping folds, compound scaling retained robust feature representations.

---

## Shortcut check (DALL·E colour bar)

Following the top-aligned centered 960×960 crop (`docs/dataset_audit.md`), performance is evaluated on images originally bearing the DALL·E watermark bar vs clean images (`accuracy_by_colour_bar.csv`):

| Seed | Accuracy, had bar (82 images) | Accuracy, no bar (121 images) |
|---|---|---|
| 42 | 0.902 | 0.967 |
| 123 | 0.902 | 0.950 |
| 2026 | 0.866 | 0.934 |

The model achieves high accuracy across both subgroups (> 86% and > 93%). In fact, clean images without watermarks show slightly higher accuracy, demonstrating that the classifier relies on real plant features rather than watermark artifacts.

**Grad-CAM analysis** ([`gradcam/`](./gradcam/README.md)):
- **87.3%** of visual attention is focused directly on plant tissue (compared to 82.0% in MobileNetV2).
- In **100% of images**, the majority of attention falls on the plant.
- The bottom-right corner where the watermark originally sat receives only **1.66%** attention mass (less than half the 4% area expected by chance).

---

## Limitations

- **Image-level evaluation only:** No plant identity metadata exists, so independence across individual plants cannot be strictly validated.
- **AI-generated image dominance:** The dataset contains synthetic DALL·E images, so findings reflect visual symptom recognition in this benchmark rather than live aeroponic fenugreek cultivation.
- **FN minority sample size:** `FN` contains only 12 images (3 per fold), so a single misclassification shifts FN recall by ~0.083.
- **Compute footprint:** EfficientNet-B0 has 4.01M parameters (vs 2.23M for MobileNetV2) and takes ~42.6 ms CPU latency (vs ~17.8 ms for MobileNetV2). The trade-off yields a significant +4.9% boost in Macro F1 and much tighter variance.
