# Classical ML experiment analysis

## Scope and evidence

This report analyzes the saved real-data outputs in this directory: [`metrics.csv`](./metrics.csv), [`predictions.csv`](./predictions.csv), [`experiment_configurations.json`](./experiment_configurations.json), [`efficiency.csv`](./efficiency.csv), and the fold/seed matrices in [`confusion_matrices/`](./confusion_matrices/). No model was retrained for this analysis, and no existing result was changed.

The prediction file contains 3,045 held-out predictions: 203 images × 3 seeds × 5 experiments. For each experiment, metrics are computed on the 203 out-of-fold predictions for each seed. The reported aggregate is the mean and sample standard deviation (`ddof=1`) across the three seed-level OOF metric values. Per-class metrics below are likewise the mean of the three seed-level values. Fold ranges describe the 12 fold-test rows and are descriptive, not independent-sample confidence intervals.

## 1. Overall comparison

| Experiment | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| E1 | 0.8768 ± 0.0226 | 0.8994 ± 0.0180 | 0.8629 ± 0.0314 | 0.8994 ± 0.0180 | 0.8764 ± 0.0270 | 0.8767 ± 0.0222 |
| E2 | 0.8834 ± 0.0186 | 0.8884 ± 0.0288 | 0.8962 ± 0.0361 | 0.8884 ± 0.0288 | 0.8917 ± 0.0318 | 0.8831 ± 0.0190 |
| E3 | 0.7356 ± 0.0103 | 0.7396 ± 0.0170 | 0.7032 ± 0.0315 | 0.7396 ± 0.0170 | 0.7093 ± 0.0195 | 0.7399 ± 0.0108 |
| E4 | **0.9130 ± 0.0075** | **0.9228 ± 0.0139** | **0.9316 ± 0.0067** | **0.9228 ± 0.0139** | **0.9263 ± 0.0103** | **0.9125 ± 0.0076** |
| E5 | 0.8834 ± 0.0103 | 0.9054 ± 0.0073 | 0.8917 ± 0.0384 | 0.9054 ± 0.0073 | 0.8961 ± 0.0252 | 0.8826 ± 0.0106 |

E4 has the highest mean for each listed aggregate metric. E3 has the lowest macro F1 and accuracy. E5 has the second-highest macro F1, 0.0302 below E4. These are descriptive comparisons; no significance test was performed.

## 2. Per-class results

Values are mean precision / recall / F1 across the three seed-level OOF evaluations. Supports are the dataset counts represented in each seed's OOF predictions.

| Experiment | Class | Support | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| E1 | FN | 12 | 0.7877 | 1.0000 | 0.8799 |
| E1 | -N | 57 | 0.8957 | 0.8012 | 0.8451 |
| E1 | -P | 66 | 0.9374 | 0.9091 | 0.9228 |
| E1 | -K | 68 | 0.8306 | 0.8873 | 0.8578 |
| E2 | FN | 12 | 0.9444 | 0.9167 | 0.9300 |
| E2 | -N | 57 | 0.8663 | 0.8246 | 0.8445 |
| E2 | -P | 66 | 0.9454 | 0.9596 | 0.9524 |
| E2 | -K | 68 | 0.8287 | 0.8529 | 0.8401 |
| E3 | FN | 12 | 0.4984 | 0.7500 | 0.5842 |
| E3 | -N | 57 | 0.7839 | 0.7602 | 0.7696 |
| E3 | -P | 66 | 0.8270 | 0.7374 | 0.7790 |
| E3 | -K | 68 | 0.7033 | 0.7108 | 0.7045 |
| E4 | FN | 12 | 1.0000 | 0.9722 | 0.9855 |
| E4 | -N | 57 | 0.9067 | 0.8421 | 0.8724 |
| E4 | -P | 66 | 0.9606 | 0.9848 | 0.9726 |
| E4 | -K | 68 | 0.8590 | 0.8922 | 0.8748 |
| E5 | FN | 12 | 0.9167 | 1.0000 | 0.9524 |
| E5 | -N | 57 | 0.8928 | 0.8187 | 0.8538 |
| E5 | -P | 66 | 0.9050 | 0.9596 | 0.9314 |
| E5 | -K | 68 | 0.8523 | 0.8431 | 0.8470 |

### Weakest class by experiment

Using mean per-class F1, E1's weakest class is -N (0.8451), E2's is -K (0.8401), E3's is FN (0.5842), E4's is -N (0.8724), and E5's is -K (0.8470). E2's -N score (0.8445) is close to its weakest -K score. The particularly low E3 FN precision (0.4984) occurs in a class with only 12 OOF examples per seed.

## 3. Confusion patterns

Across all five experiments, the 3,045 predictions contain 431 errors. Counts below pool the three seed-level OOF predictions and all five experiments, so each original image can contribute once per seed and experiment.

| True → predicted | Count | Share of all errors |
|---|---:|---:|
| -N → -K | 124 | 28.8% |
| -K → -N | 91 | 21.1% |
| -K → -P | 65 | 15.1% |
| -P → -K | 63 | 14.6% |
| -N → FN | 22 | 5.1% |

The reciprocal -N / -K confusion is the most persistent pattern. The four leading pairs account for 343 of 431 errors (79.6%). This aggregate does not mean the errors are independent: the same images are evaluated under multiple seeds and experiments.

| Experiment | Test errors (of 609) | Most frequent confusions |
|---|---:|---|
| E1 | 75 | -N → -K (20); -P → -K (17); -K → -N (15) |
| E2 | 71 | -N → -K (27); -K → -N (20); -K → -P (10) |
| E3 | 161 | -P → -K (29); -N → -K (28); -K → -N (25); -K → -P (24) |
| E4 | 53 | -N → -K (26); -K → -N (15); -K → -P (7) |
| E5 | 71 | -N → -K (23); -K → -N (16); -K → -P (16) |

E4 reduces errors on several of these common nutrient-class confusions relative to E1, E2, and E5, but -N / -K remains its dominant error pair. E3 has the most errors and also confuses -P and -K frequently.

## 4. Variability across seeds and folds

Seed-level OOF macro F1:

| Experiment | Seed 42 | Seed 123 | Seed 2026 | Seed range | Across-seed sample SD |
|---|---:|---:|---:|---:|---:|
| E1 | 0.9052 | 0.8722 | 0.8517 | 0.0536 | 0.0270 |
| E2 | 0.9077 | 0.9124 | 0.8551 | 0.0573 | 0.0318 |
| E3 | 0.6917 | 0.7303 | 0.7060 | 0.0385 | 0.0195 |
| E4 | 0.9363 | 0.9269 | 0.9158 | 0.0205 | 0.0103 |
| E5 | 0.9092 | 0.9122 | 0.8671 | 0.0451 | 0.0252 |

| Experiment | Fold-test macro F1 across 12 runs: min–max | Fold mean | Fold sample SD |
|---|---:|---:|---:|
| E1 | 0.7601–0.9218 | 0.8782 | 0.0579 |
| E2 | 0.8014–0.9531 | 0.8904 | 0.0448 |
| E3 | 0.5593–0.8141 | 0.7085 | 0.0770 |
| E4 | 0.8496–0.9836 | 0.9252 | 0.0433 |
| E5 | 0.8101–0.9537 | 0.8970 | 0.0480 |

E4 has the smallest observed seed-level macro-F1 SD and seed range; E2 has the largest seed range. E3 has the widest fold-test range and largest fold SD. Fold averages differ slightly from seed OOF aggregates because the former average fold metrics equally, whereas OOF metrics are calculated over all held-out predictions for a seed. These repeated-split summaries describe variability; they do not establish statistical significance.

## 5. Selected hyperparameters

The predefined SVM search had nine candidates: `C ∈ {0.1, 1, 10}` crossed with `gamma ∈ {"scale", 0.01, 0.001}`. The Random Forest search had eight candidates: `n_estimators ∈ {100, 300}`, `max_depth ∈ {None, 20}`, and `min_samples_leaf ∈ {1, 2}`, with `max_features="sqrt"` fixed. Macro F1 on validation selected the configuration in each run. [`experiment_configurations.json`](./experiment_configurations.json) records each of the 60 selected configurations and validation scores.

| Experiment | Selected configurations across its 12 runs |
|---|---|
| E1 | SVM `C=10, gamma=scale` (8); `C=1, gamma=scale` (2); `C=1, gamma=0.01` (1); `C=10, gamma=0.01` (1) |
| E2 | SVM `C=10, gamma=scale` (7); `C=10, gamma=0.001` (5) |
| E3 | SVM `C=1, gamma=scale` (6); `C=10, gamma=scale` (6) |
| E4 | SVM `C=10, gamma=scale` (7); `C=1, gamma=scale` (3); `C=10, gamma=0.001` (2) |
| E5 | RF 100 trees, depth None, leaf 2 (5); 100, None, leaf 1 (3); 300, None, leaf 1 (2); 300, None, leaf 2 (2) |

For E5, `max_depth=None` was selected in all 12 runs; no depth-20 candidate was selected. Each configuration was chosen from validation results only. The final fit then used train plus validation, as specified.

## 6. Colour versus texture

**Observed:** E2 (colour histograms) exceeded E1 (colour statistics) by 0.0154 macro F1. E3 (GLCM + LBP only) scored 0.7093, 0.1824 below E2. E4 (all features) exceeded E2 by 0.0346 and E3 by 0.2170 macro F1. Thus, the colour-only runs were substantially stronger than texture-only E3 in this experiment matrix, while the combined feature set performed best.

**Interpretation / hypothesis:** The results are consistent with useful complementary information from combining colour statistics, colour histograms, and texture. However, the matrix does not include every ablation (for example, colour-only statistics plus histograms, or all features minus one group), so it cannot isolate how much of E4's gain comes specifically from texture rather than the additional colour statistics, feature interactions, or the selected SVM settings.

## 7. Why E4 may outperform E1–E3 and E5

**Observed:** E4 is best on macro F1 (0.9263), accuracy (0.9130), balanced accuracy (0.9228), macro precision (0.9316), macro recall (0.9228), and weighted F1 (0.9125). Its per-class F1 is also the highest of these five runs for FN, -N, -P, and -K. E4 improves macro F1 by 0.0346 over the better single-colour experiment E2 and by 0.0302 over E5, which uses the same full feature set with a Random Forest. E5's RF configurations favored unrestricted depth, while its OOF macro F1 remained below E4.

**Hypotheses, not causal findings:** Combining features may give the RBF SVM more separable signals than any individual feature family. The SVM's scaled nonlinear boundary may also suit these features better than the searched forests on a small dataset. The results alone do not prove either explanation. The experiment has no repeated independent dataset, no significance test, and no complete feature ablation; tuning noise and sample composition could contribute to the observed differences.

## 8. Efficiency

The shared cache extracted all 128 features once for 203 unique images in 9.245 seconds (45.54 ms/image). Per-run feature time below is the sum of those measured per-image times for that fold's test set; the actual extraction was shared rather than repeated for every experiment. Times are means over 12 runs.

| Experiment | Test-fold feature extraction | Hyperparameter selection | Final refit training | Prediction latency | Serialized model size |
|---|---:|---:|---:|---:|---:|
| E1 | 2.303 s | 0.120 s | 0.00274 s | 0.0309 ms/image | 18.9 kB |
| E2 | 2.303 s | 0.125 s | 0.00313 s | 0.0359 ms/image | 95.3 kB |
| E3 | 2.303 s | 0.120 s | 0.00282 s | 0.0327 ms/image | 18.8 kB |
| E4 | 2.303 s | 0.128 s | 0.00358 s | 0.0407 ms/image | 132.2 kB |
| E5 | 2.303 s | 2.455 s | 0.21913 s | 1.3381 ms/image | 571.0 kB |

Across all 60 runs, validation search took 35.365 seconds, final refits 2.777 seconds, and prediction 0.900 seconds. Search time is reported separately because it includes fitting candidate models and scoring validation data; final-refit training time is not the total tuning cost. Model size is the serialized in-memory pipeline size recorded for each run. These wall-clock values are specific to the execution environment.

## 9. Limitations

- The audited dataset contains 203 images after exact-duplicate removal. FN has only 12 images, versus 57 -N, 66 -P, and 68 -K. Each seed's OOF FN recall is therefore based on only 12 examples; single-image changes can noticeably affect class metrics. High FN scores should be interpreted cautiously.
- The three seeds reuse the same 203 images. The OOF predictions are useful repeated-split estimates, but seed and fold results are not independent replications of a new dataset.
- No statistical significance testing was performed. Mean/std and fold ranges are descriptive only.
- Validation macro F1 selects among finite predefined grids; a different search space could select different models.
- SVM confidence is a softmax-normalized decision score and is uncalibrated. Random Forest `predict_proba` values are not separately calibrated.
- Efficiency results depend on this machine and cache/timing implementation; model size is serialized size, not peak memory or deployed runtime footprint.
- These are results on the lettuce NPK dataset and shared image-level splits. They are **not** validation results on the team's fenugreek aeroponic dataset, and do not establish performance on that domain.

## Presentation-ready findings

1. **E4 (all features + RBF SVM) led the matrix:** macro F1 0.9263 ± 0.0103 across seeds, with best aggregate accuracy and balanced accuracy.
2. **Colour-only features outperformed texture-only features:** E2 macro F1 was 0.8917 versus E3's 0.7093; all features improved over E2 to 0.9263.
3. **-N / -K was the main confusion direction pair:** across experiments, -N→-K occurred 124 times and -K→-N 91 times in the repeated OOF predictions.
4. **E4 was comparatively consistent across seeds:** macro-F1 SD 0.0103; E2 had the largest seed range (0.0573).
5. **The small FN class remains a major caveat:** only 12 FN images contribute to each seed's OOF score. Findings are descriptive and apply only to this lettuce dataset.
