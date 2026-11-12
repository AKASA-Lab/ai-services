# MobileNetV2 — results and error analysis (Person 1)

All numbers are recomputed from `predictions.csv` with the shared evaluator
(`lettuce_benchmark.evaluation`). For each seed, the four held-out folds are combined into one
out-of-fold set of 203 images, and the metrics are then averaged over the 3 seeds
(mean ± sample std).

## Headline

| Metric | Mean ± std (3 seeds) |
|---|---|
| **Macro F1 (primary)** | **0.879 ± 0.067** |
| Accuracy | 0.888 ± 0.049 |
| Balanced accuracy | 0.899 ± 0.058 |
| Macro precision / recall | 0.867 / 0.899 |
| Weighted F1 | 0.889 ± 0.047 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| FN | 0.757 | 0.944 | 0.838 |
| -N | 0.913 | 0.860 | 0.885 |
| -P | 0.916 | 0.914 | 0.913 |
| -K | 0.882 | 0.877 | 0.878 |

| Seed | Macro F1 | Accuracy | FN recall |
|---|---|---|---|
| 42 | 0.914 | 0.911 | 1.000 |
| 123 | 0.921 | 0.921 | 1.000 |
| 2026 | 0.801 | 0.833 | 0.833 |

**Efficiency (CPU, 2 threads; not comparable with GPU numbers):** 2,228,996 parameters,
9.16 MB checkpoint, 17.8 ± 1.4 ms per image at batch size 1 (224×224, after warm-up),
about 87 s training per run (17.5 min for all 12). Peak GPU memory: not applicable.

## What goes wrong

1. **-K ↔ -P is the most common confusion.** In seed 2026, 13 of the 68 -K images were called
   -P. The images that are wrong in all three seeds include dark, purple-red-leaved plants from
   both classes (`-K/k_60` → -P, `-P/p58` → -K). Leaf colour alone doesn't separate these two
   classes, and the classifier struggles on exactly those.
2. **-N is sometimes read as FN.** There are seven such errors in seed 2026 and one each in the
   others. `-N/n_38` (mostly uniformly green leaves) is called FN in all three seeds. Because FN has
   only 12 images, a few false positives drop FN precision to 0.76, even though FN recall is high
   (0.94).
3. **Lower-leaf symptoms pull predictions towards -K.** `-N/n_22` has a yellow-brown, necrotic
   lower leaf and is called -K in all three seeds.
4. **46 of the 203 images are wrong at least once, but only 6 are wrong in every seed**
   (`misclassified_images.csv`). Most errors depend on which images ended up in training, which is
   expected with about 121 training images per run.
5. **Wrong predictions are low-confidence:** mean confidence is 0.47 for errors vs 0.77 for
   correct predictions.

## Why seed 2026 is lower

Checkpoints are selected on validation macro F1 over only 31 images, and just 2 of them are FN.
One validation mistake therefore moves macro F1 a lot. In seed 2026 this rule picked early epochs in
every fold (2, 4, 8 and 8, against 10–15 for the other seeds), before fine-tuning had converged.
Folds 3 and 4 give the lowest test scores of all 12 runs (macro F1 0.74 and 0.76). The rule was fixed before seeing any test
results, so it was not changed afterwards. That would be tuning on the test set. For the team:
selecting on validation loss, or averaging the last few epochs, would probably be more stable,
but that has to be agreed on validation data only and applied to every model.

## Shortcut check

After the fixed 960×960 crop (`docs/dataset_audit.md` §4), accuracy is similar for images that
originally had the DALL·E colour bar and images that did not (`accuracy_by_colour_bar.csv`):

| Seed | Accuracy, had bar (82 images) | Accuracy, no bar (121 images) |
|---|---|---|
| 42 | 0.878 | 0.934 |
| 123 | 0.927 | 0.917 |
| 2026 | 0.829 | 0.835 |

So the model is not just "detecting the bar". FN still sits entirely in the
1024×1024 (likely AI-generated) group, though, so other style differences between the two groups
could still help on FN. Grad-CAM would be the next check.

## Limitations

- Image-level evaluation only. No plant/session IDs exist, so plant-independent generalisation
  is not shown.
- A large share of the images appear to be AI-generated (DALL·E colour bar), so these results
  say little about real hydroponic lettuce, and nothing about aeroponic fenugreek.
- FN has 12 images (3 per test fold), so FN metrics move by about 0.08 per single image.
- The best epoch was often the last (15), so a longer schedule might help. The README budget of 15
  epochs was kept so the comparison stays fair.
