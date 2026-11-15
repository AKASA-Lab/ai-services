# YOLOv8n-cls — results and error analysis (Person 4)

All numbers are recomputed from `predictions.csv` with the shared evaluator
(`lettuce_benchmark.evaluation`). For each seed, the four held-out folds are combined into one
out-of-fold set of 203 images. The metrics are then averaged over the 3 seeds (mean ± sample std).

## Headline

| Metric | Mean ± std (3 seeds) |
|---|---|
| **Macro F1 (primary)** | **0.936 ± 0.016** |
| Accuracy | 0.936 ± 0.013 |
| Balanced accuracy | 0.948 ± 0.010 |
| Macro precision / recall | 0.929 / 0.948 |
| Weighted F1 | 0.936 ± 0.013 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| FN | 0.886 | 1.000 | 0.937 |
| -N | 0.957 | 0.906 | 0.931 |
| -P | 0.974 | 0.929 | 0.951 |
| -K | 0.900 | 0.956 | 0.927 |

| Seed | Macro F1 | Accuracy | Balanced accuracy | FN recall |
|---|---|---|---|---|
| 42 | 0.946 | 0.931 | 0.945 | 1.000 |
| 123 | 0.945 | 0.951 | 0.959 | 1.000 |
| 2026 | 0.918 | 0.926 | 0.939 | 1.000 |

**Efficiency (CPU, Apple M3; latency with 2 threads, not comparable with GPU numbers):**

- 1,443,412 parameters, 3.4 GFLOPs at 224×224.
- Checkpoint: 2.98 MB as an Ultralytics `.pt` (FP16), 5.86 MB as an FP32 state dict.
- Latency: 3.1 ± 0.2 ms per image at batch size 1 (bare network, after warm-up).
- Training: about 17 s per run, 3.4 min for all 12.
- Peak GPU memory: not applicable.

The MobileNetV2 latency (17.8 ms) and training time were measured on different hardware, so they
can't be compared directly. The size and parameter numbers can be compared.

## What goes wrong

1. **Most errors are predictions of -K.** Over the 3 seeds, 13 -P images and 9 -N images were
   called -K. -K precision (0.90) is the lowest of the four classes. Three of the four images that
   are wrong in all three seeds are called -K: `-N/n_22` (also wrong in every seed for MobileNetV2),
   `-N/n_14` and `-P/p19`.
2. **-K ↔ -P remains a hard pair.** `-K/k_60`, the dark purple-red plant that MobileNetV2 also
   misses in every seed, is called -P in all three seeds here too.
3. **FN is never missed, but -N is sometimes called FN.** FN recall is 1.000 in every seed (12/12).
   FN precision is 0.886 because of 5 false positives in total (2 in seed 123, 3 in seed 2026), all
   from -N. This is the same direction of error as MobileNetV2. With only 12 FN images, each false
   positive lowers FN precision by about 0.05–0.08.
4. **26 of the 203 images are wrong at least once, and 4 are wrong in every seed**
   (`misclassified_images.csv`). MobileNetV2 had 46 and 6.
5. **Wrong predictions are less confident:** mean confidence is 0.70 for errors vs 0.94 for correct
   predictions. However, the errors that repeat in every seed are confident (0.74–0.81), so a
   confidence threshold would not catch them.

## Same images vs MobileNetV2 (Person 1)

Both models were tested on identical out-of-fold images (609 paired predictions, 3 seeds × 203):

| Seed | Both right | Only YOLOv8n-cls right | Only MobileNetV2 right | Both wrong |
|---|---|---|---|---|
| 42 | 177 | 12 | 8 | 6 |
| 123 | 182 | 11 | 5 | 5 |
| 2026 | 163 | 25 | 6 | 9 |

The two models agree more often than they disagree, and they share their hardest images (`k_60`,
`n_22`). Most of YOLOv8n-cls's lead comes from seed 2026. There, MobileNetV2's validation-based
selection picked early epochs (see the member_1 error analysis). YOLOv8n-cls was much less
sensitive to the seed (std 0.016 vs 0.067).

Do **not** read this as "YOLOv8n-cls is the better architecture". The two training recipes differ:

- this model trains every layer from epoch 1 at one learning rate, with LR decay and EMA weights;
- MobileNetV2 uses a frozen-head phase and a constant LR.

These differences could explain part of the gap. A recipe-matched comparison is integration work.

## Macro F1 ≥ 0.90: leakage / shortcut checks

The project README asks for a check whenever macro F1 reaches 0.90 or more. What was checked:

- **Split integrity.** Every run's Ultralytics folder was re-read and matched file by file to the
  shared split CSV. The test fold is never written to that folder, so the trainer could not
  validate on it or fall back to it (the Ultralytics log shows `test: None`). Each test fold was
  predicted exactly once, after the epoch had been chosen on validation data. The shared splits had
  already passed `validate_splits`.
- **Duplicates.** The 5 byte-identical copies were removed from the manifest before splitting.
  Person 1's audit found no further near-duplicates (`docs/dataset_audit.md` §1, §3).
- **Colour-bar shortcut.** The fixed 960×960 crop removes the bar. After the crop, accuracy on
  images that originally had the bar is not higher than on images without it
  (`accuracy_by_colour_bar.csv`):

| Seed | Accuracy, had bar (82 images) | Accuracy, no bar (121 images) |
|---|---|---|
| 42 | 0.890 | 0.959 |
| 123 | 0.915 | 0.975 |
| 2026 | 0.927 | 0.926 |

- **What is *not* ruled out.** All 12 FN images are in the 1024×1024 group, which looks
  AI-generated. So FN's perfect recall could partly come from the image style rather than the
  plant, and the -N → FN errors fit that picture. No explainability (Grad-CAM/LIME) was run for this
  model yet.

## Model-selection notes

- Validation macro F1 was 1.0 at the selected epoch in 3 of 12 runs, and ≥ 0.92 in all of them.
  Each validation set has only 31 images (2 FN), so many epochs tie, and the tie-break on
  validation loss decides.
- **Selected checkpoint vs training-time validation score.** In 11 of 12 runs, re-predicting the
  validation images with the saved checkpoint gives exactly the macro F1 logged during training.
  The exception is seed 123 / fold 3 (logged 0.923, re-predicted 0.947). One `-P` validation image
  scores 0.499 for both -P and -K. The trainer validates FP32 weights, while Ultralytics saves
  checkpoints in FP16, and that rounding flips this single image. The Ultralytics predictor and the
  Ultralytics validation dataloader give identical predictions for all 31 images, so the pipeline
  itself is consistent. Test predictions always come from the saved checkpoint.
- The best epoch ranges from 4 to 13, with no consistent preference for the last epoch.

## Limitations

- Image-level evaluation only. No plant/session IDs exist, so plant-independent generalisation is
  not shown.
- A large share of the images appear to be AI-generated (DALL·E colour bar), and every FN image is
  among them. These results say little about real hydroponic lettuce, and nothing about aeroponic
  fenugreek.
- FN has 12 images (3 per test fold), so FN metrics move by about 0.08 per single image.
- Latency was measured on an Apple M3 CPU. Re-measure every model on the same hardware before
  comparing efficiency.
