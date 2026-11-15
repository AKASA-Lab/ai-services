# Checkpoint Verification & Reproduction Summary (Reproduced Checkpoints)

> **Checkpoint Source:** **Reproduction** (Trained on Windows x86_64 CPU with PyTorch 2.14.1+cpu and Ultralytics 8.4.212 using unchanged `models/yolov8n_cls/config.yaml`).  
> **Important Note:** The original Apple M3 `.pt` checkpoint files were not committed to the repository (per project guidelines prohibiting large binary artifacts) and were therefore **not available and NOT verified**. Only the training, checkpoint saving, reload, and prediction code path was verified using the newly trained reproduction checkpoints.

---

## 1. Scope of Verification

### What Was Verified
1. **Pipeline Execution Code Path:** The complete training, FP16 checkpoint saving/stripping, reload via `ultralytics.YOLO`, and held-out test prediction code path was executed across all 12 shared benchmark runs (Seeds 42, 123, 2026 × Folds 1–4).
2. **Prediction and Metric Concordance:**
   - Across all 609 held-out test predictions (203 images × 3 seeds), **604 of 609 predictions (99.18%) have identical class labels** between the original published predictions and this reproduction.
   - **8 of 12 runs** have 100% identical test labels (Seeds 42 Folds 1–3, Seeds 123 Folds 1–3, Seed 2026 Folds 2 and 4).
   - Mean absolute confidence difference across all test predictions is **0.0058** (~0.58%), and the maximum confidence difference across 11 of 12 runs is $\le 0.062$.
   - Selected best epoch is identical in **11 of 12 runs** (Seed 2026 Fold 3 selected Epoch 9 in reproduction vs. Epoch 8 originally; validation macro F1 was identical at 0.94839).
3. **Model Reload Integrity:** Each of the 12 reproduced FP16 checkpoints reloads successfully via `ultralytics.YOLO` to run downstream inference, generating evaluation predictions and Grad-CAM explanations.

### What Was Not Verified
- **Original Saved Checkpoints:** Because `.pt` files are excluded by `.gitignore` and were not committed, the original binary weights from the macOS Apple M3 training runs were **not available and could NOT be verified**. Bit-for-bit numerical reproducibility within the strict `1e-4` tolerance was therefore not verified against the original checkpoints.

---

## 2. 12-Run Benchmark Comparison Table

| Seed | Fold | Test Images | Label Agreement | Label Diff | Max Conf Diff | Mean Conf Diff | Orig Best Epoch | Repro Best Epoch | Orig Val F1 | Repro Val F1 | Orig Test F1 | Repro Test F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 1 | 51 | 51 / 51 (100%) | 0 | 0.047877 | 0.004175 | 13 | 13 | 1.00000 | 1.00000 | 0.90893 | 0.90893 |
| 42 | 2 | 51 | 51 / 51 (100%) | 0 | 0.029365 | 0.003490 | 10 | 10 | 1.00000 | 1.00000 | 0.98528 | 0.98528 |
| 42 | 3 | 51 | 51 / 51 (100%) | 0 | 0.055452 | 0.010068 | 6 | 6 | 0.97494 | 0.97494 | 0.92217 | 0.92217 |
| 42 | 4 | 50 | 49 / 50 (98%) | 1 | 0.018945 | 0.004057 | 4 | 4 | 0.94941 | 0.94941 | 0.96846 | 0.98485 |
| 123 | 1 | 51 | 51 / 51 (100%) | 0 | 0.039710 | 0.006219 | 10 | 10 | 0.97494 | 0.97494 | 0.85488 | 0.85488 |
| 123 | 2 | 51 | 51 / 51 (100%) | 0 | 0.023471 | 0.002451 | 10 | 10 | 0.97494 | 0.97494 | 0.98360 | 0.98360 |
| 123 | 3 | 51 | 51 / 51 (100%) | 0 | 0.046635 | 0.005195 | 12 | 12 | 0.92335 | 0.94722 | 0.96910 | 0.96910 |
| 123 | 4 | 50 | 49 / 50 (98%) | 1 | 0.036281 | 0.002989 | 12 | 12 | 0.97339 | 0.97339 | 0.98380 | 0.96744 |
| 2026 | 1 | 51 | 49 / 51 (96%) | 2 | 0.061679 | 0.006154 | 5 | 5 | 1.00000 | 1.00000 | 0.90934 | 0.87971 |
| 2026 | 2 | 51 | 51 / 51 (100%) | 0 | 0.023344 | 0.003299 | 13 | 13 | 0.92458 | 0.92458 | 0.98360 | 0.98360 |
| 2026 | 3 | 51 | 50 / 51 (98%) | 1 | 0.163835 | 0.026622 | 8 | 9 | 0.94839 | 0.94839 | 0.85592 | 0.87063 |
| 2026 | 4 | 50 | 50 / 50 (100%) | 0 | 0.024470 | 0.002227 | 5 | 5 | 0.97339 | 0.97339 | 0.95141 | 0.95141 |
| **Total** | | **609** | **604 / 609 (99.18%)** | **5** | **0.163835** | **0.005786** | | | | | | |

---

## 3. Investigation of Seed 123 Fold 3 Discrepancy

In the original benchmark (`results/member_4/error_analysis.md` and `runs.csv`), Seed 123 Fold 3 reported a discrepancy between the logged validation macro F1 during training (`0.92335`) and the recomputed value after saving the checkpoint (`0.94722`, a gap of `+0.02387`). The original report hypothesized that FP16 weight truncation flipped a borderline validation sample near 0.499/0.499.

### Empirical Findings on Reproduced Checkpoints:
- **Logged Validation Macro F1:** `0.94722`
- **Recomputed Checkpoint Validation Macro F1 (FP16):** `0.94722`
- **Recomputed Raw State Dict (FP32):** `0.94722`
- **Gap:** **`0.00000` (The gap did NOT reproduce)**

### Assessment:
The `0.923` vs `0.947` gap did **not** reproduce in this run. Consequently:
- The true root cause of the original gap remains **unconfirmed**.
- The FP16 numerical truncation / tie-breaking hypothesis put forward in the original report remains an unverified hypothesis, not an established finding.
- In the reproduction, all 31 validation images receive identical predictions across FP32 and FP16 evaluations, with no borderline predictions near 0.50 (minimum top-1 confidence margin is 0.09285 on `p27.png`, with all other margins exceeding 0.26).
