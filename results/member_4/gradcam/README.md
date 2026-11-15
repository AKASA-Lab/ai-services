# Grad-CAM Explainability & Shortcut Learning Analysis (Person 4 — YOLOv8n-cls)

> **Checkpoint Source:** **Reproduction** (Trained on Windows x86_64 CPU with PyTorch 2.14.1+cpu and Ultralytics 8.4.212 using unchanged `models/yolov8n_cls/config.yaml`).  
> **Evaluation Protocol:** Each held-out test image is explained strictly using the fold model that never saw it during training or validation (zero test leakage).

---

## 1. Visualizations Overview

### Figure 1: `gradcam_examples.png`
A 4×4 grid displaying representative correctly classified and misclassified examples across all four nutrient deficiency classes (`FN`, `-N`, `-P`, `-K`) evaluated on Seed 42.
- **Visual finding:** Attention for healthy `FN` and nitrogen-deficient `-N` samples strongly localizes on the inner leaf blade and central venation (mean leaf-mask CAM mass of **81.8%** and **82.5%** respectively).
- **Diffuse attention in `-P` and `-K`:** Phosphorus and potassium deficiency examples exhibit more diffuse heatmaps with substantial activation along leaf margins, petiole bases, and outer border bands (border-band CAM mass of **43.9%** for `-P` and **34.7%** for `-K`), corresponding to peripheral leaf yellowing, marginal necrosis, and background contrast edges.

### Figure 2: `gradcam_fn_shortcuts.png`
A spotlight view of all 12 healthy `FN` images in the benchmark (Seed 42 holdout fold models).
- **Visual finding:** High-activation regions (cyan contour $\ge 50\%$ of peak activation) localize squarely over dense green foliage.
- **Corner watermark check:** Following the mandatory 960×960 watermark-safe crop (`docs/dataset_audit.md` §4), none of the 12 `FN` heatmaps exhibit attention parked in the bottom-right corner where DALL·E colour bars originally resided.

### Figure 3: `gradcam_misclassified.png`
Detailed inspection of 8 misclassified samples from the reproduced checkpoints, focusing on:
1. **False-FN Images:** All 5 images misclassified as `FN` in the benchmark:
   - `-N/n_24.png` (Seed 123 Fold 1: true `-N` $\rightarrow$ pred `FN`, conf 0.38)
   - `-N/n_25.png` (Seed 123 Fold 1: true `-N` $\rightarrow$ pred `FN`, conf 0.43)
   - `-N/n_37.png` (Seed 2026 Fold 3: true `-N` $\rightarrow$ pred `FN`, conf 0.72)
   - `-N/n_38.png` (Seed 2026 Fold 3: true `-N` $\rightarrow$ pred `FN`, conf 0.56)
   - `-N/n_39.png` (Seed 2026 Fold 3: true `-N` $\rightarrow$ pred `FN`, conf 0.86)
   - *Observation:* These false-FN predictions display diffuse attention spread across both leaf tissue and background boundaries, failing to identify localized chlorotic patches that would distinguish `-N` from `FN`.
2. **Deficiency Confusion Errors (`-N` vs `-K`):**
   - `-N/n_14.png` (Seed 42 Fold 1: true `-N` $\rightarrow$ pred `-K`, conf 0.80)
   - `-N/n_27.png` (Seed 42 Fold 1: true `-N` $\rightarrow$ pred `-K`, conf 0.92)
   - `-K/k_35.png` (Seed 42 Fold 3: true `-K` $\rightarrow$ pred `-N`, conf 0.44)
   - *Observation:* Misclassification occurs where generalized leaf paleing resembles potassium marginal scorch, with heatmaps focusing heavily on leaf tips and perimeter edges.

---

## 2. Quantitative Shortcut Metrics (Seed 42)

The quantitative metrics evaluate attention distribution across HSV-segmented leaf tissue versus the outer 15% image frame border:

| True Class | Number of Images | Mean Leaf CAM Mass | Mean Border-Band CAM Mass | Leaf Attention Density | Border Attention Density |
|---|---|---|---|---|---|
| **FN** | 12 | **81.77%** | 28.23% | 1.834 | 0.609 |
| **-N** | 57 | **82.49%** | 26.13% | 2.115 | 0.563 |
| **-P** | 66 | 51.68% | **43.86%** | 1.092 | 0.945 |
| **-K** | 68 | 52.64% | 34.66% | 1.378 | 0.747 |
| **Non-FN Combined** | 191 | 61.21% | 35.29% | 1.499 | 0.760 |

### By Size Group:
- **1024×1024 images (82 images):** Mean leaf mass = **54.36%**, mean border mass = **38.34%** (heavily weighted by the 48 `-P` images in this group).
- **Other sizes (121 images):** Mean leaf mass = **67.89%**, mean border mass = **32.53%**.

---

## 3. Experimental Controls

### Neutral-Background Ablation Control
Non-leaf pixels were replaced with neutral gray (RGB 128, 128, 128) using HSV segmentation, and images were re-evaluated with the unchanged fold models:
- **Overall Accuracy:** Dropped from **93.60%** (190/203) to **72.91%** (148/203). This confirms that background contrast and boundary context contribute to general classification confidence.
- **FN Recall:** Remained at **100.00% (12/12)** under neutral gray background ablation. Silencing background cues did not impair healthy leaf detection.

### Resampling Control (Simulating non-1024 Acquisition)
Each 1024×1024 test image was downscaled to random non-1024 acquisition dimensions (widths 1015–1024, heights 999–1008) and scaled back up to simulate non-1024 resampling artifacts:
- **Prediction Stability:** 81 of 82 predictions (98.78%) remained identical. Only 1 image flipped (`n_25.png` from `-N` 0.644 to `-K` 0.493).
- **FN Recall:** Remained **1.0000 (12/12)** under resampling simulation.

---

## 4. Methodological Limitations

1. **Reproduced Checkpoints:** All analyses were conducted on checkpoints reproduced on Windows x86_64 CPU (PyTorch 2.14.1+cpu). While achieving 99.18% label agreement with original results, exact floating-point activations may vary slightly across hardware platforms.
2. **Coarse Spatial Feature Maps:** In YOLOv8n-cls, feature extraction hooks Module 8 (`model.model[-2]`, the final C2f bottleneck block), which outputs 256 channels at a coarse **7×7 spatial grid** for a 224×224 input. Bilinear interpolation to image resolution produces broad attention blobs rather than fine-grained pixel boundaries.
3. **Explaining Diagnostic Intent:** Grad-CAM illustrates which feature activations maximize the target class logit; it does not constitute mathematical proof of biological feature comprehension.
4. **Acquisition Confounding:** In this Kaggle dataset, all 12 healthy `FN` images have native dimensions of 1024×1024. Class identity and acquisition parameters remain inextricably confounded in the underlying data.
5. **Statistical Sample Size:** With only 12 `FN` images across 4 folds (~3 per fold), statistical conclusions regarding the healthy class are based on small sample counts.
