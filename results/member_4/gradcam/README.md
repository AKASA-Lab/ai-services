# Grad-CAM Explainability & Shortcut Learning Analysis (Person 4 — YOLOv8n-cls)

> **Checkpoint Source:** **Reproduction** (Trained on Windows x86_64 CPU with PyTorch 2.14.1+cpu and Ultralytics 8.4.212 using unchanged `models/yolov8n_cls/config.yaml`).  
> **Evaluation Protocol:** Each held-out test image is explained strictly using the fold model that never saw it during training or validation (zero test leakage).

---

## 1. Visualizations Overview

### Figure 1: `gradcam_examples.png`
A 4×4 grid displaying representative correctly classified examples across all four nutrient deficiency classes (`FN`, `-N`, `-P`, `-K`) evaluated on Seed 42 (4 images per class: 2 highest-confidence and 2 lower-confidence correct examples).
- **Visual finding:** Attention for healthy `FN` and nitrogen-deficient `-N` samples localizes primarily on the inner leaf blade and central venation (mean leaf-mask CAM mass of **81.8%** and **82.5%** respectively in Seed 42).
- **Diffuse attention in `-P` and `-K`:** Phosphorus and potassium deficiency examples exhibit more diffuse heatmaps with substantial activation along leaf margins, petiole bases, and outer border bands (border-band CAM mass of **43.9%** for `-P` and **34.7%** for `-K` in Seed 42), corresponding to peripheral leaf yellowing, marginal necrosis, and background contrast edges.

### Figure 2: `gradcam_fn_shortcuts.png`
A spotlight view of all 12 healthy `FN` images in the benchmark (Seed 42 holdout fold models).
- **Visual finding:** High-activation regions (cyan contour $\ge 50\%$ of peak activation) localize over green leaf lamina.
- **Corner watermark check:** Following the mandatory 960×960 watermark-safe crop (`docs/dataset_audit.md` §4), none of the 12 `FN` heatmaps exhibit attention parked in the bottom-right corner where DALL·E colour bars originally resided.

### Figure 3: `gradcam_misclassified.png`
Detailed inspection of 8 misclassified samples from the reproduced checkpoints:
1. **False-FN Images (All 5 benchmark occurrences included):**
   - `-N/n_24.png` (Seed 123 Fold 1: true `-N` $\rightarrow$ pred `FN`, conf 0.38)
   - `-N/n_25.png` (Seed 123 Fold 1: true `-N` $\rightarrow$ pred `FN`, conf 0.43)
   - `-N/n_37.png` (Seed 2026 Fold 3: true `-N` $\rightarrow$ pred `FN`, conf 0.72)
   - `-N/n_38.png` (Seed 2026 Fold 3: true `-N` $\rightarrow$ pred `FN`, conf 0.56)
   - `-N/n_39.png` (Seed 2026 Fold 3: true `-N` $\rightarrow$ pred `FN`, conf 0.86)
   - *Observation:* These false-FN predictions display broad, diffuse attention across the foliage and background borders, failing to focus on localized chlorotic symptoms that distinguish `-N` from `FN`.
2. **Deficiency Confusion Errors (`-N` vs `-K`):**
   - `-N/n_14.png` (Seed 42 Fold 1: true `-N` $\rightarrow$ pred `-K`, conf 0.80)
   - `-N/n_27.png` (Seed 42 Fold 1: true `-N` $\rightarrow$ pred `-K`, conf 0.92)
   - `-K/k_35.png` (Seed 42 Fold 3: true `-K` $\rightarrow$ pred `-N`, conf 0.44)
   - *Observation:* Confusions occur where generalized chlorosis visually overlaps with potassium leaf-edge scorching, with attention concentrated along outer leaf perimeters.

---

## 2. Quantitative Shortcut Metrics (Seed 42 Only)

The quantitative attention distribution metrics across HSV-segmented leaf tissue versus the outer 15% image frame border for Seed 42:

| True Class | Number of Images | Mean Leaf CAM Mass | Mean Border-Band CAM Mass | Leaf Attention Density | Border Attention Density |
|---|---|---|---|---|---|
| **FN** | 12 | **81.77%** | 28.23% | 1.834 | 0.609 |
| **-N** | 57 | **82.49%** | 26.13% | 2.115 | 0.563 |
| **-P** | 66 | 51.68% | **43.86%** | 1.092 | 0.945 |
| **-K** | 68 | 52.64% | 34.66% | 1.378 | 0.747 |
| **Non-FN Combined** | 191 | 61.21% | 35.29% | 1.499 | 0.760 |

### By Size Group (Seed 42 Only):
- **1024×1024 images (82 images):** Mean leaf mass = **54.36%**, mean border mass = **38.34%** (heavily weighted by the 48 `-P` images in this group).
- **Other sizes (121 images):** Mean leaf mass = **67.89%**, mean border mass = **32.53%**.

### CAM Enrichment Ratio (Seed 42, Reproduced Checkpoints)

The CAM enrichment ratio measures the degree to which attention concentrates on segmented leaf tissue relative to its spatial footprint:
$$\text{CAM Enrichment Ratio} = \frac{\text{CAM mass on leaf mask}}{\text{leaf-mask pixel fraction}}$$

| True Class | Number of Images | Leaf Mask Pixel Fraction | Mean Leaf CAM Mass | CAM Enrichment Ratio |
|---|---|---|---|---|
| **FN** | 12 | 60.61% | 81.77% | **1.3493** |
| **-N** | 57 | 57.76% | 82.49% | **1.4280** |
| **-P** | 66 | 47.89% | 51.68% | **1.0792** |
| **-K** | 68 | 41.72% | 52.64% | **1.2617** |
| **Non-FN Combined** | 191 | 48.64% | 61.21% | **1.2585** |

An enrichment ratio around 1.0 indicates that model attention is distributed across the image in direct proportion to area (i.e. no selective focus on plant tissue), whereas a ratio above 1.0 signifies that attention is preferentially concentrated inside the leaf mask. Here, healthy `FN` (1.35) and nitrogen-deficient `-N` (1.43) exhibit clear concentration on leaf foliage, while phosphorus-deficient `-P` (1.08) shows attention barely exceeding area proportionality. Importantly, the standard HSV leaf mask is crude: it relies on fixed colour thresholds that inadvertently segment out chlorotic, necrotic, or purplish leaf margins into the background category, which depresses the measured leaf area and CAM mass for `-P` and `-K`. Detailed values are saved in `results/member_4/gradcam/cam_enrichment.csv`.

---

## 3. Experimental Controls

### Neutral-Background Ablation Control & Class Distribution
Non-leaf pixels were replaced with neutral gray (RGB 128, 128, 128) using HSV segmentation, and images were re-evaluated with the unchanged fold models (Seed 42):

- **Accuracy Drop:** Overall test accuracy drops from **93.60% (190/203)** down to **72.91% (148/203)** under the standard mask.
- **Class Distribution Shifts & Confusion Matrix (Standard Mask):**
  - **-K into -N:** 24 out of 68 true `-K` images flip to `-N` when non-leaf pixels are grayed out.
  - **-P into -K and -N:** 15 true `-P` flip to `-K` and 8 flip to `-N` (only 42/66 `-P` correctly predicted).
- **FN Recall vs. Precision Limitation:**
  - While all 12 true `FN` images continue to be predicted as `FN` (FN recall = 100.00%, 12/12), **this recall alone cannot be taken as reassurance that no shortcut or leakage exists**.
  - Non-FN images are falsely pulled into `FN` after background ablation: 2 non-FN images (`-N/n_8.png` and `-P/p44.png`) flip to `FN`, reducing **FN Precision from 100.00% (12/12) to 85.71% (12/14)**.
- **Mask Segmentation Limitation:**
  - The standard green HSV mask (`hue 18–95, sat >= 25, val >= 30`) replaces substantial portions of diseased tissue with neutral gray (on average **52.11%** of pixels in `-P` and **58.28%** in `-K` are grayed out, vs. 39.40% in `FN`).
  - Testing a second, looser leaf mask (`hue 10–110, sat >= 15, val >= 20`) that captures chlorotic, necrotic, and brownish lamina replaces fewer pixels (31.36% in `-P`, 41.78% in `-K`, 14.22% in `FN`) and recovers overall accuracy to **87.19% (177/203)**. Under the looser mask, FN recall remains 100.00% (12/12), but 3 non-FN images flip to `FN` (`-N/n_8.png`, `-N/n_42.png`, `-P/p44.png`), yielding an FN precision of **80.00% (12/15)**.

### Resampling Control (Mild Resolution Jitter)
Each 1024×1024 test image was downscaled to random non-1024 acquisition dimensions (widths 1015–1024, heights 999–1008) and scaled back up to simulate the slight dimension difference found in the non-1024 group:
- **Scope:** This control tests only **mild resolution jitter / aspect-ratio interpolation artifacts**; it is **not** a test of image style, acquisition illumination, sensor response, or camera setup.
- **Outcome:** 81 of 82 predictions (98.78%) remained identical; only 1 prediction flipped (`n_25.png` from `-N` 0.644 to `-K` 0.493). FN recall remained **1.0000 (12/12)** under this mild dimension jitter.

---

## 4. Methodological Limitations

1. **Reproduced Checkpoints:** All analyses were conducted on checkpoints reproduced on Windows x86_64 CPU (PyTorch 2.14.1+cpu). While achieving 99.18% label agreement with original results, exact floating-point activations may vary slightly across hardware platforms.
2. **Coarse Spatial Feature Maps:** In YOLOv8n-cls, feature extraction hooks Module 8 (`model.model[-2]`, the final C2f bottleneck block), which outputs 256 channels at a coarse **7×7 spatial grid** for a 224×224 input. Bilinear interpolation to image resolution produces broad attention blobs rather than fine-grained pixel boundaries.
3. **Explaining Diagnostic Intent:** Grad-CAM illustrates which feature activations maximize the target class logit; it does not constitute mathematical proof of biological feature comprehension.
4. **Acquisition Confounding:** In this Kaggle dataset, all 12 healthy `FN` images have native dimensions of 1024×1024. Class identity and acquisition parameters remain inextricably confounded in the underlying data.
5. **Statistical Sample Size:** With only 12 `FN` images across 4 folds (~3 per fold), statistical conclusions regarding the healthy class are based on small sample counts.
