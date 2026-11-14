# EfficientNet-B0 — Arya Pramesh (Member 2)

Owner: Arya Pramesh (Member 2).

Pretrained EfficientNet-B0 (ImageNet-1k weights from timm, `efficientnet_b0.ra_in1k`)
with a 4-class classification head, trained on every shared split file (3 seeds × 4 folds = 12 runs).
The results, Grad-CAM interpretability, and evaluation deliverables are in [`results/member_2/`](../../results/member_2/).

## Key Results (Out-of-Fold across 3 Seeds)

| Metric | EfficientNet-B0 (Member 2) | MobileNetV2 (Member 1) |
|---|---|---|
| **Macro F1 (primary)** | **0.928 ± 0.026** | 0.879 ± 0.067 |
| Accuracy | **0.926 ± 0.018** | 0.888 ± 0.049 |
| Balanced Accuracy | **0.936 ± 0.024** | 0.899 ± 0.058 |
| FN Recall (Healthy) | **0.972 ± 0.048** | 0.944 |
| FN Precision (Healthy) | **0.901 ± 0.086** | 0.757 |

## Files

| File | Purpose |
|---|---|
| `config.yaml` | All hyperparameter settings (architecture, safe crop, augmentations, optimizer, loss, selection metric) |
| `train.py` | Executes training across all 12 runs, checkpoints models, writes predictions and run metadata |
| `summarize.py` | Recomputes metrics from saved predictions with the shared evaluator and creates `results/member_2/` |
| `gradcam.py` | Computes Grad-CAM activations on held-out checkpoints and exports spotlight visualizations |
| `README.md` | Approach documentation and reproducibility guide |

## Pipeline

1. **Preprocessing (every image):** `watermark_safe_crop` (fixed 960×960 top-aligned, centered) →
   resize to 256×256 (bicubic) once and cache. This removes the DALL·E color bar described in
   `docs/dataset_audit.md` §4.
2. **Training augmentation only:** rotation ±10° (corners filled with the image's own border
   color via `BorderFillRotation`) → `RandomResizedCrop(224, scale 0.75–1.0, ratio 0.9–1.1)` → horizontal flip.
   No color jitter, because leaf color is the primary deficiency signal. Validation and test images are only resized to 224.
3. **Model:** timm `efficientnet_b0` (`ra_in1k`), dropout 0.2, linear classifier head (1280 → 4) initialized N(0, 0.01) with zero bias.
4. **Training:** batch size 16, 15 epochs max, AdamW optimizer (backbone lr 1e-4, classifier lr 1e-3, weight decay 1e-4).
   Epochs 1–3 train the classifier head only (backbone parameters and BatchNorm statistics frozen).
   Epochs 4–15 fine-tune all parameters. Loss is class-weighted cross-entropy with weights computed strictly from that run's training partition.
5. **Model selection:** best validation macro F1 (ties broken by lower validation loss). The held-out test fold is predicted once with that checkpoint.
6. **Evaluation:** `summarize.py` → `lettuce_benchmark.evaluation` (out-of-fold metrics per seed, then mean ± sample std over the 3 seeds).
7. **Explainability:** `gradcam.py` → calculates gradient-weighted class activation mapping for held-out images.

## Reproduce

```bash
pip install -r requirements.txt timm
# data/raw/{FN,-N,-P,-K}/... must exist (see data/README.md)
python -m models.efficientnet_b0.train --output-dir outputs/efficientnet_b0
python -m models.efficientnet_b0.summarize --runs-dir outputs/efficientnet_b0 --output-dir results/member_2
python -m models.efficientnet_b0.gradcam --runs-dir outputs/efficientnet_b0 --seed 42 --output-dir results/member_2/gradcam
```
