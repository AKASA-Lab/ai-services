# MobileNetV2 — Person 1 (Karthik)

Owner: Person 1 (also dataset audit, shared split generation, and integration coordination).

Pretrained MobileNetV2 (width 1.0, ImageNet-1k weights from timm, `mobilenetv2_100.ra_in1k`)
with a new 4-class head, trained on every shared split file (3 seeds × 4 folds = 12 runs).
The results are in [`results/member_1/`](../../results/member_1/).

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Every setting used (data, augmentation, optimiser, selection metric) |
| `train.py` | Trains/evaluates the runs, writes predictions, per-run history and metadata |
| `summarize.py` | Recomputes metrics from the saved predictions with the shared evaluator and builds `results/member_1/` |
| `gradcam.py` | Grad-CAM heatmaps and attention statistics → `results/member_1/gradcam/` |

## Pipeline

1. **Preprocessing (every image):** `watermark_safe_crop` (fixed 960×960, top-aligned, centred) →
   resize to 256×256 (bicubic) once and cache. This removes the DALL·E colour bar described in
   `docs/dataset_audit.md` §4.
2. **Training augmentation only:** rotation ±10° (corners filled with the image's own border
   colour) → `RandomResizedCrop(224, scale 0.75–1.0, ratio 0.9–1.1)` → horizontal flip. No colour
   jitter, because leaf colour is the signal. Validation/test images are only resized to 224.
3. **Model:** timm `mobilenetv2_100`, dropout 0.2, head initialised N(0, 0.01).
4. **Training:** batch 16, 15 epochs, AdamW (backbone lr 1e-4, head lr 1e-3, weight decay 1e-4).
   Epochs 1–3 train the head only (backbone weights and BatchNorm statistics frozen). Epochs 4–15
   fine-tune everything. Loss is class-weighted cross-entropy, with weights computed from that
   run's train partition.
5. **Model selection:** best validation macro F1 (ties go to the lower validation loss). The test
   fold is predicted once, with that checkpoint.
6. **Evaluation:** `summarize.py` → `lettuce_benchmark.evaluation` (out-of-fold metrics per seed,
   then mean ± sample std over the 3 seeds).

## Reproduce

```bash
pip install -r requirements.txt timm
# data/raw/{FN,-N,-P,-K}/... must exist (see data/README.md)
python -m models.mobilenet_v2.train --output-dir outputs/mobilenet_v2
python -m models.mobilenet_v2.summarize --runs-dir outputs/mobilenet_v2 --output-dir results/member_1
```

On a machine that can't reach Hugging Face, download the same weights from
`https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_100_ra-b33bc2c4.pth`
(SHA-256 `b33bc2c4319407fabd894326c3c758959c2c21fbc1cbac888cad6aa9f017cf2a`) and add
`--weights path/to/mobilenetv2_100_ra-b33bc2c4.pth`. The reported results were produced this way, on
CPU (2 threads), so training time and latency are CPU numbers. A Colab GPU will be much faster.
Checkpoints are written to `outputs/mobilenet_v2/checkpoints/` and are not committed (per project
rules).

## Deviations from the README starting configuration

- Weights come from timm (`mobilenetv2_100.ra_in1k`) instead of torchvision; it is the same
  MobileNetV2-1.0 architecture with ImageNet-1k weights.
- The fixed 960×960 crop is added before resizing (dataset-specific shortcut removal).
- The classifier head uses the torchvision-style N(0, 0.01) initialisation. timm's default head
  init produced very large starting logits for a 4-class head.
