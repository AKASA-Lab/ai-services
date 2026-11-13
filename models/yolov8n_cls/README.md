# YOLOv8n-cls — Person 4 (Abhinav)

Owner: Person 4. Ultralytics YOLOv8n **classification** model (`yolov8n-cls.pt`, ImageNet-1k
pretrained) with a new 4-class output layer, fine-tuned on every shared split file
(3 seeds × 4 folds = 12 runs). The results are in [`results/member_4/`](../../results/member_4/).

> This is a classification baseline built on the stock Ultralytics `yolov8n-cls` model. It is not object
> detection, and it is not a reproduction of any custom YOLO-NPK research architecture.

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Every setting used. The `training:` block is passed to `YOLO.train()` unchanged |
| `prepare_data.py` | Preprocessing cache + per-run Ultralytics folder tree, with a membership check against the split CSV |
| `train.py` | Trains/evaluates the runs, selects the epoch, writes predictions, per-run history and metadata |
| `summarize.py` | Recomputes metrics from the saved predictions with the shared evaluator and builds `results/member_4/` |

## Pipeline

1. **Preprocessing (every image):** `watermark_safe_crop` (fixed 960×960, top-aligned, centred) →
   resize to 256×256 (bicubic) once and save as PNG. This removes the DALL·E colour bar described in
   `docs/dataset_audit.md` §4. It is the same preprocessing as MobileNetV2 (Person 1).
2. **Ultralytics dataset per run:** `outputs/yolov8n_cls/datasets/seed_<s>_fold_<f>/{train,val}/<class>/<image_id>.png`
   (symlinks into the cache). These are written from the run's split CSV and re-checked against it
   file by file: same images, same partition, same label. The **test fold is not in this folder**,
   so the Ultralytics trainer can't fall back to it.
3. **Training-only augmentation:** `RandomResizedCrop(224, scale 0.75–1.0)` + horizontal flip.
   HSV jitter, RandAugment and random erasing are all switched off, because leaf colour is the
   signal. Validation/test images are only resized to 224.
4. **Training (Ultralytics trainer):** batch 16, 15 epochs, AdamW lr 1e-3 (1 warm-up epoch, linear
   decay to 1e-5), weight decay 1e-4, every layer trained. `nbs=16`, so there is one optimiser step
   per batch. Loss is class-weighted cross-entropy (`cls_pw=1.0`, i.e. inverse frequency of that
   run's **train** partition).
5. **Model selection:** a small `ClassificationTrainer` subclass also computes **validation macro F1**
   every epoch and uses it as Ultralytics' fitness. A checkpoint is saved every epoch.
   `train.py` picks the best validation macro F1. Ties go to the lower validation loss, then to the
   earlier epoch, which is the same rule as MobileNetV2. As a check, the selected checkpoint is run
   through the Ultralytics predictor on the validation images, and its macro F1 is recorded next to
   the logged value (`val_macro_f1_recomputed` in `runs.csv`). They match in 11 of 12 runs. In the
   12th, the FP16 checkpoint flips a validation image that sits on a 0.499/0.499 tie (see
   `results/member_4/error_analysis.md`).
6. **Test:** the held-out fold is predicted once, with that checkpoint, via `YOLO.predict`. The
   probability columns are mapped to `["FN", "-N", "-P", "-K"]` **by name**, because Ultralytics
   numbers classes by sorted folder name (`-K, -N, -P, FN`).
7. **Evaluation:** `summarize.py` → `lettuce_benchmark.evaluation` (out-of-fold metrics per seed,
   then mean ± sample std over the 3 seeds).

## Reproduce

```bash
pip install -r requirements.txt
# data/raw/{FN,-N,-P,-K}/... must exist (see data/README.md)
python -m models.yolov8n_cls.train --output-dir outputs/yolov8n_cls          # all 12 runs
python -m models.yolov8n_cls.summarize --runs-dir outputs/yolov8n_cls --output-dir results/member_4
```

`yolov8n-cls.pt` is downloaded once into `outputs/yolov8n_cls/pretrained/` (or pass
`--weights path/to/yolov8n-cls.pt`). The reported results were produced on CPU (Apple M3) with
`--device cpu`. On Colab add `--device 0`. A quick smoke test is
`--seeds 42 --folds 1 --epochs 2`. Checkpoints go to `outputs/yolov8n_cls/checkpoints/` and are not
committed (per project rules). Ultralytics usage analytics are switched off inside `train.py`.

## Deviations from Ultralytics defaults (all deliberate)

| Setting | Ultralytics default | Used | Why |
|---|---|---|---|
| `nbs` | 64 | 16 | The default accumulates 4 batches, which would give only 2 optimiser steps per epoch on ~121 images |
| `optimizer`, `lr0` | auto (AdamW, 0.00125 for 4 classes) | AdamW, 1e-3 | Fixed explicitly so the config is self-documenting; this matches the README classifier LR |
| `warmup_epochs`, `warmup_bias_lr` | 3, 0.1 | 1, 0.0 | 3 epochs is 20% of the budget, and bias LR 0.1 is an SGD setting |
| `weight_decay` | 5e-4 | 1e-4 | README shared value |
| `cls_pw` | 0 (unweighted) | 1.0 | Class-weighted CE from the train partition, as for the other CNNs |
| `hsv_h/s/v`, `auto_augment`, `erasing` | 0.015/0.7/0.4, randaugment, 0.4 | all off | README: avoid colour changes, since leaf colour is the signal |
| `scale` | 0.5 (RRC area 0.5–1) | 0.25 (0.75–1) | Mild crop, same range as MobileNetV2 |
| best checkpoint | top-1 accuracy | validation macro F1 (tie → loss) | README primary metric |

## Differences from the shared CNN recipe (README §5.1)

- **No two-stage freeze, single learning rate.** Every layer is trained from the first epoch at
  1e-3, which is Ultralytics' standard fine-tuning. MobileNetV2 used a frozen head-only phase and
  separate backbone/head rates.
- **No rotation augmentation.** The Ultralytics classification pipeline has no rotation. The
  random-crop aspect-ratio range is Ultralytics' fixed 3/4–4/3 (MobileNetV2: 0.9–1.1).
- **Input scaling** is Ultralytics' 0–1 scaling with no ImageNet mean/std normalisation, as the
  pretrained weights expect.
- **EMA weights.** Ultralytics validates and saves an exponential moving average of the weights.
  Its decay ramps up over ~2000 updates, so with ~120 updates per run it stays close to the raw
  weights.
- **Checkpoint format.** Ultralytics stores the weights in FP16, so the file is about half the size
  of an FP32 state dict. Both sizes are reported in `efficiency.json`.
