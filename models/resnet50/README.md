# ResNet50 — Person 3

Pretrained ResNet50 (ImageNet-1k) with a four-class classification head for lettuce NPK deficiency classification.

## Architecture

- **Backbone:** ResNet50 (`timm resnet50.a1_in1k`, ~23.5M parameters)
- **Classifier:** Linear(2048 → 4), re-initialised with N(0, 0.01) weights and zero bias
- **Input:** 224 × 224 RGB, ImageNet normalisation

## Training strategy

1. **Epochs 1–3:** Backbone frozen (including BatchNorm), only the classifier head trains at `lr=1e-3`
2. **Epochs 4–15:** Full fine-tuning — backbone at `lr=1e-4`, classifier at `lr=1e-3`
3. **Loss:** Weighted cross-entropy (weights computed from each run's training partition)
4. **Checkpoint selection:** Best validation macro F1, ties broken by lowest validation loss

## Data handling

- Uses the shared split manifests from `data/splits/` (3 seeds × 4 folds = 12 runs)
- Applies the 960px safe crop to remove the DALL·E colour bar
- Mild training augmentation: random resized crop, horizontal flip, small rotation (no colour jitter)
- Validation and test images receive only resize + normalisation

## Usage

```bash
# Train all 12 runs
python -m models.resnet50.train --output-dir outputs/resnet50

# Single seed for testing
python -m models.resnet50.train --output-dir outputs/resnet50 --seeds 42

# Specify device
python -m models.resnet50.train --output-dir outputs/resnet50 --device cuda

# Use local weights (no Hugging Face download)
python -m models.resnet50.train --output-dir outputs/resnet50 --weights path/to/resnet50_a1_in1k.pth

# Generate deliverables
python -m models.resnet50.summarize --runs-dir outputs/resnet50 --output-dir results/member_3
```

## Outputs

Training produces:

```
outputs/resnet50/
├── predictions.csv          # All test predictions (common format)
├── runs.csv                 # Per-run summaries
├── image_cache.pt           # Preprocessed image cache
├── checkpoints/             # Best model per run
│   ├── resnet50_seed_42_fold_1.pt
│   └── ...
└── runs/
    └── seed_*/fold_*/
        ├── history.csv      # Per-epoch training curves
        └── run.json         # Full config + hardware + results
```

Summarisation produces `results/member_3/` with metrics, confusion matrices, training curves, efficiency measurements, and error analysis.
