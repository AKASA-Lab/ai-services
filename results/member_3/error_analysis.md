# ResNet50 — Error Analysis

## Overall Performance

- **Macro F1:** 0.863 ± 0.045
- **Accuracy:** 0.870 ± 0.041
- **Balanced Accuracy:** 0.866 ± 0.044

## Per-Class Recall

| Class | Recall (mean ± std) |
|---|---|
| `FN` | 0.861 ± 0.127 |
| `-N` | 0.819 ± 0.071 |
| `-P` | 0.869 ± 0.070 |
| `-K` | 0.917 ± 0.008 |

## Misclassification Patterns

- **Always misclassified** (wrong in all seeds): 10 images
- **Sometimes misclassified** (wrong in some seeds): 37 images
- **Always correct**: 156 images

## Most Common Confusions

| True | Predicted As | Count |
|---|---|---:|
| `-N` | `-K` | 23 |
| `-P` | `-K` | 21 |
| `-K` | `-N` | 9 |
| `-K` | `-P` | 7 |
| `-N` | `FN` | 6 |
| `-P` | `-N` | 5 |
| `FN` | `-N` | 5 |
| `-N` | `-P` | 2 |
| `-K` | `FN` | 1 |

## Limitations

- The `FN` (healthy) class has only ~12 images — per-class metrics for this class are inherently noisy.
- Image-level splitting does not guarantee plant-independent generalisation.
- Results are from the lettuce benchmark dataset and do not transfer to fenugreek.