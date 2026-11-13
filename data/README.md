# Shared data files (Person 1)

These are the **official** shared split files. Do not regenerate them for your own model.

```text
data/
├── raw/                                  # Kaggle images. NOT committed; add them yourself (below)
├── processed/
│   ├── metadata.csv                      # 203 images used in the benchmark
│   ├── excluded_exact_duplicates.csv     # 5 byte-identical copies left out (see audit)
│   ├── audit.csv                         # every file: size, SHA-256, duplicate and colour-bar flags
│   └── audit_summary.json
└── splits/seed_{42,123,2026}/fold_{1..4}.csv   # 12 runs: train / validation / test
```

## 1. Put the images in place

Download the Kaggle dataset (`baronn/lettuce-npk-dataset`, about 354 MB) and extract it so that the
four class folders sit **directly** inside `data/raw/`:

```text
data/raw/FN/fn1.png
data/raw/-N/n_1.png
data/raw/-P/p1.png
data/raw/-K/k_1.png
```

The archive unpacks into a folder called `FNNPK/`. Move its four class folders up into `data/raw/`,
or point your code's image root at `data/raw/FNNPK`. `image_path` in every CSV is relative to the
folder that contains `FN/ -N/ -P/ -K/`.

## 2. Use a split

Each `fold_k.csv` has one row per image with columns
`image_id,image_path,label,partition,seed,fold`, where `partition` is `train`, `validation` or `test`.

```python
import pandas as pd
split = pd.read_csv("data/splits/seed_42/fold_1.csv")
train = split[split.partition == "train"]
val   = split[split.partition == "validation"]
test  = split[split.partition == "test"]   # touch only for the final evaluation of the run
```

Train on `train`, select checkpoints/hyperparameters on `validation`, and predict `test` once.
Run all 12 files (3 seeds × 4 folds). Class order is always `["FN", "-N", "-P", "-K"]`.

## 3. Read the audit before training

`docs/dataset_audit.md` §4: every 1024×1024 image has a DALL·E colour bar in the corner, and
which images have it depends strongly on the class (all FN images have it). Apply
`lettuce_benchmark.preprocessing.watermark_safe_crop` (a fixed 960×960 crop) to every image
before resizing, so your model can't learn the bar instead of the leaf.
