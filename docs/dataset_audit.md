# Dataset audit — Lettuce NPK (Kaggle `baronn/lettuce-npk-dataset`)

Audited by Person 1 (Karthik) on 2026-11-12. All numbers below come from the downloaded archive
(`archive.zip`, 353,828,161 bytes, extracted as `FNNPK/{FN,-N,-P,-K}`) and the repository tools.

## 1. What is in the archive

| Check | Result |
|---|---|
| Image files discovered | 208 (all PNG) |
| Valid / decodable | 208 |
| Corrupt / unreadable | 0 |
| Unlabelled (outside a class folder) | 0 |
| Class counts (raw) | FN 12, -N 58, -P 66, -K 72 (matches the published counts) |
| Exact duplicates (SHA-256) | 5 redundant copies, all same-label (see §3) |
| Near duplicates | none beyond the 5 exact copies (dHash ≤ 6 and 256-bit pHash screen; closest distinct pair is 72/256 bits apart and visually different) |
| Plant ID / session metadata | **not available** in the archive; none was invented |

Per-file details (dimensions, SHA-256, duplicate flags) are in `data/processed/audit.csv` and the
summary in `data/processed/audit_summary.json`. Use the SHA-256 column to confirm your local copy
matches.

## 2. Image sizes

87 images are exactly 1024 × 1024. The other 121 are slightly cropped (widths 1015–1024, heights
999–1008; most commonly 1023 × 1006 and 1023 × 1005).

## 3. Duplicate policy (decision)

Five files are byte-identical copies of another file in the same class:

| Excluded copy | Identical to |
|---|---|
| `-K/k_66.png` | `-K/k_63.png` |
| `-K/k_68.png` | `-K/k_58.png` |
| `-K/k_69.png` | `-K/k_57.png` |
| `-K/k_70.png` | `-K/k_56.png` |
| `-N/n_23.png` | `-N/n_17.png` |

**Decision: the redundant copies are excluded from the manifest; the first copy is kept.**
Keeping both copies would either leak an identical image into train and test, or double-count it
in the metrics. Excluded rows are listed in `data/processed/excluded_exact_duplicates.csv`.
No images were deleted or relabelled on disk.

**Manifest used for all splits: 203 images: FN 12, -N 57, -P 66, -K 68.**

## 4. ⚠️ Shortcut risk: DALL·E colour bar and image size are tied to the class

Every 1024 × 1024 image carries the five-square DALL·E colour bar in its bottom-right corner
(yellow, cyan, green, orange, blue). The cropped images do not have it. So at least these 87
images appear to be AI-generated, and the bar's presence depends heavily on the class:

| Class | Images with colour bar (= 1024 × 1024) |
|---|---|
| FN | **12 / 12 (100%)** |
| -P | 48 / 66 (73%) |
| -K | 14 / 68 (21%) |
| -N | 8 / 57 (14%) |

(Counts are over the 203-image manifest. The flag for every file is the `dalle_colour_bar_detected` column in `data/processed/audit.csv`.)

A model can score well just by spotting the bar or the square aspect ratio, without looking at
the leaf at all. This matters most for FN, the minority class.

**Recommendation for all five approaches:** before any resizing, crop every image to the same
fixed 960 × 960 window: horizontally centred, starting at the top row. This removes the bottom
strip where the bar sits, and every image ends up the same size whichever group it came from.
It is available as `lettuce_benchmark.preprocessing.watermark_safe_crop`, and MobileNetV2 (Person 1) uses it. It does not
remove every possible difference between the two groups, so check Grad-CAM/LIME maps for attention
on corners or background.

The benchmark write-up should also note that a large share of the images appear to be synthetic,
which further limits claims about real hydroponic (or aeroponic fenugreek) plants.

## 5. Shared split summary

Generated once with `python -m lettuce_benchmark.cli.generate_splits data/processed/metadata.csv
--output-dir data/splits` (seeds 42, 123, 2026; 4 stratified outer folds; stratified 80/20
train/validation inside the development folds). Validated with `validate_splits`.

| Partition (per run) | FN | -N | -P | -K | Total |
|---|---|---|---|---|---|
| train | 7 | 33–34 | 39–40 | 41 | 121–122 |
| validation | 2 | 9 | 10 | 10 | 31 |
| test | 3 | 14–15 | 16–17 | 17 | 50–51 |

The evaluation is **image-level**: no plant/session metadata exists, so plant-independent
generalisation is not established. With only 2 FN images per validation set, any class-wise
validation metric for FN is very noisy.
