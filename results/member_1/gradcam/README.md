# Grad-CAM — MobileNetV2 (Person 1)

Grad-CAM shows which parts of an image drove the model's prediction. Every image here is
explained by the checkpoint of the fold in which it was **held out**, so the heatmap comes from a
model that never trained on that image. Seed 42; all 203 images. The target class is the
predicted class. The script checks that each checkpoint reproduces the saved prediction exactly.

Reproduce: `python -m models.mobilenet_v2.gradcam --runs-dir outputs/mobilenet_v2 --seed 42`
(needs the training checkpoints, which are not committed).

| File | What it shows |
|---|---|
| `gradcam_examples.png` | Per class: the most confident correct predictions and up to two errors |
| `gradcam_always_wrong.png` | The 6 images misclassified in all 3 seeds: original + heatmap |
| `gradcam_class_average.png` | Mean heatmap per class, to check whether attention sits in one fixed spot |
| `gradcam_stats.csv` / `gradcam_summary.json` | Per-image and aggregated attention numbers |

How to read the figures: bright = regions that drove the prediction, dimmed = regions the model
ignored, cyan outline = strongest region (≥ 50% of the peak).

## Findings

**The model mostly looks at the plant, not the background or the corner.**

| Check | Result |
|---|---|
| Share of Grad-CAM attention on plant pixels (plant covers 69% of the frame on average) | **82%** |
| Images with more than half their attention on the plant | **198 of 203** |
| Attention in the bottom-right corner (4% of the frame, where the DALL·E bar was) | **1.8%** (less than half of what evenly spread attention would put there) |
| Same corner: images that had the bar / did not | 1.4% / 2.0% |

The class averages (`gradcam_class_average.png`) show attention centred on the plant, with a cold
bottom-right corner for every class. After the 960×960 crop there is no sign that the model uses
where the colour bar used to be.

**What each class is recognised by** (`gradcam_examples.png`):
- **FN:** the whole compact, evenly green head.
- **-N:** the yellow (chlorotic) leaves.
- **-P:** the purple/red-tinged leaf blades.
- **-K:** lower and outer leaves with brown or scorched edges.

These match the textbook symptoms for each deficiency. But Grad-CAM only shows where the model
looked; it doesn't prove the model learned the biology.

**Why the errors happen** (`gradcam_always_wrong.png`):
- `-N/n_22` → -K: attention is on a yellow-brown, necrotic lower leaf, which looks like
  potassium scorch.
- `-K/k_60` → -P: attention is on the dark purple outer leaves, plus part of the background strip
  at the top.
- `-P/p44` → -N: **only 10% of the attention is on the plant**. The model looked at the black
  background, so this error comes from a background cue, not a leaf symptom. It is the clearest
  failure case and is worth showing in the report.
- `-N/n_38` → FN: attention is on the upper, evenly green leaves, which look healthy.

Only 5 images put less than half their attention on the plant (`gradcam_stats.csv`,
`cam_mass_on_plant < 0.5`). Four of them were still classified correctly.

## Limitations

- The heatmap is 7×7 upscaled to 224×224, so it marks regions (about 32-pixel patches), not
  individual spots or veins.
- "Plant pixels" come from a simple rule: colour differs clearly from the image's own border
  colour. It is approximate where leaves and background have similar colours.
- Only seed 42 is shown. The other seeds can be generated with `--seed 123` or `--seed 2026`.
