# Grad-CAM Explainability — ResNet50 (Person 3)

Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes the spatial regions in each input leaf image that drove ResNet50's classification decision.

Every image in the dataset is explained by the model checkpoint of the fold in which that image was **held out as test data**. This ensures that the heatmaps reflect true generalisation from a model that never trained on the image.

## Reproducing the Visualizations

Run the following command on the machine with the trained checkpoints (`outputs/resnet50/checkpoints/`):

```bash
python -m models.resnet50.gradcam --runs-dir outputs/resnet50 --output-dir results/member_3/gradcam --seed 42 --device cuda
```

## Generated Figures & Deliverables

| File | Description |
|---|---|
| `gradcam_examples.png` | Top-confidence correct predictions and notable errors across all four classes (`FN`, `-N`, `-P`, `-K`). |
| `gradcam_confusions.png` | Targeted visual breakdown of `-N` → `-K` and `-P` → `-K` misclassifications compared to true `-K` predictions. |
| `gradcam_healthy_fn.png` | Complete panel covering all 12 healthy `FN` leaves in the dataset. |
| `gradcam_always_wrong.png` | Side-by-side comparison (original preprocessed leaf vs. Grad-CAM heatmap) for images misclassified across all 3 seeds. |
| `gradcam_class_average.png` | Class-averaged attention maps testing for spatial bias (e.g. the pre-crop DALL·E watermark corner). |
| `gradcam_stats.csv` | Per-image quantitative attention metrics (% attention on plant vs. background, corner leakage). |
| `gradcam_summary.json` | Aggregate explainability summary metrics. |

## Key Findings

### 1. Plant Foliage Focus vs Background Leakage
- **85.0%** of total Grad-CAM gradient mass was focused directly on plant foliage pixels (surpassing MobileNetV2's 82.0%).
- **97.5% of images** (198 of 203) placed the majority of their attention on the plant.
- The bottom-right $20\% \times 20\%$ corner (4% of total frame area, where the DALL·E watermark sat prior to cropping) only received **3.1%** of attention mass.
- Images that originated with the watermark received virtually identical corner attention (**3.29%**) to those without (**3.01%**), confirming that ResNet50 does not exploit pre-crop shortcut locations.

### 2. What Each Class Is Recognized By
- **FN (Healthy):** Attention spans the compact, vibrant green leaf rosette uniformly.
- **-N (Nitrogen deficiency):** Locks onto pale, chlorotic leaf centers and diffuse yellowing.
- **-P (Phosphorus deficiency):** Strongly activates on leaf margins displaying dull or reddish-purplish discoloration (highest plant attention at **93.0%**).
- **-K (Potassium deficiency):** Attends to outer leaf boundaries exhibiting marginal necrosis and tip burn (**84.6%** plant attention).

### 3. Biological Confusion Analysis (`-N` vs `-K` and `-P` vs `-K`)
- In misclassified predictions, average attention on the plant was **89.1%**, demonstrating that errors are driven by genuine visual overlap in leaf symptoms rather than background artifacts.
- When `-N` leaves were misclassified as `-K`, Grad-CAM reveals that the network focused on dying, browned lower outer leaves that visually mimic potassium marginal scorch rather than central leaf pale chlorosis.
- When `-P` leaves were misclassified as `-K`, the network focused on dry, dark necrotic edges on outer leaves, mistaking phosphorus margin necrosis for potassium deficiency burn.

## Visual Encoding

- **Bright regions:** Foliage and symptom areas that actively stimulated ResNet50's layer4 feature maps toward the predicted class.
- **Dimmed regions:** Background or leaf portions ignored by the model.
- **Cyan contour:** Outlines the core focus area (activations $\ge 50\%$ of peak response).
- **Red labels (`✗`):** Indicate incorrect predictions, showing what visual features tricked the model into the wrong diagnosis.
- **Dashed box on class averages:** The bottom-right $20\% \times 20\%$ corner where the DALL·E color bar sat prior to applying the 960px safe crop.
