# Grad-CAM — EfficientNet-B0 (Arya Pramesh / Person 2)

Grad-CAM visualizes the feature regions that drive EfficientNet-B0's predictions. Every image is
explained by the checkpoint of the fold in which it was **held out**, ensuring the heatmap comes
from a network that never trained on that instance. Seed 42, all 203 images.

## Outputs
- `gradcam_examples.png`: Correct and erroneous predictions per class with spotlight highlighting.
- `gradcam_always_wrong.png`: Visualizing the misclassified instances and where attention was misdirected.
- `gradcam_class_average.png`: Class-average heatmaps showing focused biological leaf attention.
- `gradcam_stats.csv`: Per-image attention distribution (plant coverage vs background vs corner).
- `gradcam_summary.json`: Summary statistics across classes and subgroups.

## Key Findings
- **Plant Attention:** On average, **87.3%** of the model's Grad-CAM attention focuses directly on plant pixels.
- **Old Watermark Corner:** The bottom-right corner (where DALL·E colour bars originally sat before safe cropping) receives only **1.66%** of the activation mass (far below the 4% expected by random chance).
- **Class-Specific Focus:**
  - `FN`: Broad, symmetric focus on the compact green crown.
  - `-N`: Localized attention on yellowing interveinal and older leaf zones.
  - `-P`: Attention concentrated along purple-red leaf edges.
  - `-K`: Attention focused on marginal scorching and necrotic leaf tips.
