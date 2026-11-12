# Project guidelines

## Shared rules

- This is four-class, single-label classification. Use `["FN", "-N", "-P", "-K"]` in that order.
- Do not claim a dataset count or audit result until the downloaded archive has been audited.
- Never commit raw/processed images, checkpoints, or generated large artifacts.
- Person 1 coordinates one dataset manifest and one set of split CSVs. Every approach consumes those same files.
- Split original images before creating training-only augmentation.
- Fit preprocessing, feature scaling, feature selection, and resampling from training data only.
- Validation is for model selection. Held-out test data is for final evaluation only.
- Do not invent plant IDs or acquisition groups. If groups are unavailable, call the protocol image-level and note the limitation.
- Record deviations, software versions, hardware, configuration, seed, and fold for every run.
- Do not fabricate results; publish metrics only from saved predictions.

## Shared prediction interface

Each model exports one row per held-out prediction with exactly these columns:

```csv
image_id,seed,fold,true_label,predicted_label,confidence
```

`confidence` is the predicted class probability/score in `[0, 1]`, not a calibrated probability unless calibration is explicitly performed using development data. See `docs/experimental_protocol.md` for aggregation and metric requirements.

## Pull requests and collaboration

- Work only in your approach directory unless coordinating a shared change.
- Do not regenerate splits for an individual model. Propose protocol changes to the team and version the shared split artifacts together.
- Keep commits focused and include tests for shared utility changes.
- Keep notebooks exploratory; place reusable logic in importable Python modules.
- Do not merge a model result until its prediction CSV and run metadata are available for independent metric recomputation.
