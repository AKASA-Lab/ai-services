"""Generate shared repeated stratified train/validation/test CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..config import configure_logging, load_config
from ..splits import generate_splits, write_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--config", type=Path, default=Path("configs/common.yaml"))
    parser.add_argument("--group-column", default=None)
    args = parser.parse_args()
    configure_logging()
    config = load_config(args.config)
    split_config = config.get("splits", {})
    group_column = args.group_column or config.get("dataset", {}).get("group_column")
    metadata = pd.read_csv(args.metadata_csv, dtype={"image_id": str, "label": str})
    splits = generate_splits(
        metadata,
        seeds=split_config.get("seeds", [42, 123, 2026]),
        n_splits=split_config.get("n_splits", 4),
        validation_fraction=split_config.get("validation_fraction", 0.2),
        group_column=group_column,
    )
    write_splits(splits, args.output_dir)
    print(f"Wrote {len(splits)} shared split CSVs to {args.output_dir}")
    print("Group-aware splitting enabled." if group_column else "Evaluation is image-level; no reliable group column was selected.")


if __name__ == "__main__":
    main()
