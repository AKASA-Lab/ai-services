"""Validate a generated split directory and report partition sizes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..splits import validate_split_collection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_dir", type=Path)
    args = parser.parse_args()
    frames = {}
    for path in sorted(args.split_dir.glob("seed_*/fold_*.csv")):
        seed = int(path.parent.name.removeprefix("seed_"))
        fold = int(path.stem.removeprefix("fold_"))
        frames[(seed, fold)] = pd.read_csv(path, dtype={"image_id": str, "label": str})
    validate_split_collection(frames)
    for (seed, fold), frame in sorted(frames.items()):
        counts = frame["partition"].value_counts().to_dict()
        print(f"seed={seed} fold={fold}: {counts}")
    print(f"Validated {len(frames)} split files.")


if __name__ == "__main__":
    main()
