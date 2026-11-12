"""Create a metadata manifest from the audited image tree."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..config import configure_logging
from ..dataset import audit_dataset, class_distribution, create_metadata, drop_exact_duplicates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/metadata.csv"))
    parser.add_argument(
        "--metadata-sidecar",
        type=Path,
        help="Optional CSV keyed by image_id or image_path with plant_id/source_group columns",
    )
    parser.add_argument(
        "--drop-exact-duplicates",
        action="store_true",
        help=(
            "Exclude redundant byte-identical copies (same SHA-256 and same label), keeping the "
            "first copy. Excluded rows are written next to the manifest for review."
        ),
    )
    args = parser.parse_args()
    configure_logging()
    audit = audit_dataset(args.image_root)
    if args.drop_exact_duplicates:
        audit, excluded = drop_exact_duplicates(audit)
        excluded_path = args.output.parent / "excluded_exact_duplicates.csv"
        excluded_path.parent.mkdir(parents=True, exist_ok=True)
        excluded[["image_id", "image_path", "label", "sha256", "exact_duplicate_of"]].to_csv(
            excluded_path, index=False
        )
        print(f"Excluded {len(excluded)} exact-duplicate copies; listed in {excluded_path}")
    optional_metadata = (
        pd.read_csv(args.metadata_sidecar, dtype=str)
        if args.metadata_sidecar is not None
        else None
    )
    metadata = create_metadata(audit, args.image_root, optional_metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.output, index=False)
    print(f"Wrote {len(metadata)} valid, labeled image rows to {args.output}")
    print(f"Class distribution: {class_distribution(metadata)}")


if __name__ == "__main__":
    main()
