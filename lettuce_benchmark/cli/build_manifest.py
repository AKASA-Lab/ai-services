"""Create a metadata manifest from the audited image tree."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import configure_logging
from ..dataset import audit_dataset, class_distribution, create_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/metadata.csv"))
    parser.add_argument(
        "--metadata-sidecar",
        type=Path,
        help="Optional CSV keyed by image_id or image_path with plant_id/source_group columns",
    )
    args = parser.parse_args()
    configure_logging()
    audit = audit_dataset(args.image_root)
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
