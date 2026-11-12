"""Audit an extracted dataset without modifying source files."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ..config import configure_logging
from ..dataset import audit_dataset, class_distribution, create_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/audit"))
    parser.add_argument("--near-duplicates", action="store_true")
    parser.add_argument("--near-duplicate-threshold", type=int, default=6)
    args = parser.parse_args()
    configure_logging()
    report = audit_dataset(
        args.image_root,
        detect_near_duplicates=args.near_duplicates,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output_dir / "audit.csv", index=False)
    metadata = create_metadata(report)
    summary = {
        "image_files_discovered": int(len(report)),
        "valid_images": int(report["valid"].sum()) if not report.empty else 0,
        "invalid_images": int((~report["valid"]).sum()) if not report.empty else 0,
        "unclassified_valid_images": int((report["valid"] & report["label"].isna()).sum()) if not report.empty else 0,
        "class_distribution_valid_labeled": class_distribution(metadata),
        "exact_duplicate_rows": int(report["exact_duplicate_of"].notna().sum()) if not report.empty else 0,
        "near_duplicate_candidates_reported": bool(args.near_duplicates),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    logging.getLogger(__name__).info("Audit summary: %s", summary)


if __name__ == "__main__":
    main()
