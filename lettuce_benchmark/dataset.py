"""Image discovery, integrity checks, metadata generation, and duplicate reports."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from . import CLASS_NAMES

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _label_from_path(path: Path, class_names: Iterable[str]) -> str | None:
    """Infer a label only when a path component exactly names a known class."""
    names = {name.casefold(): name for name in class_names}
    for component in path.parts[:-1]:
        if component.casefold() in names:
            return names[component.casefold()]
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _difference_hash(path: Path, hash_size: int = 8) -> int:
    """Return a simple perceptual difference hash for near-duplicate screening."""
    with Image.open(path) as image:
        pixels = np.asarray(
            image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        )
    bits = pixels[:, 1:] > pixels[:, :-1]
    return int("".join("1" if bit else "0" for bit in bits.flat), 2)


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def audit_dataset(
    image_root: str | Path,
    class_names: Iterable[str] = CLASS_NAMES,
    detect_near_duplicates: bool = False,
    near_duplicate_threshold: int = 6,
) -> pd.DataFrame:
    """Audit discovered images and return one report row per image file.

    Labels are inferred only from a path component matching a known class name.
    No images are changed or deleted. Near-duplicate pairs are optional and
    reported as candidate IDs for manual review.
    """
    root = Path(image_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Image root is not a directory: {root}")
    if near_duplicate_threshold < 0:
        raise ValueError("near_duplicate_threshold must be non-negative")

    paths = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS),
        key=lambda path: path.as_posix().casefold(),
    )
    rows: list[dict[str, object]] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        image_id = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]
        label = _label_from_path(path.relative_to(root), class_names)
        row: dict[str, object] = {
            "image_id": image_id,
            "image_path": relative,
            "label": label,
            "valid": False,
            "width": None,
            "height": None,
            "sha256": None,
            "error": None,
            "exact_duplicate_of": None,
            "near_duplicate_candidates": "",
        }
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                row["width"], row["height"] = image.size
            row["sha256"] = _sha256(path)
            row["valid"] = True
        except (OSError, ValueError, UnidentifiedImageError) as error:
            row["error"] = f"{type(error).__name__}: {error}"
            LOGGER.warning("Invalid image %s: %s", relative, row["error"])
        rows.append(row)

    report = pd.DataFrame(
        rows,
        columns=[
            "image_id",
            "image_path",
            "label",
            "valid",
            "width",
            "height",
            "sha256",
            "error",
            "exact_duplicate_of",
            "near_duplicate_candidates",
        ],
    )
    if report.empty:
        return report

    valid_rows = report[report["valid"]]
    for _, duplicate_group in valid_rows[valid_rows.duplicated("sha256", keep=False)].groupby("sha256"):
        ids = duplicate_group["image_id"].tolist()
        canonical = ids[0]
        for duplicate_id in ids[1:]:
            report.loc[report["image_id"] == duplicate_id, "exact_duplicate_of"] = canonical

    if detect_near_duplicates:
        hashes: list[tuple[str, int]] = []
        for row in valid_rows.itertuples(index=False):
            try:
                hashes.append((row.image_id, _difference_hash(root / Path(row.image_path))))
            except (OSError, ValueError, UnidentifiedImageError) as error:
                LOGGER.warning("Could not compute perceptual hash for %s: %s", row.image_path, error)
        candidates: dict[str, set[str]] = {image_id: set() for image_id, _ in hashes}
        for index, (left_id, left_hash) in enumerate(hashes):
            for right_id, right_hash in hashes[index + 1 :]:
                if _hamming_distance(left_hash, right_hash) <= near_duplicate_threshold:
                    candidates[left_id].add(right_id)
                    candidates[right_id].add(left_id)
        for image_id, matches in candidates.items():
            if matches:
                report.loc[report["image_id"] == image_id, "near_duplicate_candidates"] = ";".join(
                    sorted(matches)
                )
    return report


def create_metadata(
    audit: pd.DataFrame,
    image_root: str | Path | None = None,
    optional_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create metadata from valid audit rows and optional source group metadata."""
    required = {"image_id", "image_path", "label", "valid"}
    missing = required.difference(audit.columns)
    if missing:
        raise ValueError(f"Audit report is missing columns: {sorted(missing)}")
    selected = audit.loc[audit["valid"] & audit["label"].isin(CLASS_NAMES), [
        "image_id",
        "image_path",
        "label",
    ]].copy()
    if selected["image_id"].duplicated().any():
        raise ValueError("Metadata contains duplicate image_id values")
    if optional_metadata is not None:
        optional_columns = [
            column for column in ("plant_id", "source_group")
            if column in optional_metadata.columns
        ]
        if not optional_columns:
            raise ValueError("Optional metadata must contain plant_id or source_group")
        key = next(
            (column for column in ("image_id", "image_path") if column in optional_metadata.columns),
            None,
        )
        if key is None:
            raise ValueError("Optional metadata must be keyed by image_id or image_path")
        if optional_metadata[key].isna().any() or optional_metadata[key].duplicated().any():
            raise ValueError(f"Optional metadata key {key!r} must be non-null and unique")
        unmatched = set(optional_metadata[key]) - set(selected[key])
        if unmatched:
            raise ValueError(f"Optional metadata contains unknown {key} values: {len(unmatched)}")
        selected = selected.merge(
            optional_metadata[[key, *optional_columns]],
            on=key,
            how="left",
            validate="one_to_one",
        )
    if image_root is not None:
        root = Path(image_root).expanduser().resolve()
        for relative in selected["image_path"]:
            if not (root / Path(relative)).is_file():
                raise FileNotFoundError(f"Manifest image does not exist: {relative}")
    return selected.reset_index(drop=True)


def class_distribution(metadata: pd.DataFrame) -> dict[str, int]:
    """Return counts in canonical class order, including absent classes as zero."""
    if "label" not in metadata:
        raise ValueError("Metadata must contain a label column")
    counts = metadata["label"].value_counts()
    return {name: int(counts.get(name, 0)) for name in CLASS_NAMES}
