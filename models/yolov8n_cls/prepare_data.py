"""Turn the shared split CSVs into Ultralytics classification folders (Person 4).

Ultralytics trains a classifier from a directory tree::

    <run_dir>/train/<class>/<image>.png
    <run_dir>/val/<class>/<image>.png

For every (seed, fold) run this module writes exactly the ``train`` and ``validation`` rows of
that run's shared split CSV into such a tree (as symlinks, or copies where symlinks are not
supported), then re-reads the tree and checks that its membership matches the CSV row for row.

The ``test`` rows are deliberately **not** written into the tree. Ultralytics falls back to a
``test/`` folder when it cannot find ``val/``, so keeping the held-out fold outside the folder
means the trainer cannot see it. Test images are predicted once, after model selection,
straight from the preprocessed image cache.

Every image goes through the shared ``watermark_safe_crop`` (fixed 960x960, removes the DALL-E
colour bar, see ``docs/dataset_audit.md`` §4) and is resized once to ``cache_size`` before any
augmentation, the same preprocessing as MobileNetV2 (Person 1).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

from lettuce_benchmark import CLASS_NAMES
from lettuce_benchmark.preprocessing import watermark_safe_crop

#: Shared split partition name -> Ultralytics folder name.
PARTITION_FOLDERS = {"train": "train", "validation": "val"}
_STAMP = "preprocessing.json"


def preprocess_images(image_paths: list[str], image_root: Path, cache_root: Path, crop: int, size: int) -> Path:
    """Crop + resize every source image once, keeping its relative path, and return ``cache_root``.

    Output files are PNG (lossless). A stamp file records the crop/size so a cache built with
    different settings is rebuilt instead of reused.
    """
    cache_root = Path(cache_root)
    stamp_path = cache_root / _STAMP
    settings = {"safe_crop": crop, "size": size, "resample": "bicubic", "format": "png"}
    if stamp_path.is_file() and json.loads(stamp_path.read_text()) != settings:
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    for relative in image_paths:
        destination = cache_root / Path(relative).with_suffix(".png")
        if destination.is_file():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(Path(image_root) / relative) as image:
            image = watermark_safe_crop(image.convert("RGB"), crop)
            image.resize((size, size), Image.Resampling.BICUBIC).save(destination)
    stamp_path.write_text(json.dumps(settings))
    return cache_root


def cached_path(cache_root: Path, relative: str) -> Path:
    """Location of a source image (``image_path`` from the CSV) inside the preprocessed cache."""
    return Path(cache_root) / Path(relative).with_suffix(".png")


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.symlink(source.resolve(), destination)
    except (OSError, NotImplementedError):
        shutil.copy2(source, destination)


def build_split_dataset(split: pd.DataFrame, cache_root: Path, destination: Path) -> Path:
    """Write one run's train/validation rows as an Ultralytics classification tree.

    Files are named ``<image_id>.png`` inside the folder of their label, so every file maps back to
    exactly one CSV row. Any existing tree at ``destination`` is replaced.
    """
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    for partition, folder in PARTITION_FOLDERS.items():
        for label in CLASS_NAMES:
            (destination / folder / label).mkdir(parents=True, exist_ok=True)
        for row in split[split["partition"] == partition].itertuples(index=False):
            source = cached_path(cache_root, row.image_path)
            if not source.is_file():
                raise FileNotFoundError(f"Preprocessed image missing: {source}")
            _link_or_copy(source, destination / folder / row.label / f"{row.image_id}.png")
    verify_split_dataset(split, destination)
    return destination


def verify_split_dataset(split: pd.DataFrame, destination: Path) -> None:
    """Check that the folder tree holds exactly the CSV's train/validation rows and no test rows."""
    destination = Path(destination)
    expected = {
        (PARTITION_FOLDERS[row.partition], row.label, str(row.image_id))
        for row in split.itertuples(index=False)
        if row.partition in PARTITION_FOLDERS
    }
    found = {
        (path.parent.parent.name, path.parent.name, path.stem)
        for path in destination.glob("*/*/*")
        if path.is_file()
    }
    unexpected_folders = {p.name for p in destination.iterdir() if p.is_dir()} - set(PARTITION_FOLDERS.values())
    if unexpected_folders:
        raise ValueError(f"Unexpected folders in {destination}: {sorted(unexpected_folders)}")
    test_ids = set(split.loc[split["partition"] == "test", "image_id"].astype(str))
    leaked = {image_id for _, _, image_id in found} & test_ids
    if leaked:
        raise ValueError(f"{len(leaked)} test images found in the training tree, e.g. {sorted(leaked)[:3]}")
    if found != expected:
        missing, extra = expected - found, found - expected
        raise ValueError(f"Split tree does not match the CSV: {len(missing)} missing, {len(extra)} unexpected files")
