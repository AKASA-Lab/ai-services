"""Repeated stratified four-fold split generation and integrity validation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)

from . import CLASS_NAMES, N_SPLITS, SEEDS

METADATA_COLUMNS = {"image_id", "image_path", "label"}
SPLIT_COLUMNS = ["image_id", "image_path", "label", "partition", "seed", "fold"]
PARTITIONS = {"train", "validation", "test"}


def _validate_metadata(metadata: pd.DataFrame) -> None:
    missing = METADATA_COLUMNS.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
    if metadata.empty:
        raise ValueError("Metadata has no images")
    if metadata["image_id"].isna().any() or metadata["image_id"].duplicated().any():
        raise ValueError("image_id values must be non-null and unique")
    if metadata[["image_path", "label"]].isna().any().any():
        raise ValueError("image_path and label values must be non-null")
    unknown = sorted(set(metadata["label"]) - set(CLASS_NAMES))
    if unknown:
        raise ValueError(f"Unknown class labels: {unknown}")
    counts = metadata["label"].value_counts()
    too_small = {label: int(counts.get(label, 0)) for label in CLASS_NAMES if counts.get(label, 0) < N_SPLITS}
    if too_small:
        raise ValueError(f"Each class needs at least {N_SPLITS} images for stratified folds: {too_small}")


def _inner_split(
    development: pd.DataFrame,
    validation_fraction: float,
    random_state: int,
    groups: pd.Series | None,
) -> tuple[list[int], list[int]]:
    indices = list(range(len(development)))
    if groups is None:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=validation_fraction, random_state=random_state
        )
        train_idx, validation_idx = next(
            splitter.split(development, development["label"])
        )
        return train_idx.tolist(), validation_idx.tolist()

    n_splits = max(2, round(1 / validation_fraction))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    candidates = list(splitter.split(development, development["label"], groups))
    train_idx, validation_idx = min(
        candidates,
        key=lambda pair: abs(len(pair[1]) / len(development) - validation_fraction),
    )
    if not set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[validation_idx])):
        raise RuntimeError("Group leakage detected in train/validation split")
    return train_idx.tolist(), validation_idx.tolist()


def generate_splits(
    metadata: pd.DataFrame,
    seeds: Iterable[int] = SEEDS,
    n_splits: int = N_SPLITS,
    validation_fraction: float = 0.2,
    group_column: str | None = None,
) -> dict[tuple[int, int], pd.DataFrame]:
    """Generate one CSV-ready split frame per ``(seed, fold)`` pair.

    If ``group_column`` is given, all group values must be present and both
    outer and inner splits are group-aware. No image-level fallback is made.
    """
    _validate_metadata(metadata)
    if n_splits != N_SPLITS:
        raise ValueError(f"This protocol requires exactly {N_SPLITS} folds")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if group_column is not None:
        if group_column not in metadata:
            raise ValueError(f"Group column not found in metadata: {group_column}")
        if metadata[group_column].isna().any() or metadata[group_column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Group column {group_column!r} must be complete and non-empty")

    results: dict[tuple[int, int], pd.DataFrame] = {}
    labels = metadata["label"].to_numpy()
    group_values = metadata[group_column] if group_column is not None else None
    for seed in seeds:
        if group_values is None:
            outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
            outer_splits = outer.split(metadata, labels)
        else:
            outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
            outer_splits = outer.split(metadata, labels, group_values)

        for fold, (development_idx, test_idx) in enumerate(outer_splits, start=1):
            development = metadata.iloc[development_idx].reset_index(drop=True)
            development_groups = (
                development[group_column] if group_column is not None else None
            )
            train_idx, validation_idx = _inner_split(
                development,
                validation_fraction,
                int(seed) + fold,
                development_groups,
            )
            assignments = pd.concat(
                [
                    development.iloc[train_idx].assign(partition="train"),
                    development.iloc[validation_idx].assign(partition="validation"),
                    metadata.iloc[test_idx].assign(partition="test"),
                ],
                ignore_index=True,
            )
            assignments["seed"] = int(seed)
            assignments["fold"] = fold
            columns = SPLIT_COLUMNS + ([group_column] if group_column is not None else [])
            split = assignments[columns].sort_values("image_id").reset_index(drop=True)
            validate_split_frame(split)
            if group_column is not None:
                for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
                    left_groups = set(split.loc[split["partition"] == left, group_column])
                    right_groups = set(split.loc[split["partition"] == right, group_column])
                    if not left_groups.isdisjoint(right_groups):
                        raise RuntimeError(f"Group leakage between {left} and {right}")
            results[(int(seed), fold)] = split
    validate_split_collection(results)
    return results


def validate_split_frame(split: pd.DataFrame) -> None:
    """Raise ``ValueError`` when a fold has invalid columns or overlap."""
    missing = set(SPLIT_COLUMNS).difference(split.columns)
    if missing:
        raise ValueError(f"Split is missing required columns: {sorted(missing)}")
    if split["image_id"].isna().any() or split["image_id"].duplicated().any():
        raise ValueError("Each image must appear exactly once per fold")
    if not set(split["partition"]).issubset(PARTITIONS) or set(split["partition"]) != PARTITIONS:
        raise ValueError(f"Each split must contain train, validation, and test rows: {PARTITIONS}")


def validate_split_collection(splits: dict[tuple[int, int], pd.DataFrame]) -> None:
    """Validate fold completeness and ensure each image is test once per seed."""
    if not splits:
        raise ValueError("No split files were supplied")
    seeds = sorted({seed for seed, _ in splits})
    all_seed_ids: set[str] | None = None
    for seed in seeds:
        folds = sorted(fold for current_seed, fold in splits if current_seed == seed)
        if folds != list(range(1, N_SPLITS + 1)):
            raise ValueError(f"Seed {seed} must have folds 1 through {N_SPLITS}; found {folds}")
        test_counts: dict[str, int] = {}
        reference_ids: set[str] | None = None
        for fold in folds:
            frame = splits[(seed, fold)]
            validate_split_frame(frame)
            ids = set(frame["image_id"])
            if reference_ids is None:
                reference_ids = ids
            elif ids != reference_ids:
                raise ValueError(f"Images differ across folds for seed {seed}")
            for image_id in frame.loc[frame["partition"] == "test", "image_id"]:
                test_counts[image_id] = test_counts.get(image_id, 0) + 1
        if reference_ids is None or set(test_counts) != reference_ids or any(count != 1 for count in test_counts.values()):
            raise ValueError(f"Every image must be held out exactly once for seed {seed}")
        if all_seed_ids is None:
            all_seed_ids = reference_ids
        elif reference_ids != all_seed_ids:
            raise ValueError("Images differ between seeds")


def write_splits(splits: dict[tuple[int, int], pd.DataFrame], output_dir: str | Path) -> None:
    """Write all split frames to ``seed_<seed>/fold_<fold>.csv``."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for (seed, fold), frame in sorted(splits.items()):
        destination = root / f"seed_{seed}" / f"fold_{fold}.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
