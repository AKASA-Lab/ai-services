import pandas as pd
import pytest

from lettuce_benchmark.splits import (
    generate_splits,
    validate_split_collection,
    validate_split_frame,
    write_splits,
)


def make_metadata(per_class: int = 20) -> pd.DataFrame:
    rows = []
    for label in ("FN", "-N", "-P", "-K"):
        for index in range(per_class):
            rows.append({
                "image_id": f"{label}_{index}",
                "image_path": f"{label}/{index}.jpg",
                "label": label,
            })
    return pd.DataFrame(rows)


def test_split_generation_is_disjoint_and_complete(tmp_path) -> None:
    splits = generate_splits(make_metadata(), seeds=[42, 123])
    assert len(splits) == 8
    for split in splits.values():
        validate_split_frame(split)
        assert split["image_id"].nunique() == len(make_metadata())
    validate_split_collection(splits)
    write_splits(splits, tmp_path)
    saved = pd.read_csv(tmp_path / "seed_42" / "fold_1.csv")
    assert set(saved["partition"]) == {"train", "validation", "test"}
    assert set(saved.columns) == {
        "image_id", "image_path", "label", "partition", "seed", "fold"
    }


def test_group_aware_splits_keep_groups_together() -> None:
    metadata = make_metadata()
    # The same source group is intentionally shared between classes; the splitter
    # must keep each complete group in one partition.
    metadata["plant_id"] = [f"plant_{index // 2}" for index in range(len(metadata))]
    splits = generate_splits(metadata, seeds=[42], group_column="plant_id")
    for split in splits.values():
        group_partitions = split.groupby("plant_id")["partition"].nunique()
        assert group_partitions.max() == 1


def test_group_column_must_be_complete() -> None:
    metadata = make_metadata()
    metadata["plant_id"] = None
    with pytest.raises(ValueError, match="complete and non-empty"):
        generate_splits(metadata, seeds=[42], group_column="plant_id")


def test_overlapping_split_rows_are_rejected() -> None:
    split = generate_splits(make_metadata(), seeds=[42])[(42, 1)].copy()
    split.loc[split.index[0], "partition"] = split.loc[split.index[1], "partition"]
    split.loc[split.index[1], "image_id"] = split.loc[split.index[0], "image_id"]
    with pytest.raises(ValueError, match="exactly once"):
        validate_split_frame(split)
