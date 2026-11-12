from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from lettuce_benchmark.dataset import audit_dataset, class_distribution, create_metadata


def test_audit_and_metadata_only_include_valid_labeled_images(tmp_path: Path) -> None:
    image_path = tmp_path / "FN" / "healthy.png"
    image_path.parent.mkdir()
    Image.new("RGB", (12, 9), color=(20, 100, 30)).save(image_path)
    duplicate_path = tmp_path / "-N" / "copy.png"
    duplicate_path.parent.mkdir()
    image_path.replace(duplicate_path)
    # Restore a healthy image and retain an exact duplicate under another label.
    Image.new("RGB", (12, 9), color=(20, 100, 30)).save(image_path)
    duplicate_path.write_bytes(image_path.read_bytes())
    invalid_path = tmp_path / "-P" / "broken.jpg"
    invalid_path.parent.mkdir()
    invalid_path.write_text("not an image", encoding="utf-8")
    (tmp_path / "misc").mkdir()
    Image.new("RGB", (4, 4)).save(tmp_path / "misc" / "unlabeled.png")

    report = audit_dataset(tmp_path)
    assert len(report) == 4
    assert report["valid"].sum() == 3
    assert report["exact_duplicate_of"].notna().sum() == 1
    metadata = create_metadata(report, tmp_path)
    assert list(metadata.columns) == ["image_id", "image_path", "label"]
    assert len(metadata) == 2
    assert class_distribution(metadata) == {"FN": 1, "-N": 1, "-P": 0, "-K": 0}


def test_empty_audit_has_schema(tmp_path: Path) -> None:
    report = audit_dataset(tmp_path)
    assert isinstance(report, pd.DataFrame)
    assert "valid" in report.columns


def test_optional_source_metadata_is_preserved_without_invention(tmp_path: Path) -> None:
    image_path = tmp_path / "FN" / "sample.png"
    image_path.parent.mkdir()
    Image.new("RGB", (4, 4)).save(image_path)
    audit = audit_dataset(tmp_path)
    image_id = audit.loc[0, "image_id"]
    metadata = create_metadata(
        audit,
        optional_metadata=pd.DataFrame({
            "image_id": [image_id],
            "plant_id": ["source-plant-7"],
            "source_group": ["session-2"],
        }),
    )
    assert metadata.loc[0, "plant_id"] == "source-plant-7"
    assert metadata.loc[0, "source_group"] == "session-2"


def test_unknown_optional_metadata_is_rejected(tmp_path: Path) -> None:
    image_path = tmp_path / "FN" / "sample.png"
    image_path.parent.mkdir()
    Image.new("RGB", (4, 4)).save(image_path)
    audit = audit_dataset(tmp_path)
    with pytest.raises(ValueError, match="unknown image_id"):
        create_metadata(
            audit,
            optional_metadata=pd.DataFrame({
                "image_id": ["not-in-manifest"],
                "plant_id": ["plant-1"],
            }),
        )
