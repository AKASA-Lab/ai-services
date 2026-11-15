from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from lettuce_benchmark import CLASS_NAMES
from models.yolov8n_cls.prepare_data import build_split_dataset, cached_path, preprocess_images, verify_split_dataset
from models.yolov8n_cls.train import select_best_epoch, to_canonical_order


def _split(tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    raw = tmp_path / "raw"
    rows = []
    partitions = ["train", "train", "validation", "test"]
    for label in CLASS_NAMES:
        (raw / label).mkdir(parents=True)
        for index, partition in enumerate(partitions):
            image = Image.new("RGB", (1015, 1000), color=(30 * index, 100, 50))
            image.paste((255, 255, 0), (950, 984, 1015, 1000))  # colour-bar corner
            image.save(raw / label / f"img{index}.png")
            rows.append({"image_id": f"{label}_{index}", "image_path": f"{label}/img{index}.png", "label": label,
                         "partition": partition, "seed": 42, "fold": 1})
    return pd.DataFrame(rows), raw


def test_preprocess_applies_safe_crop_and_resize(tmp_path: Path) -> None:
    split, raw = _split(tmp_path)
    cache = preprocess_images(split["image_path"].tolist(), raw, tmp_path / "cache", crop=960, size=64)
    with Image.open(cached_path(cache, "FN/img0.png")) as image:
        assert image.size == (64, 64)
        assert (255, 255, 0) not in {colour for _, colour in image.getcolors(64 * 64)}


def test_split_tree_matches_csv_and_excludes_test(tmp_path: Path) -> None:
    split, raw = _split(tmp_path)
    cache = preprocess_images(split["image_path"].tolist(), raw, tmp_path / "cache", crop=960, size=32)
    tree = build_split_dataset(split, cache, tmp_path / "run")
    assert sorted(p.name for p in tree.iterdir()) == ["train", "val"]
    assert len(list((tree / "train").glob("*/*.png"))) == 8
    assert len(list((tree / "val").glob("*/*.png"))) == 4
    assert (tree / "val" / "-K" / "-K_2.png").exists()
    assert not list(tree.glob("*/*/*_3.png"))  # index 3 is the test row of every class


def test_verify_rejects_leaked_test_image(tmp_path: Path) -> None:
    split, raw = _split(tmp_path)
    cache = preprocess_images(split["image_path"].tolist(), raw, tmp_path / "cache", crop=960, size=32)
    tree = build_split_dataset(split, cache, tmp_path / "run")
    (tree / "train" / "FN" / "FN_3.png").write_bytes(b"")
    with pytest.raises(ValueError, match="test images"):
        verify_split_dataset(split, tree)


def test_verify_rejects_wrong_partition(tmp_path: Path) -> None:
    split, raw = _split(tmp_path)
    cache = preprocess_images(split["image_path"].tolist(), raw, tmp_path / "cache", crop=960, size=32)
    tree = build_split_dataset(split, cache, tmp_path / "run")
    (tree / "train" / "-N" / "-N_0.png").rename(tree / "val" / "-N" / "-N_0.png")
    with pytest.raises(ValueError, match="does not match"):
        verify_split_dataset(split, tree)


def test_probabilities_are_reordered_by_name() -> None:
    names = {0: "-K", 1: "-N", 2: "-P", 3: "FN"}  # Ultralytics' sorted-folder order
    probabilities = np.array([[0.1, 0.2, 0.3, 0.4]])
    reordered = to_canonical_order(probabilities, names)
    assert reordered.tolist() == [[0.4, 0.2, 0.3, 0.1]]  # FN, -N, -P, -K
    with pytest.raises(ValueError):
        to_canonical_order(probabilities, {0: "a", 1: "b", 2: "c", 3: "d"})


def test_best_epoch_prefers_f1_then_loss_then_earlier() -> None:
    history = pd.DataFrame({
        "epoch": [1, 2, 3, 4, 5],
        "val_macro_f1": [0.5, 0.9, 0.9, 0.9, 0.8],
        "val_loss": [1.0, 0.40, 0.30, 0.30, 0.1],
    })
    assert int(select_best_epoch(history)["epoch"]) == 3
