from pathlib import Path

from lettuce_benchmark.config import load_config


def test_common_configuration_loads() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "common.yaml")
    assert config["splits"]["seeds"] == [42, 123, 2026]
    assert config["splits"]["n_splits"] == 4
