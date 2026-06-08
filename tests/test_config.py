"""
Unit tests for configuration utilities.
"""

from pathlib import Path

import pytest
import yaml

from fashion_vision.utils.config import load_yaml_config, require_config_keys


def test_load_yaml_config(tmp_path: Path) -> None:
    """Test loading a valid YAML configuration file."""
    config_path = tmp_path / "config.yaml"

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump({"dataset": {"name": "DeepFashion2"}}, file)

    config = load_yaml_config(config_path)

    assert config["dataset"]["name"] == "DeepFashion2"


def test_load_yaml_config_missing_file(tmp_path: Path) -> None:
    """Test loading a missing YAML configuration file."""
    config_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        load_yaml_config(config_path)


def test_require_config_keys_success() -> None:
    """Test required config key validation with valid input."""
    config = {
        "dataset": {},
        "model": {},
    }

    require_config_keys(config, ["dataset", "model"])


def test_require_config_keys_failure() -> None:
    """Test required config key validation with missing key."""
    config = {
        "dataset": {},
    }

    with pytest.raises(KeyError):
        require_config_keys(config, ["dataset", "model"])
