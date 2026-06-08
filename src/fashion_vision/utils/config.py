"""
Configuration loading utilities.

This module provides reusable functions to load YAML configuration files
with basic validation and clear error messages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the YAML file is empty or does not contain a mapping.
        yaml.YAMLError: If the YAML file cannot be parsed.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Config path is not a file: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    if not isinstance(config, dict):
        raise ValueError(
            f"Config file must contain a YAML mapping at top level: {path}"
        )

    return config


def require_config_keys(config: Dict[str, Any], keys: list[str]) -> None:
    """
    Validate whether required top-level keys exist in configuration.

    Args:
        config: Configuration dictionary.
        keys: Required top-level keys.

    Raises:
        KeyError: If any required key is missing.
    """
    missing_keys = [key for key in keys if key not in config]

    if missing_keys:
        raise KeyError(f"Missing required config keys: {missing_keys}")


def get_nested_config(
    config: Dict[str, Any],
    key_path: list[str],
    default: Any | None = None,
) -> Any:
    """
    Safely get a nested configuration value.

    Args:
        config: Configuration dictionary.
        key_path: List of nested keys.
        default: Default value if the nested key does not exist.

    Returns:
        Nested configuration value or default.
    """
    current: Any = config

    for key in key_path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]

    return current
