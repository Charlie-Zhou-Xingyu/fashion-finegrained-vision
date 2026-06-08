"""
JSON input and output utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_json(json_path: str | Path) -> Dict[str, Any]:
    """
    Load a JSON file.

    Args:
        json_path: Path to JSON file.

    Returns:
        Parsed JSON dictionary.

    Raises:
        FileNotFoundError: If JSON file does not exist.
        ValueError: If the JSON root object is not a dictionary.
        json.JSONDecodeError: If JSON decoding fails.
    """
    path = Path(json_path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be a dictionary: {path}")

    return data


def save_json(data: Dict[str, Any], json_path: str | Path) -> None:
    """
    Save a dictionary to a JSON file.

    Args:
        data: Dictionary to save.
        json_path: Output JSON file path.
    """
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
