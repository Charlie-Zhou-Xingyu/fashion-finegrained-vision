"""
Path management utilities.

This module provides reusable helpers for creating and validating project
directories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure that a directory exists.

    Args:
        path: Directory path.

    Returns:
        Resolved Path object.

    Raises:
        ValueError: If the path exists but is not a directory.
    """
    directory = Path(path)

    if directory.exists() and not directory.is_dir():
        raise ValueError(f"Path exists but is not a directory: {directory}")

    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent_dir(file_path: str | Path) -> Path:
    """
    Ensure that the parent directory of a file path exists.

    Args:
        file_path: File path.

    Returns:
        Parent directory path.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.parent


def prepare_output_dirs(output_root: str) -> Dict[str, Path]:
    """
    Prepare output directories.

    Args:
        output_root: Root output directory.

    Returns:
        Dictionary of prepared output directories.
    """
    output_root = Path(output_root)

    dirs = {
        "root": output_root,
        "predictions": output_root / "predictions",
        "visualizations": output_root / "visualizations",
        "metrics": output_root / "metrics",
        "logs": output_root / "logs",
        "pred_masks": output_root / "masks" / "pred",
        "gt_masks": output_root / "masks" / "gt",
        "crops": output_root / "crops",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def validate_file_exists(path: str | Path, description: str = "file") -> Path:
    """
    Validate whether a file exists.

    Args:
        path: File path.
        description: Human-readable file description.

    Returns:
        Path object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path exists but is not a file.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"{description} does not exist: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"{description} is not a file: {file_path}")

    return file_path


def validate_dir_exists(path: str | Path, description: str = "directory") -> Path:
    """
    Validate whether a directory exists.

    Args:
        path: Directory path.
        description: Human-readable directory description.

    Returns:
        Path object.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If the path exists but is not a directory.
    """
    directory = Path(path)

    if not directory.exists():
        raise FileNotFoundError(f"{description} does not exist: {directory}")

    if not directory.is_dir():
        raise ValueError(f"{description} is not a directory: {directory}")

    return directory
