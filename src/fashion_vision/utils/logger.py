"""
Logging utilities for the fashion vision project.

This module creates a logger that writes logs to both console and file.
"""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(
    name: str,
    log_file: str | Path | None = None,
    level: str = "INFO",
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.

    Args:
        name: Logger name.
        log_file: Optional log file path.
        level: Logging level, such as ``INFO`` or ``DEBUG``.

    Returns:
        Configured logger instance.

    Raises:
        ValueError: If the logging level is invalid.
    """
    numeric_level = getattr(logging, level.upper(), None)

    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid logging level: {level}")

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] "
        "[%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
