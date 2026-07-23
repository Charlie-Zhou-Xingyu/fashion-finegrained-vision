"""Shared helpers for eval framework tests — imports the runner by file path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_SCRIPTS = PROJECT_ROOT / "eval" / "scripts"
DATASETS_DIR = PROJECT_ROOT / "eval" / "datasets"
SCHEMA_PATH = PROJECT_ROOT / "eval" / "schemas" / "eval_case_schema.json"


def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_runner():
    """Import eval/scripts/run_serving_eval.py as a module."""
    return _load_module("p1_eval_runner", EVAL_SCRIPTS / "run_serving_eval.py")


def load_summarizer():
    """Import eval/scripts/summarize_eval_report.py as a module."""
    return _load_module("p1_eval_summarizer", EVAL_SCRIPTS / "summarize_eval_report.py")
