"""Unit tests for pure helper functions in tools/eval/benchmark_attribute_latency.py.

Tests are restricted to functions that do NOT require a model checkpoint or GPU:
  - _compute_stats()
  - _resolve_num_classes()
  - _resolve_device()

The script is imported via importlib so that sys.path patching inside the module
runs correctly (the module sets up _PROJECT_ROOT / _SRC_DIR on sys.path at
import time).

No model weights, no dataset access, no GPU required.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Import the benchmark script via importlib
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BENCHMARK_SCRIPT = _PROJECT_ROOT / "tools" / "eval" / "benchmark_attribute_latency.py"


def _load_benchmark_module():
    """Load benchmark_attribute_latency as a module object (importlib)."""
    spec = importlib.util.spec_from_file_location(
        "benchmark_attribute_latency", _BENCHMARK_SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bench():
    """Return the benchmark module, loaded once per test session."""
    return _load_benchmark_module()


# ---------------------------------------------------------------------------
# _compute_stats — pure numpy, no model
# ---------------------------------------------------------------------------


def test_compute_stats_returns_dict(bench) -> None:
    result = bench._compute_stats([1.0, 2.0, 3.0])
    assert isinstance(result, dict)


def test_compute_stats_has_required_keys(bench) -> None:
    keys = {"mean", "median", "p50", "p95", "p99", "min", "max"}
    result = bench._compute_stats([1.0, 2.0, 3.0])
    assert keys.issubset(result.keys())


def test_compute_stats_mean_uniform(bench) -> None:
    result = bench._compute_stats([2.0, 2.0, 2.0, 2.0])
    assert result["mean"] == pytest.approx(2.0)


def test_compute_stats_mean_known(bench) -> None:
    result = bench._compute_stats([1.0, 2.0, 3.0, 4.0])
    assert result["mean"] == pytest.approx(2.5)


def test_compute_stats_median_even_count(bench) -> None:
    result = bench._compute_stats([1.0, 2.0, 3.0, 4.0])
    assert result["median"] == pytest.approx(2.5)


def test_compute_stats_median_odd_count(bench) -> None:
    result = bench._compute_stats([1.0, 2.0, 3.0])
    assert result["median"] == pytest.approx(2.0)


def test_compute_stats_p50_equals_median(bench) -> None:
    data = [float(i) for i in range(1, 11)]
    result = bench._compute_stats(data)
    assert result["p50"] == pytest.approx(result["median"])


def test_compute_stats_min_max(bench) -> None:
    result = bench._compute_stats([3.0, 1.0, 5.0, 2.0, 4.0])
    assert result["min"] == pytest.approx(1.0)
    assert result["max"] == pytest.approx(5.0)


def test_compute_stats_p95_above_p50(bench) -> None:
    data = [float(i) for i in range(1, 101)]
    result = bench._compute_stats(data)
    assert result["p95"] > result["p50"]


def test_compute_stats_p99_above_p95(bench) -> None:
    data = [float(i) for i in range(1, 101)]
    result = bench._compute_stats(data)
    assert result["p99"] >= result["p95"]


def test_compute_stats_single_element(bench) -> None:
    result = bench._compute_stats([7.5])
    assert result["mean"] == pytest.approx(7.5)
    assert result["min"] == pytest.approx(7.5)
    assert result["max"] == pytest.approx(7.5)


def test_compute_stats_all_values_are_floats(bench) -> None:
    result = bench._compute_stats([1.0, 2.0, 3.0])
    for key, val in result.items():
        assert isinstance(val, float), f"{key!r} is {type(val).__name__}, expected float"


def test_compute_stats_empty_raises(bench) -> None:
    with pytest.raises(ValueError, match="empty"):
        bench._compute_stats([])


# ---------------------------------------------------------------------------
# _resolve_num_classes — synthetic state dicts, no file I/O
# ---------------------------------------------------------------------------


def _fake_state(fc_out: int | None = None) -> dict[str, torch.Tensor]:
    """Return a minimal synthetic state dict with optional fc.weight."""
    state: dict[str, torch.Tensor] = {
        "layer1.weight": torch.zeros(64, 3, 3, 3),
    }
    if fc_out is not None:
        state["fc.weight"] = torch.zeros(fc_out, 512)
        state["fc.bias"] = torch.zeros(fc_out)
    return state


_FAKE_CHECKPOINT_PATH = Path("/fake/ckpt.pt")


def test_resolve_num_classes_uses_explicit_arg(bench) -> None:
    state = _fake_state(fc_out=5)
    result = bench._resolve_num_classes(8, state, _FAKE_CHECKPOINT_PATH)
    assert result == 8


def test_resolve_num_classes_infers_from_fc_weight(bench) -> None:
    state = _fake_state(fc_out=6)
    result = bench._resolve_num_classes(None, state, _FAKE_CHECKPOINT_PATH)
    assert result == 6


def test_resolve_num_classes_explicit_overrides_state(bench) -> None:
    state = _fake_state(fc_out=5)
    result = bench._resolve_num_classes(12, state, _FAKE_CHECKPOINT_PATH)
    assert result == 12


def test_resolve_num_classes_infers_from_classifier_weight(bench) -> None:
    state = {
        "classifier.weight": torch.zeros(10, 1280),
        "classifier.bias": torch.zeros(10),
    }
    result = bench._resolve_num_classes(None, state, _FAKE_CHECKPOINT_PATH)
    assert result == 10


def test_resolve_num_classes_raises_when_no_key_and_no_arg(bench) -> None:
    state = _fake_state(fc_out=None)
    with pytest.raises(ValueError, match="Cannot infer --num-classes"):
        bench._resolve_num_classes(None, state, _FAKE_CHECKPOINT_PATH)


def test_resolve_num_classes_error_message_includes_path(bench) -> None:
    state = _fake_state(fc_out=None)
    with pytest.raises(ValueError, match="fake"):
        bench._resolve_num_classes(None, state, _FAKE_CHECKPOINT_PATH)


def test_resolve_num_classes_error_mentions_explicit_flag(bench) -> None:
    state = _fake_state(fc_out=None)
    with pytest.raises(ValueError, match="--num-classes"):
        bench._resolve_num_classes(None, state, _FAKE_CHECKPOINT_PATH)


# ---------------------------------------------------------------------------
# _resolve_device
# ---------------------------------------------------------------------------


def test_resolve_device_cpu_string(bench) -> None:
    device = bench._resolve_device("cpu")
    assert device == torch.device("cpu")


def test_resolve_device_auto_returns_device(bench) -> None:
    device = bench._resolve_device("auto")
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "cuda")


def test_resolve_device_auto_cpu_when_no_cuda(bench) -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA present — auto would return cuda, not cpu.")
    device = bench._resolve_device("auto")
    assert device == torch.device("cpu")


# ---------------------------------------------------------------------------
# PRD constant sanity
# ---------------------------------------------------------------------------


def test_prd_latency_target_is_20ms(bench) -> None:
    assert bench._PRD_LATENCY_TARGET_MS == pytest.approx(20.0)
