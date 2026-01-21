"""Integration tests for multiproc package."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from multiproc import multiproc_cpu


# Module-level functions for pickling compatibility (required for macOS spawn)
def _cpu_intensive(n: int) -> int:
    """Simple CPU-bound task."""
    total = 0
    for i in range(n):
        total += i * i
    return total


def _process_dict(d: dict[str, int]) -> int:
    return sum(d.values())


def _reverse_string(s: str) -> str:
    return s[::-1]


def _double(x: int) -> int:
    return x * 2


def _simple_transform(x: int) -> int:
    return x * 2 + 1


def _array_mean(arr: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    return float(np.mean(arr))


def _classify(x: int) -> str | int:
    if x % 2 == 0:
        return "even"
    return x


def _handle_none(x: int | None) -> int:
    return 0 if x is None else x * 2


def _precise_calc(x: float) -> float:
    return x * 1.123456789


def _return_none(x: int) -> None:
    return None


class TestIntegration:
    """End-to-end integration tests."""

    def test_cpu_bound_task(self) -> None:
        """Test with a CPU-bound computation."""
        data = [1000, 2000, 3000, 4000, 5000]
        result = multiproc_cpu(data, _cpu_intensive)
        expected = [_cpu_intensive(x) for x in data]
        assert result == expected

    def test_with_object_types(self) -> None:
        """Test processing of complex object types."""
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}]
        result = multiproc_cpu(data, _process_dict)
        assert result == [3, 7, 11]

    def test_with_string_processing(self) -> None:
        """Test string processing tasks."""
        data = ["hello", "world", "test", "multiproc"]
        result = multiproc_cpu(data, _reverse_string)
        expected = ["olleh", "dlrow", "tset", "corpitlum"]
        assert result == expected

    def test_with_tuple_input(self) -> None:
        """Test with tuple input."""
        data = (1, 2, 3, 4, 5)
        result = multiproc_cpu(data, _double)
        assert result == [2, 4, 6, 8, 10]

    def test_large_dataset(self) -> None:
        """Test with a larger dataset."""
        data = list(range(1000))
        result = multiproc_cpu(data, _simple_transform)
        expected = [x * 2 + 1 for x in data]
        assert result == expected

    def test_numpy_operations(self) -> None:
        """Test with numpy-based operations."""
        data = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
        result = multiproc_cpu(data, _array_mean)
        assert result == [2.0, 5.0]

    def test_mixed_return_types(self) -> None:
        """Test function returning different types based on input."""
        data = [1, 2, 3, 4, 5]
        result = multiproc_cpu(data, _classify)
        assert result == [1, "even", 3, "even", 5]

    def test_with_none_values(self) -> None:
        """Test processing data containing None."""
        data: list[int | None] = [1, None, 3, None, 5]
        result = multiproc_cpu(data, _handle_none)
        assert result == [2, 0, 6, 0, 10]

    def test_preserves_floating_point_precision(self) -> None:
        """Test that floating point precision is preserved."""
        data = [1.0, 2.0, 3.0]
        result = multiproc_cpu(data, _precise_calc)
        expected = [x * 1.123456789 for x in data]
        for r, e in zip(result, expected, strict=False):
            assert abs(r - e) < 1e-10


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_invalid_free_cores(self) -> None:
        """Test that invalid free_cores raises ValueError."""
        with pytest.raises(ValueError):
            multiproc_cpu([1, 2, 3], _double, free_cores=1000)

    def test_function_returning_none(self) -> None:
        """Test that functions returning None work correctly."""
        data = [1, 2, 3]
        result = multiproc_cpu(data, _return_none)
        assert result == [None, None, None]


class TestImportAndAPI:
    """Tests for package import and API surface."""

    def test_import_from_package(self) -> None:
        """Test that multiproc_cpu can be imported from package root."""
        from multiproc import multiproc_cpu as mp

        assert callable(mp)

    def test_version_available(self) -> None:
        """Test that version is accessible."""
        from multiproc import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_all_exports(self) -> None:
        """Test that __all__ contains expected exports."""
        from multiproc import __all__

        assert "multiproc_cpu" in __all__
