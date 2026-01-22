"""Tests for multiproc_cpu module."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from mpviz import multiproc_cpu
from mpviz.multiproc_cpu import _parallel_split, _set_nb_of_workers

from .conftest import add_constant, identity, multiply_kwargs, square


# Module-level function for pickling compatibility (required for macOS spawn)
def _combined(x: int, add: int, *, mult: int = 1) -> int:
    return (x + add) * mult


class TestMultiprocCpu:
    """Tests for the main multiproc_cpu function."""

    def test_empty_list(self) -> None:
        """Test with empty input returns empty list."""
        result = multiproc_cpu([], identity)
        assert result == []

    def test_single_element(self) -> None:
        """Test with single element."""
        result = multiproc_cpu([42], identity)
        assert result == [42]

    def test_preserves_order(self, small_list: list[int]) -> None:
        """Test that results are returned in original order."""
        result = multiproc_cpu(small_list, identity)
        assert result == small_list

    def test_applies_function(self, small_list: list[int]) -> None:
        """Test that function is correctly applied to each element."""
        result = multiproc_cpu(small_list, square)
        expected = [x * x for x in small_list]
        assert result == expected

    def test_with_numpy_array(
        self, numpy_array: np.ndarray[Any, np.dtype[np.int64]]
    ) -> None:
        """Test that numpy arrays are handled correctly."""
        result = multiproc_cpu(numpy_array, square)
        expected = [int(x * x) for x in numpy_array]
        assert result == expected

    def test_with_args(self, small_list: list[int]) -> None:
        """Test passing positional arguments to the function."""
        result = multiproc_cpu(small_list, add_constant, 10)
        expected = [x + 10 for x in small_list]
        assert result == expected

    def test_with_kwargs(self, small_list: list[int]) -> None:
        """Test passing keyword arguments to the function."""
        result = multiproc_cpu(small_list, multiply_kwargs, multiplier=3)
        expected = [x * 3 for x in small_list]
        assert result == expected

    def test_with_args_and_kwargs(self, small_list: list[int]) -> None:
        """Test passing both positional and keyword arguments."""
        result = multiproc_cpu(small_list, _combined, 5, mult=2)
        expected = [(x + 5) * 2 for x in small_list]
        assert result == expected

    def test_free_cores_parameter(self, small_list: list[int]) -> None:
        """Test that free_cores parameter works."""
        result = multiproc_cpu(small_list, square, free_cores=1)
        expected = [x * x for x in small_list]
        assert result == expected

    def test_custom_desc_and_unit(self, small_list: list[int]) -> None:
        """Test custom progress bar description and unit."""
        result = multiproc_cpu(small_list, square, desc="Squaring", unit="numbers")
        expected = [x * x for x in small_list]
        assert result == expected

    def test_medium_list(self, medium_list: list[int]) -> None:
        """Test with a larger list to exercise multiple workers."""
        result = multiproc_cpu(medium_list, square)
        expected = [x * x for x in medium_list]
        assert result == expected


class TestSetNbOfWorkers:
    """Tests for the _set_nb_of_workers helper function."""

    def test_none_uses_all_cores(self) -> None:
        """Test that None uses all available cores."""
        with patch("mpviz.multiproc_cpu.psutil.cpu_count", return_value=8):
            result = _set_nb_of_workers(None)
            assert result == 8

    def test_free_cores_subtracted(self) -> None:
        """Test that free_cores are subtracted from total."""
        with patch("mpviz.multiproc_cpu.psutil.cpu_count", return_value=8):
            result = _set_nb_of_workers(2)
            assert result == 6

    def test_free_cores_equal_to_available_raises(self) -> None:
        """Test that free_cores >= available raises ValueError."""
        with (
            patch("mpviz.multiproc_cpu.psutil.cpu_count", return_value=4),
            pytest.raises(ValueError, match="must be less than"),
        ):
            _set_nb_of_workers(4)

    def test_free_cores_greater_than_available_raises(self) -> None:
        """Test that free_cores > available raises ValueError."""
        with (
            patch("mpviz.multiproc_cpu.psutil.cpu_count", return_value=4),
            pytest.raises(ValueError, match="must be less than"),
        ):
            _set_nb_of_workers(10)

    def test_handles_none_cpu_count(self) -> None:
        """Test graceful handling when cpu_count returns None."""
        with patch("mpviz.multiproc_cpu.psutil.cpu_count", return_value=None):
            result = _set_nb_of_workers(None)
            assert result == 1


class TestParallelSplit:
    """Tests for the _parallel_split helper function."""

    def test_splits_list_evenly(self) -> None:
        """Test even splitting of a list."""
        result = _parallel_split([1, 2, 3, 4], 2)
        assert len(result) == 2
        assert list(result[0]) == [1, 2]
        assert list(result[1]) == [3, 4]

    def test_splits_list_unevenly(self) -> None:
        """Test uneven splitting when not perfectly divisible."""
        result = _parallel_split([1, 2, 3, 4, 5], 2)
        assert len(result) == 2
        assert list(result[0]) == [1, 2, 3]
        assert list(result[1]) == [4, 5]

    def test_handles_numpy_array(self) -> None:
        """Test splitting of numpy arrays."""
        arr = np.array([1, 2, 3, 4])
        result = _parallel_split(arr, 2)
        assert len(result) == 2
        assert list(result[0]) == [1, 2]
        assert list(result[1]) == [3, 4]

    def test_more_workers_than_items(self) -> None:
        """Test when there are more workers than items."""
        result = _parallel_split([1, 2], 4)
        assert len(result) == 4
        non_empty = [r for r in result if len(r) > 0]
        assert len(non_empty) == 2
