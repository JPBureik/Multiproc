"""Shared test fixtures for multiproc tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


@pytest.fixture
def small_list() -> list[int]:
    """Small list for basic tests."""
    return [1, 2, 3, 4, 5]


@pytest.fixture
def medium_list() -> list[int]:
    """Medium list for more thorough tests."""
    return list(range(100))


@pytest.fixture
def numpy_array() -> np.ndarray[Any, np.dtype[np.int64]]:
    """NumPy array input for testing array handling."""
    return np.arange(50)


def identity(x: Any) -> Any:
    """Identity function for testing."""
    return x


def square(x: int) -> int:
    """Square function for testing."""
    return x * x


def add_constant(x: int, constant: int) -> int:
    """Function with additional argument for testing *args."""
    return x + constant


def multiply_kwargs(x: int, *, multiplier: int = 1) -> int:
    """Function with keyword argument for testing **kwargs."""
    return x * multiplier
