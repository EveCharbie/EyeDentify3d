import unittest
import numpy as np
from eyedentify3d.time_range import TimeRange


def test_time_range_init_default():
    """Test TimeRange initialization with default values."""
    time_range = TimeRange()
    assert time_range.min_time == 0
    assert time_range.max_time == float("inf")


def test_time_range_init_custom():
    """Test TimeRange initialization with custom values."""
    time_range = TimeRange(min_time=1.5, max_time=10.0)
    assert time_range.min_time == 1.5
    assert time_range.max_time == 10.0


def test_get_indices_all_in_range():
    """Test get_indices when all values are in range."""
    time_range = TimeRange(min_time=0.0, max_time=5.0)
    time_vector = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    indices = time_range.get_indices(time_vector)
    np.testing.assert_array_equal(indices, np.array([0, 1, 2, 3, 4, 5]))


def test_get_indices_some_in_range():
    """Test get_indices when only some values are in range."""
    time_range = TimeRange(min_time=2.0, max_time=4.0)
    time_vector = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    indices = time_range.get_indices(time_vector)
    np.testing.assert_array_equal(indices, np.array([2, 3, 4]))


def test_get_indices_none_in_range():
    """Test get_indices when no values are in range."""
    time_range = TimeRange(min_time=10.0, max_time=20.0)
    time_vector = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    indices = time_range.get_indices(time_vector)
    assert len(indices) == 0


def test_get_indices_empty_time_vector():
    """Test get_indices with an empty time vector."""
    time_range = TimeRange(min_time=0.0, max_time=5.0)
    time_vector = np.array([])
    indices = time_range.get_indices(time_vector)
    assert len(indices) == 0


def test_get_indices_boundary_conditions():
    """Test get_indices with values exactly at the boundaries."""
    time_range = TimeRange(min_time=1.0, max_time=5.0)
    time_vector = np.array([0.9, 1.0, 5.0, 5.1])
    indices = time_range.get_indices(time_vector)
    np.testing.assert_array_equal(indices, np.array([1, 2]))


def test_get_indices_with_inf_max():
    """Test get_indices with infinite max_time."""
    time_range = TimeRange(min_time=2.0)  # Default max_time is inf
    time_vector = np.array([1.0, 2.0, 3.0, 100.0, 1000.0])
    indices = time_range.get_indices(time_vector)
    np.testing.assert_array_equal(indices, np.array([1, 2, 3, 4]))
