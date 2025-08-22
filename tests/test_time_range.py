import pytest
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


def test_time_range_with_min_greater_than_max():
    """Test time_range when min_time is greater than max_time."""
    with pytest.raises(ValueError, match="The min_time must be less than or equal to the max_time."):
        time_range = TimeRange(min_time=5.0, max_time=3.0)


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


def test_get_indices_with_zero_min():
    """Test get_indices with default min_time."""
    time_range = TimeRange(max_time=10.0)  # Default min_time is zero
    time_vector = np.array([1.0, 2.0, 3.0, 100.0, 1000.0])
    indices = time_range.get_indices(time_vector)
    np.testing.assert_array_equal(indices, np.array([0, 1, 2]))


def test_get_indices_with_nan_values():
    """Test get_indices with NaN values in the time vector."""
    time_range = TimeRange(min_time=1.0, max_time=5.0)
    time_vector = np.array([0.5, 1.5, np.nan, 3.0, 4.5, 5.5])
    indices = time_range.get_indices(time_vector)
    np.testing.assert_array_equal(indices, np.array([1, 2, 3, 4]))


def test_get_indices_with_unsorted_time_vector():
    """Test get_indices with an unsorted time vector."""
    time_range = TimeRange(min_time=2.0, max_time=6.0)
    time_vector = np.array([5.0, 1.0, 3.0, 7.0, 2.0, 4.0])
    with pytest.raises(ValueError, match="The time vector must be strictly increasing."):
        indices = time_range.get_indices(time_vector)


def test_get_indices_with_duplicate_times():
    """Test get_indices with duplicate time values."""
    time_range = TimeRange(min_time=2.0, max_time=4.0)
    time_vector = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0])
    indices = time_range.get_indices(time_vector)
    np.testing.assert_array_equal(indices, np.array([1, 2, 3, 4, 5, 6]))


def test_get_indices_with_negative_times():
    """Test get_indices with negative time values."""
    time_range = TimeRange(min_time=-3.0, max_time=0.0)
    time_vector = np.array([-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0])
    indices = time_range.get_indices(time_vector)
    np.testing.assert_array_equal(indices, np.array([2, 3, 4, 5]))


def test_get_indices_with_min_equals_max():
    """Test get_indices when min_time equals max_time."""
    time_range = TimeRange(min_time=3.0, max_time=3.0)
    time_vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    indices = time_range.get_indices(time_vector)
    np.testing.assert_array_equal(indices, np.array([2.0]))
