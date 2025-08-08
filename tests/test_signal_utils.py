import numpy as np
import pytest
from eyedentify3d.utils.signal_utils import centered_finite_difference, filter_data


def test_centered_finite_difference_constant_data():
    """Test that centered_finite_difference returns zero velocity for constant data."""
    time_vector = np.linspace(0, 1, 10)
    data = np.ones((3, 10))
    velocity = centered_finite_difference(time_vector, data)
    assert velocity.shape == (3, 10)
    assert np.allclose(velocity, 0)


def test_centered_finite_difference_linear_data():
    """Test that centered_finite_difference returns constant velocity for linear data."""
    time_vector = np.linspace(0, 1, 10)
    data = np.zeros((3, 10))
    for i in range(3):
        data[i, :] = time_vector * (i + 1)  # Different slope for each component
    
    velocity = centered_finite_difference(time_vector, data)
    assert velocity.shape == (3, 10)
    
    # Check that velocity is constant and equal to the slope
    for i in range(3):
        assert np.allclose(velocity[i, 1:-1], i + 1)


def test_centered_finite_difference_endpoints():
    """Test that centered_finite_difference handles endpoints correctly."""
    time_vector = np.linspace(0, 1, 5)
    data = np.zeros((3, 5))
    for i in range(3):
        data[i, :] = time_vector * (i + 1)  # Different slope for each component
    
    velocity = centered_finite_difference(time_vector, data)
    
    # Check endpoints
    dt = time_vector[1] - time_vector[0]
    for i in range(3):
        assert np.isclose(velocity[i, 0], (data[i, 1] - data[i, 0]) / dt)
        assert np.isclose(velocity[i, -1], (data[i, -1] - data[i, -2]) / dt)


def test_filter_data_shape():
    """Test that filter_data preserves the shape of the input data."""
    data = np.random.rand(3, 100)
    filtered_data = filter_data(data)
    assert filtered_data.shape == data.shape


def test_filter_data_constant():
    """Test that filter_data preserves constant signals."""
    data = np.ones((3, 100))
    filtered_data = filter_data(data)
    assert np.allclose(filtered_data, data)


def test_filter_data_parameters():
    """Test that filter_data works with different parameters."""
    data = np.random.rand(3, 100)
    
    # Test with different cutoff frequencies
    filtered_data_low = filter_data(data, cutoff_freq=0.1)
    filtered_data_high = filter_data(data, cutoff_freq=0.5)
    
    # Lower cutoff should result in smoother data (less variance)
    assert np.var(filtered_data_low) < np.var(filtered_data_high)
    
    # Test with different orders
    filtered_data_low_order = filter_data(data, order=4)
    filtered_data_high_order = filter_data(data, order=8)
    
    # Both should have the same shape
    assert filtered_data_low_order.shape == filtered_data_high_order.shape
