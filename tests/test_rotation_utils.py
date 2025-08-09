import numpy as np
import pytest
from eyedentify3d.utils.rotation_utils import unwrap_rotation


def test_unwrap_rotation_no_jumps():
    """Test that unwrap_rotation doesn't change angles without jumps."""
    angles = np.zeros((3, 10))
    for i in range(3):
        angles[i, :] = np.linspace(0, 350, 10)

    unwrapped = unwrap_rotation(angles)
    assert unwrapped.shape == angles.shape
    assert np.allclose(unwrapped, angles)


def test_unwrap_rotation_with_jumps():
    """Test that unwrap_rotation correctly unwraps angles with jumps."""
    angles = np.zeros((3, 10))

    # Create a jump from 350 to 10 degrees (which should be unwrapped to 370)
    angles[0, :5] = 350
    angles[0, 5:] = 10

    # Create a jump from 10 to 350 degrees (which should be unwrapped to -10)
    angles[1, :5] = 10
    angles[1, 5:] = 350

    # No jumps in the third component
    angles[2, :] = 180

    unwrapped = unwrap_rotation(angles)

    # First component: [350, 350, 350, 350, 350, 370, 370, 370, 370, 370]
    assert np.allclose(unwrapped[0, 5:], 370)

    # Second component: [10, 10, 10, 10, 10, -10, -10, -10, -10, -10]
    assert np.allclose(unwrapped[1, 5:], -10)

    # Third component should remain unchanged
    assert np.allclose(unwrapped[2, :], 180)


def test_unwrap_rotation_multiple_jumps():
    """Test that unwrap_rotation correctly handles multiple jumps."""
    angles = np.zeros((3, 15))

    # Create multiple jumps: 350 -> 10 -> 350 -> 10
    angles[0, :3] = 350
    angles[0, 3:7] = 10
    angles[0, 7:11] = 350
    angles[0, 11:] = 10

    unwrapped = unwrap_rotation(angles)

    # Expected: [350, 350, 350, 370, 370, 370, 370, 710, 710, 710, 710, 730, 730, 730, 730]
    assert np.allclose(unwrapped[0, 3:7], 370)
    assert np.allclose(unwrapped[0, 7:11], 710)
    assert np.allclose(unwrapped[0, 11:], 730)


def test_unwrap_rotation_edge_cases():
    """Test unwrap_rotation with edge cases."""
    # Empty array
    angles = np.zeros((3, 0))
    unwrapped = unwrap_rotation(angles)
    assert unwrapped.shape == (3, 0)

    # Single value (no unwrapping possible)
    angles = np.array([[10], [20], [30]])
    unwrapped = unwrap_rotation(angles)
    assert np.array_equal(unwrapped, angles)
