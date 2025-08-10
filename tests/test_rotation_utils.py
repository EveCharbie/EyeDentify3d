import numpy as np
import numpy.testing as npt
import pytest

from eyedentify3d.utils.rotation_utils import (
    unwrap_rotation,
    rotation_matrix_from_euler_angles,
    get_angle_between_vectors,
    get_gaze_direction,
    rot_x_matrix,
    rot_y_matrix,
    rot_z_matrix,
    compute_angular_velocity,
)


def test_unwrap_rotation_no_jumps():
    """Test that unwrap_rotation doesn't change angles without jumps."""
    angles = np.zeros((3, 10))
    for i in range(3):
        angles[i, :] = np.linspace(0, 350, 10)

    unwrapped = unwrap_rotation(angles)
    assert unwrapped.shape == angles.shape
    npt.assert_almost_equal(unwrapped, angles)


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
    npt.assert_almost_equal(unwrapped[0, 5:], np.array([370, 370, 370, 370, 370]))

    # Second component: [10, 10, 10, 10, 10, -10, -10, -10, -10, -10]
    npt.assert_almost_equal(unwrapped[1, 5:], np.array([-10, -10, -10, -10, -10]))

    # Third component should remain unchanged
    npt.assert_almost_equal(unwrapped[2, :], np.array([180, 180, 180, 180, 180, 180, 180, 180, 180, 180]))


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
    npt.assert_almost_equal(unwrapped[0, :3], np.array([350.0, 350.0, 350.0]))
    npt.assert_almost_equal(unwrapped[0, 3:7], np.array([370.0, 370.0, 370.0, 370.0]))
    npt.assert_almost_equal(unwrapped[0, 7:11], np.array([350.0, 350.0, 350.0, 350.0]))
    npt.assert_almost_equal(unwrapped[0, 11:], np.array([370.0, 370.0, 370.0, 370.0]))


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


def test_rotation_matrices():
    """Test the individual rotation matrices."""
    # Test rotation around x-axis
    angle = np.pi / 2  # 90 degrees
    rot_x = rot_x_matrix(angle)
    v = np.array([0, 1, 0])  # Unit vector along y-axis
    rotated = rot_x @ v
    npt.assert_almost_equal(rotated, np.array([0, 0, 1]))  # Should rotate to z-axis

    # Test rotation around y-axis
    angle = np.pi / 2  # 90 degrees
    rot_y = rot_y_matrix(angle)
    v = np.array([0, 0, 1])  # Unit vector along z-axis
    rotated = rot_y @ v
    npt.assert_almost_equal(rotated, np.array([1, 0, 0]))  # Should rotate to x-axis

    # Test rotation around z-axis
    angle = np.pi / 2  # 90 degrees
    rot_z = rot_z_matrix(angle)
    v = np.array([1, 0, 0])  # Unit vector along x-axis
    rotated = rot_z @ v
    npt.assert_almost_equal(rotated, np.array([0, 1, 0]))  # Should rotate to y-axis


def test_rotation_matrix_from_euler_angles():
    """Test creating rotation matrix from Euler angles."""
    # Test with a simple rotation sequence
    angles = np.array([np.pi / 2, 0, 0])  # 90 degrees around x-axis
    rot_matrix = rotation_matrix_from_euler_angles("xyz", angles)

    # Should be equivalent to just the x rotation matrix
    expected = rot_x_matrix(np.pi / 2)
    npt.assert_almost_equal(rot_matrix, expected)

    # Test with a more complex rotation sequence
    angles = np.array([np.pi / 4, np.pi / 4, np.pi / 4])  # 45 degrees around each axis
    rot_matrix = rotation_matrix_from_euler_angles("xyz", angles)

    # Apply to a vector and check result
    v = np.array([1, 0, 0])
    rotated = rot_matrix @ v
    # The exact values depend on the order of rotations, but we can check it's a unit vector
    npt.assert_almost_equal(np.linalg.norm(rotated), 1.0)


def test_rotation_matrix_from_euler_angles_errors():
    """Test error cases for rotation_matrix_from_euler_angles."""
    # Test with wrong shape of angles
    angles = np.array([[np.pi / 2, 0, 0], [0, np.pi / 2, 0]])  # 2D array
    with pytest.raises(ValueError, match="The angles should be of shape"):
        rotation_matrix_from_euler_angles("xyz", angles)

    # Test with mismatched sequence length and angles
    angles = np.array([np.pi / 2, 0])  # Only 2 angles
    with pytest.raises(ValueError, match="The number of angles and the length of the angle_sequence must match"):
        rotation_matrix_from_euler_angles("xyz", angles)


def test_get_angle_between_vectors():
    """Test get_angle_between_vectors function."""
    # Test with parallel vectors
    v1 = np.array([1, 0, 0])
    v2 = np.array([2, 0, 0])  # Same direction, different magnitude
    angle = get_angle_between_vectors(v1, v2)
    npt.assert_almost_equal(angle, 0)

    # Test with perpendicular vectors
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    angle = get_angle_between_vectors(v1, v2)
    npt.assert_almost_equal(angle, 90)

    # Test with opposite vectors
    v1 = np.array([1, 0, 0])
    v2 = np.array([-1, 0, 0])
    angle = get_angle_between_vectors(v1, v2)
    npt.assert_almost_equal(angle, 180)

    # Test with identical vectors
    v1 = np.array([1, 2, 3])
    v2 = np.array([1, 2, 3])
    angle = get_angle_between_vectors(v1, v2)
    npt.assert_almost_equal(angle, 0)


def test_get_angle_between_vectors_errors():
    """Test error cases for get_angle_between_vectors."""
    # Test with wrong shape vectors
    v1 = np.array([1, 0, 0, 0])  # 4D vector
    v2 = np.array([0, 1, 0])
    with pytest.raises(ValueError, match="Both vectors must be of shape"):
        get_angle_between_vectors(v1, v2)

    # Test with zero vector
    v1 = np.array([0, 0, 0])  # Zero vector
    v2 = np.array([0, 1, 0])
    with pytest.raises(
        RuntimeError, match="The gaze vectors should be unitary. This should not happen, please contact the developer."
    ):
        get_angle_between_vectors(v1, v2)


def test_get_gaze_direction():
    """Test get_gaze_direction function."""
    # Create test data
    n_frames = 5
    head_angles = np.zeros((3, n_frames))
    eye_direction = np.zeros((3, n_frames))

    # Set up head angles (in degrees)
    head_angles[0, :] = np.linspace(0, 90, n_frames)  # Rotation around x-axis

    # Set up eye direction (unit vectors in head reference frame)
    for i in range(n_frames):
        eye_direction[:, i] = [0, 0, 1]  # Looking straight ahead in head reference frame

    # Calculate gaze direction
    gaze_direction = get_gaze_direction(head_angles, eye_direction)

    # Check shape
    assert gaze_direction.shape == (3, n_frames)

    # Check that all vectors are unit vectors
    for i in range(n_frames):
        npt.assert_almost_equal(np.linalg.norm(gaze_direction[:, i]), 1.0)

    # For 0 degree head rotation, gaze should match eye direction
    npt.assert_almost_equal(gaze_direction[:, 0], eye_direction[:, 0])

    npt.assert_almost_equal(
        gaze_direction,
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, -3.82683432e-01, -7.07106781e-01, -9.23879533e-01, -1.00000000e00],
                [1.00000000e00, 9.23879533e-01, 7.07106781e-01, 3.82683432e-01, 0],
            ]
        ),
    )


def test_compute_angular_velocity():
    """Test compute_angular_velocity function."""
    # Create a time vector and eye direction
    time_vector = np.linspace(0, 1, 10)  # 5 frames, 1 second apart
    eye_direction = np.zeros((3, 10))
    eye_direction[0, :] = np.linspace(0, 1, 10)  # x component changes linearly
    eye_direction[1, :] = np.linspace(0, 0.05, 10)  # y component changes linearly
    eye_direction[2, :] = np.linspace(1, 0, 10)  # z component changes linearly

    # Compute angular velocity
    angular_velocity = compute_angular_velocity(time_vector, eye_direction)

    # Check shape
    assert angular_velocity.shape == (len(time_vector),)

    # Check that the velocity increases and decreases as expected (and that the values are expressed in degrees per second)
    npt.assert_almost_equal(
        angular_velocity,
        np.array(
            [
                64.20442871,
                71.83935327,
                87.56931878,
                102.28682319,
                111.51632601,
                111.45979964,
                102.14284089,
                87.39104318,
                71.66929997,
                64.04596144,
            ]
        ),
    )

    # Check errors
    with pytest.raises(ValueError, match="The direction vector should be a 3D vector."):
        compute_angular_velocity(time_vector, eye_direction[:1, :])  # Not enough components in eye direction

    with pytest.raises(
        ValueError, match="The time vector should have the same number of frames as the direction vector."
    ):
        compute_angular_velocity(time_vector, eye_direction[:, :-2])  # Not the same number of frames

    with pytest.raises(
        ValueError, match="The time vector should have the same number of frames as the direction vector."
    ):
        compute_angular_velocity(time_vector[:-2], eye_direction)  # Not the same number of frames

    with pytest.raises(ValueError, match="The time vector should have at least 3 frames to compute angular velocity."):
        compute_angular_velocity(time_vector[:1], eye_direction[:, :1])  # Not enough frames in time vector
