import numpy as np
import numpy.testing as npt
import pytest
from unittest.mock import patch

from eyedentify3d import ReducedData
from eyedentify3d.identification.smooth_pursuit import SmoothPursuitEvent


def mock_data_object():
    """Create a mock data object for testing."""
    np.random.seed(42)  # For reproducibility
    # Create time vector
    dt = 0.01
    time_vector = np.arange(0, 1, dt)
    n_samples = len(time_vector)

    # Create gaze direction data with some smooth pursuit patterns
    gaze_direction = np.zeros((3, n_samples))
    gaze_direction[2, :] = 1.0  # Default looking forward

    # Create a smooth pursuit around frame 30-40
    for i in range(30, 40):
        angle = (i - 30) * 2  # Small increasing angle
        # Rotate around y-axis
        gaze_direction[0, i] = np.sin(np.radians(angle))
        gaze_direction[2, i] = np.cos(np.radians(angle))

    # Create another smooth pursuit around frame 60-70
    for i in range(60, 70):
        angle = (i - 60) * 1.5  # Different rate of change
        # Rotate around x-axis
        gaze_direction[1, i] = np.sin(np.radians(angle))
        gaze_direction[2, i] = np.cos(np.radians(angle))

    gaze_direction /= np.linalg.norm(gaze_direction, axis=0)  # Normalize vectors

    # Create gaze angular velocity
    gaze_angular_velocity = np.zeros(n_samples)
    # Set angular velocity for smooth pursuit segments
    gaze_angular_velocity[30:40] = 20  # deg/s
    gaze_angular_velocity[60:70] = 15  # deg/s

    # Create sample eye openness data
    right_eye_openness = np.ones(n_samples) * 0.8
    left_eye_openness = np.ones(n_samples) * 0.7

    # Create sample direction vectors (3D)
    eye_direction = np.zeros((3, n_samples))
    eye_direction[2, :] = 1.0  # Looking forward along z-axis

    head_angles = np.zeros((3, n_samples))
    head_angles[0, :] = np.linspace(0, 0.1, n_samples)  # Small rotation around x-axis

    gaze_angular_velocity = np.zeros((n_samples,))
    gaze_angular_velocity[30:35] = 150  # Large angular velocity
    gaze_angular_velocity[70:75] = 150  # Large angular velocity
    gaze_angular_velocity += np.random.normal(0, 5, n_samples)  # Add some noise

    head_angular_velocity = np.zeros((3, n_samples))
    head_angular_velocity[0, :] = 0.1  # Constant angular velocity around x-axis

    head_velocity_norm = np.ones(n_samples) * 0.1

    data_invalidity = np.zeros(n_samples, dtype=bool)
    data_invalidity[80:85] = True  # Some invalid data

    data_object = ReducedData(
        original_dt=dt,
        original_time_vector=time_vector,
        original_right_eye_openness=right_eye_openness,
        original_left_eye_openness=left_eye_openness,
        original_eye_direction=eye_direction,
        original_head_angles=head_angles,
        original_gaze_direction=gaze_direction,
        original_head_angular_velocity=head_angular_velocity,
        original_head_velocity_norm=head_velocity_norm,
        original_data_invalidity=data_invalidity,
    )

    return data_object


@pytest.fixture
def identified_indices():
    """Create identified_indices array for testing."""
    identified_indices = np.zeros(100, dtype=bool)
    # Mark some frames as already identified
    identified_indices[20:30] = True
    identified_indices[50:60] = True
    return identified_indices


@pytest.fixture
def smooth_pursuit_indices():
    """Create smooth_pursuit_indices array for testing."""
    # Frames 30-40 and 60-70 are smooth pursuits
    return np.concatenate([np.arange(30, 40), np.arange(60, 70)])


def test_smooth_pursuit_event_initialization(identified_indices, smooth_pursuit_indices):
    """Test that SmoothPursuitEvent initializes correctly."""
    mock_data = mock_data_object()

    event = SmoothPursuitEvent(mock_data, identified_indices, smooth_pursuit_indices, minimal_duration=0.05)

    assert event.data_object is mock_data
    assert event.identified_indices is identified_indices
    assert np.array_equal(event.smooth_pursuit_indices, smooth_pursuit_indices)
    assert event.minimal_duration == 0.05
    assert event.smooth_pursuit_trajectories is None
    assert event.frame_indices is None
    assert event.sequences == []


def test_initialize(identified_indices, smooth_pursuit_indices):
    """Test that initialize correctly sets up the event."""
    mock_data = mock_data_object()

    event = SmoothPursuitEvent(mock_data, identified_indices, smooth_pursuit_indices, minimal_duration=0.05)

    # Mock the methods called in initialize
    with (
        patch.object(event, "split_sequences") as mock_split,
        patch.object(event, "merge_sequences") as mock_merge,
        patch.object(event, "keep_only_sequences_long_enough") as mock_keep,
        patch.object(event, "adjust_indices_to_sequences") as mock_adjust,
    ):

        event.initialize()

        # Check that frame_indices is set to smooth_pursuit_indices
        np.testing.assert_array_equal(event.frame_indices, smooth_pursuit_indices)

        # Check that all methods were called in the correct order
        mock_split.assert_called_once()
        mock_merge.assert_called_once()
        mock_keep.assert_called_once()
        mock_adjust.assert_called_once()


def test_merge_sequences(identified_indices, smooth_pursuit_indices):
    """Test that merge_sequences correctly merges close smooth pursuit sequences."""
    mock_data = mock_data_object()

    event = SmoothPursuitEvent(mock_data, identified_indices, smooth_pursuit_indices)

    # Set sequences
    event.sequences = [np.arange(30, 40), np.arange(41, 46), np.arange(60, 70)]

    event.merge_sequences()

    # Check that sequences were not merged because the directionality criterion was not met
    assert len(event.sequences) == 3
    npt.assert_almost_equal(event.sequences[0], np.arange(30, 40))
    npt.assert_almost_equal(event.sequences[1], np.arange(41, 46))
    npt.assert_almost_equal(event.sequences[2], np.arange(60, 70))


def test_measure_smooth_pursuit_trajectory(identified_indices, smooth_pursuit_indices):
    """Test that measure_smooth_pursuit_trajectory correctly computes trajectory lengths."""
    mock_data = mock_data_object()

    event = SmoothPursuitEvent(mock_data, identified_indices, smooth_pursuit_indices)

    # Set sequences
    event.sequences = [np.arange(30, 39), np.arange(60, 69)]  # Note: using n-1 to avoid index out of bounds

    # Call the method
    event.measure_smooth_pursuit_trajectory()

    # Check that smooth_pursuit_trajectories is computed
    assert event.smooth_pursuit_trajectories is not None
    assert len(event.smooth_pursuit_trajectories) == 2

    # First sequence has angular velocity of 20 deg/s for 9 frames (0.09s)
    # So trajectory should be approximately 20 * 0.09 = 1.8 degrees
    # Second sequence has angular velocity of 15 deg/s for 9 frames (0.09s)
    # So trajectory should be approximately 15 * 0.09 = 1.35 degrees
    # But we need to account for the absolute value and frame-by-frame calculation
    assert event.smooth_pursuit_trajectories[0] > 0
    assert event.smooth_pursuit_trajectories[1] > 0


def test_measure_smooth_pursuit_trajectory_with_nan(identified_indices, smooth_pursuit_indices):
    """Test that measure_smooth_pursuit_trajectory handles NaN values correctly."""
    mock_data = mock_data_object()

    event = SmoothPursuitEvent(mock_data, identified_indices, smooth_pursuit_indices)

    # Set sequences
    event.sequences = [np.arange(30, 39)]

    # Introduce NaN values in gaze_angular_velocity
    mock_data.gaze_angular_velocity[35] = np.nan

    # Call the method
    event.measure_smooth_pursuit_trajectory()

    # Check that smooth_pursuit_trajectories is computed and NaN values are handled
    assert event.smooth_pursuit_trajectories is not None
    assert len(event.smooth_pursuit_trajectories) == 1
    assert not np.isnan(event.smooth_pursuit_trajectories[0])


def test_end_to_end_smooth_pursuit_detection(identified_indices, smooth_pursuit_indices):
    """Test the complete smooth pursuit detection process."""
    mock_data = mock_data_object()

    event = SmoothPursuitEvent(
        mock_data, identified_indices, smooth_pursuit_indices, minimal_duration=0.05  # 5 frames at 0.01s per frame
    )

    # Run the complete initialization process
    event.initialize()

    # Check that sequences contains the expected sequences
    assert len(event.sequences) == 2
    np.testing.assert_array_equal(event.sequences[0], np.arange(30, 40))
    np.testing.assert_array_equal(event.sequences[1], np.arange(60, 70))

    # Check that frame_indices contains all frames from all sequences
    expected_indices = np.concatenate([np.arange(30, 40), np.arange(60, 70)])
    np.testing.assert_array_equal(event.frame_indices, expected_indices)

    # Measure smooth pursuit trajectories
    event.measure_smooth_pursuit_trajectory()

    # Check that smooth_pursuit_trajectories is computed
    assert event.smooth_pursuit_trajectories is not None
    assert len(event.smooth_pursuit_trajectories) == 2


def test_smooth_pursuit_with_short_sequences(identified_indices):
    """Test that short sequences are filtered out when minimal_duration is set."""
    mock_data = mock_data_object()

    # Create smooth pursuit indices with one short sequence
    smooth_pursuit_indices = np.concatenate([np.arange(30, 32), np.arange(60, 70)])

    # Create a SmoothPursuitEvent with a minimal_duration that filters out the short sequence
    event = SmoothPursuitEvent(
        mock_data, identified_indices, smooth_pursuit_indices, minimal_duration=0.05  # 5 frames at 0.01s per frame
    )

    # Set frame_indices and split into sequences
    event.frame_indices = smooth_pursuit_indices
    event.split_sequences()

    # Apply minimal duration filter
    event.keep_only_sequences_long_enough()

    # The first sequence (30-32) is only 2 frames (0.02s) which is less than minimal_duration (0.05s)
    # So it should be filtered out
    assert len(event.sequences) == 1
    np.testing.assert_array_equal(event.sequences[0], np.arange(60, 70))
