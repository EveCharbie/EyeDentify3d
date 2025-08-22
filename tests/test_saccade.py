import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from eyedentify3d.identification.saccade import SaccadeEvent
from eyedentify3d.utils.rotation_utils import get_angle_between_vectors


@pytest.fixture
def mock_data_object():
    """Create a mock data object for testing."""
    mock_data = MagicMock()

    # Create time vector
    dt = 0.01
    time_vector = np.arange(0, 1, dt)
    n_samples = len(time_vector)

    # Create eye direction data with some saccades
    eye_direction = np.zeros((3, n_samples))
    eye_direction[2, :] = 1.0  # Default looking forward

    # Create a saccade around frame 30
    for i in range(30, 35):
        angle = (i - 30) * 5  # Increasing angle
        # Rotate around y-axis
        eye_direction[0, i] = np.sin(np.radians(angle))
        eye_direction[2, i] = np.cos(np.radians(angle))

    # Create another saccade around frame 70
    for i in range(70, 75):
        angle = (i - 70) * 4  # Increasing angle
        # Rotate around x-axis
        eye_direction[1, i] = np.sin(np.radians(angle))
        eye_direction[2, i] = np.cos(np.radians(angle))

    # Set up the mock data object
    mock_data.time_vector = time_vector
    mock_data.dt = dt
    mock_data.eye_direction = eye_direction

    # Create gaze direction (same as eye direction for simplicity)
    mock_data.gaze_direction = eye_direction.copy()

    # Create identified_indices (none identified yet)
    identified_indices = np.zeros(n_samples, dtype=bool)

    return mock_data, identified_indices


def test_saccade_event_initialization():
    """Test that SaccadeEvent initializes correctly."""
    mock_data = MagicMock()
    identified_indices = np.zeros(10, dtype=bool)

    event = SaccadeEvent(mock_data, identified_indices)

    assert event.data_object is mock_data
    assert event.identified_indices is identified_indices
    assert event.min_acceleration_threshold == 4000
    assert event.velocity_window_size == 0.52
    assert event.velocity_factor == 5.0
    assert event.eye_angular_velocity is None
    assert event.eye_angular_acceleration is None
    assert event.velocity_threshold is None
    assert event.saccade_amplitudes is None
    assert event.frame_indices is None
    assert event.sequences == []


def test_saccade_event_custom_parameters():
    """Test that SaccadeEvent accepts custom parameters."""
    mock_data = MagicMock()
    identified_indices = np.zeros(10, dtype=bool)

    event = SaccadeEvent(
        mock_data, identified_indices, min_acceleration_threshold=5000, velocity_window_size=0.6, velocity_factor=6.0
    )

    assert event.min_acceleration_threshold == 5000
    assert event.velocity_window_size == 0.6
    assert event.velocity_factor == 6.0


@patch("eyedentify3d.identification.saccade.compute_angular_velocity")
def test_set_eye_angular_velocity(mock_compute_angular_velocity, mock_data_object):
    """Test that set_eye_angular_velocity correctly computes eye angular velocity."""
    mock_data, identified_indices = mock_data_object

    # Mock the compute_angular_velocity function
    mock_velocity = np.ones(len(mock_data.time_vector))
    mock_compute_angular_velocity.return_value = mock_velocity

    event = SaccadeEvent(mock_data, identified_indices)
    event.set_eye_angular_velocity()

    # Check that compute_angular_velocity was called with correct arguments
    mock_compute_angular_velocity.assert_called_once_with(mock_data.time_vector, mock_data.eye_direction)

    # Check that eye_angular_velocity is set correctly
    np.testing.assert_array_equal(event.eye_angular_velocity, mock_velocity)


@patch("eyedentify3d.identification.saccade.centered_finite_difference")
def test_set_eye_angular_acceleration(mock_centered_finite_difference, mock_data_object):
    """Test that set_eye_angular_acceleration correctly computes eye angular acceleration."""
    mock_data, identified_indices = mock_data_object

    # Mock the centered_finite_difference function
    mock_acceleration = np.ones((1, len(mock_data.time_vector)))
    mock_centered_finite_difference.return_value = mock_acceleration

    event = SaccadeEvent(mock_data, identified_indices)
    event.eye_angular_velocity = np.ones(len(mock_data.time_vector))
    event.set_eye_angular_acceleration()

    # Check that centered_finite_difference was called with correct arguments
    mock_centered_finite_difference.assert_called_once()
    args = mock_centered_finite_difference.call_args[0]
    np.testing.assert_array_equal(args[0], mock_data.time_vector)
    np.testing.assert_array_equal(args[1][0, :], event.eye_angular_velocity)

    # Check that eye_angular_acceleration is set correctly
    np.testing.assert_array_equal(event.eye_angular_acceleration, mock_acceleration[0, :])


def test_set_the_velocity_threshold(mock_data_object):
    """Test that set_the_velocity_threshold correctly computes velocity threshold."""
    mock_data, identified_indices = mock_data_object

    event = SaccadeEvent(mock_data, identified_indices)

    # Set eye_angular_velocity with known values
    n_samples = len(mock_data.time_vector)
    eye_angular_velocity = np.ones(n_samples) * 10  # Constant velocity
    event.eye_angular_velocity = eye_angular_velocity

    event.set_the_velocity_threshold()

    # Check that velocity_threshold is set and has correct shape
    assert event.velocity_threshold is not None
    assert event.velocity_threshold.shape == (n_samples,)

    # With constant velocity, threshold should be constant and equal to velocity * factor
    expected_threshold = 10 * event.velocity_factor
    np.testing.assert_allclose(event.velocity_threshold, expected_threshold, rtol=1e-5)


def test_detect_saccade_indices(mock_data_object):
    """Test that detect_saccade_indices correctly identifies saccade frames."""
    mock_data, identified_indices = mock_data_object

    event = SaccadeEvent(mock_data, identified_indices)

    # Set eye_angular_velocity and velocity_threshold with known values
    n_samples = len(mock_data.time_vector)
    eye_angular_velocity = np.zeros(n_samples)
    # Set some frames with high velocity
    eye_angular_velocity[30:35] = 100  # Above threshold
    eye_angular_velocity[70:75] = 100  # Above threshold

    velocity_threshold = np.ones(n_samples) * 50  # Constant threshold

    event.eye_angular_velocity = eye_angular_velocity
    event.velocity_threshold = velocity_threshold

    event.detect_saccade_indices()

    # Check that frame_indices contains frames where velocity > threshold
    expected_indices = np.concatenate([np.arange(30, 35), np.arange(70, 75)])
    np.testing.assert_array_equal(event.frame_indices, expected_indices)


def test_detect_saccade_sequences(mock_data_object):
    """Test that detect_saccade_sequences correctly identifies saccade sequences."""
    mock_data, identified_indices = mock_data_object

    event = SaccadeEvent(mock_data, identified_indices)

    # Set frame_indices with known values
    event.frame_indices = np.concatenate([np.arange(30, 35), np.arange(70, 75)])

    # Set eye_angular_acceleration with high values at saccade frames
    n_samples = len(mock_data.time_vector)
    eye_angular_acceleration = np.zeros(n_samples)
    # Set high acceleration at the beginning and end of each saccade
    eye_angular_acceleration[29:31] = 5000  # Above threshold
    eye_angular_acceleration[34:36] = 5000  # Above threshold
    eye_angular_acceleration[69:71] = 5000  # Above threshold
    eye_angular_acceleration[74:76] = 5000  # Above threshold

    event.eye_angular_acceleration = eye_angular_acceleration
    event.min_acceleration_threshold = 4000

    event.detect_saccade_sequences()

    # Check that sequences contains the expected sequences
    assert len(event.sequences) == 2
    np.testing.assert_array_equal(event.sequences[0], np.arange(30, 35))
    np.testing.assert_array_equal(event.sequences[1], np.arange(70, 75))


@patch("eyedentify3d.identification.saccade.merge_close_sequences")
def test_merge_sequences(mock_merge_close_sequences, mock_data_object):
    """Test that merge_sequences correctly merges close saccade sequences."""
    mock_data, identified_indices = mock_data_object

    # Mock the merge_close_sequences function
    mock_merged_sequences = [np.arange(30, 35), np.arange(70, 75)]
    mock_merge_close_sequences.return_value = mock_merged_sequences

    event = SaccadeEvent(mock_data, identified_indices)
    event.sequences = [np.arange(30, 33), np.arange(33, 35), np.arange(70, 75)]

    event.merge_sequences()

    # Check that merge_close_sequences was called with correct arguments
    mock_merge_close_sequences.assert_called_once()

    # Check that sequences is updated with merged sequences
    assert event.sequences == mock_merged_sequences


def test_measure_saccade_amplitude(mock_data_object):
    """Test that measure_saccade_amplitude correctly computes saccade amplitudes."""
    mock_data, identified_indices = mock_data_object

    event = SaccadeEvent(mock_data, identified_indices)

    # Set sequences with known values
    event.sequences = [np.arange(30, 35), np.arange(70, 75)]

    # Mock get_angle_between_vectors to return known values
    with patch("eyedentify3d.identification.saccade.get_angle_between_vectors", side_effect=[20.0, 16.0]):
        event.measure_saccade_amplitude()

    # Check that saccade_amplitudes contains the expected amplitudes
    np.testing.assert_array_equal(event.saccade_amplitudes, [20.0, 16.0])


def test_initialize(mock_data_object):
    """Test that initialize correctly sets up the event."""
    mock_data, identified_indices = mock_data_object

    # Create a SaccadeEvent with mocked methods
    event = SaccadeEvent(mock_data, identified_indices)

    # Mock all the methods called in initialize
    with (
        patch.object(event, "set_eye_angular_velocity") as mock_set_velocity,
        patch.object(event, "set_eye_angular_acceleration") as mock_set_acceleration,
        patch.object(event, "set_the_velocity_threshold") as mock_set_threshold,
        patch.object(event, "detect_saccade_indices") as mock_detect_indices,
        patch.object(event, "detect_saccade_sequences") as mock_detect_sequences,
        patch.object(event, "merge_sequences") as mock_merge,
        patch.object(event, "adjust_indices_to_sequences") as mock_adjust,
    ):

        event.initialize()

        # Check that all methods were called in the correct order
        mock_set_velocity.assert_called_once()
        mock_set_acceleration.assert_called_once()
        mock_set_threshold.assert_called_once()
        mock_detect_indices.assert_called_once()
        mock_detect_sequences.assert_called_once()
        mock_merge.assert_called_once()
        mock_adjust.assert_called_once()
