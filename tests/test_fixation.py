import numpy as np
import numpy.testing as npt
import pytest
from unittest.mock import patch

from eyedentify3d import ReducedData
from eyedentify3d.identification.fixation import FixationEvent


def mock_data_object():
    """Create a mock data object for testing."""
    np.random.seed(42)  # For reproducibility

    # Create time vector
    dt = 0.01
    time_vector = np.arange(0, 1, dt)
    n_samples = len(time_vector)

    # Create gaze direction data with some fixation patterns
    gaze_direction = np.zeros((3, n_samples))
    gaze_direction[2, :] = 1.0  # Default looking forward

    # Create a fixation around frame 30-40
    for i in range(30, 40):
        # Small random movements around a central point
        np.random.seed(i)  # For reproducibility
        gaze_direction[0, i] = np.sin(np.radians(np.random.uniform(-1, 1)))
        gaze_direction[1, i] = np.sin(np.radians(np.random.uniform(-1, 1)))
        gaze_direction[2, i] = np.sqrt(1 - gaze_direction[0, i] ** 2 - gaze_direction[1, i] ** 2)

    # Create another fixation around frame 60-70
    for i in range(60, 70):
        # Small random movements around a different central point
        np.random.seed(i + 100)  # Different seed for different pattern
        gaze_direction[0, i] = 0.1 + np.sin(np.radians(np.random.uniform(-1, 1)))
        gaze_direction[1, i] = 0.1 + np.sin(np.radians(np.random.uniform(-1, 1)))
        gaze_direction[2, i] = np.sqrt(1 - gaze_direction[0, i] ** 2 - gaze_direction[1, i] ** 2)

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
def fixation_indices():
    """Create fixation_indices array for testing."""
    # Frames 30-40 and 60-70 are fixations
    return np.concatenate([np.arange(30, 40), np.arange(60, 70)])


def test_fixation_event_initialization(identified_indices, fixation_indices):
    """Test that FixationEvent initializes correctly."""
    mock_data = mock_data_object()
    event = FixationEvent(mock_data, identified_indices, fixation_indices, minimal_duration=0.05)

    assert event.data_object is mock_data
    assert event.identified_indices is identified_indices
    assert event.fixation_indices is fixation_indices
    assert event.minimal_duration == 0.05
    assert event.search_rate is None
    assert event.frame_indices is None
    assert event.sequences == []


def test_initialize(identified_indices, fixation_indices):
    """Test that initialize correctly sets up the event."""
    mock_data = mock_data_object()

    event = FixationEvent(mock_data, identified_indices, fixation_indices, minimal_duration=0.05)

    # Mock the methods called in initialize
    with (
        patch.object(event, "split_sequences") as mock_split,
        patch.object(event, "merge_sequences") as mock_merge,
        patch.object(event, "keep_only_sequences_long_enough") as mock_keep,
        patch.object(event, "adjust_indices_to_sequences") as mock_adjust,
    ):

        event.initialize()

        # Check that frame_indices is set to fixation_indices
        np.testing.assert_array_equal(event.frame_indices, fixation_indices)

        # Check that all methods were called in the correct order
        mock_split.assert_called_once()
        mock_merge.assert_called_once()
        mock_keep.assert_called_once()
        mock_adjust.assert_called_once()


def test_merge_sequences(identified_indices, fixation_indices):
    """Test that merge_sequences correctly merges close fixation sequences."""
    mock_data = mock_data_object()

    event = FixationEvent(
        mock_data,
        identified_indices,
        fixation_indices,
        minimal_duration=0.01,
    )

    # Set sequences
    event.sequences = [np.arange(30, 40), np.arange(41, 46), np.arange(60, 70)]

    event.merge_sequences()

    # Check that sequences is updated with merged sequences
    assert len(event.sequences) == 2
    npt.assert_array_equal(event.sequences[0], np.arange(30, 46))
    npt.assert_array_equal(event.sequences[1], np.arange(60, 70))


def test_measure_search_rate(identified_indices, fixation_indices):
    """Test that measure_search_rate correctly computes search rate."""
    mock_data = mock_data_object()

    event = FixationEvent(mock_data, identified_indices, fixation_indices)

    # Set sequences and frame_indices
    event.sequences = [np.arange(30, 40), np.arange(60, 70)]
    event.frame_indices = fixation_indices

    # Mock duration method to return a known value
    with patch.object(event, "mean_duration", return_value=0.1):
        event.measure_search_rate()

        # Check that search_rate is computed correctly
        # nb_events = 2, mean_duration = 0.1, so search_rate = 2 / 0.1 = 20
        assert event.search_rate == 20.0


def test_measure_search_rate_with_no_events(identified_indices):
    """Test that measure_search_rate handles the case with no events."""
    mock_data = mock_data_object()

    event = FixationEvent(mock_data, identified_indices, np.array([]))  # No fixation indices

    # Set empty sequences
    event.sequences = []

    event.measure_search_rate()

    # Check that search_rate is None when there are no events
    assert event.search_rate is None


def test_end_to_end_fixation_detection(identified_indices, fixation_indices):
    """Test the complete fixation detection process."""
    mock_data = mock_data_object()

    event = FixationEvent(
        mock_data, identified_indices, fixation_indices, minimal_duration=0.05  # 5 frames at 0.01s per frame
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

    # Measure search rate
    event.measure_search_rate()

    # Check that search_rate is computed
    assert event.search_rate is not None


def test_fixation_with_short_sequences(identified_indices):
    """Test that short sequences are filtered out when minimal_duration is set."""
    mock_data = mock_data_object()

    # Create fixation indices with one short sequence
    fixation_indices = np.concatenate([np.arange(30, 32), np.arange(60, 70)])

    # Create a FixationEvent with a minimal_duration that filters out the short sequence
    event = FixationEvent(
        mock_data, identified_indices, fixation_indices, minimal_duration=0.05  # 5 frames at 0.01s per frame
    )

    # Set frame_indices and split into sequences
    event.frame_indices = fixation_indices
    event.split_sequences()

    # Apply minimal duration filter
    event.keep_only_sequences_long_enough()

    # The first sequence (30-32) is only 2 frames (0.02s) which is less than minimal_duration (0.05s)
    # So it should be filtered out
    assert len(event.sequences) == 1
    np.testing.assert_array_equal(event.sequences[0], np.arange(60, 70))
