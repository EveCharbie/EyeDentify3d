import numpy as np
import numpy.testing as npt
from unittest.mock import patch

from eyedentify3d import ReducedData
from eyedentify3d.identification.visual_scanning import VisualScanningEvent


def mock_data_object():
    """Create mock data object for testing."""
    np.random.seed(42)  # For reproducibility

    # Create time vector
    dt = 0.01
    time_vector = np.arange(0, 1, dt)
    n_samples = len(time_vector)

    # Create gaze angular velocity data with some visual scanning
    gaze_angular_velocity = np.zeros(n_samples)

    # Set some frames with high velocity (visual scanning)
    gaze_angular_velocity[30:35] = 120  # Above threshold
    gaze_angular_velocity[70:75] = 150  # Above threshold

    # Create gaze direction data
    gaze_direction = np.zeros((3, n_samples))
    gaze_direction[2, :] = 1.0  # Default looking forward

    # Create some variation in gaze direction during visual scanning
    for i in range(30, 35):
        angle = (i - 30) * 2  # Small increasing angle
        # Rotate around y-axis
        gaze_direction[0, i] = np.sin(np.radians(angle))
        gaze_direction[2, i] = np.cos(np.radians(angle))

    for i in range(70, 75):
        angle = (i - 70) * 2  # Small increasing angle
        # Rotate around x-axis
        gaze_direction[1, i] = np.sin(np.radians(angle))
        gaze_direction[2, i] = np.cos(np.radians(angle))

    # Create sample eye openness data
    right_eye_openness = np.ones(n_samples) * 0.8
    left_eye_openness = np.ones(n_samples) * 0.7

    # Create sample direction vectors (3D)
    eye_direction = np.zeros((3, n_samples))
    eye_direction[2, :] = 1.0  # Looking forward along z-axis

    head_angles = np.zeros((3, n_samples))
    head_angles[0, :] = np.linspace(0, 0.1, n_samples)  # Small rotation around x-axis

    gaze_direction = np.zeros((3, n_samples))
    gaze_direction[2, :] = 1.0  # Looking forward along z-axis

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
    data_object.gaze_angular_velocity = gaze_angular_velocity

    # Create identified_indices (none identified yet)
    identified_indices = np.zeros(n_samples, dtype=bool)

    return data_object, identified_indices


def test_visual_scanning_event_initialization():
    """Test that VisualScanningEvent initializes correctly."""
    mock_data, mock_indices = mock_data_object()
    identified_indices = np.zeros(10, dtype=bool)

    event = VisualScanningEvent(mock_data, identified_indices, min_velocity_threshold=100, minimal_duration=0.05)

    assert event.data_object is mock_data
    assert event.identified_indices is identified_indices
    assert event.min_velocity_threshold == 100
    assert event.minimal_duration == 0.05
    assert event.frame_indices is None
    assert event.sequences == []


def test_detect_visual_scanning_indices():
    """Test that detect_visual_scanning_indices correctly identifies visual scanning frames."""
    mock_data, identified_indices = mock_data_object()

    event = VisualScanningEvent(mock_data, identified_indices, min_velocity_threshold=100)

    event.detect_visual_scanning_indices()

    # Check that frame_indices contains frames where gaze_angular_velocity > threshold
    expected_indices = np.concatenate([np.arange(30, 35), np.arange(70, 75)])
    np.testing.assert_array_equal(event.frame_indices, expected_indices)


def test_detect_visual_scanning_indices_with_identified_frames():
    """Test that detect_visual_scanning_indices excludes already identified frames."""
    mock_data, identified_indices = mock_data_object()

    # Mark some frames as already identified
    identified_indices[30:32] = True

    event = VisualScanningEvent(mock_data, identified_indices, min_velocity_threshold=100)

    event.detect_visual_scanning_indices()

    # Check that frame_indices excludes already identified frames
    expected_indices = np.concatenate([np.arange(32, 35), np.arange(70, 75)])
    np.testing.assert_array_equal(event.frame_indices, expected_indices)


def test_merge_sequences():
    """Test that merge_sequences correctly merges close visual scanning sequences."""
    mock_data, identified_indices = mock_data_object()

    event = VisualScanningEvent(mock_data, identified_indices)
    event.sequences = [np.arange(30, 33), np.arange(33, 35), np.arange(70, 75)]

    event.merge_sequences()

    # Check that sequences were not merged because the directionality criteria was not met
    assert len(event.sequences) == 3
    npt.assert_almost_equal(event.sequences[0], np.arange(30, 33))
    npt.assert_almost_equal(event.sequences[1], np.arange(33, 35))
    npt.assert_almost_equal(event.sequences[2], np.arange(70, 75))


def test_initialize():
    """Test that initialize correctly sets up the event."""
    mock_data, identified_indices = mock_data_object()

    # Create a VisualScanningEvent with mocked methods
    event = VisualScanningEvent(mock_data, identified_indices, min_velocity_threshold=100, minimal_duration=0.05)

    # Mock all the methods called in initialize
    with (
        patch.object(event, "detect_visual_scanning_indices") as mock_detect,
        patch.object(event, "split_sequences") as mock_split,
        patch.object(event, "merge_sequences") as mock_merge,
        patch.object(event, "keep_only_sequences_long_enough") as mock_keep,
        patch.object(event, "adjust_indices_to_sequences") as mock_adjust,
    ):

        event.initialize()

        # Check that all methods were called in the correct order
        mock_detect.assert_called_once()
        mock_split.assert_called_once()
        mock_merge.assert_called_once()
        mock_keep.assert_called_once()
        mock_adjust.assert_called_once()


def test_end_to_end_visual_scanning_detection():
    """Test the complete visual scanning detection process."""
    mock_data, identified_indices = mock_data_object()

    # Create a VisualScanningEvent
    event = VisualScanningEvent(
        mock_data, identified_indices, min_velocity_threshold=100, minimal_duration=0.03  # 3 frames at 0.01s per frame
    )

    # Run the complete initialization process
    event.initialize()

    # Check that sequences contains the expected sequences
    assert len(event.sequences) == 2
    np.testing.assert_array_equal(event.sequences[0], np.arange(30, 35))
    np.testing.assert_array_equal(event.sequences[1], np.arange(70, 75))

    # Check that frame_indices contains all frames from all sequences
    expected_indices = np.concatenate([np.arange(30, 35), np.arange(70, 75)])
    np.testing.assert_array_equal(event.frame_indices, expected_indices)


def test_visual_scanning_with_short_sequences():
    """Test that short sequences are filtered out when minimal_duration is set."""
    mock_data, identified_indices = mock_data_object()

    # Create a VisualScanningEvent with a high minimal_duration
    event = VisualScanningEvent(
        mock_data, identified_indices, min_velocity_threshold=100, minimal_duration=0.1  # 10 frames at 0.01s per frame
    )

    # Set frame_indices with known values
    event.frame_indices = np.concatenate([np.arange(30, 35), np.arange(70, 75)])

    # Split into sequences
    event.split_sequences()

    # Apply minimal duration filter
    event.keep_only_sequences_long_enough()

    # Both sequences are 5 frames (0.05s) which is less than minimal_duration (0.1s)
    # So both should be filtered out
    assert len(event.sequences) == 0
