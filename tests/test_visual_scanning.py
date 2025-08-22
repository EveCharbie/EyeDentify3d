import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from eyedentify3d.identification.visual_scanning import VisualScanningEvent


@pytest.fixture
def mock_data_object():
    """Create a mock data object for testing."""
    mock_data = MagicMock()
    
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
    
    # Set up the mock data object
    mock_data.time_vector = time_vector
    mock_data.dt = dt
    mock_data.gaze_angular_velocity = gaze_angular_velocity
    mock_data.gaze_direction = gaze_direction
    
    # Create identified_indices (none identified yet)
    identified_indices = np.zeros(n_samples, dtype=bool)
    
    return mock_data, identified_indices


def test_visual_scanning_event_initialization():
    """Test that VisualScanningEvent initializes correctly."""
    mock_data = MagicMock()
    identified_indices = np.zeros(10, dtype=bool)
    
    event = VisualScanningEvent(
        mock_data,
        identified_indices,
        min_velocity_threshold=100,
        minimal_duration=0.05
    )
    
    assert event.data_object is mock_data
    assert event.identified_indices is identified_indices
    assert event.min_velocity_threshold == 100
    assert event.minimal_duration == 0.05
    assert event.frame_indices is None
    assert event.sequences == []


def test_detect_visual_scanning_indices(mock_data_object):
    """Test that detect_visual_scanning_indices correctly identifies visual scanning frames."""
    mock_data, identified_indices = mock_data_object
    
    event = VisualScanningEvent(
        mock_data,
        identified_indices,
        min_velocity_threshold=100
    )
    
    event.detect_visual_scanning_indices()
    
    # Check that frame_indices contains frames where gaze_angular_velocity > threshold
    expected_indices = np.concatenate([np.arange(30, 35), np.arange(70, 75)])
    np.testing.assert_array_equal(event.frame_indices, expected_indices)


def test_detect_visual_scanning_indices_with_identified_frames(mock_data_object):
    """Test that detect_visual_scanning_indices excludes already identified frames."""
    mock_data, identified_indices = mock_data_object
    
    # Mark some frames as already identified
    identified_indices[30:32] = True
    
    event = VisualScanningEvent(
        mock_data,
        identified_indices,
        min_velocity_threshold=100
    )
    
    event.detect_visual_scanning_indices()
    
    # Check that frame_indices excludes already identified frames
    expected_indices = np.concatenate([np.arange(32, 35), np.arange(70, 75)])
    np.testing.assert_array_equal(event.frame_indices, expected_indices)


@patch('eyedentify3d.identification.visual_scanning.merge_close_sequences')
def test_merge_sequences(mock_merge_close_sequences, mock_data_object):
    """Test that merge_sequences correctly merges close visual scanning sequences."""
    mock_data, identified_indices = mock_data_object
    
    # Mock the merge_close_sequences function
    mock_merged_sequences = [np.arange(30, 35), np.arange(70, 75)]
    mock_merge_close_sequences.return_value = mock_merged_sequences
    
    event = VisualScanningEvent(mock_data, identified_indices)
    event.sequences = [np.arange(30, 33), np.arange(33, 35), np.arange(70, 75)]
    
    event.merge_sequences()
    
    # Check that merge_close_sequences was called with correct arguments
    mock_merge_close_sequences.assert_called_once_with(
        event.sequences,
        mock_data.time_vector,
        mock_data.gaze_direction,
        identified_indices,
        max_gap=0.040,
        check_directionality=True,
        max_angle=30.0
    )
    
    # Check that sequences is updated with merged sequences
    assert event.sequences == mock_merged_sequences


def test_initialize(mock_data_object):
    """Test that initialize correctly sets up the event."""
    mock_data, identified_indices = mock_data_object
    
    # Create a VisualScanningEvent with mocked methods
    event = VisualScanningEvent(
        mock_data,
        identified_indices,
        min_velocity_threshold=100,
        minimal_duration=0.05
    )
    
    # Mock all the methods called in initialize
    with patch.object(event, 'detect_visual_scanning_indices') as mock_detect, \
         patch.object(event, 'split_sequences') as mock_split, \
         patch.object(event, 'merge_sequences') as mock_merge, \
         patch.object(event, 'keep_only_sequences_long_enough') as mock_keep, \
         patch.object(event, 'adjust_indices_to_sequences') as mock_adjust:
        
        event.initialize()
        
        # Check that all methods were called in the correct order
        mock_detect.assert_called_once()
        mock_split.assert_called_once()
        mock_merge.assert_called_once()
        mock_keep.assert_called_once()
        mock_adjust.assert_called_once()


def test_end_to_end_visual_scanning_detection(mock_data_object):
    """Test the complete visual scanning detection process."""
    mock_data, identified_indices = mock_data_object
    
    # Create a VisualScanningEvent
    event = VisualScanningEvent(
        mock_data,
        identified_indices,
        min_velocity_threshold=100,
        minimal_duration=0.03  # 3 frames at 0.01s per frame
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


def test_visual_scanning_with_short_sequences(mock_data_object):
    """Test that short sequences are filtered out when minimal_duration is set."""
    mock_data, identified_indices = mock_data_object
    
    # Create a VisualScanningEvent with a high minimal_duration
    event = VisualScanningEvent(
        mock_data,
        identified_indices,
        min_velocity_threshold=100,
        minimal_duration=0.1  # 10 frames at 0.01s per frame
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
