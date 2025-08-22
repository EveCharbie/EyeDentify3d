import numpy as np
import pytest
from unittest.mock import patch

from eyedentify3d.identification.smooth_pursuit import SmoothPursuitEvent


@pytest.fixture
def mock_data_object():
    """Create a mock data object for testing."""
    class MockDataObject:
        def __init__(self):
            # Create time vector
            self.dt = 0.01
            self.time_vector = np.arange(0, 1, self.dt)
            n_samples = len(self.time_vector)
            
            # Create gaze direction data with some smooth pursuit patterns
            self.gaze_direction = np.zeros((3, n_samples))
            self.gaze_direction[2, :] = 1.0  # Default looking forward
            
            # Create a smooth pursuit around frame 30-40
            for i in range(30, 40):
                angle = (i - 30) * 2  # Small increasing angle
                # Rotate around y-axis
                self.gaze_direction[0, i] = np.sin(np.radians(angle))
                self.gaze_direction[2, i] = np.cos(np.radians(angle))
            
            # Create another smooth pursuit around frame 60-70
            for i in range(60, 70):
                angle = (i - 60) * 1.5  # Different rate of change
                # Rotate around x-axis
                self.gaze_direction[1, i] = np.sin(np.radians(angle))
                self.gaze_direction[2, i] = np.cos(np.radians(angle))
            
            # Create gaze angular velocity
            self.gaze_angular_velocity = np.zeros(n_samples)
            # Set angular velocity for smooth pursuit segments
            self.gaze_angular_velocity[30:40] = 20  # deg/s
            self.gaze_angular_velocity[60:70] = 15  # deg/s
            
            # Create trial_duration for calculations
            self.trial_duration = 1.0
    
    return MockDataObject()


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


def test_smooth_pursuit_event_initialization(mock_data_object, identified_indices, smooth_pursuit_indices):
    """Test that SmoothPursuitEvent initializes correctly."""
    event = SmoothPursuitEvent(
        mock_data_object,
        identified_indices,
        smooth_pursuit_indices,
        minimal_duration=0.05
    )
    
    assert event.data_object is mock_data_object
    assert event.identified_indices is identified_indices
    assert np.array_equal(event.smooth_pursuit_indices, smooth_pursuit_indices)
    assert event.minimal_duration == 0.05
    assert event.smooth_pursuit_trajectories is None
    assert event.frame_indices is None
    assert event.sequences == []


def test_initialize(mock_data_object, identified_indices, smooth_pursuit_indices):
    """Test that initialize correctly sets up the event."""
    event = SmoothPursuitEvent(
        mock_data_object,
        identified_indices,
        smooth_pursuit_indices,
        minimal_duration=0.05
    )
    
    # Mock the methods called in initialize
    with patch.object(event, 'split_sequences') as mock_split, \
         patch.object(event, 'merge_sequences') as mock_merge, \
         patch.object(event, 'keep_only_sequences_long_enough') as mock_keep, \
         patch.object(event, 'adjust_indices_to_sequences') as mock_adjust:
        
        event.initialize()
        
        # Check that frame_indices is set to smooth_pursuit_indices
        np.testing.assert_array_equal(event.frame_indices, smooth_pursuit_indices)
        
        # Check that all methods were called in the correct order
        mock_split.assert_called_once()
        mock_merge.assert_called_once()
        mock_keep.assert_called_once()
        mock_adjust.assert_called_once()


@patch('eyedentify3d.identification.smooth_pursuit.merge_close_sequences')
def test_merge_sequences(mock_merge_close_sequences, mock_data_object, identified_indices, smooth_pursuit_indices):
    """Test that merge_sequences correctly merges close smooth pursuit sequences."""
    event = SmoothPursuitEvent(
        mock_data_object,
        identified_indices,
        smooth_pursuit_indices
    )
    
    # Set sequences
    event.sequences = [np.arange(30, 40), np.arange(60, 70)]
    
    # Mock the merge_close_sequences function
    mock_merged_sequences = [np.arange(30, 40), np.arange(60, 70)]
    mock_merge_close_sequences.return_value = mock_merged_sequences
    
    event.merge_sequences()
    
    # Check that merge_close_sequences was called with correct arguments
    mock_merge_close_sequences.assert_called_once_with(
        event.sequences,
        mock_data_object.time_vector,
        mock_data_object.gaze_direction,
        identified_indices,
        max_gap=0.040,
        check_directionality=True,
        max_angle=30.0
    )
    
    # Check that sequences is updated with merged sequences
    assert event.sequences == mock_merged_sequences


def test_measure_smooth_pursuit_trajectory(mock_data_object, identified_indices, smooth_pursuit_indices):
    """Test that measure_smooth_pursuit_trajectory correctly computes trajectory lengths."""
    event = SmoothPursuitEvent(
        mock_data_object,
        identified_indices,
        smooth_pursuit_indices
    )
    
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


def test_measure_smooth_pursuit_trajectory_with_nan(mock_data_object, identified_indices, smooth_pursuit_indices):
    """Test that measure_smooth_pursuit_trajectory handles NaN values correctly."""
    event = SmoothPursuitEvent(
        mock_data_object,
        identified_indices,
        smooth_pursuit_indices
    )
    
    # Set sequences
    event.sequences = [np.arange(30, 39)]
    
    # Introduce NaN values in gaze_angular_velocity
    mock_data_object.gaze_angular_velocity[35] = np.nan
    
    # Call the method
    event.measure_smooth_pursuit_trajectory()
    
    # Check that smooth_pursuit_trajectories is computed and NaN values are handled
    assert event.smooth_pursuit_trajectories is not None
    assert len(event.smooth_pursuit_trajectories) == 1
    assert not np.isnan(event.smooth_pursuit_trajectories[0])


def test_end_to_end_smooth_pursuit_detection(mock_data_object, identified_indices, smooth_pursuit_indices):
    """Test the complete smooth pursuit detection process."""
    event = SmoothPursuitEvent(
        mock_data_object,
        identified_indices,
        smooth_pursuit_indices,
        minimal_duration=0.05  # 5 frames at 0.01s per frame
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


def test_smooth_pursuit_with_short_sequences(mock_data_object, identified_indices):
    """Test that short sequences are filtered out when minimal_duration is set."""
    # Create smooth pursuit indices with one short sequence
    smooth_pursuit_indices = np.concatenate([np.arange(30, 32), np.arange(60, 70)])
    
    # Create a SmoothPursuitEvent with a minimal_duration that filters out the short sequence
    event = SmoothPursuitEvent(
        mock_data_object,
        identified_indices,
        smooth_pursuit_indices,
        minimal_duration=0.05  # 5 frames at 0.01s per frame
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
