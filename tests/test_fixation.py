import numpy as np
import pytest
from unittest.mock import patch

from eyedentify3d.identification.fixation import FixationEvent


@pytest.fixture
def mock_data_object():
    """Create a mock data object for testing."""
    class MockDataObject:
        def __init__(self):
            # Create time vector
            self.dt = 0.01
            self.time_vector = np.arange(0, 1, self.dt)
            n_samples = len(self.time_vector)
            
            # Create gaze direction data with some fixation patterns
            self.gaze_direction = np.zeros((3, n_samples))
            self.gaze_direction[2, :] = 1.0  # Default looking forward
            
            # Create a fixation around frame 30-40
            for i in range(30, 40):
                # Small random movements around a central point
                np.random.seed(i)  # For reproducibility
                self.gaze_direction[0, i] = np.sin(np.radians(np.random.uniform(-1, 1)))
                self.gaze_direction[1, i] = np.sin(np.radians(np.random.uniform(-1, 1)))
                self.gaze_direction[2, i] = np.sqrt(1 - self.gaze_direction[0, i]**2 - self.gaze_direction[1, i]**2)
            
            # Create another fixation around frame 60-70
            for i in range(60, 70):
                # Small random movements around a different central point
                np.random.seed(i + 100)  # Different seed for different pattern
                self.gaze_direction[0, i] = 0.1 + np.sin(np.radians(np.random.uniform(-1, 1)))
                self.gaze_direction[1, i] = 0.1 + np.sin(np.radians(np.random.uniform(-1, 1)))
                self.gaze_direction[2, i] = np.sqrt(1 - self.gaze_direction[0, i]**2 - self.gaze_direction[1, i]**2)
            
            # Create trial_duration for search_rate calculation
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
def fixation_indices():
    """Create fixation_indices array for testing."""
    # Frames 30-40 and 60-70 are fixations
    return np.concatenate([np.arange(30, 40), np.arange(60, 70)])


def test_fixation_event_initialization(mock_data_object, identified_indices, fixation_indices):
    """Test that FixationEvent initializes correctly."""
    event = FixationEvent(
        mock_data_object,
        identified_indices,
        fixation_indices,
        minimal_duration=0.05
    )
    
    assert event.data_object is mock_data_object
    assert event.identified_indices is identified_indices
    assert event.fixation_indices is fixation_indices
    assert event.minimal_duration == 0.05
    assert event.search_rate is None
    assert event.frame_indices is None
    assert event.sequences == []


def test_initialize(mock_data_object, identified_indices, fixation_indices):
    """Test that initialize correctly sets up the event."""
    event = FixationEvent(
        mock_data_object,
        identified_indices,
        fixation_indices,
        minimal_duration=0.05
    )
    
    # Mock the methods called in initialize
    with patch.object(event, 'split_sequences') as mock_split, \
         patch.object(event, 'merge_sequences') as mock_merge, \
         patch.object(event, 'keep_only_sequences_long_enough') as mock_keep, \
         patch.object(event, 'adjust_indices_to_sequences') as mock_adjust:
        
        event.initialize()
        
        # Check that frame_indices is set to fixation_indices
        np.testing.assert_array_equal(event.frame_indices, fixation_indices)
        
        # Check that all methods were called in the correct order
        mock_split.assert_called_once()
        mock_merge.assert_called_once()
        mock_keep.assert_called_once()
        mock_adjust.assert_called_once()


@patch('eyedentify3d.identification.fixation.merge_close_sequences')
def test_merge_sequences(mock_merge_close_sequences, mock_data_object, identified_indices, fixation_indices):
    """Test that merge_sequences correctly merges close fixation sequences."""
    event = FixationEvent(
        mock_data_object,
        identified_indices,
        fixation_indices
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
        check_directionality=False,
        max_angle=30.0
    )
    
    # Check that sequences is updated with merged sequences
    assert event.sequences == mock_merged_sequences


def test_measure_search_rate(mock_data_object, identified_indices, fixation_indices):
    """Test that measure_search_rate correctly computes search rate."""
    event = FixationEvent(
        mock_data_object,
        identified_indices,
        fixation_indices
    )
    
    # Set sequences and frame_indices
    event.sequences = [np.arange(30, 40), np.arange(60, 70)]
    event.frame_indices = fixation_indices
    
    # Mock duration method to return a known value
    with patch.object(event, 'mean_duration', return_value=0.1):
        event.measure_search_rate()
        
        # Check that search_rate is computed correctly
        # nb_events = 2, mean_duration = 0.1, so search_rate = 2 / 0.1 = 20
        assert event.search_rate == 20.0


def test_measure_search_rate_with_no_events(mock_data_object, identified_indices):
    """Test that measure_search_rate handles the case with no events."""
    event = FixationEvent(
        mock_data_object,
        identified_indices,
        np.array([])  # No fixation indices
    )
    
    # Set empty sequences
    event.sequences = []
    
    event.measure_search_rate()
    
    # Check that search_rate is None when there are no events
    assert event.search_rate is None


def test_end_to_end_fixation_detection(mock_data_object, identified_indices, fixation_indices):
    """Test the complete fixation detection process."""
    event = FixationEvent(
        mock_data_object,
        identified_indices,
        fixation_indices,
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
    
    # Measure search rate
    event.measure_search_rate()
    
    # Check that search_rate is computed
    assert event.search_rate is not None


def test_fixation_with_short_sequences(mock_data_object, identified_indices):
    """Test that short sequences are filtered out when minimal_duration is set."""
    # Create fixation indices with one short sequence
    fixation_indices = np.concatenate([np.arange(30, 32), np.arange(60, 70)])
    
    # Create a FixationEvent with a minimal_duration that filters out the short sequence
    event = FixationEvent(
        mock_data_object,
        identified_indices,
        fixation_indices,
        minimal_duration=0.05  # 5 frames at 0.01s per frame
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
