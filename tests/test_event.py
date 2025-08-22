import numpy as np
import numpy.testing as npt
import pytest
from unittest.mock import MagicMock

from eyedentify3d.identification.event import Event


class MockEvent(Event):
    """Mock implementation of the abstract Event class for testing."""

    def __init__(self, data_object=None, minimal_duration=None):
        super().__init__()
        self.data_object = data_object
        if minimal_duration is not None:
            self.minimal_duration = minimal_duration

    def initialize(self):
        """Implementation of abstract method."""
        pass


@pytest.fixture
def mock_data_object():
    """Create a mock data object for testing."""
    mock_data = MagicMock()
    mock_data.time_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    mock_data.trial_duration = 0.9
    return mock_data


def test_event_initialization():
    """Test that Event initializes correctly."""
    event = MockEvent()
    assert event.frame_indices is None
    assert event.sequences == []


def test_split_sequences():
    """Test that split_sequences correctly splits indices into sequences."""
    event = MockEvent()
    # Create non-consecutive indices
    event.frame_indices = np.array([0, 1, 3, 4, 6, 7, 8])

    event.split_sequences()

    # Should split into 3 sequences: [0,1], [3,4], [6,7,8]
    assert len(event.sequences) == 3
    np.testing.assert_array_equal(event.sequences[0], np.array([0, 1]))
    np.testing.assert_array_equal(event.sequences[1], np.array([3, 4]))
    np.testing.assert_array_equal(event.sequences[2], np.array([6, 7, 8]))


def test_keep_only_sequences_long_enough(mock_data_object):
    """Test that keep_only_sequences_long_enough removes short sequences."""
    event = MockEvent(data_object=mock_data_object, minimal_duration=0.15)

    # Create sequences with different durations
    event.sequences = [
        np.array([0, 1]),  # Duration: 0.1s (too short)
        np.array([3, 4, 5]),  # Duration: 0.2s (long enough)
        np.array([7, 8, 9]),  # Duration: 0.2s (long enough)
    ]

    event.keep_only_sequences_long_enough()

    # Should keep only the sequences with duration >= 0.15s
    assert len(event.sequences) == 2
    np.testing.assert_array_equal(event.sequences[0], np.array([3, 4, 5]))
    np.testing.assert_array_equal(event.sequences[1], np.array([7, 8, 9]))


def test_keep_only_sequences_long_enough_missing_attribute():
    """Test that keep_only_sequences_long_enough raises AttributeError when minimal_duration is not set."""
    event = MockEvent()

    with pytest.raises(AttributeError, match="The 'minimal_duration' attribute is not set for this event."):
        event.keep_only_sequences_long_enough()


def test_adjust_indices_to_sequences():
    """Test that adjust_indices_to_sequences correctly updates frame_indices from sequences."""
    event = MockEvent()

    # Set sequences
    event.sequences = [np.array([1, 2]), np.array([4, 5, 6])]

    event.adjust_indices_to_sequences()

    # frame_indices should be the concatenation of all sequences
    np.testing.assert_array_equal(event.frame_indices, np.array([1, 2, 4, 5, 6]))


def test_adjust_indices_to_sequences_empty():
    """Test that adjust_indices_to_sequences handles empty sequences correctly."""
    event = MockEvent()

    # Set empty sequences
    event.sequences = []

    event.adjust_indices_to_sequences()

    # frame_indices should be an empty array
    np.testing.assert_array_equal(event.frame_indices, np.array([], dtype=int))


def test_from_sequences():
    """Test that from_sequences correctly sets sequences and frame_indices."""
    event = MockEvent()

    # Create sequences
    sequences = [np.array([1, 2]), np.array([4, 5, 6])]

    # Call from_sequences
    result = event.from_sequences(sequences)

    # Check that sequences are set
    assert event.sequences == sequences

    # Check that frame_indices is the concatenation of all sequences
    np.testing.assert_array_equal(event.frame_indices, np.array([1, 2, 4, 5, 6]))

    # Check that the method returns self
    assert result is event


def test_from_sequences_empty():
    """Test that from_sequences handles empty sequences correctly."""
    event = MockEvent()

    # Call from_sequences with empty list
    result = event.from_sequences([])

    # Check that sequences are set to empty list
    assert event.sequences == []

    # Check that frame_indices is an empty array
    np.testing.assert_array_equal(event.frame_indices, np.array([], dtype=int))

    # Check that the method returns self
    assert result is event


def test_nb_events():
    """Test that nb_events returns the correct number of events."""
    event = MockEvent()

    # Set sequences
    event.sequences = [np.array([1, 2]), np.array([4, 5, 6]), np.array([8, 9])]

    # Check that nb_events returns the number of sequences
    assert event.nb_events() == 3


def test_duration(mock_data_object):
    """Test that duration returns the correct durations for each event."""
    event = MockEvent(data_object=mock_data_object)

    # Set sequences
    event.sequences = [
        np.array([1, 2]),  # From 0.1 to 0.2 = 0.1s
        np.array([4, 5, 6]),  # From 0.4 to 0.6 = 0.2s
        np.array([8, 9]),  # From 0.8 to 0.9 = 0.1s
    ]

    # Get durations
    durations = event.duration()

    # Check durations
    np.testing.assert_array_almost_equal(durations, np.array([0.1, 0.2, 0.1]))


def test_duration_with_sequence_at_end(mock_data_object):
    """Test that duration handles sequences that end at the last time point."""
    event = MockEvent(data_object=mock_data_object)

    # Set a sequence that includes the last index
    event.sequences = [np.array([8, 9])]  # From 0.8 to 0.9 = 0.1s

    # Get durations
    durations = event.duration()

    # Check durations
    np.testing.assert_array_almost_equal(durations, np.array([0.1]))


def test_mean_duration(mock_data_object):
    """Test that mean_duration returns the correct mean duration."""
    event = MockEvent(data_object=mock_data_object)

    # Set sequences with different durations
    event.sequences = [np.array([1, 2]), np.array([4, 5, 6]), np.array([8, 9])]  # 0.1s  # 0.2s  # 0.1s

    # Check mean duration
    npt.assert_almost_equal(event.mean_duration(), 0.13333333333333333)  # (0.1 + 0.2 + 0.1) / 3


def test_mean_duration_empty():
    """Test that mean_duration returns None for empty sequences."""
    event = MockEvent()

    # Set empty sequences
    event.sequences = []

    # Check mean duration
    assert event.mean_duration() is None


def test_max_duration(mock_data_object):
    """Test that max_duration returns the correct maximum duration."""
    event = MockEvent(data_object=mock_data_object)

    # Set sequences with different durations
    event.sequences = [np.array([1, 2]), np.array([4, 5, 6]), np.array([8, 9])]  # 0.1s  # 0.2s  # 0.1s

    # Check max duration
    npt.assert_almost_equal(event.max_duration(), 0.2)


def test_max_duration_empty():
    """Test that max_duration returns None for empty sequences."""
    event = MockEvent()

    # Set empty sequences
    event.sequences = []

    # Check max duration
    assert event.max_duration() is None


def test_total_duration(mock_data_object):
    """Test that total_duration returns the correct total duration."""
    event = MockEvent(data_object=mock_data_object)

    # Set sequences with different durations
    event.sequences = [np.array([1, 2]), np.array([4, 5, 6]), np.array([8, 9])]  # 0.1s  # 0.2s  # 0.1s

    # Check total duration
    npt.assert_almost_equal(event.total_duration(), 0.4)  # 0.1 + 0.2 + 0.1


def test_total_duration_empty():
    """Test that total_duration returns None for empty sequences."""
    event = MockEvent()

    # Set empty sequences
    event.sequences = []

    # Check total duration
    assert event.total_duration() is None


def test_ratio(mock_data_object):
    """Test that ratio returns the correct proportion of time spent in events."""
    event = MockEvent(data_object=mock_data_object)

    # Set sequences with total duration of 0.4s
    event.sequences = [np.array([1, 2]), np.array([4, 5, 6]), np.array([8, 9])]  # 0.1s  # 0.2s  # 0.1s

    # Check ratio (total duration / trial duration)
    npt.assert_almost_equal(event.ratio(), 0.4444444444444444)  # 0.4 / 0.9


def test_ratio_empty(mock_data_object):
    """Test that ratio returns 0.0 for empty sequences."""
    event = MockEvent(data_object=mock_data_object)

    # Set empty sequences
    event.sequences = []

    # Check ratio
    npt.assert_almost_equal(event.ratio(), 0.0)
