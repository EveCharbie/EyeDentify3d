import numpy as np
import pytest
from unittest.mock import MagicMock

from eyedentify3d.identification.invalid import InvalidEvent


@pytest.fixture
def mock_data_object():
    """Create a mock data object for testing."""
    mock_data = MagicMock()
    mock_data.time_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # Create data_invalidity with some invalid frames
    data_invalidity = np.zeros(10, dtype=bool)
    data_invalidity[[2, 3, 6, 8]] = True  # Frames 2, 3, 6, and 8 are invalid
    mock_data.data_invalidity = data_invalidity

    return mock_data


@pytest.fixture
def mock_data_object_two_sequences():
    """Create a mock data object with two sequences for testing."""
    mock_data = MagicMock()
    mock_data.time_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # Create data_invalidity with some invalid frames
    data_invalidity = np.zeros(10, dtype=bool)
    data_invalidity[[2, 3, 6, 7, 8]] = True  # Frames 2, 3, 6, and 8 are invalid
    mock_data.data_invalidity = data_invalidity

    return mock_data


def test_invalid_event_initialization():
    """Test that InvalidEvent initializes correctly."""
    mock_data = MagicMock()
    event = InvalidEvent(mock_data)

    assert event.data_object is mock_data
    assert event.frame_indices is None
    assert event.sequences == []


def test_detect_invalid_indices(mock_data_object):
    """Test that detect_invalid_indices correctly identifies invalid frames."""
    event = InvalidEvent(mock_data_object)

    event.detect_invalid_indices()

    # Check that frame_indices contains the indices where data_invalidity is True
    np.testing.assert_array_equal(event.frame_indices, np.array([2, 3, 6, 8]))


def test_initialize(mock_data_object):
    """Test that initialize correctly sets up the event."""
    event = InvalidEvent(mock_data_object)

    event.initialize()

    # Check that frame_indices contains the invalid frames
    np.testing.assert_array_equal(event.frame_indices, np.array([2, 3, 6, 8]))

    # Check that sequences are correctly split
    assert len(event.sequences) == 1
    # Only one sequence because the other two are too short (only one frame)
    np.testing.assert_array_equal(event.sequences[0], np.array([2, 3]))


def test_initialize_two_sequences(mock_data_object_two_sequences):
    """Test that initialize correctly sets up the event."""
    event = InvalidEvent(mock_data_object_two_sequences)

    event.initialize()

    # Check that frame_indices contains the invalid frames
    np.testing.assert_array_equal(event.frame_indices, np.array([2, 3, 6, 7, 8]))

    # Check that sequences are correctly split
    assert len(event.sequences) == 2
    # Only one sequence because the other two are too short (only one frame)
    np.testing.assert_array_equal(event.sequences[0], np.array([2, 3]))
    np.testing.assert_array_equal(event.sequences[1], np.array([6, 7, 8]))


def test_initialize_with_no_invalid_frames():
    """Test that initialize handles the case where there are no invalid frames."""
    mock_data = MagicMock()
    mock_data.data_invalidity = np.zeros(10, dtype=bool)  # No invalid frames

    event = InvalidEvent(mock_data)
    event.initialize()

    # Check that frame_indices is an empty array
    np.testing.assert_array_equal(event.frame_indices, np.array([]))

    # Check that sequences is an empty list
    assert event.sequences == []


def test_initialize_with_all_invalid_frames():
    """Test that initialize handles the case where all frames are invalid."""
    mock_data = MagicMock()
    mock_data.data_invalidity = np.ones(10, dtype=bool)  # All frames are invalid

    event = InvalidEvent(mock_data)
    event.initialize()

    # Check that frame_indices contains all indices
    np.testing.assert_array_equal(event.frame_indices, np.arange(10))

    # Check that sequences contains a single sequence with all indices
    assert len(event.sequences) == 1
    np.testing.assert_array_equal(event.sequences[0], np.arange(10))
