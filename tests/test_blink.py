import numpy as np
import pytest
from unittest.mock import MagicMock

from eyedentify3d.identification.blink import BlinkEvent


@pytest.fixture
def mock_data_object():
    """Create a mock data object for testing."""
    mock_data = MagicMock()
    mock_data.time_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # Create eye openness data with some blinks
    right_eye_openness = np.ones(10) * 0.8  # Default is open
    left_eye_openness = np.ones(10) * 0.8  # Default is open

    # Set some frames as blinks (both eyes closed)
    right_eye_openness[[2, 3, 4, 7, 8]] = 0.3  # Below threshold
    left_eye_openness[[2, 3, 4, 7, 8]] = 0.2  # Below threshold

    # Set some frames with only one eye closed
    right_eye_openness[6] = 0.3  # Below threshold
    left_eye_openness[9] = 0.2  # Below threshold

    mock_data.right_eye_openness = right_eye_openness
    mock_data.left_eye_openness = left_eye_openness

    return mock_data


def test_blink_event_initialization():
    """Test that BlinkEvent initializes correctly."""
    mock_data = MagicMock()
    event = BlinkEvent(mock_data)

    assert event.data_object is mock_data
    assert event.eye_openness_threshold == 0.5  # Default threshold
    assert event.frame_indices is None
    assert event.sequences == []


def test_blink_event_custom_threshold():
    """Test that BlinkEvent accepts custom threshold."""
    mock_data = MagicMock()
    event = BlinkEvent(mock_data, eye_openness_threshold=0.4)

    assert event.eye_openness_threshold == 0.4


def test_detect_blink_indices(mock_data_object):
    """Test that detect_blink_indices correctly identifies blink frames."""
    event = BlinkEvent(mock_data_object)

    event.detect_blink_indices()

    # Check that frame_indices contains frames where both eyes are below threshold
    np.testing.assert_array_equal(event.frame_indices, np.array([2, 3, 4, 7, 8]))


def test_detect_blink_indices_with_custom_threshold(mock_data_object):
    """Test that detect_blink_indices works with custom threshold."""
    # Set a lower threshold so fewer frames are considered blinks
    event = BlinkEvent(mock_data_object, eye_openness_threshold=0.1)

    event.detect_blink_indices()

    # With lower threshold, no frames should be considered blinks
    np.testing.assert_array_equal(event.frame_indices, np.array([]))


def test_initialize(mock_data_object):
    """Test that initialize correctly sets up the event."""
    event = BlinkEvent(mock_data_object)

    event.initialize()

    # Check that frame_indices contains the blink frames
    np.testing.assert_array_equal(event.frame_indices, np.array([2, 3, 4, 7, 8]))

    # Check that sequences are correctly split
    assert len(event.sequences) == 2
    np.testing.assert_array_equal(event.sequences[0], np.array([2, 3, 4]))
    np.testing.assert_array_equal(event.sequences[1], np.array([7, 8]))


def test_initialize_with_no_blinks():
    """Test that initialize handles the case where there are no blinks."""
    mock_data = MagicMock()
    mock_data.right_eye_openness = np.ones(10) * 0.8  # All eyes open
    mock_data.left_eye_openness = np.ones(10) * 0.8  # All eyes open

    event = BlinkEvent(mock_data)
    event.initialize()

    # Check that frame_indices is an empty array
    np.testing.assert_array_equal(event.frame_indices, np.array([]))

    # Check that sequences is an empty list
    assert event.sequences == []


def test_initialize_with_all_blinks():
    """Test that initialize handles the case where all frames are blinks."""
    mock_data = MagicMock()
    mock_data.right_eye_openness = np.ones(10) * 0.3  # All eyes closed
    mock_data.left_eye_openness = np.ones(10) * 0.2  # All eyes closed

    event = BlinkEvent(mock_data)
    event.initialize()

    # Check that frame_indices contains all indices
    np.testing.assert_array_equal(event.frame_indices, np.arange(10))

    # Check that sequences contains a single sequence with all indices
    assert len(event.sequences) == 1
    np.testing.assert_array_equal(event.sequences[0], np.arange(10))


def test_one_eye_closed_not_counted_as_blink(mock_data_object):
    """Test that frames where only one eye is closed are not counted as blinks."""
    event = BlinkEvent(mock_data_object)

    event.detect_blink_indices()

    # Frames 6 and 9 have only one eye closed, so they should not be in frame_indices
    assert 6 not in event.frame_indices
    assert 9 not in event.frame_indices
