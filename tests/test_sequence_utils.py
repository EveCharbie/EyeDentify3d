import numpy as np

from eyedentify3d.utils.sequence_utils import (
    split_sequences,
    apply_minimal_duration,
    apply_minimal_number_of_frames,
    _check_direction_alignment,
    _can_merge_sequences,
    merge_close_sequences,
)



def test_split_sequences_single_sequence():
    """Test split_sequences with a single continuous sequence."""
    indices = np.array([0, 1, 2, 3, 4])
    sequences = split_sequences(indices)

    assert len(sequences) == 1
    assert np.array_equal(sequences[0], indices)


def test_split_sequences_multiple_sequences():
    """Test split_sequences with multiple sequences."""
    indices = np.array([0, 1, 2, 4, 5, 7, 8, 9])
    sequences = split_sequences(indices)

    assert len(sequences) == 3
    assert np.array_equal(sequences[0], np.array([0, 1, 2]))
    assert np.array_equal(sequences[1], np.array([4, 5]))
    assert np.array_equal(sequences[2], np.array([7, 8, 9]))


def test_split_sequences_empty():
    """Test split_sequences with an empty array."""
    indices = np.array([])
    sequences = split_sequences(indices)

    assert len(sequences) == 0


def test_split_sequences_single_value():
    """Test split_sequences with a single value."""
    indices = np.array([42])
    sequences = split_sequences(indices)

    assert len(sequences) == 1
    assert np.array_equal(sequences[0], np.array([42]))


def test_split_sequences_non_consecutive():
    """Test split_sequences with all non-consecutive indices."""
    indices = np.array([1, 3, 5, 7, 9])
    sequences = split_sequences(indices)

    assert len(sequences) == 5
    for i, seq in enumerate(sequences):
        assert np.array_equal(seq, np.array([indices[i]]))


def test_split_sequences_large_gaps():
    """Test split_sequences with large gaps between indices."""
    indices = np.array([1, 2, 10, 11, 100, 101])
    sequences = split_sequences(indices)

    assert len(sequences) == 3
    assert np.array_equal(sequences[0], np.array([1, 2]))
    assert np.array_equal(sequences[1], np.array([10, 11]))
    assert np.array_equal(sequences[2], np.array([100, 101]))


def test_apply_minimal_duration():
    """Test apply_minimal_duration."""
    indices = np.array([1, 2, 10, 11, 100, 101, 102, 103, 200, 300, 301, 302, 303])
    sequences = split_sequences(indices)
    assert len(sequences) == 5

    sequence_modified = apply_minimal_duration(sequences, np.linspace(0, 100, 400), minimal_duration=0.4)
    assert len(sequence_modified) == 2

    assert np.array_equal(sequence_modified[0], np.array([100, 101, 102, 103]))
    assert np.array_equal(sequence_modified[1], np.array([300, 301, 302, 303]))


def test_apply_minimal_duration_empty_sequence():
    """Test apply_minimal_duration."""
    indices = np.array([])
    sequences = split_sequences(indices)
    assert len(sequences) == 0

    sequence_modified = apply_minimal_duration(sequences, np.linspace(0, 100, 400), minimal_duration=0.4)
    assert len(sequence_modified) == 0


def test_apply_minimal_number_of_frames():
    """Test apply_minimal_duration."""
    indices = np.array([1, 2, 10, 11, 100, 101, 102, 103, 200, 300, 301, 302, 303])
    sequences = split_sequences(indices)
    assert len(sequences) == 5

    # Actually does something
    sequence_modified = apply_minimal_number_of_frames(sequences, minimal_number_of_frames=3)
    assert len(sequence_modified) == 2

    assert np.array_equal(sequence_modified[0], np.array([100, 101, 102, 103]))
    assert np.array_equal(sequence_modified[1], np.array([300, 301, 302, 303]))

    # Nothing to do, sequences are already long enough
    sequence_modified = apply_minimal_number_of_frames(sequences, minimal_number_of_frames=1)
    assert len(sequence_modified) == 5
    assert sequence_modified == sequences


def test_apply_minimal_number_of_frames_empty_sequence():
    """Test apply_minimal_duration."""
    indices = np.array([])
    sequences = split_sequences(indices)
    assert len(sequences) == 0

    sequence_modified = apply_minimal_number_of_frames(sequences, minimal_number_of_frames=1)
    assert len(sequence_modified) == 0


