import numpy as np
import pytest
from eyedentify3d.utils.sequence_utils import split_sequences


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
    
    assert len(sequences) == 1
    assert len(sequences[0]) == 0


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
