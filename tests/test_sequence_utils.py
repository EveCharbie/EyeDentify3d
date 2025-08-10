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

    # Sequences with only one value are filtered out
    assert sequences == []


def test_split_sequences_two_values():
    """Test split_sequences with a single value."""
    indices = np.array([42, 43])
    sequences = split_sequences(indices)

    assert len(sequences) == 1
    assert np.array_equal(sequences[0], indices)


def test_split_sequences_non_consecutive():
    """Test split_sequences with all non-consecutive indices."""
    indices = np.array([1, 3, 5, 7, 9])
    sequences = split_sequences(indices)

    # One frame elements are filtered out
    assert len(sequences) == 0


def test_split_sequences_with_consecutive_values():
    """Test split_sequences with consecutive indices."""
    indices = np.array([1, 2, 3, 5, 7, 8, 9])
    sequences = split_sequences(indices)

    # One frame elements are filtered out
    assert len(sequences) == 2
    assert np.array_equal(sequences[0], np.array([1, 2, 3]))
    assert np.array_equal(sequences[1], np.array([7, 8, 9]))


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
    assert len(sequences) == 4

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
    assert len(sequences) == 4

    # Actually does something
    sequence_modified = apply_minimal_number_of_frames(sequences, minimal_number_of_frames=3)
    assert len(sequence_modified) == 2

    assert np.array_equal(sequence_modified[0], np.array([100, 101, 102, 103]))
    assert np.array_equal(sequence_modified[1], np.array([300, 301, 302, 303]))

    # Nothing to do, sequences are already long enough
    sequence_modified = apply_minimal_number_of_frames(sequences, minimal_number_of_frames=1)
    assert len(sequence_modified) == 4
    assert sequence_modified == sequences


def test_apply_minimal_number_of_frames_empty_sequence():
    """Test apply_minimal_duration."""
    indices = np.array([])
    sequences = split_sequences(indices)
    assert len(sequences) == 0

    sequence_modified = apply_minimal_number_of_frames(sequences, minimal_number_of_frames=1)
    assert len(sequence_modified) == 0


def test_check_direction_alignment():
    """Test _check_direction_alignment function."""
    # Create two sequences
    sequence1 = np.array([0, 1, 2])
    sequence2 = np.array([5, 6, 7])

    # Create gaze direction data
    gaze_direction = np.zeros((3, 10))

    # Set up first sequence direction (moving right)
    gaze_direction[:, 0] = [0, 0, 1]  # Start
    gaze_direction[:, 2] = [0.2, 0, 1]  # End

    # Test 1: Similar direction (also moving right)
    gaze_direction[:, 5] = [0, 0, 1]  # Start
    gaze_direction[:, 7] = [0.3, 0, 1]  # End

    # Should be aligned (angle < 30 degrees)
    assert _check_direction_alignment(sequence1, sequence2, gaze_direction, max_angle=30)

    # Test 2: Different direction (moving left)
    gaze_direction[:, 5] = [0.3, 0, 1]  # Start
    gaze_direction[:, 7] = [0, 0, 1]  # End

    # Should not be aligned (angle > 30 degrees)
    assert not _check_direction_alignment(sequence1, sequence2, gaze_direction, max_angle=30)

    # Test 3: Perpendicular direction (moving up)
    gaze_direction[:, 5] = [0, 0, 1]  # Start
    gaze_direction[:, 7] = [0, 0.3, 1]  # End

    # Should not be aligned (angle = 90 degrees)
    assert not _check_direction_alignment(sequence1, sequence2, gaze_direction, max_angle=30)

    # Test 4: With larger max_angle tolerance
    assert _check_direction_alignment(sequence1, sequence2, gaze_direction, max_angle=100)


def test_can_merge_sequences():
    """Test _can_merge_sequences function."""
    # Create two sequences
    sequence1 = np.array([0, 1, 2])
    sequence2 = np.array([4, 5, 6])

    # Create time vector with 0.1s intervals
    time_vector = np.arange(10) * 0.1

    # Create gaze direction data (all moving in same direction)
    gaze_direction = np.zeros((3, 10))
    for i in range(10):
        gaze_direction[:, i] = [i * 0.1, 0, 1]

    # Create identified indices array (all False)
    identified_indices = np.zeros((10,), dtype=bool)

    # Test 1: Small time gap, no directionality check
    assert _can_merge_sequences(
        sequence1,
        sequence2,
        time_vector,
        gaze_direction,
        identified_indices,
        max_time_gap=0.3,
        check_directionality=False,
        max_angle=30,
    )

    # Test 2: Large time gap, should not merge
    assert not _can_merge_sequences(
        sequence1,
        sequence2,
        time_vector,
        gaze_direction,
        identified_indices,
        max_time_gap=0.01,
        check_directionality=False,
        max_angle=30,
    )

    # Test 3: With directionality check (should pass as they move in same direction)
    assert _can_merge_sequences(
        sequence1,
        sequence2,
        time_vector,
        gaze_direction,
        identified_indices,
        max_time_gap=0.3,
        check_directionality=True,
        max_angle=30,
    )

    # Test 4: With directionality check (should not pass as they move in the opposite direction)
    gaze_direction[:, sequence2] *= -1
    assert not _can_merge_sequences(
        sequence1,
        sequence2,
        time_vector,
        gaze_direction,
        identified_indices,
        max_time_gap=0.3,
        check_directionality=True,
        max_angle=30,
    )

    # Test 5: With identified frames in the gap
    identified_indices[3] = True
    assert not _can_merge_sequences(
        sequence1,
        sequence2,
        time_vector,
        gaze_direction,
        identified_indices,
        max_time_gap=0.3,
        check_directionality=False,
        max_angle=30,
    )


def test_merge_close_sequences():
    """Test merge_close_sequences function."""
    # Create candidate sequences
    sequence1 = np.array([0, 1, 2])
    sequence2 = np.array([4, 5, 6])
    sequence3 = np.array([10, 11, 12])
    sequences = [sequence1, sequence2, sequence3]

    # Create time vector with 0.1s intervals
    time_vector = np.arange(15) * 0.1

    # Create gaze direction data (all moving in same direction)
    gaze_direction = np.zeros((3, 15))
    for i in range(15):
        gaze_direction[:, i] = [i * 0.1, 0, 1]

    # Create identified indices array (all False)
    identified_indices = np.zeros((15,), dtype=bool)

    # Test 1: Merge with small gap
    merged = merge_close_sequences(
        sequences, time_vector, gaze_direction, identified_indices, max_gap=0.3, check_directionality=False
    )

    # Should merge sequence1 and sequence2, but not sequence3
    assert len(merged) == 2
    assert np.array_equal(merged[0], np.arange(0, 7))
    assert np.array_equal(merged[1], sequence3)

    # Test 2: No merging as identified frame in gap
    identified_indices[3] = True  # Add identified frame in gap
    merged = merge_close_sequences(
        sequences, time_vector, gaze_direction, identified_indices, max_gap=0.3, check_directionality=False
    )
    # Should not merge any sequences
    assert len(merged) == 3

    # Test 3: Empty input
    merged = merge_close_sequences(
        [], time_vector, gaze_direction, identified_indices, max_gap=0.3, check_directionality=False
    )
    assert len(merged) == 0

    # Test 4: Single-frame sequences (should be filtered out)
    single_frame_seq = [np.array([0]), np.array([2]), np.array([4])]
    merged = merge_close_sequences(
        single_frame_seq, time_vector, gaze_direction, identified_indices, max_gap=0.3, check_directionality=False
    )
    assert len(merged) == 0

    # Test 5: Merge with same directionality
    identified_indices = np.zeros((15,), dtype=bool)
    merged = merge_close_sequences(
        sequences, time_vector, gaze_direction, identified_indices, max_gap=0.3, check_directionality=True
    )
    # Should merge sequence1 and sequence2, but not sequence3
    assert len(merged) == 2
    assert np.array_equal(merged[0], np.arange(0, 7))
    assert np.array_equal(merged[1], sequence3)

    # Test 6: Do not merge with bad directionality
    gaze_direction[:, 4:7] *= -1  # Change direction for sequence2
    merged = merge_close_sequences(
        sequences, time_vector, gaze_direction, identified_indices, max_gap=0.3, check_directionality=True
    )
    # Should not merge sequence1 and sequence2
    assert len(merged) == 3
    assert np.array_equal(merged[0], sequence1)
    assert np.array_equal(merged[1], sequence2)
    assert np.array_equal(merged[2], sequence3)
