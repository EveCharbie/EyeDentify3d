import numpy as np

from .rotation_utils import get_angle_between_vectors
from ..time_range import TimeRange
from ..utils.signal_utils import find_time_index


# TODO: This could be transformed into a class that handles the sequences and their properties


def split_sequences(indices: np.ndarray) -> list[np.ndarray]:
    """
    Split an array of indices into an array of sequences of consecutive indices.
    :param indices:
    :return:
    """
    if indices.size == 0:
        return []
    else:
        sequence = np.array_split(
            np.array(indices),
            np.flatnonzero(np.diff(np.array(indices)) > 1) + 1,
        )
        sequence = apply_minimal_number_of_frames(sequence, minimal_number_of_frames=2)
        return sequence


def apply_minimal_duration(
    original_sequences: list[np.ndarray], time_vector: np.ndarray, minimal_duration
) -> list[np.ndarray]:
    """
    Go through the original sequences and remove all sequences that are shorter than the minimal duration.
    This is useful for example to impose that an event can only be categorized as a fixation if it lasts more than 100 ms.

    Parameters
    ----------
    original_sequences: A list of sequences, each sequence is a numpy array of indices.
    time_vector: A numpy array of the time of acquisition of each data frame.
    minimal_duration: The minimal duration in seconds of the event.
    """
    sequences = []
    for i_sequence in original_sequences:
        if len(i_sequence) < 2:
            continue
        event_duration = time_vector[i_sequence[-1]] - time_vector[i_sequence[0]]
        if event_duration < minimal_duration:
            continue
        sequences += [i_sequence]
    return sequences


def apply_minimal_number_of_frames(
    original_sequences: list[np.ndarray], minimal_number_of_frames: int = 2
) -> list[np.ndarray]:
    """
    Go through the original sequences and remove all sequences that are shorter than the minimal number of frames.
    Typically, sequences shorter than 2 frames are (hard to handle and are not considered as valid events anyway).

    Parameters
    ----------
    original_sequences: A list of sequences, each sequence is a numpy array of indices.
    minimal_number_of_frames: The minimal number of frames to consider a sequence.
    """
    sequences = []
    for i_sequence in original_sequences:
        if len(i_sequence) < minimal_number_of_frames:
            continue
        sequences += [i_sequence]
    return sequences


def _check_direction_alignment(
    sequence1: np.ndarray, sequence2: np.ndarray, gaze_direction: np.ndarray, max_angle: float
):
    """
    Check if the gaze is moving in a similar direction during the two sequences.

    Parameters
    ----------
    sequence1: The first sequence of indices.
    sequence2: The second sequence of indices.
    gaze_direction: The gaze direction unit vector, shape (3, n_frames).
    max_angle: The maximum tolerance angle in deg between the movement of the sequences. If the angle of the
        movement during the sequences is smaller than this threshold, the sequences are considered to be similarly
        aligned.
    """
    # The direction vectors are not supposed to be unitary, but we normalize them anyway as they would have been
    # normalized at the angle determination step anyway

    # Calculate direction in which the gaze is moving throughout the sequences
    direction1 = gaze_direction[:, sequence1[-1]] - gaze_direction[:, sequence1[0]]
    direction1 /= np.linalg.norm(direction1)

    direction2 = gaze_direction[:, sequence2[-1]] - gaze_direction[:, sequence2[0]]
    direction2 /= np.linalg.norm(direction2)

    # Calculate angle between directions
    angle = get_angle_between_vectors(direction1, direction2)

    return angle < max_angle


def _can_merge_sequences(
    sequence1: np.ndarray,
    sequence2: np.ndarray,
    time_vector: np.ndarray,
    gaze_direction: np.ndarray,
    identified_indices: np.ndarray,
    max_time_gap: float,
    check_directionality: bool,
    max_angle: float,
):
    """
    Check if the two sequences can be merged together (if the gap between the sequences si smaller than max_time_gap and
    if the frames in the sequences have not already been identified.

    Parameters
    ----------
    sequence1: The first sequence of indices.
    sequence2: The second sequence of indices.
    time_vector: The time vector of the data acquisition.
    gaze_direction: The gaze direction unit vector, shape (3, n_frames).
    identified_indices: A boolean array indicating which frames have already been identified as events.
    max_time_gap: The maximum time gap in seconds between the two sequences to consider them as mergeable.
    check_directionality: If True, check if the gaze is moving in the same direction during the two sequences.
    max_angle: The maximum angle in degrees between the movement of the sequences to consider them as similarly aligned.
        This parameter is only used if check_directionality is True.
    """
    # Define the gap between the two sequences
    gap_start_index = sequence1[-1]
    gap_end_index = sequence2[0]

    # Check if there is a too large gap between them
    time_gap = time_vector[gap_end_index] - time_vector[gap_start_index]
    if time_gap >= max_time_gap:
        return False

    # Check if there was not already an event identified in the gap
    if gap_end_index > gap_start_index and np.any(identified_indices[gap_start_index : gap_end_index + 1]):
        return False

    # Check if gaze is moving in the same direction
    if check_directionality:
        return _check_direction_alignment(sequence1, sequence2, gaze_direction, max_angle)

    return True


def merge_close_sequences(
    sequence_candidates: list[np.ndarray],
    time_vector: np.ndarray,
    gaze_direction: np.ndarray,
    identified_indices: np.ndarray,
    max_gap: float,
    check_directionality: bool = False,
    max_angle: float = 30,
):
    """
    Merge event sequences that are temporally close and similarly aligned (if check_directionality is True).

    Parameters
    ----------
    sequence_candidates: A list of candidate sequences that we consider for merging, each sequence is a numpy array of indices.
    time_vector: The time vector of the data acquisition.
    gaze_direction: The gaze direction unit vector, shape (3, n_frames).
    identified_indices: A boolean array indicating which frames have already been identified as events.
    max_gap: The maximum time gap in seconds between the two sequences to consider them as mergeable.
    check_directionality: If True, check if the gaze is moving in the same direction during the two sequences.
    max_angle: The maximum angle in degrees between the movement of the sequences to consider them as similarly aligned.
        This parameter is only used if check_directionality is True.
    """
    # Remove all candidates that have only one frame
    sequence_candidates = apply_minimal_number_of_frames(sequence_candidates)

    if not sequence_candidates:
        return []

    # The first candidate is always added as a starting point for merging
    merged_sequences = [sequence_candidates[0].copy()]

    for candidate in sequence_candidates[1:]:
        merged_with_existing = False

        # Try to merge with each existing merged sequence
        # TODO: could be optimized by using a more efficient search method
        for i, merged_seq in enumerate(merged_sequences):
            if _can_merge_sequences(
                merged_seq,
                candidate,
                time_vector,
                gaze_direction,
                identified_indices,
                max_gap,
                check_directionality,
                max_angle,
            ):
                # Merge by extending the sequence range
                merged_sequences[i] = np.arange(merged_seq[0], candidate[-1] + 1)
                merged_with_existing = True
                break

        # If couldn't merge with any existing sequence, copy the candidate
        if not merged_with_existing:
            merged_sequences.append(candidate.copy())

    return merged_sequences


def merge_sequence_lists(sequences_1: list[np.ndarray], sequences_2: list[np.ndarray]):
    """
    Merges two lists of sequences and sort them based on the first element of the sequences.
    Note that this function assumes that there is no overlap between the sequences from the two lists.
    """
    all_sequences = sequences_1 + sequences_2
    # Sort by the first index of each sequence
    return sorted(all_sequences, key=lambda seq: seq[0] if len(seq) > 0 else float("inf"))


def get_sequences_in_range(
    time_vector: np.ndarray[float], time_range: TimeRange, sequences: list[np.ndarray[int]]
) -> list[np.ndarray[int]]:
    """
    Get the sequences before and after the timing cue.
    Note: the event occurring during the cue is removed.
    """
    # If sequences is empty, return an empty sequences
    if len(sequences) == 0 or sequences[0].shape == (0,) or sequences[0].shape == (1, 0):
        return sequences
    if time_range.min_time < 1e-6:
        new_first_idx = 0
    else:
        new_first_idx = find_time_index(time_vector, time_range.min_time - 1e-6, method="last")

    sequences_in_range = []
    for sequence in sequences:
        if time_vector[sequence[-1]] <= time_range.max_time and time_vector[sequence[0]] >= time_range.min_time:
            sequences_in_range += [sequence - new_first_idx]

    return sequences_in_range
