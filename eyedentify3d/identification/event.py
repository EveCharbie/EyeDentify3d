import numpy as np

from ..utils.data_utils import DataObject
from ..utils.sequence_utils import split_sequences, apply_minimal_duration


class Event:
    """
    Generic class to detect event sequences.
    """

    def __init__(self):
        # Extended attributes to be filled by the subclasses
        self.frame_indices: np.ndarray | None = None
        self.sequences: list[np.ndarray] = []

    def split_sequences(self):
        """
        Split the indices into sequences.
        """
        self.sequences = split_sequences(self.frame_indices)

    def keep_only_sequences_long_enough(self, data_object: DataObject):
        """
        Remove sequences that are too short.
        """
        if not hasattr(self, "minimal_duration"):
            raise AttributeError("The 'minimal_duration' attribute is not set for this event.")
        self.sequences = apply_minimal_duration(self.sequences, data_object.time_vector, self.minimal_duration)

    def adjust_indices_to_sequences(self):
        """
        Adjust the frame indices to the sequences after merging and applying minimal duration.
        """
        if len(self.sequences) > 0:
            self.frame_indices = np.concatenate(self.sequences)
        else:
            self.frame_indices = np.array([], dtype=int)
