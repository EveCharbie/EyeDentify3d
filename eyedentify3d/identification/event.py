from typing import Self
from abc import ABC, abstractmethod
import numpy as np

from ..utils.sequence_utils import split_sequences, apply_minimal_duration


class Event(ABC):
    """
    Generic class to detect event sequences.
    """

    def __init__(self):
        # Extended attributes to be filled by the subclasses
        self.frame_indices: np.ndarray | None = None
        self.sequences: list[np.ndarray] = []

    @abstractmethod
    def initialize(self):
        """
        Initialize the event detection.
        This method should be implemented by subclasses to set up the necessary parameters and attributes.
        """
        pass

    def split_sequences(self):
        """
        Split the indices into sequences.
        """
        self.sequences = split_sequences(self.frame_indices)

    def keep_only_sequences_long_enough(self):
        """
        Remove sequences that are too short.
        """
        if not hasattr(self, "minimal_duration"):
            raise AttributeError("The 'minimal_duration' attribute is not set for this event.")
        self.sequences = apply_minimal_duration(self.sequences, self.data_object.time_vector, self.minimal_duration)

    def adjust_indices_to_sequences(self):
        """
        Adjust the frame indices to the sequences after merging and applying minimal duration.
        """
        if len(self.sequences) > 0:
            self.frame_indices = np.concatenate(self.sequences)
        else:
            self.frame_indices = np.array([], dtype=int)

    def from_sequences(self, sequences: list[np.ndarray]) -> Self:
        """
        Set the frame indices from the sequences.
        """
        self.sequences = sequences
        self.frame_indices = np.concatenate(sequences) if sequences else np.array([], dtype=int)
        return self
