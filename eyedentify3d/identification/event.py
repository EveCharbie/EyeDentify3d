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

    def nb_events(self) -> int:
        """
        Get the number of events detected.
        """
        return len(self.sequences)

    def duration(self) -> np.ndarray[float]:
        """
        Get the duration of each event detected.
        """
        time_vector = self.data_object.time_vector

        durations = []
        for sequence in self.sequences:
            beginning_time = time_vector[sequence[0]]
            if len(time_vector) > sequence[-1]:
                end_time = time_vector[sequence[-1]]
            else:
                end_time = time_vector[-1]

            durations += [end_time - beginning_time]
        return np.array(durations, dtype=float)

    def mean_duration(self) -> float | None:
        """
        Get the mean duration of the events detected.
        """
        return np.mean(self.duration()) if self.nb_events() > 0 else None

    def max_duration(self) -> float | None:
        """
        Get the maximum duration of the events detected.
        """
        return np.max(self.duration()) if self.nb_events() > 0 else None

    def total_duration(self) -> float | None:
        """
        Get the total duration of all events detected.
        """
        return np.sum(self.duration()) if self.nb_events() > 0 else None

    def ratio(self):
        """
        The proportion of the time spent in events compared to the total time of the data object.
        """
        return self.total_duration() / self.data_object.trial_duration if self.nb_events() > 0 else 0.0
