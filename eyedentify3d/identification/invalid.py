import numpy as np

from ..utils.data_utils import DataObject
from ..utils.sequence_utils import split_sequences


class InvalidEvent:
    """
    Class to detect invalid sequences.
    An invalid event is detected when the eye-tracker declares the frame as invalid.
    """

    def __init__(self, data_object: DataObject):
        """
        Parameters:
        data_object: The EyeDentify3d object containing the parsed eye-tracking data.
        """

        # Extended attributes
        self.frame_indices: np.ndarray | None = None
        self.sequences: list[np.ndarray] = []

        # Detect invalid sequences
        self.detect_invalid_indices(data_object)
        self.detect_invalid_sequences()

    def detect_invalid_indices(self, data_object: DataObject):
        """
        Detect the frames declared as invalid by the eye-tracker.
        """
        self.frame_indices = np.where(data_object.data_invalidity)[0]

    def detect_invalid_sequences(self):
        """
        Identify invalid sequences.
        """
        self.sequences = split_sequences(self.frame_indices)
