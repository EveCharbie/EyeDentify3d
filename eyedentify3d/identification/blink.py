import numpy as np

from ..utils.data_utils import DataObject
from ..utils.sequence_utils import split_sequences


class BlinkEvent:
    """
    Class to detect blink sequences.
    A blink event is detected when both eye openness drop bellow the threshold (default 0.5).
    ref: https://ieeexplore.ieee.org/abstract/document/9483841
    """
    def __init__(self, data_object: DataObject, eye_openness_threshold: float = 0.5):
        """
        Parameters:
        data_object: The EyeDentify3d object containing the parsed eye-tracking data.
        """

        # Original attributes
        self.eye_openness_threshold = eye_openness_threshold

        # Extended attributes
        self.frame_indices: np.ndarray | None = None
        self.sequences: list[np.ndarray] = []

        # Detect blink sequences
        self.detect_blink_indices(data_object)
        self.detect_blink_sequences()

    def detect_blink_indices(self, data_object: DataObject):
        """
        Detect the frames declared as invalid by the eye-tracker.
        """
        self.frame_indices = np.where(np.logical_and(data_object.right_eye_openness < self.eye_openness_threshold,
                                                     data_object.left_eye_openness < self.eye_openness_threshold))[0]

    def detect_blink_sequences(self):
        """
        Identify invalid sequences.
        """
        self.sequences = split_sequences(self.frame_indices)
