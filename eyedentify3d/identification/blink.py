import numpy as np

from .event import Event
from ..utils.data_utils import DataObject


class BlinkEvent(Event):
    """
    Class to detect blink sequences.
    A blink event is detected when both eye openness drop bellow the threshold (default 0.5).
    ref: https://ieeexplore.ieee.org/abstract/document/9483841
    """

    def __init__(self, data_object: DataObject, eye_openness_threshold: float = 0.5):
        """
        Parameters:
        data_object: The EyeDentify3d object containing the parsed eye-tracking data.
        eye_openness_threshold: The threshold for the eye openness to consider a blink event. Default is 0.5.
        """

        super().__init__()

        # Original attributes
        self.eye_openness_threshold = eye_openness_threshold

        # Detect blink sequences
        self.detect_blink_indices(data_object)
        self.split_sequences()

    def detect_blink_indices(self, data_object: DataObject):
        """
        Detect the frames declared as invalid by the eye-tracker.
        """
        self.frame_indices = np.where(
            np.logical_and(
                data_object.right_eye_openness < self.eye_openness_threshold,
                data_object.left_eye_openness < self.eye_openness_threshold,
            )
        )[0]
