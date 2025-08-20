import numpy as np

from .event import Event
from ..utils.data_utils import DataObject


class InvalidEvent(Event):
    """
    Class to detect invalid sequences.
    An invalid event is detected when the eye-tracker declares the frame as invalid.
    """

    def __init__(self, data_object: DataObject):
        """
        Parameters:
        data_object: The EyeDentify3d object containing the parsed eye-tracking data.
        """
        super().__init__()

        # Original attributes
        self.data_object = data_object

    def initialize(self):
        self.detect_invalid_indices()
        self.split_sequences()

    def detect_invalid_indices(self):
        """
        Detect the frames declared as invalid by the eye-tracker.
        """
        self.frame_indices = np.where(self.data_object.data_invalidity)[0]
