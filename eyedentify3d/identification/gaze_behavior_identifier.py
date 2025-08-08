import numpy as np

from ..utils.data_utils import DataObject
from ..error_type import ErrorType
from ..identification.invalid import InvalidEvent
from ..identification.blink import BlinkEvent


class GazeBehaviorIdentifier:
    """
    The main object to identify gaze behavior.
    Please note that the `data_object` will be modified each time an event is detected, so that only the frames
    available for detection are left in the object.
    """

    def __init__(
        self,
        data_object: DataObject,
        error_type: ErrorType = ErrorType.PRINT,
    ):
        """
        Parameters
        ----------
        data_object: The EyeDentify3d object containing the parsed eye-tracking data.
        error_type: How to handle errors. Default is ErrorType.PRINT, which prints the error message.
        """
        # Initial attributes
        self.data_object = data_object
        self.error_type = error_type

        # Extended attributes
        self.available_frames = np.arange(len(self.data_object.time_vector))
        self.blink = None
        self.invalid = None

    @property
    def data_object(self):
        return self._data_object

    @data_object.setter
    def data_object(self, value: DataObject):
        if not isinstance(value, DataObject):
            raise ValueError(f"The data_object must be an instance of HtcViveProData, got {value}.")
        self._data_object = value

    def remove_identified_frames(self, event_identifier):
        """
        When an event is identified at a frame, this frame is removed from the data object, as events are mutually
        exclusive and have a detection priority ordering.
        This method should be called each time an event is identified, so that the data object is updated accordingly.
        """
        if not hasattr(event_identifier, "frame_indices"):
            raise RuntimeError("The event identifier must have a 'frame_indices' attribute. This should not happen, please contact the developer.")

        self.data_object.time_vector[event_identifier.frame_indices] = np.nan
        self.data_object.eye_direction[:, event_identifier.frame_indices] = np.nan
        self.data_object.head_angles[:, event_identifier.frame_indices] = np.nan
        self.data_object.head_angular_velocity[:, event_identifier.frame_indices] = np.nan
        self.data_object.head_velocity_norm[event_identifier.frame_indices] = np.nan
        self.data_object.data_validity[event_identifier.frame_indices] = np.nan
        self.available_frames[event_identifier.frame_indices] = False

    def detect_blink_sequences(self):
        self.blink = BlinkEvent(self.data_object)
        self.remove_identified_frames(self.blink)

    def detect_invalid_sequences(self):
        self.invalid = InvalidEvent(self.data_object)
        self.remove_identified_frames(self.invalid)
