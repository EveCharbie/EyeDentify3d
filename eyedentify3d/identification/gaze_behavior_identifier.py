import numpy as np

from ..utils.data_utils import DataObject
from ..error_type import ErrorType
from ..identification.invalid import InvalidEvent
from ..identification.blink import BlinkEvent
from ..identification.saccade import SaccadeEvent


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
        self.blink = None
        self.invalid = None
        self.saccade = None
        self.identified_indices = None
        self._initialize_identified_indices()

    @property
    def data_object(self):
        return self._data_object

    @data_object.setter
    def data_object(self, value: DataObject):
        if not isinstance(value, DataObject):
            raise ValueError(f"The data_object must be an instance of HtcViveProData, got {value}.")
        self._data_object = value

    def _initialize_identified_indices(self):
        self.identified_indices = np.empty((self.data_object.time_vector.shape[0],), dtype=bool)
        self.identified_indices.fill(False)

    def remove_bad_frames(self, event_identifier):
        """
        Removing the date when the eyes are closed (blink is detected) or when the data is invalid, as it does not make
        sense to have a gaze orientation if the eyes are closed.
        """
        if not hasattr(event_identifier, "frame_indices"):
            raise RuntimeError(
                "The event identifier must have a 'frame_indices' attribute. This should not happen, please contact the developer."
            )

        self.data_object.time_vector[event_identifier.frame_indices] = np.nan
        self.data_object.eye_direction[:, event_identifier.frame_indices] = np.nan
        self.data_object.head_angles[:, event_identifier.frame_indices] = np.nan
        self.data_object.gaze_direction[:, event_identifier.frame_indices] = np.nan
        self.data_object.head_angular_velocity[:, event_identifier.frame_indices] = np.nan
        self.data_object.head_velocity_norm[event_identifier.frame_indices] = np.nan
        self.data_object.data_invalidity[event_identifier.frame_indices] = np.nan

    def set_identified_frames(self, event_identifier):
        """
        When an event is identified at a frame, this frame becomes identified and is not available for further
        identification, as events are mutually exclusive and have a detection priority ordering.
        This method should be called each time an event is identified, so that the data object is updated accordingly.
        """
        if not hasattr(event_identifier, "frame_indices"):
            raise RuntimeError(
                "The event identifier must have a 'frame_indices' attribute. This should not happen, please contact the developer."
            )

        self.identified_indices[event_identifier.frame_indices] = True

    def detect_blink_sequences(self):
        self.blink = BlinkEvent(self.data_object)
        self.remove_bad_frames(self.blink)
        self.set_identified_frames(self.blink)

    def detect_invalid_sequences(self):
        self.invalid = InvalidEvent(self.data_object)
        self.remove_bad_frames(self.invalid)
        self.set_identified_frames(self.invalid)

    def detect_saccade_sequences(self):
        self.saccade = SaccadeEvent(self.data_object, self.identified_indices)
        self.set_identified_frames(self.saccade)
