import numpy as np

from ..utils.data_utils import DataObject
from ..error_type import ErrorType
from ..identification.invalid import InvalidEvent
from ..identification.blink import BlinkEvent
from ..identification.saccade import SaccadeEvent
from ..identification.visual_scanning import VisualScanningEvent
from ..identification.inter_saccades import InterSaccadicEvent
from ..identification.fixation import FixationEvent
from ..identification.smooth_pursuit import SmoothPursuitEvent


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
        self.blink: BlinkEvent = None
        self.invalid: InvalidEvent = None
        self.saccade: SaccadeEvent = None
        self.visual_scanning: VisualScanningEvent = None
        self.inter_saccadic_sequences: InterSaccadicEvent = None
        self.fixation: FixationEvent = None
        self.smooth_pursuit: SmoothPursuitEvent = None
        self.identified_indices: np.ndarray[int] = None
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

    def detect_blink_sequences(self, eye_openness_threshold: float = 0.5):
        """
        Detects blink sequences in the data object.

        Parameters
        ----------
        eye_openness_threshold: The threshold for the eye openness to consider a blink event. Default is 0.5.
        """
        self.blink = BlinkEvent(self.data_object, eye_openness_threshold)
        self.remove_bad_frames(self.blink)
        self.set_identified_frames(self.blink)

    def detect_invalid_sequences(self):
        """
        Detects invalid sequences in the data object.
        """
        self.invalid = InvalidEvent(self.data_object)
        self.remove_bad_frames(self.invalid)
        self.set_identified_frames(self.invalid)

    def detect_saccade_sequences(
        self,
        min_acceleration_threshold: float = 4000,
        velocity_window_size: float = 0.52,
        velocity_factor: float = 5.0,
    ):
        """
        Detects saccade sequences in the data object.

        Parameters
        ----------
        min_acceleration_threshold: The minimal threshold for the eye angular acceleration to consider a saccade in deg/s².
        velocity_window_size: The length in seconds of the window used to compute the rolling median of the eye angular
            velocity. This rolling median is used to identify when the eye angular velocity is larger than usual.
        velocity_factor: The factor by which the eye angular velocity must be larger than the rolling median to consider
            a saccade. Default is 5, meaning that the eye angular velocity must be larger than 5 times the rolling
            median to be considered a saccade.
        """
        self.saccade = SaccadeEvent(
            self.data_object,
            self.identified_indices,
            min_acceleration_threshold,
            velocity_window_size,
            velocity_factor,
        )
        self.set_identified_frames(self.saccade)

    def detect_visual_scanning_sequences(self,
                                         min_velocity_threshold: float = 100,
                                         minimal_duration: float = 0.1):
        """
        Detects visual scanning sequences in the data object.

        Parameters
        ----------
        min_velocity_threshold: The minimal threshold for the gaze angular velocity to consider a visual scanning
            event, in deg/s. Default is 100 deg/s.
        minimal_duration: The minimal duration of the visual scanning event, in seconds. Default is 0.1 seconds.
        """
        self.visual_scanning = VisualScanningEvent(
            self.data_object,
            self.identified_indices,
            min_velocity_threshold,
            minimal_duration,
        )
        # Remove frames where visual scanning events are detected
        self.set_identified_frames(self.visual_scanning)
        # Also remove all frames where the velocity is above threshold, as these frames are not available for the
        # detection of other events. Please note that these frames might not be part of a visual scanning event if the
        # velocity is not maintained for at least minimal_duration.
        high_velocity_condition = np.abs(self.visual_scanning.gaze_angular_velocity) > self.visual_scanning.min_velocity_threshold
        self.identified_indices[high_velocity_condition] = True

    def detect_fixation_and_smooth_pursuit_sequences(
        self,
        minimal_duration: float = 0.045,
        window_duration: float = 0.022,
        window_overlap: float = 0.006,
        eta_p: float = 0.01,
        eta_d: float = 0.45,
        eta_cd: float = 0.5,
        eta_pd: float = 0.2,
        eta_max_fixation: float = 1.9,
        eta_min_smooth_pursuit: float = 1.7,
        phi: float = 45,
    ):
        """
        Detects fixation and smooth pursuit sequences in the data object.

        Parameters
        ----------
        minimal_duration: The minimal duration of the fixation or smooth pursuit event, in seconds. This minimal
            duration is also applied to the inter-saccadic sequences.
        window_duration: The duration of the window (in seconds) used to compute the coherence of the inter-saccadic
            sequences.
        window_overlap: The overlap between two consecutive windows (in seconds)
        eta_p: The threshold for the p-value of the Rayleigh test to classify the inter-saccadic sequences as coherent
            or incoherent.
        eta_d: The threshold for the gaze direction dispersion (without units).
        eta_cd: The threshold for the consistency of direction (without units).
        eta_pd: The threshold for the position displacement (without units).
        phi: The threshold for the similar angular range (in degrees).
        eta_max_fixation: The threshold for the maximum fixation range (in degrees).
        eta_min_smooth_pursuit: The threshold for the minimum smooth pursuit range (in degrees).

        Note that the default values for the parameters
            `minimal_duration` = 40 ms
            `window_duration` = 22 ms
            `window_overlap` = 6 ms
            `eta_p` = 0.01
            `eta_d` = 0.45
            `eta_cd` = 0.5
            `eta_pd` = 0.2
            `eta_max_fixation` = 1.9 deg
            `eta_min_smooth_pursuit` = 1.7 deg
            `phi` = 45 deg
            are taken from Larsson et al. (2015), but they should be modified to fit your experimental setup
            (acquisition frequency and task).
        """
        self.inter_saccadic_sequences = InterSaccadicEvent(
            self.data_object,
            self.identified_indices,
            minimal_duration,
            window_duration,
            window_overlap,
            eta_p,
            eta_d,
            eta_cd,
            eta_pd,
            eta_max_fixation,
            eta_min_smooth_pursuit,
            phi,
        )

        self.fixation = FixationEvent(
            self.data_object, self.identified_indices, self.inter_saccadic_sequences.fixation_indices, minimal_duration
        )
        self.smooth_pursuit = SmoothPursuitEvent(
            self.data_object,
            self.identified_indices,
            self.inter_saccadic_sequences.smooth_pursuit_indices,
            minimal_duration,
        )
        self.set_identified_frames(self.fixation)
        self.set_identified_frames(self.smooth_pursuit)
