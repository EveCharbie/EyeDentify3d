import numpy as np

from .event import Event
from ..utils.data_utils import DataObject
from ..utils.sequence_utils import merge_close_sequences
from ..utils.rotation_utils import compute_angular_velocity


class VisualScanningEvent(Event):
    """
    Class to detect visual scanning sequences.
    A visual scanning event is detected when the gaze velocity is larger than 100 deg/s, but which are not saccades.
    Please note that the gaze (head + eyes) movements were used to identify visual scanning.
    """

    def __init__(
        self,
        data_object: DataObject,
        identified_indices: np.ndarray,
        min_velocity_threshold: float,
        minimal_duration: float,
    ):
        """
        Parameters:
        ----------
        data_object: The EyeDentify3d object containing the parsed eye-tracking data.
        identified_indices: A boolean array indicating which frames have already been identified as events.
        min_velocity_threshold: The minimal threshold for the gaze angular velocity to consider a visual scanning
            event, in deg/s.
        minimal_duration: The minimal duration of the visual scanning event, in seconds.
        """
        super().__init__()

        # Original attributes
        self.min_velocity_threshold = min_velocity_threshold
        self.minimal_duration = minimal_duration

        # Extended attributes
        self.gaze_angular_velocity = None

        # Detect visual scanning sequences
        self.set_gaze_angular_velocity(data_object)
        self.detect_visual_scanning_indices(identified_indices)
        self.split_sequences()
        self.merge_sequences(data_object, identified_indices)
        self.keep_only_sequences_long_enough(data_object)
        self.adjust_indices_to_sequences()

    def set_gaze_angular_velocity(self, data_object: DataObject):
        """
        Computes the gaze (eye + head) angular velocity in deg/s as the angle difference between two frames divided by
        the time difference between them. It is computed like a centered finite difference, meaning that the frame i+1
        and i-1 are used to set the value for the frame i.
        """
        self.gaze_angular_velocity = compute_angular_velocity(data_object.time_vector, data_object.gaze_direction)

    def detect_visual_scanning_indices(self, identified_indices: np.ndarray):
        """
        Detect when velocity is above the threshold and if the frames are not already identified.
        """
        visual_scanning = np.abs(self.gaze_angular_velocity) > self.min_velocity_threshold
        unique_visual_scanning = np.logical_and(visual_scanning, ~identified_indices)
        self.frame_indices = np.where(unique_visual_scanning)[0]

    def merge_sequences(self, data_object: DataObject, identified_indices: np.ndarray):
        """
        Modify the sequences detected to merge visual scanning sequences that are close in time and have a similar
        direction of movement.
        """
        self.sequences = merge_close_sequences(
            self.sequences,
            data_object.time_vector,
            data_object.gaze_direction,
            identified_indices,
            max_gap=0.040,  # TODO: make modulable
            check_directionality=True,
            max_angle=30.0,  # TODO: make modulable
        )
