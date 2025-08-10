import numpy as np

from ..utils.data_utils import DataObject
from ..utils.sequence_utils import split_sequences, merge_close_sequences
from ..utils.rotation_utils import compute_angular_velocity


class VisualScanningEvent:
    """
    Class to detect visual scanning sequences.
    A visual scanning event is detected when the gaze velocity is larger than 100 deg/s, but which are not saccades.
    Please note that the gaze (head + eyes) movements were used to identify visual scanning.
    """

    def __init__(
        self,
        data_object: DataObject,
        identified_indices: np.ndarray,
        min_velocity_threshold: float = 100,
    ):
        """
        Parameters:
        ----------
        data_object: The EyeDentify3d object containing the parsed eye-tracking data.
        identified_indices: A boolean array indicating which frames have already been identified as events.
        min_velocity_threshold: The minimal threshold for the gaze angular velocity to consider a visual scanning
            event, in deg/s.
        """

        # Original attributes
        self.min_velocity_threshold = min_velocity_threshold

        # Extended attributes
        self.frame_indices: np.ndarray | None = None
        self.sequences: list[np.ndarray] = []
        self.gaze_angular_velocity = None

        # Detect visual scanning sequences
        self.set_gaze_angular_velocity(data_object)
        self.detect_visual_scanning_indices(identified_indices)
        self.detect_visual_scanning_sequences()
        self.merge_sequences(data_object, identified_indices)

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

    def detect_visual_scanning_sequences(self):
        """
        Detect the frames where there is a visual scanning.
        """
        self.sequences = split_sequences(self.frame_indices)

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
        if len(self.sequences) > 0:
            self.frame_indices = np.concatenate(self.sequences)

    #
    # def compute_saccade_amplitude(self, data_object: DataObject):
    #     """
    #     Compute the amplitude of each saccade sequence. It is defined as the angle between the beginning and end of the
    #     saccade in degrees.
    #     Note that there is no check made to detect if there is a larger amplitude reached during the saccade. If you'd
    #     prefer this option, you can open an issue on the GitHub repository.
    #     """
    #     saccade_amplitudes = []
    #     for sequence in self.sequences:
    #         vector_before = data_object.eye_direction[:, sequence[0]]
    #         vector_after = data_object.eye_direction[:, sequence[-1]]
    #         angle = get_angle_between_vectors(vector_before, vector_after)
    #         saccade_amplitudes += [angle]
    #     self.saccade_amplitudes = saccade_amplitudes
