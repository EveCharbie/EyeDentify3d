import numpy as np

from ..utils.data_utils import DataObject
from ..utils.sequence_utils import split_sequences, merge_close_sequences
from ..utils.rotation_utils import get_angle_between_vectors
from ..utils.signal_utils import centered_finite_difference


class SaccadeEvent:
    """
    Class to detect saccade sequences.
    A saccade event is detected when both conditions are met:
        1. The eye angular velocity is larger than `velocity_factor` times the rolling median on the current window of
        length `velocity_window_size`.
        2. The eye angular acceleration is larger than `min_acceleration_threshold` deg/s² for at least two frames
    Please note that only the eye (not gaze) movements were used to identify saccades.
    """

    def __init__(
        self,
        data_object: DataObject,
        identified_indices: np.ndarray,
        min_acceleration_threshold: float = 4000,
        velocity_window_size: float = 0.52,  # TODO: make modulable
        velocity_factor: float = 5.0,
    ):
        """
        Parameters:
        ----------
        data_object: The EyeDentify3d object containing the parsed eye-tracking data.
        identified_indices: A boolean array indicating which frames have already been identified as events.
        min_acceleration_threshold: The minimal threshold for the eye angular acceleration to consider a saccade in deg/s².
        velocity_window_size: The length in seconds of the window used to compute the rolling median of the eye angular
        velocity. This rolling median is used to identify when the eye angular velocity is larger than usual.
        velocity_factor: The factor by which the eye angular velocity must be larger than the rolling median to consider
            a saccade. Default is 5, meaning that the eye angular velocity must be larger than 5 times the rolling
            median to be considered a saccade.
        """

        # Original attributes
        self.min_acceleration_threshold = min_acceleration_threshold
        self.velocity_window_size = velocity_window_size
        self.velocity_factor = velocity_factor

        # Extended attributes
        self.frame_indices: np.ndarray | None = None
        self.sequences: list[np.ndarray] = []
        self.eye_angular_velocity = None
        self.eye_angular_acceleration = None
        self.velocity_threshold = None
        self.saccade_amplitudes = None

        # Detect blink sequences
        self.set_eye_angular_velocity(data_object)
        self.set_eye_angular_acceleration(data_object)
        self.set_the_velocity_threshold(data_object)
        self.detect_saccade_indices()
        self.detect_saccade_sequences()
        self.merge_sequences(data_object, identified_indices)
        self.compute_saccade_amplitude(data_object)

    def set_eye_angular_velocity(self, data_object: DataObject):
        """
        Computes the eye angular velocity in deg/s as the angle difference between two frames divided by
        the time difference between them. It is computed like a centered finite difference, meaning that the frame i+1
        and i-1 are used to set the value for the frame i.
        """
        eye_angular_velocity = np.zeros((data_object.eye_direction.shape[1],))
        for i_frame in range(1, data_object.eye_direction.shape[1] - 1):  # Skipping the first and last frames
            vector_before = data_object.eye_direction[:, i_frame - 1]
            vector_after = data_object.eye_direction[:, i_frame + 1]
            angle = get_angle_between_vectors(vector_before, vector_after)
            eye_angular_velocity[i_frame] = angle / (
                data_object.time_vector[i_frame + 1] - data_object.time_vector[i_frame - 1]
            )

        # Deal with the first and last frames separately
        first_angle = get_angle_between_vectors(data_object.eye_direction[:, 0], data_object.eye_direction[:, 1])
        eye_angular_velocity[0] = first_angle / (data_object.time_vector[1] - data_object.time_vector[0])
        last_angle = get_angle_between_vectors(data_object.eye_direction[:, -2], data_object.eye_direction[:, -1])
        eye_angular_velocity[-1] = last_angle / (data_object.time_vector[-1] - data_object.time_vector[-2])

        self.eye_angular_velocity = eye_angular_velocity * 180 / np.pi  # Convert to degrees per second

    def set_eye_angular_acceleration(self, data_object: DataObject):
        """
        Computes the eye angular acceleration in deg/s² as a centered finite difference of the eye angular
        velocity.
        """
        self.eye_angular_acceleration = centered_finite_difference(
            data_object.time_vector, self.eye_angular_velocity[np.newaxis, :]
        )[0, :]

    def set_the_velocity_threshold(self, data_object: DataObject):
        """
        Set the velocity threshold based in the velocity_window_size and velocity_factor.
        Note that the velocity threshold changes in time as it is computed using the rolling median of the eye angular
        velocity.
        """
        # Get a number of frames corresponding to the velocity window size
        frame_window_size = int(self.velocity_window_size / data_object.dt)

        # Compute the velocity threshold
        velocity_threshold = np.zeros((self.eye_angular_velocity.shape[0],))
        # Deal with the first frames separately
        velocity_threshold[: int(frame_window_size / 2)] = np.nanmedian(
            np.abs(self.eye_angular_velocity[:frame_window_size])
        )
        for i_frame in range(self.eye_angular_velocity.shape[0] - frame_window_size):
            velocity_threshold[int(i_frame + frame_window_size / 2)] = np.nanmedian(
                np.abs(self.eye_angular_velocity[i_frame : i_frame + frame_window_size])
            )
        # Deal with the last frames separately
        velocity_threshold[int(-frame_window_size / 2) :] = np.nanmedian(
            np.abs(self.eye_angular_velocity[-frame_window_size:])
        )
        self.velocity_threshold = velocity_threshold * self.velocity_factor

    def detect_saccade_indices(self):
        """
        Detect when velocity is above the threshold.
        """
        self.frame_indices = np.where(np.abs(self.eye_angular_velocity) > self.velocity_threshold)[0]

    def detect_saccade_sequences(self):
        """
        Detect the frames where there is a saccade.
        """
        # Get saccade sequences
        saccade_sequence_candidates = split_sequences(self.frame_indices)

        # Only seep the sequences where the eye angular acceleration is above the threshold for at least two frames
        # There should be at least one acceleration to leave the current fixation and one deceleration on target arrival.
        self.sequences = []
        if saccade_sequence_candidates[0].shape != (0,):
            for i in saccade_sequence_candidates:
                if len(i) <= 1:
                    # One frame is not long enough for a sequence
                    continue
                acceleration_above_threshold = np.where(
                    np.abs(self.eye_angular_acceleration[i[0] - 1 : i[-1] + 1]) > self.min_acceleration_threshold
                )[0]
                if len(acceleration_above_threshold) > 1:
                    self.sequences += [i]

    def merge_sequences(self, data_object: DataObject, identified_indices: np.ndarray):
        """
        Modify the sequences detected to merge saccade sequences that are close in time and have a similar direction of
        movement.
        """
        self.sequences = merge_close_sequences(
            self.sequences,
            data_object.time_vector,
            data_object.gaze_direction,
            identified_indices,
            max_gap=0.041656794425087115,  # TODO: make modulable
            check_directionality=True,
            max_angle=30.0,  # TODO: make modulable
        )
        self.frame_indices = np.concatenate(self.sequences)

    def compute_saccade_amplitude(self, data_object: DataObject):
        """
        Compute the amplitude of each saccade sequence. It is defined as the angle between the beginning and end of the
        saccade in degrees.
        Note that there is no check made to detect if there is a larger amplitude reached during the saccade. If you'd
        prefer this option, you can open an issue on the GitHub repository.
        """
        saccade_amplitudes = []
        for sequence in self.sequences:
            vector_before = data_object.eye_direction[:, sequence[0]]
            vector_after = data_object.eye_direction[:, sequence[-1]]
            angle = get_angle_between_vectors(vector_before, vector_after)
            saccade_amplitudes += [angle * 180 / np.pi]
        self.saccade_amplitudes = saccade_amplitudes
