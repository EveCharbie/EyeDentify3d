import numpy as np

from .event import Event
from ..utils.data_utils import DataObject
from ..utils.sequence_utils import merge_close_sequences


class SmoothPursuitEvent(Event):
    """
    Class to detect smooth pursuit sequences.
    See eyedentify3d/identification/inter_sacadic.py for more details on the identification if smooth pursuit indices.
    """

    def __init__(
        self,
        data_object: DataObject,
        identified_indices: np.ndarray = None,
        smooth_pursuit_indices: np.ndarray = None,
        minimal_duration: float = None,
    ):
        """
        Parameters:
        ----------
        data_object: The EyeDentify3d object containing the parsed eye-tracking data.
        identified_indices: A boolean array indicating which frames have already been identified as events.
        smooth_pursuit_indices: A numpy array of indices where smooth pursuits were detected in the InterSaccadicEvent.
        minimal_duration: The minimal duration of the fixation event, in seconds.
        """
        super().__init__()

        # Original attributes
        self.data_object = data_object
        self.identified_indices = identified_indices
        self.smooth_pursuit_indices = smooth_pursuit_indices
        self.minimal_duration = minimal_duration

        # Extended attributes
        self.smooth_pursuit_trajectories: list[float] = None

    def initialize(self):
        self.frame_indices = self.smooth_pursuit_indices
        self.split_sequences()
        self.merge_sequences()
        self.keep_only_sequences_long_enough()
        self.adjust_indices_to_sequences()

    def merge_sequences(self):
        """
        Modify the sequences detected to merge smooth pursuit sequences that are close in time and have a similar
        direction of movement.
        """
        self.sequences = merge_close_sequences(
            self.sequences,
            self.data_object.time_vector,
            self.data_object.gaze_direction,
            self.identified_indices,
            max_gap=0.040,  # TODO: make modulable
            check_directionality=True,
            max_angle=30.0,  # TODO: make modulable
        )

    def measure_smooth_pursuit_trajectory(self):
        """
        Compute the length of the smooth pursuit trajectory as the sum of the angle between two frames in degrees.
        It can be seen as the integral of the angular velocity.
        """
        smooth_pursuit_trajectories = []
        for sequence in self.sequences:
            trajectory_this_time = 0
            for idx in sequence:
                time_beginning = self.data_object.time_vector[idx]
                if idx + 1 < len(self.data_object.time_vector):
                    time_end = self.data_object.time_vector[idx + 1]
                else:
                    time_end = self.data_object.time_vector[-1] + self.data_object.dt
                d_trajectory = np.abs(self.data_object.gaze_angular_velocity[idx]) * (time_end - time_beginning)
                if not np.isnan(d_trajectory):
                    trajectory_this_time += d_trajectory
            smooth_pursuit_trajectories += [trajectory_this_time]
        self.smooth_pursuit_trajectories = smooth_pursuit_trajectories
