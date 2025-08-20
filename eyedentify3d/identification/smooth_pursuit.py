import numpy as np

from .event import Event
from ..utils.data_utils import DataObject
from ..utils.sequence_utils import split_sequences, merge_close_sequences


class SmoothPursuitEvent(Event):
    """
    Class to detect smooth pursuit sequences.
    See eyedentify3d/identification/inter_sacadic.py for more details on the identification if smooth pursuit indices.
    """

    def __init__(
            self,
            data_object: DataObject,
            identified_indices: np.ndarray,
            smooth_pursuit_indices: np.ndarray,
            minimal_duration: float,
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
        self.minimal_duration = minimal_duration

        # Detect fixation sequences
        self.frame_indices = smooth_pursuit_indices
        self.split_sequences()
        self.merge_sequences(data_object, identified_indices)
        self.keep_only_sequences_long_enough(data_object)
        self.adjust_indices_to_sequences()

    def merge_sequences(self, data_object: DataObject, identified_indices: np.ndarray):
        """
        Modify the sequences detected to merge smooth pursuit sequences that are close in time and have a similar
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
