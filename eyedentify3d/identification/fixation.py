import numpy as np

from .event import Event
from ..utils.data_utils import DataObject
from ..utils.sequence_utils import merge_close_sequences


class FixationEvent(Event):
    """
    Class to detect fixation sequences.
    See eyedentify3d/identification/inter_sacadic.py for more details on the identification if fixation indices.
    """

    def __init__(
        self,
        data_object: DataObject,
        identified_indices: np.ndarray = None,
        fixation_indices: np.ndarray = None,
        minimal_duration: float = None,
    ):
        """
        Parameters:
        ----------
        data_object: The EyeDentify3d object containing the parsed eye-tracking data.
        identified_indices: A boolean array indicating which frames have already been identified as events.
        fixation_indices: A numpy array of indices where fixations were detected in the InterSaccadicEvent.
        minimal_duration: The minimal duration of the fixation event, in seconds.
        """
        super().__init__()

        # Original attributes
        self.data_object = data_object
        self.identified_indices = identified_indices
        self.fixation_indices = fixation_indices
        self.minimal_duration = minimal_duration

        # Extended attributes
        self.search_rate: float | None = None

    def initialize(self):
        self.frame_indices = self.fixation_indices
        self.split_sequences()
        self.merge_sequences()
        self.adjust_indices_to_sequences()

    def merge_sequences(self):
        """
        Modify the sequences detected to merge fixation sequences that are close in time and have a similar
        direction of movement.
        """
        self.sequences = merge_close_sequences(
            self.sequences,
            self.data_object.time_vector,
            self.data_object.gaze_direction,
            self.identified_indices,
            max_gap=0.040,  # TODO: make modulable
            check_directionality=False,
            max_angle=30.0,  # TODO: make modulable
        )

    def measure_search_rate(self):
        """
        Compute the search rate, which is the number of fixations divided by the mean fixation duration.
        """
        nb_fixations = self.nb_events()
        if nb_fixations == 0:
            self.search_rate = None
        else:
            mean_fixation_duration = self.mean_duration()
            self.search_rate = nb_fixations / mean_fixation_duration
