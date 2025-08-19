import numpy as np

from ..utils.data_utils import DataObject
from ..utils.sequence_utils import split_sequences, merge_close_sequences


class FixationEvent:
    """
    Class to detect fixation sequences.
    See eyedentify3d/identification/inter_sacadic.py for more details on the identification if fixation indices.
    """

    def __init__(
        self,
        data_object: DataObject,
        identified_indices: np.ndarray,
        fixation_indices: np.ndarray,
    ):
        """
        Parameters:
        ----------
        data_object: The EyeDentify3d object containing the parsed eye-tracking data.
        identified_indices: A boolean array indicating which frames have already been identified as events.
        fixation_indices: A numpy array of indices where fixations were detected in the InterSaccadicEvent.
        """

        # Extended attributes
        self.frame_indices = fixation_indices
        self.sequences: list[np.ndarray] = []

        # Detect fixation sequences
        self.detect_fixation_sequences()
        self.merge_sequences(data_object, identified_indices)

    def detect_fixation_sequences(self):
        """
        Detect the frames where there is a fixation.
        """
        self.sequences = split_sequences(self.frame_indices)

    def merge_sequences(self, data_object: DataObject, identified_indices: np.ndarray):
        """
        Modify the sequences detected to merge fixation sequences that are close in time and have a similar
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
