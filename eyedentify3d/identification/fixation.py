import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .event import Event
from ..utils.data_utils import DataObject
from ..utils.sequence_utils import merge_close_sequences
from ..utils.check_utils import check_save_name


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
        self.keep_only_sequences_long_enough()
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

    def add_sequence_to_plot(self, ax: Axes):
        """
        Plot the detected fixation events on the provided axis.

        Parameters:
        ax: The matplotlib axis to plot on.
        """
        for sequence in self.sequences:
            start = sequence[0]
            end = sequence[-1]
            ax.axvspan(
                self.data_object.time_vector[start],
                self.data_object.time_vector[end],
                color="tab:purple",
                alpha=0.5,
                edgecolor=None,
            )
        ax.axvspan(
            0,
            0,
            color="tab:purple",
            alpha=0.5,
            edgecolor=None,
            label="Fixations",
        )

    def plot(self, save_name: str = None, live_show: bool = True) -> plt.Figure:
        """
        Plot the detected fixation events.

        Parameters
        ----------
        save_name: The name under which to save the figure. If None is provided, the figure is not saved.
        live_show: If the figure should be shown immediately. Please note that showing the figure is blocking.
        """

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.set_title("Detected fixation events")

        # Plot the gaze vector and the identified fixations
        self.data_object.plot_gaze_vector(ax=ax)
        self.add_sequence_to_plot(ax)
        ax.set_xlim((self.data_object.time_vector[0], self.data_object.time_vector[-1]))
        ax.set_ylabel("Gaze orientation [without units]")
        ax.legend(bbox_to_anchor=(1.025, 0.5), loc="center left")

        plt.subplots_adjust(bottom=0.15, top=0.90, left=0.1, right=0.7, hspace=0.15)

        # If wanted, save the figure
        if save_name is not None:
            extension = check_save_name(save_name)
            plt.savefig(save_name, format=extension)

        if live_show:
            plt.show()

        return fig  # for plot tests
