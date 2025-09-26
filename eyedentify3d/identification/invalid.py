import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .event import Event
from ..utils.data_utils import DataObject
from ..utils.check_utils import check_save_name


class InvalidEvent(Event):
    """
    Class to detect invalid sequences.
    An invalid event is detected when the eye-tracker declares the frame as invalid.
    """

    def __init__(self, data_object: DataObject):
        """
        Parameters:
        data_object: The EyeDentify3d object containing the parsed eye-tracking data.
        """
        super().__init__()

        # Original attributes
        self.data_object = data_object

    def initialize(self):
        self.detect_invalid_indices()
        self.split_sequences()

    def detect_invalid_indices(self):
        """
        Detect the frames declared as invalid by the eye-tracker.
        """
        self.frame_indices = np.where(self.data_object.data_invalidity)[0]

    def add_sequence_to_plot(self, ax: Axes):
        """
        Plot the detected invalid events on the provided axis.

        Parameters:
        ax: The matplotlib axis to plot on.
        """
        for sequence in self.sequences:
            start = sequence[0]
            end = sequence[-1]
            ax.axvspan(
                self.data_object.time_vector[start],
                self.data_object.time_vector[end],
                color="tab:red",
                alpha=0.5,
                edgecolor=None,
            )
        ax.axvspan(
            0,
            0,
            color="tab:red",
            alpha=0.5,
            edgecolor=None,
            label="Invalid",
        )

    def plot(self, save_name: str = None, live_show: bool = True) -> plt.Figure:
        """
        Plot the detected invalid frames.

        Parameters
        ----------
        save_name: The name under which to save the figure. If None is provided, the figure is not saved.
        live_show: If the figure should be shown immediately. Please note that showing the figure is blocking.
        """

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.set_title("Detected invalid frames")

        # Plot the gaze vector and the identified invalid frames
        self.data_object.plot_gaze_vector(ax=ax)
        self.add_sequence_to_plot(ax)
        ax.set_xlim((self.data_object.time_vector[0], self.data_object.time_vector[-1]))
        ax.set_ylabel("Gaze orientation [without units]")
        ax.legend(bbox_to_anchor=(1.02, 0.7))
        ax.set_xlabel("Time [s]")

        plt.subplots_adjust(bottom=0.15, top=0.90, left=0.1, right=0.7, hspace=0.15)

        # If wanted, save the figure
        if save_name is not None:
            extension = check_save_name(save_name)
            plt.savefig(save_name, format=extension)

        if live_show:
            plt.show()

        return fig  # for plot tests
