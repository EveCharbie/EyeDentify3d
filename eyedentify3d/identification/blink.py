import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .event import Event
from ..utils.data_utils import DataObject
from ..utils.check_utils import check_save_name


class BlinkEvent(Event):
    """
    Class to detect blink sequences.
    A blink event is detected when both eye openness drop bellow the threshold (default 0.5).
    ref: https://ieeexplore.ieee.org/abstract/document/9483841
    """

    def __init__(self, data_object: DataObject, eye_openness_threshold: float = 0.5):
        """
        Parameters:
        data_object: The EyeDentify3d object containing the parsed eye-tracking data.
        eye_openness_threshold: The threshold for the eye openness to consider a blink event. Default is 0.5.
        """

        super().__init__()

        # Original attributes
        self.data_object = data_object
        self.eye_openness_threshold = eye_openness_threshold

    def initialize(self):
        self.detect_blink_indices()
        self.split_sequences()

    def detect_blink_indices(self):
        """
        Detect the frames declared as invalid by the eye-tracker.
        """
        self.frame_indices = np.where(
            np.logical_and(
                self.data_object.right_eye_openness < self.eye_openness_threshold,
                self.data_object.left_eye_openness < self.eye_openness_threshold,
            )
        )[0]

    def add_sequence_to_plot(self, ax: Axes):
        """
        Plot the detected blink events on the provided axis.

        Parameters:
        ax: The matplotlib axis to plot on.
        """
        for sequence in self.sequences:
            start = sequence[0]
            end = sequence[-1]
            ax.axvspan(
                self.data_object.time_vector[start],
                self.data_object.time_vector[end],
                color="tab:green",
                alpha=0.5,
                edgecolor=None,
            )
        ax.axvspan(
            0,
            0,
            color="tab:green",
            alpha=0.5,
            edgecolor=None,
            label="Blinks",
        )

    def plot(self, save_name: str = None, live_show: bool = True) -> plt.Figure:
        """
        Plot the eye openness and detected blink events.

        Parameters
        ----------
        save_name: The name under which to save the figure. If None is provided, the figure is not saved.
        live_show: If the figure should be shown immediately. Please note that showing the figure is blocking.
        """

        fig, axs = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [2, 1]})
        axs[0].set_title("Detected blink events")

        # Plot the gaze vector and the identified blinks
        self.data_object.plot_gaze_vector(ax=axs[0])
        self.add_sequence_to_plot(axs[0])
        axs[0].set_xlim((self.data_object.time_vector[0], self.data_object.time_vector[-1]))
        axs[0].set_ylabel("Gaze orientation [without units]")
        axs[0].legend(bbox_to_anchor=(1.025, 0.5), loc="center left")

        # Plot the eye openness
        axs[1].plot(
            self.data_object.time_vector,
            self.data_object.right_eye_openness,
            label="Right Eye Openness",
            color="tab:blue",
        )
        axs[1].plot(
            self.data_object.time_vector,
            self.data_object.right_eye_openness,
            label="Left Eye Openness",
            color="tab:orange",
        )
        axs[1].axhline(
            self.eye_openness_threshold,
            color="k",
            linestyle="--",
            label="Eye Openness Threshold",
        )
        axs[1].set_xlim((self.data_object.time_vector[0], self.data_object.time_vector[-1]))
        axs[1].legend(bbox_to_anchor=(1.025, 0.5), loc="center left")
        axs[1].set_xlabel("Time [s]")
        axs[1].set_ylabel("Eye Openness [without units]")

        plt.subplots_adjust(bottom=0.07, top=0.95, left=0.1, right=0.7, hspace=0.15)

        # If wanted, save the figure
        if save_name is not None:
            extension = check_save_name(save_name)
            plt.savefig(save_name, format=extension)

        if live_show:
            plt.show()

        return fig  # for plot tests
