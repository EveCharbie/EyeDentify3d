import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .event import Event
from ..utils.data_utils import DataObject
from ..utils.sequence_utils import merge_close_sequences
from ..utils.check_utils import check_save_name


class VisualScanningEvent(Event):
    """
    Class to detect visual scanning sequences.
    A visual scanning event is detected when the gaze velocity is larger than 100 deg/s, but which are not saccades.
    Please note that the gaze (head + eyes) movements were used to identify visual scanning.
    """

    def __init__(
        self,
        data_object: DataObject,
        identified_indices: np.ndarray = None,
        min_velocity_threshold: float = None,
        minimal_duration: float = None,
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
        self.data_object = data_object
        self.identified_indices = identified_indices
        self.min_velocity_threshold = min_velocity_threshold
        self.minimal_duration = minimal_duration

    def initialize(self):
        self.detect_visual_scanning_indices()
        self.split_sequences()
        self.merge_sequences()
        self.keep_only_sequences_long_enough()
        self.adjust_indices_to_sequences()

    def detect_visual_scanning_indices(self):
        """
        Detect when velocity is above the threshold and if the frames are not already identified.
        """
        visual_scanning = np.abs(self.data_object.gaze_angular_velocity) > self.min_velocity_threshold
        unique_visual_scanning = np.logical_and(visual_scanning, ~self.identified_indices)
        self.frame_indices = np.where(unique_visual_scanning)[0]

    def merge_sequences(self):
        """
        Modify the sequences detected to merge visual scanning sequences that are close in time and have a similar
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

    def add_sequence_to_plot(self, ax: Axes):
        """
        Plot the detected visual scanning events on the provided axis.

        Parameters:
        ax: The matplotlib axis to plot on.
        """
        for sequence in self.sequences:
            start = sequence[0]
            end = sequence[-1]
            ax.axvspan(
                self.data_object.time_vector[start],
                self.data_object.time_vector[end],
                color="tab:pink",
                alpha=0.5,
                edgecolor=None,
            )
        ax.axvspan(
            0,
            0,
            color="tab:pink",
            alpha=0.5,
            edgecolor=None,
            label="Visual scanning",
        )

    def plot(self, save_name: str = None, live_show: bool = True) -> plt.Figure:
        """
        Plot the gaze velocity and detected visual scanning events.

        Parameters
        ----------
        save_name: The name under which to save the figure. If None is provided, the figure is not saved.
        live_show: If the figure should be shown immediately. Please note that showing the figure is blocking.
        """

        fig, axs = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [2, 1]})
        axs[0].set_title("Detected visual scanning events")

        # Plot the gaze vector and the identified visual scanning events
        self.data_object.plot_gaze_vector(ax=axs[0])
        self.add_sequence_to_plot(axs[0])
        axs[0].set_xlim((self.data_object.time_vector[0], self.data_object.time_vector[-1]))
        axs[0].set_ylabel("Gaze orientation [without units]")
        axs[0].legend(bbox_to_anchor=(1.025, 0.5), loc="center left")

        # Plot the gaze velocity
        axs[1].plot(
            self.data_object.time_vector,
            np.abs(self.data_object.gaze_angular_velocity),
            color="tab:pink",
            label="Gaze velocity",
        )
        axs[1].plot(
            np.array([self.data_object.time_vector[0], self.data_object.time_vector[-1]]),
            np.array([self.min_velocity_threshold, self.min_velocity_threshold]),
            "--",
            color="k",
            label=f"Velocity threshold",
        )
        axs[1].set_xlim((self.data_object.time_vector[0], self.data_object.time_vector[-1]))
        axs[1].set_ylabel(r"Gaze velocity [$^\circ/s$]")
        axs[1].legend(bbox_to_anchor=(1.025, 0.5), loc="center left")
        axs[1].set_xlabel("Time [s]")

        plt.subplots_adjust(bottom=0.07, top=0.95, left=0.1, right=0.7, hspace=0.15)

        # If wanted, save the figure
        if save_name is not None:
            extension = check_save_name(save_name)
            plt.savefig(save_name, format=extension)

        if live_show:
            plt.show()

        return fig  # for plot tests
