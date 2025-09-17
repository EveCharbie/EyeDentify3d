import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .event import Event
from ..utils.data_utils import DataObject
from ..utils.sequence_utils import split_sequences, merge_close_sequences
from ..utils.rotation_utils import get_angle_between_vectors, compute_angular_velocity
from ..utils.signal_utils import centered_finite_difference
from ..utils.check_utils import check_save_name


class SaccadeEvent(Event):
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
        identified_indices: np.ndarray = None,
        min_acceleration_threshold: float = 4000,
        velocity_window_size: float = 0.52,
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
        super().__init__()

        # Original attributes
        self.data_object = data_object
        self.identified_indices = identified_indices
        self.min_acceleration_threshold = min_acceleration_threshold
        self.velocity_window_size = velocity_window_size
        self.velocity_factor = velocity_factor

        # Extended attributes
        self.eye_angular_velocity: np.ndarray[float] = None
        self.eye_angular_acceleration: np.ndarray[float] = None
        self.velocity_threshold: np.ndarray[float] = None
        self.saccade_amplitudes: np.ndarray[float] = None

    def initialize(self):
        self.set_eye_angular_velocity()
        self.set_eye_angular_acceleration()
        self.set_the_velocity_threshold()
        self.detect_saccade_indices()
        self.detect_saccade_sequences()
        self.merge_sequences()
        self.adjust_indices_to_sequences()

    def set_eye_angular_velocity(self):
        """
        Computes the eye angular velocity in deg/s as the angle difference between two frames divided by
        the time difference between them. It is computed like a centered finite difference, meaning that the frame i+1
        and i-1 are used to set the value for the frame i.
        """
        self.eye_angular_velocity = compute_angular_velocity(
            self.data_object.time_vector, self.data_object.eye_direction
        )

    def set_eye_angular_acceleration(self):
        """
        Computes the eye angular acceleration in deg/s² as a centered finite difference of the eye angular
        velocity.
        """
        self.eye_angular_acceleration = centered_finite_difference(
            self.data_object.time_vector, self.eye_angular_velocity[np.newaxis, :]
        )[0, :]

    def set_the_velocity_threshold(self):
        """
        Set the velocity threshold based in the velocity_window_size and velocity_factor.
        Note that the velocity threshold changes in time as it is computed using the rolling median of the eye angular
        velocity.
        """
        # Get a number of frames corresponding to the velocity window size
        frame_window_size = int(self.velocity_window_size / self.data_object.dt)

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
        Note that it is not possible to detect a blinks during a saccade, although it is physiologically possible. This
            is due to the fact that we consider sequences to be mutually exclusive (and it is hard to detect if the
            saccade events before and after the blink should have been merges together of not). It is enforced
            implicitly here as if there is a blink detected earlier, the eye_angular_velocity will be nan and the
            saccade condition will not be met.
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

    def merge_sequences(self):
        """
        Modify the sequences detected to merge saccade sequences that are close in time and have a similar direction of
        movement.
        """
        self.sequences = merge_close_sequences(
            self.sequences,
            self.data_object.time_vector,
            self.data_object.gaze_direction,
            self.identified_indices,
            max_gap=0.041656794425087115,  # TODO: make modulable
            check_directionality=True,
            max_angle=30.0,  # TODO: make modulable
        )

    def measure_saccade_amplitude(self):
        """
        Compute the amplitude of each saccade sequence. It is defined as the angle between the beginning and end of the
        saccade in degrees.
        Note that there is no check made to detect if there is a larger amplitude reached during the saccade. If you'd
        prefer this option, you can open an issue on the GitHub repository.
        """
        saccade_amplitudes = []
        for sequence in self.sequences:
            vector_before = self.data_object.eye_direction[:, sequence[0]]
            vector_after = self.data_object.eye_direction[:, sequence[-1]]
            angle = get_angle_between_vectors(vector_before, vector_after)
            saccade_amplitudes += [angle]
        self.saccade_amplitudes = saccade_amplitudes

    def add_sequence_to_plot(self, ax: Axes):
        """
        Plot the detected saccade events on the provided axis.

        Parameters:
        ax: The matplotlib axis to plot on.
        """
        for sequence in self.sequences:
            start = sequence[0]
            end = sequence[-1]
            ax.axvspan(
                self.data_object.time_vector[start],
                self.data_object.time_vector[end],
                color="tab:blue",
                alpha=0.5,
                edgecolor=None,
            )
        ax.axvspan(
            0,
            0,
            color="tab:blue",
            alpha=0.5,
            edgecolor=None,
            label="Saccades",
        )

    def plot(self, save_name: str = None, live_show: bool = True) -> plt.Figure:
        """
        Plot the eye velocity, eye acceleration and detected saccade events.

        Parameters
        ----------
        save_name: The name under which to save the figure. If None is provided, the figure is not saved.
        live_show: If the figure should be shown immediately. Please note that showing the figure is blocking.
        """

        fig, axs = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1, 1]})
        axs[0].set_title("Detected saccade events")

        # Plot the gaze vector and the identified saccades
        self.data_object.plot_gaze_vector(ax=axs[0])
        self.add_sequence_to_plot(axs[0])
        axs[0].set_xlim((self.data_object.time_vector[0], self.data_object.time_vector[-1]))
        axs[0].set_ylabel("Gaze orientation [without units]")
        axs[0].legend(bbox_to_anchor=(1.025, 0.5), loc="center left")

        # Plot the eye velocity
        axs[1].plot(
            self.data_object.time_vector, np.abs(self.eye_angular_velocity), color="tab:blue", label="Eye velocity"
        )
        axs[1].plot(
            self.data_object.time_vector,
            self.velocity_threshold,
            "--",
            color="k",
            label=f"Velocity threshold \n({self.velocity_factor} * medians over \n{self.velocity_window_size}s sliding window)",
        )
        axs[1].set_xlim((self.data_object.time_vector[0], self.data_object.time_vector[-1]))
        axs[1].set_ylabel(r"Eye velocity [$^\circ/s$]")
        axs[1].legend(bbox_to_anchor=(1.025, 0.5), loc="center left")

        # Plot the eye acceleration
        axs[2].plot(
            self.data_object.time_vector, self.eye_angular_acceleration, color="tab:blue", label="Eye acceleration"
        )
        axs[2].axhline(
            self.min_acceleration_threshold,
            color="k",
            linestyle="--",
            label=f"Acceleration threshold",
        )
        axs[2].set_ylabel(r"Eye acceleration [$^\circ/s^2$]")
        axs[2].set_xlim((self.data_object.time_vector[0], self.data_object.time_vector[-1]))
        axs[2].legend(bbox_to_anchor=(1.025, 0.5), loc="center left")
        axs[2].set_xlabel("Time [s]")

        plt.subplots_adjust(bottom=0.07, top=0.95, left=0.1, right=0.7, hspace=0.15)

        # If wanted, save the figure
        if save_name is not None:
            extension = check_save_name(save_name)
            plt.savefig(save_name, format=extension)

        if live_show:
            plt.show()

        return fig  # for plot tests
