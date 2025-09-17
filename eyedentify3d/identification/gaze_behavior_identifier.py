from typing import Self
import numpy as np
import matplotlib.pyplot as plt


from ..utils.data_utils import DataObject
from ..utils.sequence_utils import get_sequences_in_range
from ..utils.check_utils import check_save_name
from ..time_range import TimeRange
from ..data_parsers.reduced_data import ReducedData
from ..error_type import ErrorType
from ..identification.invalid import InvalidEvent
from ..identification.blink import BlinkEvent
from ..identification.saccade import SaccadeEvent
from ..identification.visual_scanning import VisualScanningEvent
from ..identification.inter_saccades import InterSaccadicEvent
from ..identification.fixation import FixationEvent
from ..identification.smooth_pursuit import SmoothPursuitEvent


class GazeBehaviorIdentifier:
    """
    The main object to identify gaze behavior.
    Please note that the `data_object` will be modified each time an event is detected, so that only the frames
    available for detection are left in the object.
    """

    def __init__(
        self,
        data_object: DataObject,
    ):
        """
        Parameters
        ----------
        data_object: The EyeDentify3d object containing the parsed eye-tracking data.
        """
        # Initial attributes
        self.data_object = data_object

        # Extended attributes
        self.blink: BlinkEvent = None
        self.invalid: InvalidEvent = None
        self.saccade: SaccadeEvent = None
        self.visual_scanning: VisualScanningEvent = None
        self.inter_saccadic_sequences: InterSaccadicEvent = None
        self.fixation: FixationEvent = None
        self.smooth_pursuit: SmoothPursuitEvent = None
        self.identified_indices: np.ndarray[int] = None
        self.fast_frame_indices: np.ndarray[bool] = None
        self._initialize_identified_indices()
        self.is_finalized = False

    @property
    def data_object(self):
        return self._data_object

    @data_object.setter
    def data_object(self, value: DataObject):
        if not isinstance(value, DataObject):
            raise ValueError(f"The data_object must be an instance of HtcViveProData, got {value}.")
        self._data_object = value

    def _initialize_identified_indices(self):
        self.identified_indices = np.empty((self.data_object.time_vector.shape[0],), dtype=bool)
        self.identified_indices.fill(False)

    def remove_bad_frames(self, event_identifier):
        """
        Removing the frames where the eyes are closed (blink is detected) or when the data is invalid, as it does not
        make sense to have a gaze orientation when the eyes are closed.
        """
        if not hasattr(event_identifier, "frame_indices"):
            raise RuntimeError(
                "The event identifier must have a 'frame_indices' attribute. This should not happen, please contact the developer."
            )

        self.data_object.eye_direction[:, event_identifier.frame_indices] = np.nan
        self.data_object.gaze_direction[:, event_identifier.frame_indices] = np.nan

    def set_identified_frames(self, event_identifier):
        """
        When an event is identified at a frame, this frame becomes identified and is not available for further
        identification, as events are mutually exclusive and have a detection priority ordering.
        This method should be called each time an event is identified, so that the data object is updated accordingly.
        """
        if not hasattr(event_identifier, "frame_indices"):
            raise RuntimeError(
                "The event identifier must have a 'frame_indices' attribute. This should not happen, please contact the developer."
            )

        self.identified_indices[event_identifier.frame_indices] = True

    def detect_blink_sequences(self, eye_openness_threshold: float = 0.5):
        """
        Detects blink sequences in the data object.

        Parameters
        ----------
        eye_openness_threshold: The threshold for the eye openness to consider a blink event. Default is 0.5.
        """
        self.blink = BlinkEvent(self.data_object, eye_openness_threshold)
        self.blink.initialize()
        self.remove_bad_frames(self.blink)
        self.set_identified_frames(self.blink)

    def detect_invalid_sequences(self):
        """
        Detects invalid sequences in the data object.
        """
        self.invalid = InvalidEvent(self.data_object)
        self.invalid.initialize()
        self.remove_bad_frames(self.invalid)
        self.set_identified_frames(self.invalid)

    def detect_saccade_sequences(
        self,
        min_acceleration_threshold: float = 4000,
        velocity_window_size: float = 0.52,
        velocity_factor: float = 5.0,
    ):
        """
        Detects saccade sequences in the data object.

        Parameters
        ----------
        min_acceleration_threshold: The minimal threshold for the eye angular acceleration to consider a saccade in deg/s².
        velocity_window_size: The length in seconds of the window used to compute the rolling median of the eye angular
            velocity. This rolling median is used to identify when the eye angular velocity is larger than usual.
        velocity_factor: The factor by which the eye angular velocity must be larger than the rolling median to consider
            a saccade. Default is 5, meaning that the eye angular velocity must be larger than 5 times the rolling
            median to be considered a saccade.
        """
        self.saccade = SaccadeEvent(
            self.data_object,
            self.identified_indices,
            min_acceleration_threshold,
            velocity_window_size,
            velocity_factor,
        )
        self.saccade.initialize()
        self.set_identified_frames(self.saccade)

    def detect_visual_scanning_sequences(self, min_velocity_threshold: float = 100, minimal_duration: float = 0.040):
        """
        Detects visual scanning sequences in the data object.

        Parameters
        ----------
        min_velocity_threshold: The minimal threshold for the gaze angular velocity to consider a visual scanning
            event, in deg/s. Default is 100 deg/s.
        minimal_duration: The minimal duration of the visual scanning event, in seconds. Default is 0.1 seconds.
        """
        self.visual_scanning = VisualScanningEvent(
            self.data_object,
            self.identified_indices,
            min_velocity_threshold,
            minimal_duration,
        )
        self.visual_scanning.initialize()

        # Remove frames where visual scanning events are detected
        self.set_identified_frames(self.visual_scanning)
        # Also remove all frames where the velocity is above threshold, as these frames are not available for the
        # detection of other events. Please note that these frames might not be part of a visual scanning event if the
        # velocity is not maintained for at least minimal_duration.
        self.fast_frame_indices = np.where(
            np.abs(self.data_object.gaze_angular_velocity) > self.visual_scanning.min_velocity_threshold
        )[0]
        self.identified_indices[self.fast_frame_indices] = True

    def detect_fixation_and_smooth_pursuit_sequences(
        self,
        inter_saccade_minimal_duration: float = 0.04,
        fixation_minimal_duration: float = 0.04,
        smooth_pursuit_minimal_duration: float = 0.04,
        window_duration: float = 0.022,
        window_overlap: float = 0.006,
        eta_p: float = 0.001,
        eta_d: float = 0.45,
        eta_cd: float = 0.5,
        eta_pd: float = 0.2,
        eta_max_fixation: float = 1.9,
        eta_min_smooth_pursuit: float = 1.7,
        phi: float = 45,
    ):
        """
        Detects fixation and smooth pursuit sequences in the data object.

        Parameters
        ----------
        inter_saccade_minimal_duration: The minimal duration of the intersaccadic events, in seconds.
        fixation_minimal_duration: The minimal duration of the fixation events, in seconds.
        smooth_pursuit_minimal_duration: The minimal duration of the smooth pursuit events, in seconds.
        window_duration: The duration of the window (in seconds) used to compute the coherence of the inter-saccadic
            sequences.
        window_overlap: The overlap between two consecutive windows (in seconds)
        eta_p: The threshold for the p-value of the Rayleigh test to classify the inter-saccadic sequences as coherent
            or incoherent.
        eta_d: The threshold for the gaze direction dispersion (without units).
        eta_cd: The threshold for the consistency of direction (without units).
        eta_pd: The threshold for the position displacement (without units).
        phi: The threshold for the similar angular range (in degrees).
        eta_max_fixation: The threshold for the maximum fixation range (in degrees).
        eta_min_smooth_pursuit: The threshold for the minimum smooth pursuit range (in degrees).

        Note that the default values for the parameters
            `minimal_duration` = 40 ms
            `window_duration` = 22 ms
            `window_overlap` = 6 ms
            `eta_p` = 0.01 -> in 3D we rather recommend to use 0.001
            `eta_d` = 0.45
            `eta_cd` = 0.5
            `eta_pd` = 0.2
            `eta_max_fixation` = 1.9 deg
            `eta_min_smooth_pursuit` = 1.7 deg
            `phi` = 45 deg
            are taken from Larsson et al. (2015), but they should be modified to fit your experimental setup
            (acquisition frequency and task).
        """
        self.inter_saccadic_sequences = InterSaccadicEvent(
            self.data_object,
            self.identified_indices,
            inter_saccade_minimal_duration,
            window_duration,
            window_overlap,
            eta_p,
            eta_d,
            eta_cd,
            eta_pd,
            eta_max_fixation,
            eta_min_smooth_pursuit,
            phi,
        )
        self.inter_saccadic_sequences.initialize()

        self.fixation = FixationEvent(
            self.data_object,
            self.identified_indices,
            self.inter_saccadic_sequences.fixation_indices,
            fixation_minimal_duration,
        )
        self.fixation.initialize()

        self.smooth_pursuit = SmoothPursuitEvent(
            self.data_object,
            self.identified_indices,
            self.inter_saccadic_sequences.smooth_pursuit_indices,
            smooth_pursuit_minimal_duration,
        )
        self.smooth_pursuit.initialize()
        self.inter_saccadic_sequences.finalize(self.fixation.sequences, self.smooth_pursuit.sequences)

        self.set_identified_frames(self.fixation)
        self.set_identified_frames(self.smooth_pursuit)

    @property
    def unidentified_indices(self):
        """
        Returns the indices of the frames that are not identified as any event.
        Here we deliberately exclude frames that have a large gaze velocity but that are not part of a visual scanning
        """
        unidentified_indices = np.ones(
            (self.data_object.time_vector.shape[0]), dtype=bool
        )  # Initialize as all frames identified
        for sequence in (
            self.blink.sequences
            + self.invalid.sequences
            + self.saccade.sequences
            + self.visual_scanning.sequences
            + self.fixation.sequences
            + self.smooth_pursuit.sequences
        ):
            if len(sequence) != 0:
                unidentified_indices[sequence] = False  # Mark identified frames as False

        return unidentified_indices

    def validate_sequences(self):
        """
        Check if there were problems in the classification algorithm by making sure that there is no overlapping between
        sequences as each frame can only be part of one sequence (except invalid and blink).
        """
        # Blinks and invalid data must not overlap with any other sequences
        for blink in self.blink.sequences + self.invalid.sequences:
            for sequence in (
                self.saccade.sequences
                + self.visual_scanning.sequences
                + self.inter_saccadic_sequences.sequences
                + self.fixation.sequences
                + self.smooth_pursuit.sequences
            ):
                if any(item in blink for item in sequence):
                    raise RuntimeError(
                        "Problem: Blink or Invalid data sequence overlap with another sequence."
                        "This should not happen, please contact the developer."
                    )

        # Saccades, visual scanning, fixations, and smooth pursuits must not overlap with each other
        for saccade in self.saccade.sequences:
            for sequence in (
                self.visual_scanning.sequences
                + self.inter_saccadic_sequences.sequences
                + self.fixation.sequences
                + self.smooth_pursuit.sequences
            ):
                if any(item in saccade for item in sequence):
                    raise RuntimeError(
                        "Problem: Saccade sequence overlap with Inter-saccadic, Visual scanning, Fixation, or Smooth pursuit sequences"
                        "This should not happen, please contact the developer."
                    )

        for visual_scanning in self.visual_scanning.sequences:
            for sequence in (
                self.inter_saccadic_sequences.sequences + self.fixation.sequences + self.smooth_pursuit.sequences
            ):
                if any(item in visual_scanning for item in sequence):
                    raise RuntimeError(
                        "Problem:  Visual scanning sequence overlap with Inter-saccadic, Fixation, or Smooth pursuit sequences"
                        "This should not happen, please contact the developer."
                    )

        for fixation in self.fixation.sequences:
            for sequence in self.smooth_pursuit.sequences:
                if any(item in fixation for item in sequence):
                    raise RuntimeError(
                        "Problem: Fixation sequence overlap with Smooth pursuit sequences"
                        "This should not happen, please contact the developer."
                    )

        # Check that inter-saccadic sequences are all fixations, smooth pursuits, or unknown sequences
        if (len(self.fixation.sequences) + len(self.smooth_pursuit.sequences)) > len(
            self.inter_saccadic_sequences.sequences
        ):
            raise RuntimeError(
                "There is more inter-saccadic sequences than there are fixations + smooth pursuits."
                "This should not happen, please contact the developer."
            )
        for sequence in self.fixation.sequences + self.smooth_pursuit.sequences:
            for frame in sequence:
                if frame not in self.inter_saccadic_sequences.frame_indices:
                    raise RuntimeError(
                        "There is a fixation or smooth pursuit sequence that is not part of an inter-saccadic sequence."
                        "This should not happen, please contact the developer."
                    )

        # Check that the identified frames are the frames in the sequences
        for sequence in (
            self.blink.sequences
            + self.invalid.sequences
            + self.saccade.sequences
            + self.visual_scanning.sequences
            + self.inter_saccadic_sequences.sequences
            + self.fixation.sequences
            + self.smooth_pursuit.sequences
        ):
            if not np.all(self.identified_indices[sequence]):
                raise RuntimeError(
                    "There are frames that are not considered as identified, but that are part of an event sequence."
                    "This should not happen, please contact the developer."
                )
        for frame in np.where(self.unidentified_indices)[0]:
            if frame in np.where(self.identified_indices)[0] and frame not in self.fast_frame_indices:
                raise RuntimeError(
                    "There are frames that are considered as unidentified, but that are also considered as identified."
                    "This should not happen, please contact the developer."
                )
            for sequence in (
                self.blink.sequences
                + self.invalid.sequences
                + self.saccade.sequences
                + self.visual_scanning.sequences
                + self.inter_saccadic_sequences.sequences
                + self.fixation.sequences
                + self.smooth_pursuit.sequences
            ):
                if frame in sequence:
                    raise RuntimeError(
                        "There are frames that are considered as unidentified, but that are part of an event sequence."
                        "This should not happen, please contact the developer."
                    )

    def compute_metrics(self):
        """
        The metrics are computed at the end for each event type to allow skipping the initialization of the Events.
        """
        self.saccade.measure_saccade_amplitude()
        self.fixation.measure_search_rate()
        self.smooth_pursuit.measure_smooth_pursuit_trajectory()

    def finalize(self):
        self.validate_sequences()
        self.compute_metrics()
        self.is_finalized = True

    def _get_event_at_split_timing(
        self, timing: float, time_vector: np.ndarray[float], dt: float, error_type: ErrorType
    ) -> tuple[str, int, int]:

        for sequence_type, sequence_list in (
            ("Blink", self.blink.sequences),
            ("Saccade", self.saccade.sequences),
            ("Visual scanning", self.visual_scanning.sequences),
            ("Fixation", self.fixation.sequences),
            ("Smooth pursuit", self.smooth_pursuit.sequences),
        ):
            for sequence in sequence_list:
                beginning_time = time_vector[sequence[0]]
                end_time = time_vector[sequence[-1]]
                if beginning_time <= timing <= end_time:
                    # We found the event at the split timing
                    event_at_split = sequence_type

                    # Remove this event but write it in a file so that we know what was removed
                    error_str = f"{sequence_type} : {np.round(end_time - beginning_time, decimals=5)} s ----"
                    error_type(error_str)

                    return event_at_split, time_vector[sequence[0]], time_vector[sequence[-1]]

        # There was no event happening at the split timing
        return None, timing, timing

    def _get_a_reduced_gaze_behavior_identifier(self, time_range: TimeRange) -> Self:

        # Build a reduced data object based on the time range
        reduced_data_object = ReducedData(
            self.data_object.dt,
            self.data_object.time_vector,
            self.data_object.right_eye_openness,
            self.data_object.left_eye_openness,
            self.data_object.eye_direction,
            self.data_object.head_angles,
            self.data_object.gaze_direction,
            self.data_object.head_angular_velocity,
            self.data_object.head_velocity_norm,
            self.data_object.data_invalidity,
            time_range,
        )

        # Build a reduced GazeBehaviorIdentifier
        reduced_gaze_behavior_identifier = GazeBehaviorIdentifier(reduced_data_object)

        # Blink
        reduced_gaze_behavior_identifier.blink = BlinkEvent(reduced_data_object).from_sequences(
            get_sequences_in_range(self.data_object.time_vector, time_range, self.blink.sequences)
        )

        # Invalid
        reduced_gaze_behavior_identifier.invalid = InvalidEvent(reduced_data_object).from_sequences(
            get_sequences_in_range(self.data_object.time_vector, time_range, self.invalid.sequences)
        )

        # Saccade
        reduced_gaze_behavior_identifier.saccade = SaccadeEvent(reduced_data_object).from_sequences(
            get_sequences_in_range(self.data_object.time_vector, time_range, self.saccade.sequences)
        )

        # Visual Scanning
        reduced_gaze_behavior_identifier.visual_scanning = VisualScanningEvent(reduced_data_object).from_sequences(
            get_sequences_in_range(self.data_object.time_vector, time_range, self.visual_scanning.sequences)
        )

        # Inter-saccadic
        reduced_gaze_behavior_identifier.inter_saccadic_sequences = InterSaccadicEvent(
            reduced_data_object
        ).from_sequences(
            get_sequences_in_range(self.data_object.time_vector, time_range, self.inter_saccadic_sequences.sequences)
        )

        # Fixation
        reduced_gaze_behavior_identifier.fixation = FixationEvent(reduced_data_object).from_sequences(
            get_sequences_in_range(self.data_object.time_vector, time_range, self.fixation.sequences)
        )

        # Smooth Pursuit
        reduced_gaze_behavior_identifier.smooth_pursuit = SmoothPursuitEvent(reduced_data_object).from_sequences(
            get_sequences_in_range(self.data_object.time_vector, time_range, self.smooth_pursuit.sequences)
        )

        # Identified indices
        reduced_gaze_behavior_identifier.identified_indices = ~reduced_gaze_behavior_identifier.unidentified_indices

        # Fast frame indices
        first_index_in_old_time_vector = time_range.get_indices(self.data_object.time_vector)[0]
        fast_frame_indices = []
        for i in self.fast_frame_indices:
            if i in time_range.get_indices(self.data_object.time_vector):
                fast_frame_indices += [int(i)]
        reduced_gaze_behavior_identifier.fast_frame_indices = (
            np.array(fast_frame_indices) - first_index_in_old_time_vector
        )

        # Finalize
        reduced_gaze_behavior_identifier.finalize()

        return reduced_gaze_behavior_identifier

    def split(
        self, split_timings: list[float], event_at_split_handling: ErrorType = ErrorType.PRINT
    ) -> list["GazeBehaviorIdentifier"]:
        """
        Split the GazeBehaviorIdentifier into multiple instances based on the provided split timings.
        Note: the event sequence happening at the splitting will be excluded from both sides of the split.
        Please open an issue on GitHub if you want tohave the possibility to split the event to have one part of the
        event on each side of the split.

        Parameters
        ----------
        split_timings: A list of timestamps (in seconds) where the data should be split.
        event_at_split_handling: How to handle the event that is happening at the split timing. Default is
        ErrorType.PRINT, which prints the type of event and its duration.

        Returns
        -------
        A list of GazeBehaviorIdentifier instances, each containing data from a specific time segment.
        """

        if not self.is_finalized:
            raise RuntimeError(
                "The GazeBehaviorIdentifier must be finalized before splitting. Please call finalize() first."
            )

        gaze_behavior_identifiers = []
        beginning_time = self.data_object.time_vector[0]
        for timing in split_timings:
            event_at_split, end_time, new_beginning_time = self._get_event_at_split_timing(
                timing, self.data_object.time_vector, self.data_object.dt, event_at_split_handling
            )
            time_range = TimeRange(beginning_time, end_time + 1e-6)
            reduced_gaze_behavior_identifier = self._get_a_reduced_gaze_behavior_identifier(time_range)
            gaze_behavior_identifiers += [reduced_gaze_behavior_identifier]

            beginning_time = new_beginning_time

        time_range = TimeRange(beginning_time, self.data_object.time_vector[-1])
        reduced_gaze_behavior_identifier = self._get_a_reduced_gaze_behavior_identifier(time_range)
        gaze_behavior_identifiers += [reduced_gaze_behavior_identifier]

        return gaze_behavior_identifiers

    def plot(self, save_name: str = None, live_show: bool = True) -> plt.Figure:
        """
        Plot all the detected gaze behaviors.

        Parameters
        ----------
        save_name: The name under which to save the figure. If None is provided, the figure is not saved.
        live_show: If the figure should be shown immediately. Please note that showing the figure is blocking.
        """

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.set_title("Detected gaze behaviors")

        # Plot the gaze vector and the identified sequences
        self.data_object.plot_gaze_vector(ax=ax)
        if self.invalid is not None:
            self.invalid.add_sequence_to_plot(ax=ax)
        if self.blink is not None:
            self.blink.add_sequence_to_plot(ax=ax)
        if self.saccade is not None:
            self.saccade.add_sequence_to_plot(ax=ax)
        if self.visual_scanning is not None:
            self.visual_scanning.add_sequence_to_plot(ax=ax)
        if self.fixation is not None:
            self.fixation.add_sequence_to_plot(ax=ax)
        if self.smooth_pursuit is not None:
            self.smooth_pursuit.add_sequence_to_plot(ax=ax)

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
