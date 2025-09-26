import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pingouin as pg
from copy import deepcopy

from .event import Event
from ..utils.data_utils import DataObject
from ..utils.rotation_utils import get_angle_between_vectors
from ..utils.sequence_utils import split_sequences, apply_minimal_duration, merge_sequence_lists
from ..utils.signal_utils import find_time_index
from ..utils.check_utils import check_save_name


class InterSaccadicEvent(Event):
    """
    Class to extract 'inter-saccadic' sequences. These sequences are defined as the sequences of frames that have not
    been yet identified (this is partly why order matters in the identification of the behaviors).
    Please note that an intersaccadic sequence is not a behavior in itself, but rather a collection of frames that will
    be used for the identification of fixation and smooth pursuit behaviors.
    """

    def __init__(
        self,
        data_object: DataObject,
        identified_indices: np.ndarray = None,
        minimal_duration: float = None,
        window_duration: float = None,
        window_overlap: float = None,
        eta_p: float = None,
        eta_d: float = None,
        eta_cd: float = None,
        eta_pd: float = None,
        eta_max_fixation: float = None,
        eta_min_smooth_pursuit: float = None,
        phi: float = None,
    ):
        """
        Parameters:
        ----------
        data_object: The EyeDentify3d object containing the parsed eye-tracking data.
        identified_indices: A boolean array indicating which frames have already been identified as events.
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
        """

        super().__init__()

        # Checks
        if window_duration is not None and window_overlap is not None and window_duration < 2 * window_overlap:
            raise ValueError(
                f"The window_duration ({window_duration} s) must be at least twice the window_overlap ({window_overlap} s)."
            )

        # Original attributes
        self.data_object = data_object
        self.identified_indices = identified_indices
        self.minimal_duration = minimal_duration
        self.window_duration = window_duration
        self.window_overlap = window_overlap
        self.eta_p = eta_p
        self.eta_d = eta_d
        self.eta_cd = eta_cd
        self.eta_pd = eta_pd
        self.eta_max_fixation = eta_max_fixation
        self.eta_min_smooth_pursuit = eta_min_smooth_pursuit
        self.phi = phi

        # Extended attributes
        self.gaze_displacement_angle: np.ndarray[float] = None
        self.mean_p_values: np.ndarray[float] = None
        self.parameter_D: list[float] = []
        self.parameter_CD: list[float] = []
        self.parameter_PD: list[float] = []
        self.parameter_R: list[float] = []
        self.criteria_sequences: list[np.ndarray[int]] = None
        self.coherent_sequences: list[np.ndarray[int]] = None
        self.incoherent_sequences: list[np.ndarray[int]] = None
        self.fixation_indices: np.ndarray[int] = None
        self.smooth_pursuit_indices: np.ndarray[int] = None
        self.uncertain_sequences: list[np.ndarray[int]] = None

    def initialize(self):
        self.detect_intersaccadic_indices()
        self.split_sequences()
        self.keep_only_sequences_long_enough()
        self.set_coherent_and_incoherent_sequences()
        self.set_intersaccadic_sequences()
        self.classify_sequences()

    def finalize(self, fixation_sequences: list[np.ndarray[int]], smooth_pursuit_sequences: list[np.ndarray[int]]):
        self.sequences = merge_sequence_lists(fixation_sequences, smooth_pursuit_sequences)
        self.frame_indices = np.concatenate(self.sequences) if self.sequences else np.array([], dtype=int)

    def detect_intersaccadic_indices(self):
        """
        Detect when velocity is above the threshold and if the frames are not already identified.
        """
        self.frame_indices = np.where(self.identified_indices == False)[0]

    @staticmethod
    def detect_directionality_coherence_on_axis(
        gaze_direction: np.ndarray, component_to_keep: int
    ) -> tuple[float, float]:
        """
        Detects the coherence of the gaze direction inside a window using a Rayleigh z-test on the axis specified.
        This function first computes the gaze displacement vector between two consecutive frames, and then find the
        associated angle relatively to the axis specified. Finally, a Rayleigh z-test is applied on the angles to check
        if the gaze displacement is uniformly distributed on the selected window. If the gaze direction is coherent
        (i.e., the p-value is smaller than eta_p), the gaze is moving in a particular direction.

        Parameters
        ----------
        gaze_direction: A 2D numpy array of shape (3, nb_frames) expressing the gaze (head + eyes) direction in 3D space
            through a unit vector.
        component_to_keep: The index of the component to keep for the Rayleigh test.
        """
        if component_to_keep not in [0, 1, 2]:
            raise ValueError("component_to_keep must be 0, 1, or 2.")

        nb_frames = gaze_direction.shape[1]
        gaze_displacement_angle = np.ones((nb_frames,)) * np.nan
        for i_frame in range(nb_frames - 1):
            gaze_displacement = gaze_direction[:, i_frame + 1] - gaze_direction[:, i_frame]
            gaze_displacement_angle[i_frame] = np.arcsin(
                gaze_displacement[component_to_keep] / np.linalg.norm(gaze_displacement)
            )
        # The last frame does not have a displacement, so we copy the previous one
        gaze_displacement_angle[-1] = gaze_displacement_angle[-2]

        # Test that the gaze displacement and orientation are coherent inside the window
        z_value, p_value = pg.circ_rayleigh(gaze_displacement_angle)

        # Convert the angle in degrees as the circ_rayleigh requires radians, but degrees are easier to interpret
        gaze_displacement_angle = gaze_displacement_angle * 180 / np.pi
        return p_value, gaze_displacement_angle

    @staticmethod
    def variability_decomposition(gaze_direction: np.ndarray) -> tuple[float, float]:
        """
        Computes the gaze direction variability and decompose it into a principal and second axis. It returns the
        length of the principal and second components (d_pc1 and d_pc2 in Larsson et al. 2015).
        """

        nb_frames = gaze_direction.shape[1]
        if nb_frames < 3:
            raise ValueError(
                "The gaze direction must contain at least 3 frames for the variability decomposition to be "
                "meaningful. Please consider changing the window_duration."
            )

        # Center the gaze direction around its mean
        mean_gaze_direction = np.nanmean(gaze_direction, axis=1)
        gaze_direction_centered = gaze_direction - mean_gaze_direction[:, np.newaxis]

        # Get the principal and second axis of the gaze direction
        # Note: The third axis should be almost null as the gaze vector is unitary so the gaze direction is contained
        # on the sphere.
        cov = np.ma.cov(np.ma.masked_invalid(gaze_direction_centered)).data
        eigen_values, eigen_vectors = np.linalg.eig(cov)
        if np.sum(cov) == 0:
            raise RuntimeError(
                "There was no variability in the gaze direction on this window. "
                "This should not happen, please contact the developer."
            )
        else:
            # Sort the eigen values in descending order
            sorted_indices = np.argsort(eigen_values)[::-1]
            principal_axis = eigen_vectors[:, sorted_indices[0]]
            second_axis = eigen_vectors[:, sorted_indices[1]]

        principal_projection = np.dot(gaze_direction_centered.T, principal_axis)
        second_projection = np.dot(gaze_direction_centered.T, second_axis)

        length_principal_component = np.max(principal_projection) - np.min(principal_projection)
        length_second_component = np.max(second_projection) - np.min(second_projection)
        if np.abs(np.dot(principal_axis, second_axis)) > 0.0001:
            raise RuntimeError(
                "The principal and second axis are not orthogonal. This should not happen, please contact the developer."
            )

        return length_principal_component, length_second_component

    @staticmethod
    def compute_gaze_travel_distance(gaze_direction: np.ndarray) -> float:
        gaze_travel_distance = np.linalg.norm(gaze_direction[:, -1] - gaze_direction[:, 0])
        return gaze_travel_distance

    @staticmethod
    def compute_gaze_trajectory_length(gaze_direction: np.ndarray) -> float:
        gaze_trajectory_length = np.sum(np.linalg.norm(gaze_direction[:, 1:] - gaze_direction[:, :-1], axis=0))
        return gaze_trajectory_length

    @staticmethod
    def compute_mean_gaze_direction_radius_range(gaze_direction: np.ndarray) -> float:
        mean_gaze_direction_radius_range = np.sqrt(
            (np.max(gaze_direction[0, :]) - np.min(gaze_direction[0, :])) ** 2
            + (np.max(gaze_direction[1, :]) - np.min(gaze_direction[1, :])) ** 2
            + (np.max(gaze_direction[2, :]) - np.min(gaze_direction[2, :])) ** 2
        )
        return mean_gaze_direction_radius_range

    def compute_larsson_parameters(self, gaze_direction: np.ndarray) -> tuple[float, float, float, float]:

        # Compute helpful metrics
        length_principal_component, length_second_component = self.variability_decomposition(gaze_direction)
        gaze_travel_distance = self.compute_gaze_travel_distance(gaze_direction)
        gaze_trajectory_length = self.compute_gaze_trajectory_length(gaze_direction)
        mean_gaze_direction_radius_range = self.compute_mean_gaze_direction_radius_range(gaze_direction)

        # Compute the parameters as defined in Larsson et al. (2015)
        parameter_D = length_second_component / length_principal_component
        parameter_CD = gaze_travel_distance / length_principal_component
        parameter_PD = gaze_travel_distance / gaze_trajectory_length
        parameter_R = np.arctan(mean_gaze_direction_radius_range) * 180 / np.pi

        return parameter_D, parameter_CD, parameter_PD, parameter_R

    def get_window_sequences(self, time_vector: np.ndarray) -> list[np.ndarray]:
        """
        Get the sequences of indices of the windows in which the inter-saccadic sequence should be split.
        The windows are of duration window_duration and overlap with the previous window by window_overlap seconds.
        """
        window_sequences = []

        for intersaccadic_sequence in self.sequences:
            sequence_start_idx, sequence_end_idx = intersaccadic_sequence[0], intersaccadic_sequence[-1]
            sequence_start_time, sequence_end_time = time_vector[sequence_start_idx], time_vector[sequence_end_idx]

            # Calculate window positions using time-based approach
            current_window_start_time = sequence_start_time
            window_start_idx = sequence_start_idx

            while current_window_start_time < sequence_end_time:

                # Calculate end time and find corresponding index
                window_end_time = current_window_start_time + self.window_duration
                window_end_idx = find_time_index(time_vector, window_end_time, method="last")

                # Handle edge case: if remaining sequence is very short, extend to sequence end
                remaining_duration = sequence_end_time - time_vector[window_end_idx]
                if remaining_duration < self.window_overlap:
                    window_end_idx = sequence_end_idx

                window_sequences.append(np.arange(window_start_idx, window_end_idx))

                # Only add windows with sufficient samples
                if window_end_idx - window_start_idx < 3:
                    raise RuntimeError("The merging went wrong")

                # Break if we've reached the end
                if window_end_idx >= sequence_end_idx:
                    break

                # Calculate next window start time with overlap
                current_window_start_time = time_vector[window_end_idx - 1] - self.window_overlap

                # Find start index for next window start
                window_start_idx = find_time_index(time_vector, current_window_start_time, method="first")

        return window_sequences

    def set_coherent_and_incoherent_sequences(self):
        """
        Split the inter-saccadic sequences into overlapping windows. Merge the consecutive windows that are similar in
        nature based on if they are coherent or incoherent in terms of gaze direction.
        """
        # Split this sequence into overlapping windows
        intersaccadic_window_sequences = self.get_window_sequences(time_vector=self.data_object.time_vector)

        # We store the p-values for each frame in a list of lists
        # Each frame can be part of several windows, so we store the p-values for each of the window it was part of and
        # use the mean p-value to classify this frame.
        p_values = [[] for _ in range(len(self.data_object.time_vector))]
        self.gaze_displacement_angle = np.zeros_like(self.data_object.time_vector)
        for current_window in intersaccadic_window_sequences:
            # Sanity check
            nb_elements = int(np.prod(np.array(self.data_object.gaze_direction[:, current_window].shape)))
            if int(np.sum(np.isnan(self.data_object.gaze_direction[:, current_window]))) > (nb_elements - 6):
                # Too much NaN values in the window, skip it
                continue

            # Compute the directionality coherence p-value for the current window
            p_value, gaze_displacement_angle = self.detect_directionality_coherence_on_axis(
                self.data_object.gaze_direction[:, current_window], component_to_keep=0
            )
            self.gaze_displacement_angle[current_window] = gaze_displacement_angle
            for i_idx in current_window:
                p_values[i_idx] += [p_value]

        # The mean p-value for each timestamp is used for the coherence/incoherence classification
        self.mean_p_values = np.array([np.nanmean(np.array(p)) for p in p_values])

        incoherent_indices = np.where(self.mean_p_values <= self.eta_p)[0]
        incoherent_sequences = split_sequences(incoherent_indices)
        self.incoherent_sequences = apply_minimal_duration(
            incoherent_sequences, self.data_object.time_vector, self.minimal_duration
        )

        coherent_indices = np.where(self.mean_p_values > self.eta_p)[0]
        coherent_sequences = split_sequences(coherent_indices)
        self.coherent_sequences = apply_minimal_duration(
            coherent_sequences, self.data_object.time_vector, self.minimal_duration
        )

    def classify_obvious_sequences(
        self, data_object: DataObject, all_intersaccadic_sequences: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sequences where all (smooth pursuit) or none (fixation) of the criteria are met are classified as such and
        other sequences are classified as ambiguous.
        """
        fixation_indices = []
        smooth_pursuit_indices = []
        ambiguous_indices = []
        self.criteria_sequences = deepcopy(all_intersaccadic_sequences)
        for i_sequence, sequence in enumerate(all_intersaccadic_sequences):
            parameter_D, parameter_CD, parameter_PD, parameter_R = self.compute_larsson_parameters(
                data_object.gaze_direction[:, sequence]
            )
            self.parameter_D += [parameter_D]
            self.parameter_CD += [parameter_CD]
            self.parameter_PD += [parameter_PD]
            self.parameter_R += [parameter_R]

            criteria_1 = parameter_D < self.eta_d
            criteria_2 = parameter_CD > self.eta_cd
            criteria_3 = parameter_PD > self.eta_pd
            criteria_4 = parameter_R > self.eta_max_fixation

            sum_criteria = int(criteria_1) + int(criteria_2) + int(criteria_3) + int(criteria_4)
            if sum_criteria == 0:
                fixation_indices += sequence.tolist()
            elif sum_criteria == 4:
                smooth_pursuit_indices += sequence.tolist()
            else:
                # These will be further classified later
                ambiguous_indices += sequence.tolist()

        fixation_indices = np.array(fixation_indices, dtype=int)
        smooth_pursuit_indices = np.array(smooth_pursuit_indices, dtype=int)
        ambiguous_indices = np.array(ambiguous_indices, dtype=int)

        return fixation_indices, smooth_pursuit_indices, ambiguous_indices

    def classify_ambiguous_sequences(
        self,
        data_object: DataObject,
        all_intersaccadic_sequences: list[np.ndarray],
        ambiguous_indices: np.ndarray,
        fixation_indices: np.ndarray,
        smooth_pursuit_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """
        Classify ambiguous sequences as smooth pursuit-like or fixation-like segments.

        Logic:
        if criteria_3 (parameter_PD > eta_pd):
            Merge the sequence with other adjacent sequences where the gaze is not too far away (< phi), but that are
            not fixations.
            Compute the gaze direction range for the merged sequence.
            if the gaze moves more than eta_min_smooth_pursuit:
                Classify as smooth pursuit
            else:
                Classify as fixation
        else:
            if criteria_4 (parameter_R > eta_max_fixation):
                Classify as smooth pursuit
            else:
                Classify as fixation
        """
        ambiguous_sequences = split_sequences(ambiguous_indices)

        sequences_to_remove = []
        for i_sequence, sequence in enumerate(ambiguous_sequences):
            # Compute Larsson parameters for this sequence
            parameters = self.compute_larsson_parameters(data_object.gaze_direction[:, sequence])
            parameter_D, parameter_CD, parameter_PD, parameter_R = parameters

            criteria_3 = parameter_PD > self.eta_pd
            criteria_4 = parameter_R > self.eta_max_fixation

            if criteria_3:
                # Smooth pursuit-like: try to merge with adjacent compatible segments
                merged_range = self._find_mergeable_segment_range(
                    i_sequence, sequence, all_intersaccadic_sequences, data_object, fixation_indices
                )

                if merged_range is not None:
                    fixation_indices, smooth_pursuit_indices = self._classify_criteria3_segment(
                        merged_range, data_object, fixation_indices, smooth_pursuit_indices
                    )

                    # Mark all sequences in this range for removal
                    sequences_to_remove += self._find_sequences_in_range(merged_range, ambiguous_sequences)
            else:
                if criteria_4:
                    smooth_pursuit_indices = np.hstack((smooth_pursuit_indices, sequence))
                else:
                    fixation_indices = np.hstack((fixation_indices, sequence))
                sequences_to_remove += [i_sequence]

        # Return remaining uncertain sequences
        uncertain_sequences = [seq for i, seq in enumerate(ambiguous_sequences) if i not in sequences_to_remove]

        return fixation_indices, smooth_pursuit_indices, uncertain_sequences

    def _find_mergeable_segment_range(
        self,
        sequence_idx: int,
        sequence: np.ndarray,
        all_sequences: list[np.ndarray],
        data_object: DataObject,
        fixation_indices: np.ndarray,
    ) -> tuple[int, int] | None:
        """
        Find the range of indices that can be merged with the current sequence.

        Returns
        -------
        start_idx, end_idx if mergeable range found, None otherwise
        """
        sequence_mean = np.nanmean(data_object.gaze_direction[:, sequence], axis=1)

        # Find backward extension
        start_idx = self._extend_segment_backward(
            sequence_idx, sequence[0], sequence_mean, all_sequences, data_object, fixation_indices
        )

        # Find forward extension
        end_idx = self._extend_segment_forward(
            sequence_idx, sequence[-1], sequence_mean, all_sequences, data_object, fixation_indices
        )

        if end_idx - start_idx <= 2:
            raise RuntimeError(
                "The merged sequence is too short. This should not happen, please contact the developer."
            )

        return start_idx, end_idx

    def _extend_segment_backward(
        self,
        current_idx: int,
        boundary_idx: int,
        reference_mean: np.ndarray,
        all_sequences: list[np.ndarray],
        data_object: DataObject,
        fixation_indices: np.ndarray,
    ) -> int:
        """Extend segment backward while segments are compatible."""
        if current_idx == 0:
            return boundary_idx

        search_idx = current_idx - 1

        while search_idx >= 0:
            candidate_seq = all_sequences[search_idx]

            # Stop if we hit a fixation or non-adjacent sequence
            if boundary_idx not in candidate_seq or any(idx in fixation_indices for idx in candidate_seq):
                break

            # Check angular compatibility
            if self._sequences_are_compatible(reference_mean, candidate_seq, data_object):
                boundary_idx = candidate_seq[0]
                search_idx -= 1
            else:
                break

        return boundary_idx

    def _extend_segment_forward(
        self,
        current_idx: int,
        boundary_idx: int,
        reference_mean: np.ndarray,
        all_sequences: list[np.ndarray],
        data_object: DataObject,
        fixation_indices: np.ndarray,
    ) -> int:
        """Extend segment forward while segments are compatible."""
        if current_idx >= len(all_sequences) - 1:
            return boundary_idx

        search_idx = current_idx + 1

        while search_idx < len(all_sequences):
            candidate_seq = all_sequences[search_idx]

            # Stop if we hit a fixation or non-adjacent sequence
            if boundary_idx not in candidate_seq or any(idx in fixation_indices for idx in candidate_seq):
                break

            # Check angular compatibility
            if self._sequences_are_compatible(reference_mean, candidate_seq, data_object):
                boundary_idx = candidate_seq[-1]
                search_idx += 1
            else:
                break

        return boundary_idx

    def _sequences_are_compatible(
        self, reference_mean: np.ndarray, candidate_seq: np.ndarray, data_object: DataObject
    ) -> bool:
        """Check if two sequences are compatible based on angular difference."""
        candidate_mean = np.nanmean(data_object.gaze_direction[:, candidate_seq], axis=1)
        angle = get_angle_between_vectors(reference_mean, candidate_mean)
        return angle < self.phi

    def _classify_criteria3_segment(
        self,
        merged_range: tuple[int, int],
        data_object: DataObject,
        fixation_indices: np.ndarray,
        smooth_pursuit_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Classify a merged segment as fixation or smooth pursuit."""
        start_idx, end_idx = merged_range
        indices = list(range(start_idx, end_idx + 1))

        # Compute parameters for the merged segment
        _, _, _, parameter_R = self.compute_larsson_parameters(data_object.gaze_direction[:, indices])

        if parameter_R > self.eta_min_smooth_pursuit:
            # The gaze moves
            smooth_pursuit_indices = np.hstack((smooth_pursuit_indices, indices))
        else:
            # the gaze does not move much
            fixation_indices = np.hstack((fixation_indices, indices))

        return fixation_indices, smooth_pursuit_indices

    @staticmethod
    def _find_sequences_in_range(merged_range: tuple[int, int], ambiguous_sequences: list[np.ndarray]) -> list[int]:
        """Find which ambiguous sequences overlap with the merged range."""
        start_idx, end_idx = merged_range
        merged_set = set(range(start_idx, end_idx + 1))

        overlapping_sequences = []
        for i_sequence, sequence in enumerate(ambiguous_sequences):
            if any(idx in merged_set for idx in sequence):
                overlapping_sequences.append(i_sequence)

        return overlapping_sequences

    def set_intersaccadic_sequences(self) -> None:
        """
        Define the inter-saccadic sequences as the combination of coherent and incoherent sequences.
        """
        self.sequences = merge_sequence_lists(
            self.coherent_sequences,
            self.incoherent_sequences,
        )
        if len(self.sequences) == 1 and self.sequences[0].shape == (0,):
            raise RuntimeError(
                "There should be at least one intersaccadic sequence even if there is no saccade. "
                "This should not happen, please contact the developer."
            )
        return

    def classify_sequences(self) -> None:
        """
        Classify the inter-saccadic sequences into coherent and incoherent sequences based on the gaze direction.
        TODO: add _on_sphere option.
        """
        fixation_indices, smooth_pursuit_indices, ambiguous_indices = self.classify_obvious_sequences(
            self.data_object,
            self.sequences,
        )
        fixation_indices, smooth_pursuit_indices, uncertain_sequences = self.classify_ambiguous_sequences(
            self.data_object,
            self.sequences,
            ambiguous_indices,
            fixation_indices,
            smooth_pursuit_indices,
        )

        self.smooth_pursuit_indices = np.sort(smooth_pursuit_indices)
        self.fixation_indices = np.sort(fixation_indices)
        self.uncertain_sequences = uncertain_sequences

        return

    def add_sequence_to_plot(self, ax: Axes):
        """
        Plot the detected coherent and incoherent inter-saccade sequences on the provided axis.

        Parameters:
        ax: The matplotlib axis to plot on.
        """
        for sequence in self.coherent_sequences:
            start = sequence[0]
            end = sequence[-1]
            ax.axvspan(
                self.data_object.time_vector[start],
                self.data_object.time_vector[end],
                color="tab:cyan",
                alpha=0.5,
                edgecolor=None,
            )
        for sequence in self.incoherent_sequences:
            start = sequence[0]
            end = sequence[-1]
            ax.axvspan(
                self.data_object.time_vector[start],
                self.data_object.time_vector[end],
                color="tab:olive",
                alpha=0.5,
                edgecolor=None,
            )
        ax.axvspan(
            0,
            0,
            color="tab:cyan",
            alpha=0.5,
            edgecolor=None,
            label="Coherent sequences",
        )
        ax.axvspan(
            0,
            0,
            color="tab:olive",
            alpha=0.5,
            edgecolor=None,
            label="Incoherent sequences",
        )

    def add_criteria_to_plot(self, ax: Axes):
        """
        Plot the detected Larsson criteria for the inter-saccade sequences.

        Parameters:
        ax: The matplotlib axis to plot on.
        """
        max_d = max(np.nanmax(self.parameter_D), self.eta_d)
        max_cd = max(np.nanmax(self.parameter_CD), self.eta_cd)
        max_pd = max(np.nanmax(self.parameter_PD), self.eta_pd)
        max_r = max(np.nanmax(self.parameter_R), self.eta_max_fixation)
        for i_sequence, sequence in enumerate(self.criteria_sequences):
            start = sequence[0]
            end = sequence[-1]
            ax.plot(
                np.array([self.data_object.time_vector[start], self.data_object.time_vector[end]]),
                np.array([self.parameter_D[i_sequence] / max_d, self.parameter_D[i_sequence] / max_d]),
                color="tab:red",
            )
            ax.plot(
                np.array([self.data_object.time_vector[start], self.data_object.time_vector[end]]),
                np.array([self.parameter_CD[i_sequence] / max_cd, self.parameter_CD[i_sequence] / max_cd]),
                color="tab:green",
            )
            ax.plot(
                np.array([self.data_object.time_vector[start], self.data_object.time_vector[end]]),
                np.array([self.parameter_PD[i_sequence] / max_pd, self.parameter_PD[i_sequence] / max_pd]),
                color="tab:blue",
            )
            ax.plot(
                np.array([self.data_object.time_vector[start], self.data_object.time_vector[end]]),
                np.array([self.parameter_R[i_sequence] / max_r, self.parameter_R[i_sequence] / max_r]),
                color="tab:orange",
            )
        # Add the labels
        ax.plot(np.array([0, 0]), np.array([0, 0]), color="tab:red", label=r"Dispersion ($< \eta_D$)")
        ax.plot(
            np.array([self.data_object.time_vector[0], self.data_object.time_vector[-1]]),
            np.array([self.eta_d / max_d, self.eta_d / max_d]),
            "--",
            color="tab:red",
            label=r"$\eta_D$ threshold = " + f"{self.eta_d:.2f}",
        )
        ax.plot(np.array([0, 0]), np.array([0, 0]), color="tab:green", label=r"Consistent direction ($> \eta_{CD}$)")
        ax.plot(
            np.array([self.data_object.time_vector[0], self.data_object.time_vector[-1]]),
            np.array([self.eta_cd / max_cd, self.eta_cd / max_cd]),
            "--",
            color="tab:green",
            label=r"$\eta_{CD}$ threshold = " + f"{self.eta_cd:.2f}",
        )
        ax.plot(np.array([0, 0]), np.array([0, 0]), color="tab:blue", label=r"Positional displacement ($> \eta_{PD}$)")
        ax.plot(
            np.array([self.data_object.time_vector[0], self.data_object.time_vector[-1]]),
            np.array([self.eta_pd / max_pd, self.eta_pd / max_pd]),
            "--",
            color="tab:blue",
            label=r"$\eta_{PD}$ threshold = " + f"{self.eta_pd:.2f}",
        )
        ax.plot(np.array([0, 0]), np.array([0, 0]), color="tab:orange", label=r"Spatial range ($> \eta_{maxFix}$)")
        ax.plot(
            np.array([self.data_object.time_vector[0], self.data_object.time_vector[-1]]),
            np.array([self.eta_max_fixation / max_r, self.eta_max_fixation / max_r]),
            "--",
            color="tab:orange",
            label=r"$\eta_{maxFix}$ threshold = " + f"{self.eta_max_fixation:.2f}",
        )

    def plot(self, save_name: str = None, live_show: bool = True) -> plt.Figure:
        """
        Plot the angle for the Rayleigh z-test, the associated p-value, the inter-saccadic sequences, and the four Larsson criteria.

        Parameters
        ----------
        save_name: The name under which to save the figure. If None is provided, the figure is not saved.
        live_show: If the figure should be shown immediately. Please note that showing the figure is blocking.
        """

        fig, axs = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1, 1]})
        axs[0].set_title("Detected inter-saccadic sequences")

        # Plot the gaze vector and the identified inter-saccadic sequences
        self.data_object.plot_gaze_vector(ax=axs[0])
        self.add_sequence_to_plot(axs[0])
        axs[0].set_xlim((self.data_object.time_vector[0], self.data_object.time_vector[-1]))
        axs[0].set_ylabel("Gaze orientation [without units]")
        axs[0].legend(bbox_to_anchor=(1.025, 0.5), loc="center left")

        # Plot the angle
        axs[1].plot(self.data_object.time_vector, np.abs(self.gaze_displacement_angle), color="tab:olive")
        axs[1].set_xlim((self.data_object.time_vector[0], self.data_object.time_vector[-1]))
        axs[1].set_ylabel(r"Gaze displacement [$^\circ$]", color="tab:olive")
        axs[1].tick_params(axis="y", labelcolor="tab:olive")

        # Add the associated p-value
        twin_ax = axs[1].twinx()
        twin_ax.plot(
            self.data_object.time_vector, np.abs(self.mean_p_values), color="tab:cyan", label="p-value (Rayleigh test)"
        )
        twin_ax.plot(
            np.array([self.data_object.time_vector[0], self.data_object.time_vector[-1]]),
            np.array([self.eta_p, self.eta_p]),
            "--",
            color="k",
            label=f"p-value threshold",
        )
        twin_ax.set_ylabel("p-value", color="tab:cyan")
        twin_ax.tick_params(axis="y", labelcolor="tab:cyan")

        # Plot the criteria
        self.add_criteria_to_plot(axs[2])
        axs[2].set_xlim((self.data_object.time_vector[0], self.data_object.time_vector[-1]))
        axs[2].legend(bbox_to_anchor=(1.025, 0.5), loc="center left")
        axs[2].set_ylabel("Criteria [normalized]")
        axs[2].set_xlabel("Time [s]")

        plt.subplots_adjust(bottom=0.07, top=0.95, left=0.1, right=0.7, hspace=0.15)

        # If wanted, save the figure
        if save_name is not None:
            extension = check_save_name(save_name)
            plt.savefig(save_name, format=extension)

        if live_show:
            plt.show()

        return fig  # for plot tests
