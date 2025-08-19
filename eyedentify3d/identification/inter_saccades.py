import numpy as np
import pingouin as pg

from ..utils.data_utils import DataObject
from ..utils.rotation_utils import get_angle_between_vectors
from ..utils.sequence_utils import split_sequences, apply_minimal_number_of_frames, merge_sequence_lists


class InterSaccadicEvent:
    """
    Class to extract 'inter-saccadic' sequences. These sequences are defined as the sequences of frames that have not
    been yet identified (this is partly why order matters in the identification of the behaviors).
    Please note that an intersaccadic sequence is not a behavior in itself, but rather a collection of frames that will
    be used for the identification of fixation and smooth pursuit behaviors.
    """

    def __init__(
        self,
        data_object: DataObject,
        identified_indices: np.ndarray,
        window_duration: float = 0.022,
        window_overlap: float = 0.006,
        eta_p: float = 0.01,
        eta_d: float = 0.45,
        eta_cd: float = 0.5,
        eta_pd: float = 0.2,
        eta_max_fixation: float = 1.9,
        eta_min_smooth_pursuit: float = 1.7,
        phi: float = 45,
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

        # Original attributes
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
        self.frame_indices: np.ndarray | None = None
        self.sequences: list[np.ndarray] = []
        self.coherent_sequences = None
        self.incoherent_sequences = None
        self.fixation_indices = None
        self.smooth_pursuit_indices = None
        self.uncertain_sequences = None

        # Detect visual scanning sequences
        self.detect_intersaccadic_indices(identified_indices)
        self.detect_intersaccadic_sequences()
        self.set_coherent_and_incoherent_sequences(data_object)
        self.classify_sequences(data_object, identified_indices)

    def detect_intersaccadic_indices(self, identified_indices: np.ndarray):
        """
        Detect when velocity is above the threshold and if the frames are not already identified.
        """
        self.frame_indices = np.where(identified_indices == False)[0]

    def detect_intersaccadic_sequences(self):
        """
        Detect the frames where there is a visual scanning.
        """
        sequence_candidates = split_sequences(self.frame_indices)
        self.sequences = apply_minimal_number_of_frames(sequence_candidates, minimal_number_of_frames=3)

    @staticmethod
    def detect_directionality_coherence_on_axis(gaze_direction: np.ndarray, component_to_keep: int) -> float:
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
        angle = np.zeros((nb_frames,))
        for i_frame in range(nb_frames - 1):
            # TODO: Check the logic of the acrsin (arccos instead ?)
            gaze_displacement = (
                gaze_direction[component_to_keep, i_frame + 1] - gaze_direction[component_to_keep, i_frame]
            )
            # angle[i_frame] = np.arcsin(gaze_displacement / np.linalg.norm(gaze_displacement))  # 0
            angle[i_frame] = np.arccos(gaze_displacement / np.linalg.norm(gaze_displacement))  # 1

        # Test that the gaze displacement and orientation are coherent inside the window
        z_value, p_value = pg.circ_rayleigh(angle)
        return p_value

    # def detect_directionality_coherence_on_sphere(self, gaze_direction: np.ndarray) -> float:
    #     """
    #     Detects the coherence of the gaze direction inside a window using a Rayleigh z-test on both spherical
    #     coordinates. The directionality is considered coherent if one of the two spherical coordinates is coherent.
    #     This function first computes the gaze displacement vector between two consecutive frames, and then
    #     transforms it into spherical coordinates (angle_1 and angle_2). Finally, a Rayleigh z-test is applied
    #     on the angles to check if the gaze displacement is uniformly distributed on the selected window. If the gaze
    #     direction is coherent (i.e., the p-value is smaller than eta_p), the gaze is moving in a particular direction.
    #
    #     Parameters
    #     ----------
    #     gaze_direction: A 2D numpy array of shape (3, nb_frames) expressing the gaze (head + eyes) direction in 3D space
    #         through a unit vector.
    #     """
    #     nb_frames = gaze_direction.shape[1]
    #     angle_1 = np.zeros((nb_frames,))
    #     angle_2 = np.zeros((nb_frames,))
    #     for i_frame in range(nb_frames - 1):
    #         gaze_displacement = gaze_direction[:, i_frame + 1] - gaze_direction[:, i_frame]
    #         angle_1[i_frame] = np.arctan2(gaze_displacement[1], gaze_displacement[0])  # Azimuth angle if XYZ coordinates
    #         angle_2[i_frame] = np.arccos(gaze_displacement[2], np.linalg.norm(gaze_displacement[:2]))  # Elevation angle if XYZ coordinates
    #
    #     # Test that the gaze displacement and orientation are coherent inside the window
    #     z_value_1, p_value_1 = pg.circ_rayleigh(angle_1)
    #     z_value_2, p_value_2 = pg.circ_rayleigh(angle_2)
    #     return np.logical_or(p_value_1 < self.eta_p, p_value_2 < self.eta_p)

    @staticmethod
    def variability_decomposition(gaze_direction: np.ndarray) -> tuple[float, float]:
        """
        Computes the gaze direction variability and decompose it into a principal and second axis. It returns the
        length of the principal and second components (d_pc1 and d_pc2 in Larsson et al. 2015).
        """

        nb_frames = gaze_direction.shape[1]
        if nb_frames < 4:
            raise ValueError(
                "The gaze direction must contain at least 4 frames for the variability decomposition to be "
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
            # No variability means something went wrong
            principal_axis = np.array([np.nan, np.nan, np.nan])
            second_axis = np.array([np.nan, np.nan, np.nan])
            raise RuntimeError(cov)
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
        parameter_R = np.arctan(mean_gaze_direction_radius_range)

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

            while current_window_start_time < sequence_end_time:
                # Find start index for current window
                window_start_idx = self._find_time_index(
                    time_vector, current_window_start_time, sequence_start_idx, sequence_end_idx
                )

                # Calculate end time and find corresponding index
                window_end_time = current_window_start_time + self.window_duration
                window_end_idx = self._find_time_index(
                    time_vector, window_end_time, sequence_start_idx, sequence_end_idx
                )

                # Handle edge case: if remaining sequence is very short, extend to sequence end
                remaining_duration = sequence_end_time - time_vector[window_end_idx]
                if remaining_duration <= self.window_overlap:
                    window_end_idx = sequence_end_idx

                # # Only add windows with sufficient samples
                # if window_end_idx - window_start_idx >= 3:  # Minimum 3 samples
                if window_end_idx - window_start_idx < 3:
                    raise RuntimeError("The merging went wrong")

                window_sequences.append(np.arange(window_start_idx, window_end_idx + 1))

                # Break if we've reached the end
                if window_end_idx >= sequence_end_idx:
                    break

                # Calculate next window start time with overlap
                current_window_start_time = time_vector[window_end_idx] - self.window_overlap

        return window_sequences

    @staticmethod
    def _find_time_index(time_vector: np.ndarray, target_time: float, start_bound: int, end_bound: int) -> int:
        """
        Find the index corresponding to a target time within specified bounds.

        Parameters
        ----------
        time_vector: Array of time values
        target_time: Time to find index for
        start_bound: Minimum valid index
        end_bound: Maximum valid index

        Returns
        -------
            idx: The index closest to target_time within bounds
        """
        candidate_idx = np.searchsorted(time_vector, target_time, side="left")

        # Ensure index is within valid bounds
        idx = np.clip(candidate_idx, start_bound, end_bound)

        return idx

        #
        #
        # window_indices = []
        # for intersaccadic_sequence in self.sequences:
        #
        #     # Initialize the indices of the window
        #     window_start_idx = intersaccadic_sequence[0]
        #     current_window_end = 0
        #     end_of_intersaccadic_gap = intersaccadic_sequence[-1]
        #     while current_window_end < end_of_intersaccadic_gap:
        #         current_window_end = \
        #         np.where(data_object.time_vector > data_object.time_vector[window_start_idx] + self.window_duration)[0]
        #         if len(current_window_end) != 0 and current_window_end[0] < end_of_intersaccadic_gap:
        #             if end_of_intersaccadic_gap - current_window_end[0] <= 2:
        #                 current_window_end = end_of_intersaccadic_gap
        #             else:
        #                 current_window_end = current_window_end[0]
        #         else:
        #             current_window_end = end_of_intersaccadic_gap
        #         if current_window_end - window_start_idx > 2:
        #             window_indices.append(np.arange(window_start_idx, current_window_end))
        #         if current_window_end == end_of_intersaccadic_gap:
        #             break
        #         window_start_idx = \
        #         np.where(data_object.time_vector < data_object.time_vector[current_window_end - 1] - self.window_overlap)[0][-1]
        #
        # return window_indices

    def set_coherent_and_incoherent_sequences(self, data_object: DataObject):
        """
        Split the inter-saccadic sequences into overlapping windows. Merge the consecutive windows that are similar in
        nature based on if they are coherent or incoherent in terms of gaze direction.
        """
        # Split this sequence into overlapping windows
        intersaccadic_window_sequences = self.get_window_sequences(time_vector=data_object.time_vector)

        # We store the p-values for each frame in a list of lists
        # Each frame can be part of several windows, so we store the p-values for each of the window it was part of and
        # use the mean p-value to classify this frame.
        p_values = [[] for _ in range(len(data_object.time_vector))]

        for current_window in intersaccadic_window_sequences:
            # Sanity check
            nb_elements = int(np.prod(np.array(data_object.gaze_direction[:, current_window].shape)))
            if int(np.sum(np.isnan(data_object.gaze_direction[:, current_window]))) > (nb_elements - 6):
                # Too much NaN values in the window, skip it
                continue

            # Compute the directionality coherence p-value for the current window
            p_value = self.detect_directionality_coherence_on_axis(
                data_object.gaze_direction[:, current_window], component_to_keep=1
            )
            for i_idx in current_window:
                p_values[i_idx] += [p_value]

        # The mean p-value for each timestamp is used for the coherence/incoherence classification
        mean_p_values = np.array([np.nanmean(np.array(p)) for p in p_values])

        incoherent_indices = np.where(mean_p_values <= self.eta_p)[0]
        self.incoherent_sequences = split_sequences(incoherent_indices)

        coherent_indices = np.where(mean_p_values > self.eta_p)[0]
        self.coherent_sequences = split_sequences(coherent_indices)

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
        for i_sequence, sequence in enumerate(all_intersaccadic_sequences):
            parameter_D, parameter_CD, parameter_PD, parameter_R = self.compute_larsson_parameters(
                data_object.gaze_direction[:, sequence]
            )
            criteria_1 = parameter_D < self.eta_d
            criteria_2 = parameter_CD > self.eta_cd
            criteria_3 = parameter_PD > self.eta_pd
            criteria_4 = parameter_R * 180 / np.pi > self.eta_max_fixation

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

    # def classify_ambiguous_sequences(self,
    #                                  data_object: DataObject,
    #                                  all_intersaccadic_sequences: list[np.ndarray],
    #                                  ambiguous_indices: np.ndarray,
    #                                  fixation_indices: np.ndarray,
    #                                  smooth_pursuit_indices: np.ndarray,
    #                                  ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    #     """
    #     Sequences where some of the criteria are met are classified as smooth pursuit like or fixation like segments.
    #      - smooth pursuit like if ...
    #      - fixation like if ...
    #     """
    #     ambiguous_sequences = split_sequences(ambiguous_indices)
    #
    #     uncertain_sequences_to_remove = []
    #     for i_sequence, sequence in enumerate(ambiguous_sequences):
    #         parameter_D, parameter_CD, parameter_PD, parameter_R = self.compute_larsson_parameters(
    #             data_object.gaze_direction[:, sequence]
    #         )
    #         criteria_3 = parameter_PD > self.eta_pd
    #         criteria_4 = parameter_R * 180 / np.pi > self.eta_max_fixation
    #         if criteria_3:
    #             # Smooth pursuit like segment
    #             same_segment_backward = True
    #             before_idx = sequence[0]
    #             current_i_sequence = i_sequence
    #             if i_sequence == 0:
    #                 same_segment_backward = False
    #             while same_segment_backward:
    #                 current_i_sequence -= 1
    #                 if before_idx in all_intersaccadic_sequences[current_i_sequence]:
    #                     if before_idx in fixation_indices:
    #                         same_segment_backward = False
    #                     else:
    #                         uncertain_mean = np.nanmean(data_object.gaze_direction[:, sequence], axis=1)
    #                         current_mean = np.nanmean(
    #                             data_object.gaze_direction[:, all_intersaccadic_sequences[current_i_sequence]], axis=1
    #                         )
    #                         angle = get_angle_between_vectors(uncertain_mean, current_mean)
    #                         if np.abs(angle) * 180 / np.pi < self.phi:
    #                             before_idx = all_intersaccadic_sequences[current_i_sequence][0]
    #                         else:
    #                             same_segment_backward = False
    #                 else:
    #                     same_segment_backward = False
    #             same_segment_forward = True
    #             after_idx = sequence[-1]
    #             current_i_sequence = i_sequence
    #             if i_sequence == len(all_intersaccadic_sequences) - 1:
    #                 same_segment_forward = False
    #             while same_segment_forward:
    #                 current_i_sequence += 1
    #                 if len(all_intersaccadic_sequences) <= current_i_sequence:
    #                     same_segment_forward = False
    #                 elif after_idx in all_intersaccadic_sequences[current_i_sequence]:
    #                     if after_idx in fixation_indices:
    #                         same_segment_forward = False
    #                     else:
    #                         uncertain_mean = np.nanmean(data_object.gaze_direction[:, sequence], axis=1)
    #                         current_mean = np.nanmean(
    #                             data_object.gaze_direction[:, all_intersaccadic_sequences[current_i_sequence]], axis=1
    #                         )
    #                         angle = get_angle_between_vectors(uncertain_mean, current_mean)
    #                         if np.abs(angle) * 180 / np.pi < self.phi:
    #                             after_idx = all_intersaccadic_sequences[current_i_sequence][-1]
    #                         else:
    #                             same_segment_forward = False
    #                 else:
    #                     same_segment_forward = False
    #             if len(range(before_idx, after_idx)) > 2:
    #                 parameter_D, parameter_CD, parameter_PD, parameter_R = self.compute_larsson_parameters(
    #                     data_object.gaze_direction[:, range(before_idx, after_idx + 1)]
    #                 )
    #                 if parameter_R * 180 / np.pi > self.eta_min_smooth_pursuit:
    #                     smooth_pursuit_indices += list(range(before_idx, after_idx))
    #                 else:
    #                     fixation_indices += list(range(before_idx, after_idx))
    #
    #                 for i_uncertain_sequences, uncertain in enumerate(ambiguous_sequences):
    #                     if any(item in uncertain for item in list(range(before_idx, after_idx))):
    #                         uncertain_sequences_to_remove += [i_uncertain_sequences]
    #         else:
    #             # Fixation like segment
    #             if criteria_4:
    #                 smooth_pursuit_indices += list(sequence)
    #             else:
    #                 fixation_indices += list(sequence)
    #             uncertain_sequences_to_remove += [i_sequence]
    #
    #     uncertain_sequences = []
    #     for i_sequence in range(len(ambiguous_sequences)):
    #         if i_sequence not in uncertain_sequences_to_remove:
    #             uncertain_sequences += [ambiguous_sequences[i_sequence]]
    #
    #     return fixation_indices, smooth_pursuit_indices, uncertain_sequences

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
            criteria_4 = parameter_R * 180 / np.pi > self.eta_max_fixation

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

        if parameter_R * 180 / np.pi > self.eta_min_smooth_pursuit:
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

    def classify_sequences(self, data_object: DataObject, identified_indices: np.ndarray):
        """
        Classify the inter-saccadic sequences into coherent and incoherent sequences based on the gaze direction.
        TODO: add _on_sphere option.
        """

        all_intersaccadic_sequences = merge_sequence_lists(
            self.coherent_sequences,
            self.incoherent_sequences,
        )
        if len(all_intersaccadic_sequences) == 1 and all_intersaccadic_sequences[0].shape == (0,):
            raise RuntimeError(
                "There should be at least one even if there is no saccades. "
                "This should not happen, please contact the developer."
            )

        fixation_indices, smooth_pursuit_indices, ambiguous_indices = self.classify_obvious_sequences(
            data_object,
            all_intersaccadic_sequences,
        )
        fixation_indices, smooth_pursuit_indices, uncertain_sequences = self.classify_ambiguous_sequences(
            data_object,
            all_intersaccadic_sequences,
            ambiguous_indices,
            fixation_indices,
            smooth_pursuit_indices,
        )

        self.smooth_pursuit_indices = np.sort(smooth_pursuit_indices)
        self.fixation_indices = np.sort(fixation_indices)
        self.uncertain_sequences = uncertain_sequences

        return
