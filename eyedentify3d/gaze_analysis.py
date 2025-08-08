"""
This code aims to identify visual behavior sequences, namely blink, fixations, saccades, smooth pursuit, and visual scanning.
We consider that when the head rotates, the image in the VR helmet (eye-tracker) rotates by the same amount, making it
as if the head was rotating around the subjects eyes instead of the neck joint center.
"""

from pathlib import Path
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
import biorbd
import pingouin as pg
from scipy import signal

"""
Variables extracted from the raw data and used for the current analysis.
'time(100ns)' = time stamps of the recorded frames
'eye_valid_L' = if the data is valid for the left eye
'eye_valid_R' = if the data is valid for the right eye
'gaze_direct_L.x' = gaze direction vector in the helmet reference frame for the left eye
'gaze_direct_L.y' = gaze direction vector in the helmet reference frame for the left eye
'gaze_direct_L.z' = gaze direction vector in the helmet reference frame for the left eye
'gaze_direct_R.x' = gaze direction vector in the helmet reference frame for the right eye
'gaze_direct_R.y' = gaze direction vector in the helmet reference frame for the right eye
'gaze_direct_R.z' = gaze direction vector in the helmet reference frame for the right eye
'helmet_rot_x' = rotation of the helmet in degrees (downward rotation is positive)
'helmet_rot_y' = rotation of the helmet in degrees (leftward rotation is positive)
'helmet_rot_z' = rotation of the helmet in degrees (right tilt rotation is positive)
"""

# Plot flags ---------------------------------------
PLOT_BAD_DATA_FLAG = False
PLOT_SACCADES_FLAG = False
PLOT_CLASSIFICATION_FLAG = True  # Only one that should be True not when debugging
PLOT_INTERSACCADIC_FLAG = False
PLOT_CRITERIA_FLAG = False
PLOT_ROTATION_VELOCITIES_FLAG = False
duration_after_cue = 2  # Seconds  # This is the length of the time period between the auditory cue and the black screen
# ------------------------------------------------------------


def detect_blinks(data):
    """
    Blinks are detected when both eye openness drops bellow 0.5 il line with
    https://ieeexplore.ieee.org/abstract/document/9483841
    """
    blink_threshold = 0.5
    blink_timing_both = np.where((data["openness_R"] < blink_threshold) & (data["openness_L"] < blink_threshold))[0]

    # Group the indices into sequences
    blink_sequences = np.array_split(blink_timing_both, np.flatnonzero(np.diff(blink_timing_both) > 1) + 1)

    return blink_sequences


def detect_saccades(time_vector, eye_direction, gaze_direction):
    """
    Detecting sequences where the :
    - eye angular velocity is larger than 5 times the median of the trial (over a 61 frames/500ms window)
    - eye angular acceleration is larger than 4000 deg/s² for at least one frame
    Only the eye movements were used to identify saccades.
    """

    eye_angular_velocity_rad = np.zeros((eye_direction.shape[1],))
    for i_frame in range(1, eye_direction.shape[1] - 1):  # Skipping the first and last frames
        vector_before = eye_direction[:, i_frame - 1]
        vector_after = eye_direction[:, i_frame + 1]
        eye_angular_velocity_rad[i_frame] = np.arccos(
            np.dot(vector_before, vector_after) / np.linalg.norm(vector_before) / np.linalg.norm(vector_after)
        ) / (time_vector[i_frame + 1] - time_vector[i_frame - 1])
        if np.isnan(eye_angular_velocity_rad[i_frame]) and not (
            any(np.isnan(vector_before)) or any(np.isnan(vector_after))
        ):
            if (
                vector_before[0] == vector_after[0]
                and vector_before[1] == vector_after[1]
                and vector_before[2] == vector_after[2]
            ):
                eye_angular_velocity_rad[i_frame] = 0
            else:
                raise RuntimeError(
                    f" Please review these variable values : \n "
                    f"eye_angular_velocity_rad[i_frame] = {eye_angular_velocity_rad[i_frame]} \n"
                    f"vector_before = {vector_before} \n"
                    f"vector_after = {vector_after} \n"
                )
    eye_angular_velocity_rad[0] = np.arccos(
        np.dot(eye_direction[:, 0], eye_direction[:, 1])
        / np.linalg.norm(eye_direction[:, 0])
        / np.linalg.norm(eye_direction[:, 1])
    ) / (time_vector[1] - time_vector[0])
    eye_angular_velocity_rad[-1] = np.arccos(
        np.dot(eye_direction[:, -2], eye_direction[:, -1])
        / np.linalg.norm(eye_direction[:, -2])
        / np.linalg.norm(eye_direction[:, -1])
    ) / (time_vector[-1] - time_vector[-2])

    eye_angular_acceleration_rad = np.zeros((eye_direction.shape[1],))
    eye_angular_acceleration_rad[:-1] = (eye_angular_velocity_rad[1:] - eye_angular_velocity_rad[:-1]) / (
        time_vector[1:] - time_vector[:-1]
    )

    acceleration_threshold = 4000  # deg/s²
    velocity_threshold = np.zeros((eye_angular_velocity_rad.shape[0],))
    window_size = 62  # Actually 61 frames

    velocity_threshold[: int(window_size / 2)] = (
        np.nanmedian(np.abs(eye_angular_velocity_rad[:window_size]) * 180 / np.pi) * 5
    )
    for i_frame in range(eye_angular_velocity_rad.shape[0] - window_size):
        velocity_threshold[int(i_frame + window_size / 2)] = (
            np.nanmedian(np.abs(eye_angular_velocity_rad[i_frame : i_frame + window_size]) * 180 / np.pi) * 5
        )
    velocity_threshold[int(-window_size / 2) :] = (
        np.nanmedian(np.abs(eye_angular_velocity_rad[-window_size:]) * 180 / np.pi) * 5
    )

    # Velocity + 1 frames acceleration classification
    saccade_timing = np.where(np.abs(eye_angular_velocity_rad * 180 / np.pi) > velocity_threshold)[0]
    saccade_sequences_tempo = np.array_split(saccade_timing, np.flatnonzero(np.diff(saccade_timing) > 1) + 1)
    saccade_sequences = []

    if saccade_sequences_tempo[0].shape != (0,):
        for i in saccade_sequences_tempo:
            if len(i) <= 1:
                continue
            acceleration_above_threshold = np.where(
                np.abs(eye_angular_acceleration_rad[i[0] - 1 : i[-1] + 1] * 180 / np.pi) > acceleration_threshold
            )[0]
            if len(acceleration_above_threshold) > 1:
                saccade_sequences += [i]

    # merge saccades events that are less than 5 frames appart and the gaze is moving in the same direction
    saccade_sequences_merged = merge_close_sequences(saccade_sequences, gaze_direction, check_directionnality=True)

    # Saccade amplitude
    # Defined as the angle between the beginning and end of the saccade,
    # note that there is no check made to detect if there is a larger amplitude reached during the saccade.
    saccade_amplitudes = []
    for sequence in saccade_sequences_merged:
        vector_before = eye_direction[:, sequence[0]]
        vector_after = eye_direction[:, sequence[-1]]
        angle = np.arccos(
            np.dot(vector_before, vector_after) / np.linalg.norm(vector_before) / np.linalg.norm(vector_after)
        )
        saccade_amplitudes += [angle * 180 / np.pi]

    return (
        saccade_sequences_merged,
        eye_angular_velocity_rad,
        eye_angular_acceleration_rad,
        saccade_amplitudes,
        velocity_threshold,
        acceleration_threshold,
    )


def get_gaze_direction(helmet_rotation_unwrapped_deg, eye_direction):
    """
    Gaze direction is the sum of the head + eye rotation.
    """
    helmet_rotation_in_rad = helmet_rotation_unwrapped_deg * np.pi / 180

    gaze_direction = np.zeros(eye_direction.shape)
    for i_frame in range(helmet_rotation_in_rad.shape[1]):
        rotation_matrix = biorbd.Rotation.fromEulerAngles(helmet_rotation_in_rad[:, i_frame], "xyz").to_array()
        gaze_direction[:, i_frame] = rotation_matrix @ eye_direction[:, i_frame]

    return gaze_direction


def detect_visual_scanning(time_vector, gaze_direction, saccade_sequences, helmet_rotation_unwrapped_deg):
    """
    Identify sequences where the gaze velocity is larger than 100 deg/s, but which are not saccades.
    """

    # Compute gaze (head + eye) velocity
    gaze_angular_velocity_rad = np.zeros((gaze_direction.shape[1],))
    for i_frame in range(1, gaze_direction.shape[1] - 1):  # Skipping the first and last frames
        vector_before = gaze_direction[:, i_frame - 1]
        vector_after = gaze_direction[:, i_frame + 1]
        gaze_angular_velocity_rad[i_frame] = np.arccos(
            np.dot(vector_before, vector_after) / np.linalg.norm(vector_before) / np.linalg.norm(vector_after)
        ) / (time_vector[i_frame + 1] - time_vector[i_frame - 1])
        if np.isnan(gaze_angular_velocity_rad[i_frame]) and not (
            any(np.isnan(vector_before)) or any(np.isnan(vector_after))
        ):
            raise RuntimeError(
                f" Please review these variable values : \n "
                f"gaze_angular_velocity_rad[i_frame] = {gaze_angular_velocity_rad[i_frame]} \n"
                f"vector_before = {vector_before} \n"
                f"vector_after = {vector_after} \n"
            )
    gaze_angular_velocity_rad[0] = np.arccos(
        np.dot(gaze_direction[:, 0], gaze_direction[:, 1])
        / np.linalg.norm(gaze_direction[:, 0])
        / np.linalg.norm(gaze_direction[:, 1])
    ) / (time_vector[1] - time_vector[0])
    gaze_angular_velocity_rad[-1] = np.arccos(
        np.dot(gaze_direction[:, -2], gaze_direction[:, -1])
        / np.linalg.norm(gaze_direction[:, -2])
        / np.linalg.norm(gaze_direction[:, -1])
    ) / (time_vector[-1] - time_vector[-2])

    velocity_threshold_visual_scanning = 100

    saccade_sequences_timing = (
        np.hstack(saccade_sequences) if len(saccade_sequences) > 1 else np.array(saccade_sequences)
    )
    visual_scanning_candidates = np.where(
        np.abs(gaze_angular_velocity_rad * 180 / np.pi) > velocity_threshold_visual_scanning
    )[0]
    visual_scanning_timing = np.array([i for i in visual_scanning_candidates if i not in saccade_sequences_timing])

    # Group the indices into sequences
    visual_scanning_sequences = np.array_split(
        visual_scanning_timing, np.flatnonzero(np.diff(visual_scanning_timing) > 1) + 1
    )

    visual_scanning_sequences = merge_close_sequences(
        visual_scanning_sequences, gaze_direction, check_directionnality=True
    )

    return visual_scanning_sequences, gaze_angular_velocity_rad, velocity_threshold_visual_scanning


def apply_minimal_duration(sequences_tempo, number_of_frames_min):
    """
    Consider only sequences that are longer than a certain number of frames
    """
    sequences = []
    for i_sequence in sequences_tempo:
        if len(i_sequence) < number_of_frames_min:
            continue
        sequences += [i_sequence]
    return sequences


def sliding_window(time_vector, intersaccadic_sequences, gaze_direction):
    """
    Parameters t_wind (22000 micros), t_overlap (6000 micros) and eta_p (0.1) taken from the original paper
    https://doi.org/10.1016/j.bspc.2014.12.008
    """
    t_wind = 0.022 * 5  # Window size in ms 0.022
    t_overlap = 0.006 * 5  # Window overlap in ms 0.006
    eta_p = 0.001  # Threshold for the p-value of the Rayleigh test

    intersaccadic_window_idx = []
    for i_intersaccadic_gap in intersaccadic_sequences:

        # Index of the windows
        window_start_idx = i_intersaccadic_gap[0]
        window_end_idx = 0
        end_of_intersaccadic_gap = i_intersaccadic_gap[-1]
        while window_end_idx < end_of_intersaccadic_gap:
            window_end_idx = np.where(time_vector > time_vector[window_start_idx] + t_wind)[0]
            if len(window_end_idx) != 0 and window_end_idx[0] < end_of_intersaccadic_gap:
                if end_of_intersaccadic_gap - window_end_idx[0] <= 2:
                    window_end_idx = end_of_intersaccadic_gap
                else:
                    window_end_idx = window_end_idx[0]
            else:
                window_end_idx = end_of_intersaccadic_gap
            if window_end_idx - window_start_idx > 2:
                intersaccadic_window_idx.append(np.arange(window_start_idx, window_end_idx))
            if window_end_idx == end_of_intersaccadic_gap:
                break
            window_start_idx = np.where(time_vector < time_vector[window_end_idx - 1] - t_overlap)[0][-1]

    # The mean p-value for each timestamp is used for the coherence/incoherence classification
    p_values = [[] for i in range(len(time_vector))]
    for i_window in intersaccadic_window_idx:
        nb_elements = int(np.prod(np.array(gaze_direction[:, i_window].shape)))
        if int(np.sum(np.isnan(gaze_direction[:, i_window]))) > (nb_elements - 6):
            continue
        p_value_alpha = detect_directionality_coherence(gaze_direction[:, i_window])
        for i_idx in i_window:
            p_values[i_idx] += [p_value_alpha]

    mean_p_values = np.array([np.nanmean(np.array(p)) for p in p_values])
    incoherent_windows = np.where(mean_p_values <= eta_p)[0]
    coherent_windows = np.where(mean_p_values > eta_p)[0]

    intersaccadic_coherent_sequences = np.array_split(
        coherent_windows, np.flatnonzero(np.diff(coherent_windows) > 1) + 1
    )
    intersaccadic_incoherent_sequences = np.array_split(
        incoherent_windows, np.flatnonzero(np.diff(incoherent_windows) > 1) + 1
    )

    intersaccadic_gouped_sequences = intersaccadic_coherent_sequences[:]
    for i_sequence in intersaccadic_incoherent_sequences:
        for j in range(len(intersaccadic_gouped_sequences)):
            if len(intersaccadic_gouped_sequences[j]) == 0:
                intersaccadic_gouped_sequences = [i_sequence]
            elif i_sequence[0] < intersaccadic_gouped_sequences[0][0]:
                intersaccadic_gouped_sequences.insert(0, i_sequence)
            else:
                if i_sequence[0] > intersaccadic_gouped_sequences[j][0]:
                    if j == len(intersaccadic_gouped_sequences) - 1:
                        intersaccadic_gouped_sequences.append(i_sequence)
                    elif (
                        i_sequence[0] > intersaccadic_gouped_sequences[j][0]
                        and i_sequence[0] < intersaccadic_gouped_sequences[j + 1][0]
                    ):
                        intersaccadic_gouped_sequences.insert(j + 1, i_sequence)

    return intersaccadic_gouped_sequences, intersaccadic_coherent_sequences, intersaccadic_incoherent_sequences


def plot_intersaccadic_interval_subinterval_cut(
    time_vector, intersaccadic_coherent_sequences, intersaccadic_incoherent_sequences, fig_name
):
    plt.figure()
    coherent_label = False
    if intersaccadic_coherent_sequences[0].shape != (0,):
        for i_sequence in intersaccadic_coherent_sequences:
            if not coherent_label:
                plt.axvspan(
                    time_vector[i_sequence[0]],
                    time_vector[i_sequence[-1]],
                    edgecolor=None,
                    color="tab:green",
                    alpha=0.5,
                    label="Coherent",
                )
                coherent_label = True
            else:
                plt.axvspan(
                    time_vector[i_sequence[0]],
                    time_vector[i_sequence[-1]],
                    edgecolor=None,
                    color="tab:green",
                    alpha=0.5,
                )
    incoherent_label = False
    if intersaccadic_incoherent_sequences[0].shape != (0,):
        for i_sequence in intersaccadic_incoherent_sequences:
            if not incoherent_label:
                plt.axvspan(
                    time_vector[i_sequence[0]],
                    time_vector[i_sequence[-1]],
                    edgecolor=None,
                    color="tab:red",
                    alpha=0.5,
                    label="Incoherent",
                )
                incoherent_label = True
            else:
                plt.axvspan(
                    time_vector[i_sequence[0]], time_vector[i_sequence[-1]], edgecolor=None, color="tab:red", alpha=0.5
                )
    plt.xlim(time_vector[0], time_vector[-1])
    plt.legend()
    plt.savefig(f"figures/intersaccadic_interval_{fig_name}.png")
    return


def detect_directionality_coherence(gaze_direction):

    alpha = np.zeros((gaze_direction.shape[1],))
    for i_frame in range(gaze_direction.shape[1] - 1):
        gaze_displacement_this_time = gaze_direction[:, i_frame + 1] - gaze_direction[:, i_frame]
        alpha[i_frame] = np.arcsin(gaze_displacement_this_time[0] / np.linalg.norm(gaze_displacement_this_time))

    # Test that the gaze displacement and orientation are coherent inside the window
    z_value_alpha, p_value_alpha = pg.circ_rayleigh(alpha)
    return p_value_alpha


def discriminate_fixations_and_smooth_pursuit(gaze_direction):

    mean_gaze_direction = np.nanmean(gaze_direction, axis=1)
    zeros_mean_gaze_direction = gaze_direction - mean_gaze_direction[:, np.newaxis]
    cov = np.ma.cov(np.ma.masked_invalid(zeros_mean_gaze_direction)).data
    eigen_values_decomposition = np.linalg.eig(cov)
    if np.sum(cov) == 0:
        principal_axis = np.array([np.nan, np.nan, np.nan])
        second_axis = np.array([np.nan, np.nan, np.nan])
    else:
        largest_eigen_value = np.argmax(eigen_values_decomposition[0])
        smallest_eigen_value = np.argmin(eigen_values_decomposition[0])
        index_list = [0, 1, 2]
        index_list.remove(largest_eigen_value)
        index_list.remove(smallest_eigen_value)
        second_largest_eigen_value = index_list[0]
        principal_axis = eigen_values_decomposition[1][:, largest_eigen_value]
        second_axis = eigen_values_decomposition[1][:, second_largest_eigen_value]
    principal_projection = np.zeros((gaze_direction.shape[1]))
    second_projection = np.zeros((gaze_direction.shape[1]))
    for i in range(gaze_direction.shape[1]):
        principal_projection[i] = np.dot(zeros_mean_gaze_direction[:, i], principal_axis)
        second_projection = np.dot(zeros_mean_gaze_direction[:, i], second_axis)  # d_pc2
    d_pc1 = np.max(principal_projection) - np.min(principal_projection)
    d_pc2 = np.max(second_projection) - np.min(second_projection)
    if np.abs(np.dot(principal_axis, second_axis)) > 0.0001:
        # The third axis should be almost null since the data is projected on a sphere (no variance radially)
        raise ValueError("The principal and second axis are not orthogonal.")

    gaze_distance_parcourrue = np.linalg.norm(gaze_direction[:, -1] - gaze_direction[:, 0])
    trajectory_length = np.sum(np.linalg.norm(gaze_direction[:, 1:] - gaze_direction[:, :-1], axis=0))
    mean_radius_range_gaze_direction = np.sqrt(
        (np.max(gaze_direction[0, :]) - np.min(gaze_direction[0, :])) ** 2
        + (np.max(gaze_direction[1, :]) - np.min(gaze_direction[1, :])) ** 2
        + (np.max(gaze_direction[2, :]) - np.min(gaze_direction[2, :])) ** 2
    )

    parameter_D = d_pc2 / d_pc1
    parameter_CD = gaze_distance_parcourrue / d_pc1
    parameter_PD = gaze_distance_parcourrue / trajectory_length
    parameter_R = np.arctan(mean_radius_range_gaze_direction / 1)

    return parameter_D, parameter_CD, parameter_PD, parameter_R


def merge_close_sequences(sequences_candidates, gaze_direction, check_directionnality=False):
    """
    Merge events that are less than 5 frames appart and the gaze is moving in the same direction (if check_directionnality=True)
    """

    sequences_candidates = apply_minimal_duration(sequences_candidates, 2)
    if len(sequences_candidates) == 0:
        sequences_merged = []
    else:
        sequences_merged = [sequences_candidates[0]]
        current_merged_sequence = 0
        current_sequence = 0
        while current_sequence < len(sequences_candidates) - 1:
            for i in range(current_sequence + 1, len(sequences_candidates)):
                beginning_of_merged_sequence = sequences_merged[current_merged_sequence][0]
                end_of_merged_sequence = sequences_merged[current_merged_sequence][-1]
                beginning_of_new_sequence = sequences_candidates[i][0]
                end_of_new_sequence = sequences_candidates[i][-1]
                not_invalid = (
                    np.sum(np.isnan(gaze_direction[:, range(end_of_merged_sequence, beginning_of_new_sequence + 1)]))
                    == 0
                )
                if beginning_of_new_sequence - end_of_merged_sequence < 5 and not_invalid:
                    if check_directionnality:
                        beginning_of_merged_sequence_direction = gaze_direction[:, beginning_of_merged_sequence]
                        end_of_merged_sequence_direction = gaze_direction[:, end_of_merged_sequence]
                        merged_sequence_direction = (
                            end_of_merged_sequence_direction - beginning_of_merged_sequence_direction
                        )

                        beginning_of_new_sequence_direction = gaze_direction[:, beginning_of_new_sequence]
                        end_of_new_sequence_direction = gaze_direction[:, end_of_new_sequence]
                        new_sequence_direction = end_of_new_sequence_direction - beginning_of_new_sequence_direction

                        angle = np.arccos(
                            np.dot(merged_sequence_direction, new_sequence_direction)
                            / np.linalg.norm(merged_sequence_direction)
                            / np.linalg.norm(new_sequence_direction)
                        )
                        candidate_interval = np.array(
                            range(sequences_merged[current_merged_sequence][0], sequences_candidates[i][-1] + 1)
                        )

                        if (
                            angle < 30 * np.pi / 180
                            and candidate_interval.shape != (0,)
                            and np.sum(np.isnan(gaze_direction[:, candidate_interval])) == 0
                        ):
                            criteria = True
                        else:
                            criteria = False
                    else:
                        candidate_interval = np.array(
                            range(sequences_merged[current_merged_sequence][0], sequences_candidates[i][-1] + 1)
                        )
                        criteria = True

                    if criteria:
                        sequences_merged[current_merged_sequence] = candidate_interval
                    else:
                        sequences_merged += [sequences_candidates[i]]
                        current_merged_sequence += 1
                        current_sequence = i
                    if i == len(sequences_candidates) - 1:
                        current_sequence += 1
                else:
                    sequences_merged += [sequences_candidates[i]]
                    current_merged_sequence += 1
                    current_sequence = i
                    break
    return sequences_merged


def detect_fixations_and_smooth_pursuit(
    time_vector, gaze_direction, intersaccadic_gouped_sequences, fig_name, PLOT_CRITERIA_FLAG
):
    """
    This gaze behavior classification is based on the algorithm described in Larsson et al. (2015).
    https://doi.org/10.1016/j.bspc.2014.12.008
    """
    # Parameters to define ---------------------------------------
    eta_D = 0.45  #  is the threshold for dispersion (without units)
    eta_CD = 0.5  # is the threshold for consistency of direction (without units)
    eta_PD = 0.5  # is the threshold for position displacement (without units)
    eta_maxFix = 3 * np.pi / 180  # is the threshold for spacial range (in degrees)
    phi = 45 * np.pi / 180  # is the threshold for similar angular range (in degrees)
    eta_minSmp = 2 * np.pi / 180  # is the threshold for merged segments spacial range (in degrees)

    if PLOT_CRITERIA_FLAG:
        fig, axs = plt.subplots(4, 1, figsize=(15, 10))

    # Classify the obvious timings
    fixation_timing = []
    smooth_pursuit_timing = []
    uncertain_timing = []
    if len(intersaccadic_gouped_sequences) == 1 and intersaccadic_gouped_sequences[0].shape == (0,):
        raise RuntimeError("No intersaccadic interval! There should be at least one even if there is no saccades.")
    else:
        for i_sequence, sequence in enumerate(intersaccadic_gouped_sequences):
            parameter_D, parameter_CD, parameter_PD, parameter_R = discriminate_fixations_and_smooth_pursuit(
                gaze_direction[:, sequence]
            )
            criteria_1 = parameter_D < eta_D
            criteria_2 = parameter_CD > eta_CD
            criteria_3 = parameter_PD > eta_PD
            criteria_4 = parameter_R > eta_maxFix

            if PLOT_CRITERIA_FLAG:
                if criteria_1:
                    axs[0].axvspan(
                        time_vector[sequence[0]],
                        time_vector[sequence[-1]],
                        edgecolor=None,
                        color="tab:green",
                        alpha=0.5,
                    )
                if criteria_2:
                    axs[1].axvspan(
                        time_vector[sequence[0]], time_vector[sequence[-1]], edgecolor=None, color="tab:blue", alpha=0.5
                    )
                if criteria_3:
                    axs[2].axvspan(
                        time_vector[sequence[0]],
                        time_vector[sequence[-1]],
                        edgecolor=None,
                        color="tab:purple",
                        alpha=0.5,
                    )
                if criteria_4:
                    axs[3].axvspan(
                        time_vector[sequence[0]],
                        time_vector[sequence[-1]],
                        edgecolor=None,
                        color="tab:orange",
                        alpha=0.5,
                    )

            sum_criteria = int(criteria_1) + int(criteria_2) + int(criteria_3) + int(criteria_4)
            if sum_criteria == 0:
                fixation_timing += list(sequence)
            elif sum_criteria == 4:
                smooth_pursuit_timing += list(sequence)
            else:
                uncertain_timing += list(sequence)

    if PLOT_CRITERIA_FLAG:
        axs[0].set_ylabel("Dispersion\nD < eta_D")
        axs[1].set_ylabel("Consistency of direction\nCD > eta_CD")
        axs[2].set_ylabel("Position displacement\nPD > eta_PD")
        axs[3].set_ylabel("Spacial range\nR > eta_maxFix")
        for i_ax in range(4):
            axs[i_ax].set_xlim(time_vector[0], time_vector[-1])
        plt.savefig(f"figures/criteria_{fig_name}.png")
        plt.close()

    # Classify the ambiguous timings
    uncertain_sequences_tempo = np.array_split(uncertain_timing, np.flatnonzero(np.diff(uncertain_timing) > 1) + 1)
    uncertain_sequences_to_remove = []
    for i_sequence, sequence in enumerate(uncertain_sequences_tempo):
        parameter_D, parameter_CD, parameter_PD, parameter_R = discriminate_fixations_and_smooth_pursuit(
            gaze_direction[:, sequence]
        )
        criteria_3 = parameter_PD > eta_PD
        criteria_4 = parameter_R > eta_maxFix
        if criteria_3:
            # Smooth pursuit like segment
            same_segment_backward = True
            before_idx = sequence[0]
            current_i_sequence = i_sequence
            if i_sequence == 0:
                same_segment_backward = False
            while same_segment_backward:
                current_i_sequence -= 1
                if before_idx in intersaccadic_gouped_sequences[current_i_sequence]:
                    if before_idx in fixation_timing:
                        same_segment_backward = False
                    else:
                        uncertain_mean = np.nanmean(gaze_direction[:, sequence], axis=1)
                        current_mean = np.nanmean(
                            gaze_direction[:, intersaccadic_gouped_sequences[current_i_sequence]], axis=1
                        )
                        angle = np.arccos(
                            np.dot(uncertain_mean, current_mean)
                            / np.linalg.norm(uncertain_mean)
                            / np.linalg.norm(current_mean)
                        )
                        if np.abs(angle) < phi:
                            before_idx = intersaccadic_gouped_sequences[current_i_sequence][0]
                        else:
                            same_segment_backward = False
                else:
                    same_segment_backward = False
            same_segment_forward = True
            after_idx = sequence[-1]
            current_i_sequence = i_sequence
            if i_sequence == len(intersaccadic_gouped_sequences) - 1:
                same_segment_forward = False
            while same_segment_forward:
                current_i_sequence += 1
                if len(intersaccadic_gouped_sequences) <= current_i_sequence:
                    same_segment_forward = False
                elif after_idx in intersaccadic_gouped_sequences[current_i_sequence]:
                    if after_idx in fixation_timing:
                        same_segment_forward = False
                    else:
                        uncertain_mean = np.nanmean(gaze_direction[:, sequence], axis=1)
                        current_mean = np.nanmean(
                            gaze_direction[:, intersaccadic_gouped_sequences[current_i_sequence]], axis=1
                        )
                        angle = np.arccos(
                            np.dot(uncertain_mean, current_mean)
                            / np.linalg.norm(uncertain_mean)
                            / np.linalg.norm(current_mean)
                        )
                        if np.abs(angle) < phi:
                            after_idx = intersaccadic_gouped_sequences[current_i_sequence][-1]
                        else:
                            same_segment_forward = False
                else:
                    same_segment_forward = False
            if len(range(before_idx, after_idx)) > 2:
                parameter_D, parameter_CD, parameter_PD, parameter_R = discriminate_fixations_and_smooth_pursuit(
                    gaze_direction[:, range(before_idx, after_idx + 1)]
                )
                if parameter_R > eta_minSmp:
                    smooth_pursuit_timing += list(range(before_idx, after_idx))
                else:
                    fixation_timing += list(range(before_idx, after_idx))

                for i_uncertain_sequences, uncertain in enumerate(uncertain_sequences_tempo):
                    if any(item in uncertain for item in list(range(before_idx, after_idx))):
                        uncertain_sequences_to_remove += [i_uncertain_sequences]
        else:
            # Fixation like segment
            if criteria_4:
                smooth_pursuit_timing += list(sequence)
            else:
                fixation_timing += list(sequence)
            uncertain_sequences_to_remove += [i_sequence]

    uncertain_sequences = []
    for i_sequence in range(len(uncertain_sequences_tempo)):
        if i_sequence not in uncertain_sequences_to_remove:
            uncertain_sequences += [uncertain_sequences_tempo[i_sequence]]

    smooth_pursuit_timing = np.sort(smooth_pursuit_timing)
    fixation_timing = np.sort(fixation_timing)
    fixation_sequences_candidates = np.array_split(
        np.array(fixation_timing), np.flatnonzero(np.diff(np.array(fixation_timing)) > 1) + 1
    )
    smooth_pursuit_sequences_cadidates = np.array_split(
        np.array(smooth_pursuit_timing), np.flatnonzero(np.diff(np.array(smooth_pursuit_timing)) > 1) + 1
    )

    fixation_sequences_merged = merge_close_sequences(
        fixation_sequences_candidates, gaze_direction, check_directionnality=False
    )
    smooth_pursuit_sequences_merged = merge_close_sequences(
        smooth_pursuit_sequences_cadidates, gaze_direction, check_directionnality=True
    )

    return fixation_sequences_merged, smooth_pursuit_sequences_merged, uncertain_sequences


def measure_smooth_pursuit_trajectory(time_vector, smooth_pursuit_sequences, gaze_angular_velocity_rad, dt):
    """
    The length of the smooth pursuit trajectory is computed as the sum of the angle between two frames in degrees.
    It can be seen as the integral of the angular velocity.
    """
    smooth_pursuit_trajectories = []
    for sequence in smooth_pursuit_sequences:
        trajectory_this_time = 0
        for idx in sequence:
            time_beginning = time_vector[idx]
            time_end = time_vector[idx + 1] if idx + 1 < len(time_vector) else time_vector[-1] + dt
            d_trajectory = np.abs(gaze_angular_velocity_rad[idx] * 180 / np.pi) * (time_end - time_beginning)
            trajectory_this_time += 0 if np.isnan(d_trajectory) else d_trajectory
        smooth_pursuit_trajectories += [trajectory_this_time]
    return smooth_pursuit_trajectories


def plot_bad_data_timing(time_vector, eye_direction, figname):
    """
    Plot the timing of the data to remove either because of blinks or because of invalid data
    """
    plt.figure()
    plt.plot(time_vector, eye_direction[0], label="eye_direction_x")
    plt.plot(time_vector, eye_direction[1], label="eye_direction_y")
    plt.plot(time_vector, eye_direction[2], label="eye_direction_z")
    label_flag = True
    for i in blink_sequences:
        if len(i) < 1:
            continue
        if label_flag:
            plt.axvspan(time_vector[i[0]], time_vector[i[-1] + 1], color="g", alpha=0.5, label="Blink Sequences")
            label_flag = False
        else:
            plt.axvspan(time_vector[i[0]], time_vector[i[-1] + 1], color="g", alpha=0.5)
    plt.legend()
    plt.savefig(f"figures/bad_data_{figname}.png")
    plt.show()
    return


def plot_gaze_classification(
    time_vector,
    gaze_direction,
    blink_sequences,
    saccade_sequences,
    fixation_sequences,
    smooth_pursuit_sequences,
    eyetracker_invalid_sequences,
    visual_scanning_sequences,
    duration_after_cue,
    gaze_angular_velocity_rad,
    eye_angular_velocity_rad,
    eye_angular_acceleration_rad,
    dt,
    figname,
    fixation_duration_threshold,
    smooth_pursuit_duration_threshold,
    velocity_threshold_saccades,
    acceleration_threshold_saccades,
    velocity_threshold_visual_scanning,
    helmet_rotation_unwrapped_deg,
):
    """
    Plot the final gaze classification
    """
    time_vector_step = np.hstack((time_vector, time_vector[-1] + dt))
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1, 1]})
    axs[0].plot(time_vector, gaze_direction[0, :], "-k", label="Gaze x (head + eye)")
    axs[0].plot(time_vector, gaze_direction[1, :], "--k", label="Gaze y (head + eye)")
    axs[0].plot(time_vector, gaze_direction[2, :], ":k", label="Gaze z (head + eye)")

    axs[1].plot(time_vector, np.abs(eye_angular_velocity_rad * 180 / np.pi), color="tab:red", label="Eye velocity norm")
    axs[1].plot(
        time_vector, np.abs(gaze_angular_velocity_rad * 180 / np.pi), color="tab:purple", label="Gaze velocity norm"
    )

    axs[1].plot(time_vector, np.abs(helmet_rotation_unwrapped_deg), color="b", label="Head velocity norm")
    axs[1].plot(
        np.array([time_vector[0], time_vector[-1]]),
        np.array([velocity_threshold_visual_scanning, velocity_threshold_visual_scanning]),
        ":",
        color="tab:purple",
        label=r"100 $^\circ/s$ gaze velocity",
    )
    axs[1].plot(
        time_vector,
        velocity_threshold_saccades,
        "--",
        color="tab:red",
        label="5 medians (sliding window)",
    )

    axs[2].plot(
        time_vector, np.abs(eye_angular_acceleration_rad * 180 / np.pi), color="tab:red", label="Eye acceleration norm"
    )
    axs[2].plot(
        np.array([time_vector[0], time_vector[-1]]),
        np.array([acceleration_threshold_saccades, acceleration_threshold_saccades]),
        ":",
        color="tab:red",
        label=r"4000 $^\circ/s^2$",
    )

    label_flag = True
    for i in blink_sequences:
        if i.shape == (1, 0) or i.shape == (0,) or len(i) < 1:
            continue
        if label_flag:
            axs[0].axvspan(
                time_vector[i[0]],
                time_vector_step[i[-1] + 1],
                edgecolor=None,
                color="tab:green",
                alpha=0.5,
                label="Blink sequences",
            )
            label_flag = False
        else:
            axs[0].axvspan(time_vector[i[0]], time_vector_step[i[-1] + 1], edgecolor=None, color="tab:green", alpha=0.5)
    label_flag = True
    for i in saccade_sequences:
        if i.shape == (1, 0) or i.shape == (0,) or len(i) < 1:
            continue
        if label_flag:
            axs[0].axvspan(
                time_vector[i[0]],
                time_vector_step[i[-1] + 1],
                edgecolor=None,
                color="tab:blue",
                alpha=0.5,
                label="Saccade sequences",
            )
            label_flag = False
        else:
            axs[0].axvspan(time_vector[i[0]], time_vector_step[i[-1] + 1], edgecolor=None, color="tab:blue", alpha=0.5)
    label_flag = True
    for i in fixation_sequences:
        if i.shape == (1, 0) or i.shape == (0,) or len(i) < 1:
            continue
        if time_vector_step[i[-1] + 1] - time_vector[i[0]] < fixation_duration_threshold:
            continue
        if label_flag:
            axs[0].axvspan(
                time_vector[i[0]],
                time_vector_step[i[-1] + 1],
                edgecolor=None,
                color="tab:purple",
                alpha=0.5,
                label="Fixation sequences",
            )
            label_flag = False
        else:
            axs[0].axvspan(
                time_vector[i[0]], time_vector_step[i[-1] + 1], edgecolor=None, color="tab:purple", alpha=0.5
            )
    label_flag = True
    for i in smooth_pursuit_sequences:
        if i.shape == (1, 0) or i.shape == (0,) or len(i) < 1:
            continue
        if time_vector_step[i[-1] + 1] - time_vector[i[0]] < smooth_pursuit_duration_threshold:
            continue
        if label_flag:
            axs[0].axvspan(
                time_vector[i[0]],
                time_vector_step[i[-1] + 1],
                edgecolor=None,
                color="tab:orange",
                alpha=0.5,
                label="Smooth pursuit sequences",
            )
            label_flag = False
        else:
            axs[0].axvspan(
                time_vector[i[0]], time_vector_step[i[-1] + 1], edgecolor=None, color="tab:orange", alpha=0.5
            )
    label_flag = True
    for i in eyetracker_invalid_sequences:
        if i.shape == (1, 0) or i.shape == (0,) or len(i) < 1:
            continue
        if label_flag:
            axs[0].axvspan(
                time_vector[i[0]],
                time_vector_step[i[-1] + 1],
                edgecolor=None,
                color="tab:red",
                alpha=0.5,
                label="Eye-tracker invalid sequences",
            )
            label_flag = False
        else:
            axs[0].axvspan(time_vector[i[0]], time_vector_step[i[-1] + 1], edgecolor=None, color="tab:red", alpha=0.5)
    label_flag = True
    for i in visual_scanning_sequences:
        if i.shape == (1, 0) or i.shape == (0,) or len(i) < 1:
            continue
        if label_flag:
            axs[0].axvspan(
                time_vector[i[0]],
                time_vector_step[i[-1] + 1],
                edgecolor=None,
                color="tab:pink",
                alpha=0.5,
                label="Visual scanning sequences",
            )
            label_flag = False
        else:
            axs[0].axvspan(time_vector[i[0]], time_vector_step[i[-1] + 1], edgecolor=None, color="tab:pink", alpha=0.5)

    axs[0].plot(
        np.array([time_vector[-1] - duration_after_cue, time_vector[-1]]),
        np.array([np.nanmax(gaze_direction) + 0.1, np.nanmax(gaze_direction) + 0.1]),
        "k",
    )
    axs[0].plot(
        np.array([time_vector[-1] - duration_after_cue, time_vector[-1] - duration_after_cue]),
        np.array([np.nanmax(gaze_direction) + 0.09, np.nanmax(gaze_direction) + 0.1]),
        "k",
    )
    axs[0].plot(
        np.array([time_vector[-1], time_vector[-1]]),
        np.array([np.nanmax(gaze_direction) + 0.09, np.nanmax(gaze_direction) + 0.1]),
        "k",
    )
    axs[0].text(
        time_vector[-1] - duration_after_cue / 2,
        np.nanmax(gaze_direction) + 0.11,
        f"last {duration_after_cue} sec",
    )

    axs[0].fill_between(np.array([0, 0]), np.array([0, 0]), color="w", label="Unclassified events")

    axs[0].set_xlim(0, time_vector[-1])
    axs[1].set_xlim(0, time_vector[-1])
    axs[2].set_xlim(0, time_vector[-1])

    axs[0].set_ylabel("Gaze orientation [without units]")
    axs[1].set_ylabel(r"Velocity [$^\circ/s$]")
    axs[2].set_ylabel(r"Acceleration [$^\circ/s^2$]")
    axs[2].set_xlabel("Time [s]")
    axs[0].legend(bbox_to_anchor=(1.02, 0.7))
    axs[1].legend(bbox_to_anchor=(1.02, 0.8))
    axs[2].legend(bbox_to_anchor=(1.02, 0.7))
    plt.subplots_adjust(bottom=0.07, top=0.95, left=0.1, right=0.7, hspace=0.15)
    plt.savefig(f"figures/gaze_classification_{figname}.png")
    # plt.show()
    plt.close()
    return


def fix_helmet_rotation(time_vector, helmet_rotation):
    """
    Thi function allows to unwrap the helmet rotation to avoid 360 jumps and get appropriate head rotation velocity
    """
    # Unwrap the helmet rotation to avoid 360 jumps
    helmet_rotation_unwrapped_deg = np.unwrap(helmet_rotation, period=360, axis=1)

    # Interpolate to avoid frames being repeated, which will mess up with the velocities thresholds
    i = 0
    while i < len(time_vector) - 1:
        j = i + 1
        if (
            np.abs(
                np.linalg.norm(helmet_rotation_unwrapped_deg[:, j])
                - np.linalg.norm(helmet_rotation_unwrapped_deg[:, i])
            )
            < 1e-10
        ):
            while (
                np.abs(
                    np.linalg.norm(helmet_rotation_unwrapped_deg[:, j])
                    - np.linalg.norm(helmet_rotation_unwrapped_deg[:, i])
                )
                < 1e-10
            ):
                if j + 1 < len(time_vector) - 1:
                    j += 1
                else:
                    break
            for i_component in range(3):
                helmet_rotation_unwrapped_deg[i_component, i:j] = np.linspace(
                    helmet_rotation_unwrapped_deg[i_component, i],
                    helmet_rotation_unwrapped_deg[i_component, j],
                    j - i + 1,
                )[:-1]
        i = j

    # Deal with the before last frame
    if (
        np.abs(
            np.linalg.norm(helmet_rotation_unwrapped_deg[:, -3]) - np.linalg.norm(helmet_rotation_unwrapped_deg[:, -2])
        )
        < 1e-10
    ):
        for i_component in range(3):
            helmet_rotation_unwrapped_deg[i_component, -2] = np.linspace(
                helmet_rotation_unwrapped_deg[i_component, -3],
                helmet_rotation_unwrapped_deg[i_component, -1],
                3,
            )[1:-1][0]

    # Head angular velocity
    head_3angular_velocity_deg = np.zeros((3, helmet_rotation.shape[1]))
    for i_component in range(3):
        head_3angular_velocity_deg[i_component, 0] = (
            helmet_rotation_unwrapped_deg[i_component, 1] - helmet_rotation_unwrapped_deg[i_component, 0]
        ) / (time_vector[1] - time_vector[0])
        head_3angular_velocity_deg[i_component, -1] = (
            helmet_rotation_unwrapped_deg[i_component, -1] - helmet_rotation_unwrapped_deg[i_component, -2]
        ) / (time_vector[-1] - time_vector[-2])
        head_3angular_velocity_deg[i_component, 1:-1] = (
            helmet_rotation_unwrapped_deg[i_component, 2:] - helmet_rotation_unwrapped_deg[i_component, :-2]
        ) / (time_vector[2:] - time_vector[:-2])

    head_angular_velocity_deg = np.linalg.norm(head_3angular_velocity_deg, axis=0)

    # Filter head angular velocity
    b, a = signal.butter(8, 0.2)
    head_angular_velocity_deg_filtered = signal.filtfilt(b, a, head_angular_velocity_deg, padlen=150)

    # plt.figure()
    # plt.plot(helmet_rotation[0, :])
    # plt.plot(helmet_rotation[1, :])
    # plt.plot(helmet_rotation[2, :])
    # plt.plot(head_angular_velocity_deg_filtered, '-r')
    # plt.plot(head_angular_velocity_deg, 'k-')
    # plt.savefig("figures/test.png")
    # plt.show()

    return head_angular_velocity_deg_filtered, helmet_rotation_unwrapped_deg


def compute_intermediary_metrics(
    time_vector,
    smooth_pursuit_sequences,
    fixation_sequences,
    blink_sequences,
    saccade_sequences,
    visual_scanning_sequences,
    gaze_angular_velocity_rad,
    dt,
    duration_after_cue,
    cut_file,
    fixation_duration_threshold,
    smooth_pursuit_duration_threshold,
    head_angular_velocity_deg_filtered,
):

    def split_sequences_before_and_after_quiet_eye(post_cue_timing_idx, sequences):
        """
        Get the sequences before and after the cue.
        Note that the event occurring during the cue is removed.
        """
        sequences_pre_cue = []
        sequences_post_cue = []
        if len(sequences) == 0 or sequences[0].shape == (0,) or sequences[0].shape == (1, 0):
            return sequences, sequences  # returning two empty sequences
        for i_sequence in sequences:
            if i_sequence[-1] < post_cue_timing_idx:
                sequences_pre_cue.append(i_sequence)
            elif i_sequence[0] > post_cue_timing_idx:
                sequences_post_cue.append(i_sequence)

        return sequences_pre_cue, sequences_post_cue

    def split_durations_before_and_after_quiet_eye(
        sequence_type, cut_file, sequences, post_cue_timing_idx, dt, time_vector, duration_threshold=0
    ):
        durations = []
        durations_pre_cue = []
        durations_post_cue = []
        for i in sequences:
            if len(i) > 0:
                duration = time_vector[i[-1]] - time_vector[i[0]] + dt
                if duration > duration_threshold:
                    durations.append(duration)
                    if i[-1] < post_cue_timing_idx:
                        durations_pre_cue.append(time_vector[i[-1]] - time_vector[i[0]] + dt)
                    elif post_cue_timing_idx in i:
                        # Remove this event but write it in a file so that we know what was removed
                        if cut_file is None:
                            print(
                                f"{sequence_type} : {np.round(time_vector[i[-1]] - time_vector[i[0]] + dt, decimals=5)} s ----",
                                end="",
                            )
                        else:
                            cut_file.write(f"{sequence_type} : {time_vector[i[-1]] - time_vector[i[0]] + dt} s \n")
                    elif i[0] > post_cue_timing_idx:
                        durations_post_cue.append(time_vector[i[-1]] - time_vector[i[0]] + dt)
        return durations, durations_pre_cue, durations_post_cue

    # Instant at which the "post cue switch" is happening
    post_cue_timing_idx = np.where(time_vector > time_vector[-1] - duration_after_cue)[0][0]
    smooth_pursuit_sequences_pre_cue, smooth_pursuit_sequences_post_cue = split_sequences_before_and_after_quiet_eye(
        post_cue_timing_idx, smooth_pursuit_sequences
    )
    fixation_sequences_pre_cue, fixation_sequences_post_cue = split_sequences_before_and_after_quiet_eye(
        post_cue_timing_idx, fixation_sequences
    )
    blink_sequences_pre_cue, blink_sequences_post_cue = split_sequences_before_and_after_quiet_eye(
        post_cue_timing_idx, blink_sequences
    )
    saccade_sequences_pre_cue, saccade_sequences_post_cue = split_sequences_before_and_after_quiet_eye(
        post_cue_timing_idx, saccade_sequences
    )
    visual_scanning_sequences_pre_cue, visual_scanning_sequences_post_cue = split_sequences_before_and_after_quiet_eye(
        post_cue_timing_idx, visual_scanning_sequences
    )

    # Intermediary metrics
    smooth_pursuit_trajectories = measure_smooth_pursuit_trajectory(
        time_vector, smooth_pursuit_sequences, gaze_angular_velocity_rad, dt
    )
    smooth_pursuit_trajectories_pre_cue = measure_smooth_pursuit_trajectory(
        time_vector, smooth_pursuit_sequences_pre_cue, gaze_angular_velocity_rad, dt
    )
    smooth_pursuit_trajectories_post_cue = measure_smooth_pursuit_trajectory(
        time_vector, smooth_pursuit_sequences_post_cue, gaze_angular_velocity_rad, dt
    )

    # Total time spent in fixations
    fixation_duration, fixation_duration_pre_cue, fixation_duration_post_cue = (
        split_durations_before_and_after_quiet_eye(
            "Fixation",
            cut_file,
            fixation_sequences,
            post_cue_timing_idx,
            dt,
            time_vector,
            duration_threshold=fixation_duration_threshold,
        )
    )
    total_fixation_duration = np.sum(np.array(fixation_duration))
    total_fixation_duration_pre_cue = np.sum(np.array(fixation_duration_pre_cue))
    total_fixation_duration_post_cue = np.sum(np.array(fixation_duration_post_cue))

    # Total time spent in smooth pursuit
    smooth_pursuit_duration, smooth_pursuit_duration_pre_cue, smooth_pursuit_duration_post_cue = (
        split_durations_before_and_after_quiet_eye(
            "Smooth pursuit",
            cut_file,
            smooth_pursuit_sequences,
            post_cue_timing_idx,
            dt,
            time_vector,
            duration_threshold=smooth_pursuit_duration_threshold,
        )
    )
    total_smooth_pursuit_duration = np.sum(np.array(smooth_pursuit_duration))
    total_smooth_pursuit_duration_pre_cue = np.sum(np.array(smooth_pursuit_duration_pre_cue))
    total_smooth_pursuit_duration_post_cue = np.sum(np.array(smooth_pursuit_duration_post_cue))

    # Total time spent in blinks
    blink_duration, blink_duration_pre_cue, blink_duration_post_cue = split_durations_before_and_after_quiet_eye(
        "Blink", cut_file, blink_sequences, post_cue_timing_idx, dt, time_vector
    )
    total_blink_duration = np.sum(np.array(blink_duration))
    total_blink_duration_pre_cue = np.sum(np.array(blink_duration_pre_cue))
    total_blink_duration_post_cue = np.sum(np.array(blink_duration_post_cue))

    # Total time spent in saccades
    saccade_duration, saccade_duration_pre_cue, saccade_duration_post_cue = split_durations_before_and_after_quiet_eye(
        "Saccade", cut_file, saccade_sequences, post_cue_timing_idx, dt, time_vector
    )
    total_saccade_duration = np.sum(np.array(saccade_duration))
    total_saccade_duration_pre_cue = np.sum(np.array(saccade_duration_pre_cue))
    total_saccade_duration_post_cue = np.sum(np.array(saccade_duration_post_cue))

    # Total time spent in visual scanning
    visual_scanning_duration, visual_scanning_duration_pre_cue, visual_scanning_duration_post_cue = (
        split_durations_before_and_after_quiet_eye(
            "Visual scanning", cut_file, visual_scanning_sequences, post_cue_timing_idx, dt, time_vector
        )
    )
    total_visual_scanning_duration = np.sum(np.array(visual_scanning_duration))
    total_visual_scanning_duration_pre_cue = np.sum(np.array(visual_scanning_duration_pre_cue))
    total_visual_scanning_duration_post_cue = np.sum(np.array(visual_scanning_duration_post_cue))

    # Head velocity
    mean_head_angular_velocity_deg = np.mean(head_angular_velocity_deg_filtered)
    mean_head_angular_velocity_deg_pre_cue = np.mean(head_angular_velocity_deg_filtered[:post_cue_timing_idx])
    mean_head_angular_velocity_deg_post_cue = np.mean(head_angular_velocity_deg_filtered[post_cue_timing_idx:])

    return (
        smooth_pursuit_sequences_pre_cue,
        smooth_pursuit_sequences_post_cue,
        fixation_sequences_pre_cue,
        fixation_sequences_post_cue,
        blink_sequences_pre_cue,
        blink_sequences_post_cue,
        saccade_sequences_pre_cue,
        saccade_sequences_post_cue,
        visual_scanning_sequences_pre_cue,
        visual_scanning_sequences_post_cue,
        fixation_duration,
        fixation_duration_pre_cue,
        fixation_duration_post_cue,
        smooth_pursuit_duration,
        smooth_pursuit_duration_pre_cue,
        smooth_pursuit_duration_post_cue,
        blink_duration,
        blink_duration_pre_cue,
        blink_duration_post_cue,
        saccade_duration,
        saccade_duration_pre_cue,
        saccade_duration_post_cue,
        visual_scanning_duration,
        visual_scanning_duration_pre_cue,
        visual_scanning_duration_post_cue,
        smooth_pursuit_trajectories,
        smooth_pursuit_trajectories_pre_cue,
        smooth_pursuit_trajectories_post_cue,
        total_fixation_duration,
        total_fixation_duration_pre_cue,
        total_fixation_duration_post_cue,
        total_smooth_pursuit_duration,
        total_smooth_pursuit_duration_pre_cue,
        total_smooth_pursuit_duration_post_cue,
        total_blink_duration,
        total_blink_duration_pre_cue,
        total_blink_duration_post_cue,
        total_saccade_duration,
        total_saccade_duration_pre_cue,
        total_saccade_duration_post_cue,
        total_visual_scanning_duration,
        total_visual_scanning_duration_pre_cue,
        total_visual_scanning_duration_post_cue,
        mean_head_angular_velocity_deg,
        mean_head_angular_velocity_deg_pre_cue,
        mean_head_angular_velocity_deg_post_cue,
        post_cue_timing_idx,
    )


def check_if_there_is_sequence_overlap(
    fixation_sequences,
    smooth_pursuit_sequences,
    visual_scanning_sequences,
    blink_sequences,
    saccade_sequences,
    eyetracker_invalid_sequences,
):
    """
    This function just check if there was problems in the classification algorithm
    """

    # Blinks and invalid data must not overlap with any other sequences
    for i_blink in blink_sequences + eyetracker_invalid_sequences:
        for i_sequence in fixation_sequences + smooth_pursuit_sequences + visual_scanning_sequences + saccade_sequences:
            if any(item in i_blink for item in i_sequence):
                raise ValueError("Problem: Blink or Invalid data sequences overlap with another sequence")

    # Fixations, smooth pursuit and visual scanning must not overlap with each other
    for i_fixation in fixation_sequences:
        for i_sequence in smooth_pursuit_sequences + visual_scanning_sequences:
            if any(item in i_fixation for item in i_sequence):
                raise ValueError("Problem: Fixation sequences overlap with Smooth pursuit or Visual scanning sequences")
    for i_smooth_pursuit in smooth_pursuit_sequences:
        for i_sequence in visual_scanning_sequences:
            if any(item in i_smooth_pursuit for item in i_sequence):
                raise ValueError("Problem: Smooth pursuit sequences overlap with Visual scanning sequences")


def main():

    if "figures" not in os.listdir():
        os.mkdir("figures")

    # ------------------------------------------------------------
    # Define the path to the data
    datapath = "../AllData/"
    black_screen_timing_file_path = "length_before_black_screen.xlsx"
    black_screen_timing_data = pd.read_excel(datapath + black_screen_timing_file_path)
    trial_names = list(black_screen_timing_data["Video Name"])
    fixation_duration_threshold = 0.1  # 100 ms
    smooth_pursuit_duration_threshold = 0.1  # 100 ms

    # ----------------------------------------------------------------
    # Create a file with the name of the data files that were excluded for poor quality
    bad_data_file = open("bad_data_files.txt", "w")
    bad_data_file.write(
        "The following files were excluded because more than 50% of the points were excluded by the eye-tracker : \n\n"
    )

    # ----------------------------------------------------------------
    # Create a file with the length of the event happening at the 2s cut
    cut_file = open("event_excluded_at_cut_off.txt", "w")
    cut_file.write("The following events were excluded because they happened at the moment of the 2s cut : \n\n")

    # ----------------------------------------------------------------
    output_metrics_dataframe = None
    # for path, folders, files in os.walk(datapath):
    #     for file in files:
    #         if file.endswith(".csv"):
    #
    #             if file != "20231030161241_eye_tracking_VideoListOne_TESTNA01_Experiment_Mode_2D_Fist3_014.csv":
    #                 continue

    current_path_file = Path(__file__).parent
    data_path = f"{current_path_file}/../examples/data/HTC_Vive_Pro/"
    length_before_black_screen = {
        # "TESTNA01_2D_Fist3": 7.180,  # s
        # "TESTNA01_360VR_Fist3": 7.180,
        "TESTNA05_2D_Spread7": 5.060,
        "TESTNA05_360VR_Spread7": 5.060,
        "TESTNA15_2D_Pen3": 4.230,
        "TESTNA15_360VR_Pen3": 4.230,
        "TESTVA03_2D_Spread9": 6.150,  # Bad data (no data)
        "TESTNA10_360VR_Fist3": 7.180,  # Bad data (more than 50% of the data is invalid)
    }
    for file_name in length_before_black_screen.keys():
        file = data_path + file_name + ".csv"

        figname = file_name
        print(f"Treating the data from file : {file}")
        data = pd.read_csv(file, sep=";")

        if len(data["time(100ns)"]) == 0:
            print("\n\n ****************************************************************************** \n")
            print(f"Data from file {file} is empty")
            print("\n ****************************************************************************** \n\n")
            bad_data_file.write(f"{file} \n")
            continue

        if np.sum(np.logical_or(data["eye_valid_L"] != 31, data["eye_valid_R"] != 31)) > len(data["eye_valid_L"]) / 2:
            print("\n\n ****************************************************************************** \n")
            print(f"More than 50% of the data from file {file} is declared invalid by the eye-tracker")
            print("\n ****************************************************************************** \n\n")
            bad_data_file.write(f"{file} \n")
            continue

        time_vector = np.array((data["time(100ns)"] - data["time(100ns)"][0]) / 10000000)
        length_trial = length_before_black_screen[file_name]
        # cut the data after the black screen
        black_screen_index = (
            np.where(time_vector > length_trial)[0][0] if length_trial < time_vector[-1] else len(time_vector)
        )
        time_vector = time_vector[:black_screen_index]
        data = data.iloc[:black_screen_index, :]

        # Remove the duplicated timestamps in the data
        bad_timestamps_index = list(np.where(np.abs(time_vector[1:] - time_vector[:-1]) < 1e-10)[0])
        good_timestamps_index = [i for i in range(len(time_vector)) if i not in bad_timestamps_index]
        time_vector = time_vector[good_timestamps_index]
        data = data.iloc[good_timestamps_index, :]

        eye_direction = np.array([data["gaze_direct_L.x"], data["gaze_direct_L.y"], data["gaze_direct_L.z"]])
        eye_norm = np.linalg.norm(eye_direction, axis=0)
        eye_direction = eye_direction / eye_norm
        helmet_rotation = np.array([data["helmet_rot_x"], data["helmet_rot_y"], data["helmet_rot_z"]])
        head_angular_velocity_deg_filtered, helmet_rotation_unwrapped_deg = fix_helmet_rotation(
            time_vector, helmet_rotation
        )

        eyetracker_invalid_data_index = np.array([])
        if np.sum(data["eye_valid_L"]) != 31 * len(data["eye_valid_L"]) or np.sum(data["eye_valid_R"]) != 31 * len(
            data["eye_valid_R"]
        ):
            if PLOT_BAD_DATA_FLAG:
                plt.figure()
                plt.plot(data["eye_valid_L"] / 31, label="eye_valid_L")
                plt.plot(data["eye_valid_R"] / 31, label="eye_valid_R")
                plt.plot(data["openness_L"], label="openness_L")
                plt.plot(data["openness_R"], label="openness_R")
                plt.legend()
                plt.show()
            eyetracker_invalid_data_index = np.where(
                np.logical_or(data["eye_valid_L"] != 31, data["eye_valid_R"] != 31)
            )[0]
        eyetracker_invalid_sequences = np.array_split(
            np.array(eyetracker_invalid_data_index),
            np.flatnonzero(np.diff(np.array(eyetracker_invalid_data_index)) > 1) + 1,
        )

        # Remove blinks
        blink_sequences = detect_blinks(data)

        if PLOT_BAD_DATA_FLAG:
            plot_bad_data_timing(time_vector, eye_direction, figname)

        # Remove blink sequences from the variable vectors
        for blink in blink_sequences:
            eye_direction[:, blink] = np.nan
        # Remove the data that the eye-tracker declares invalid
        if eyetracker_invalid_data_index.shape != (0,):
            eye_direction[:, eyetracker_invalid_data_index] = np.nan

        # Detect saccades
        gaze_direction = get_gaze_direction(helmet_rotation_unwrapped_deg, eye_direction)
        (
            saccade_sequences,
            eye_angular_velocity_rad,
            eye_angular_acceleration_rad,
            saccade_amplitudes,
            velocity_threshold_saccades,
            acceleration_threshold_saccades,
        ) = detect_saccades(time_vector, eye_direction, gaze_direction)

        # Detect visual scanning
        visual_scanning_sequences, gaze_angular_velocity_rad, velocity_threshold_visual_scanning = (
            detect_visual_scanning(time_vector, gaze_direction, saccade_sequences, head_angular_velocity_deg_filtered)
        )

        # Detect fixations
        intersaccadic_interval = np.zeros((len(time_vector),))
        all_index = np.arange(len(time_vector))
        for i in all_index:
            i_in_saccades = True if any(i in sequence for sequence in saccade_sequences) else False
            i_in_visual_scanning = True if any(i in sequence for sequence in visual_scanning_sequences) else False
            i_in_blinks = True if any(i in sequence for sequence in blink_sequences) else False
            i_in_eyetracker_invalid = True if i in eyetracker_invalid_data_index else False
            gaze_velocity_criteria = True if gaze_angular_velocity_rad[i] * np.pi / 180 > 100 else False
            if (
                i_in_saccades
                or i_in_visual_scanning
                or i_in_blinks
                or i_in_eyetracker_invalid
                or gaze_velocity_criteria
            ):
                continue
            else:
                intersaccadic_interval[i] = 1
        intersaccadic_timing = np.where(intersaccadic_interval == 1)[0]
        intersaccadic_sequences_temporary = np.array_split(
            intersaccadic_timing, np.flatnonzero(np.diff(intersaccadic_timing) > 1) + 1
        )
        intersaccadic_sequences = []
        for i in range(len(intersaccadic_sequences_temporary)):
            if len(intersaccadic_sequences_temporary[i]) > 2:
                intersaccadic_sequences += [intersaccadic_sequences_temporary[i]]

        intersaccadic_gouped_sequences, intersaccadic_coherent_sequences, intersaccadic_incoherent_sequences = (
            sliding_window(time_vector, intersaccadic_sequences, gaze_direction)
        )
        if PLOT_INTERSACCADIC_FLAG:
            plot_intersaccadic_interval_subinterval_cut(
                time_vector, intersaccadic_coherent_sequences, intersaccadic_incoherent_sequences, figname
            )
        fixation_sequences, smooth_pursuit_sequences, uncertain_sequences = detect_fixations_and_smooth_pursuit(
            time_vector, gaze_direction, intersaccadic_gouped_sequences, figname, PLOT_CRITERIA_FLAG
        )

        visual_scanning_sequences = apply_minimal_duration(visual_scanning_sequences, number_of_frames_min=5)
        check_if_there_is_sequence_overlap(
            fixation_sequences,
            smooth_pursuit_sequences,
            visual_scanning_sequences,
            blink_sequences,
            saccade_sequences,
            eyetracker_invalid_sequences,
        )

        if PLOT_CLASSIFICATION_FLAG:
            # The mean duration of a frame because we only have data at the frame and the duration of the events
            # should be computed as a step function, thus we have to add a step after the last index.
            dt = np.mean(time_vector[1:] - time_vector[:-1])
            plot_gaze_classification(
                time_vector,
                gaze_direction,
                blink_sequences,
                saccade_sequences,
                fixation_sequences,
                smooth_pursuit_sequences,
                eyetracker_invalid_sequences,
                visual_scanning_sequences,
                duration_after_cue,
                gaze_angular_velocity_rad,
                eye_angular_velocity_rad,
                eye_angular_acceleration_rad,
                dt,
                figname,
                fixation_duration_threshold,
                smooth_pursuit_duration_threshold,
                velocity_threshold_saccades,
                acceleration_threshold_saccades,
                velocity_threshold_visual_scanning,
                head_angular_velocity_deg_filtered,
            )

        (
            smooth_pursuit_sequences_pre_cue,
            smooth_pursuit_sequences_post_cue,
            fixation_sequences_pre_cue,
            fixation_sequences_post_cue,
            blink_sequences_pre_cue,
            blink_sequences_post_cue,
            saccade_sequences_pre_cue,
            saccade_sequences_post_cue,
            visual_scanning_sequences_pre_cue,
            visual_scanning_sequences_post_cue,
            fixation_duration,
            fixation_duration_pre_cue,
            fixation_duration_post_cue,
            smooth_pursuit_duration,
            smooth_pursuit_duration_pre_cue,
            smooth_pursuit_duration_post_cue,
            blink_duration,
            blink_duration_pre_cue,
            blink_duration_post_cue,
            saccade_duration,
            saccade_duration_pre_cue,
            saccade_duration_post_cue,
            visual_scanning_duration,
            visual_scanning_duration_pre_cue,
            visual_scanning_duration_post_cue,
            smooth_pursuit_trajectories,
            smooth_pursuit_trajectories_pre_cue,
            smooth_pursuit_trajectories_post_cue,
            total_fixation_duration,
            total_fixation_duration_pre_cue,
            total_fixation_duration_post_cue,
            total_smooth_pursuit_duration,
            total_smooth_pursuit_duration_pre_cue,
            total_smooth_pursuit_duration_post_cue,
            total_blink_duration,
            total_blink_duration_pre_cue,
            total_blink_duration_post_cue,
            total_saccade_duration,
            total_saccade_duration_pre_cue,
            total_saccade_duration_post_cue,
            total_visual_scanning_duration,
            total_visual_scanning_duration_pre_cue,
            total_visual_scanning_duration_post_cue,
            mean_head_angular_velocity_deg,
            mean_head_angular_velocity_deg_pre_cue,
            mean_head_angular_velocity_deg_post_cue,
            post_cue_timing_idx,
        ) = compute_intermediary_metrics(
            time_vector,
            smooth_pursuit_sequences,
            fixation_sequences,
            blink_sequences,
            saccade_sequences,
            visual_scanning_sequences,
            gaze_angular_velocity_rad,
            dt,
            duration_after_cue,
            cut_file,
            fixation_duration_threshold,
            smooth_pursuit_duration_threshold,
            head_angular_velocity_deg_filtered,
        )

        # Metrics
        nb_fixations = len(fixation_duration)
        nb_fixations_pre_cue = len(fixation_duration_pre_cue)
        nb_fixations_post_cue = len(fixation_duration_post_cue)

        mean_fixation_duration = np.nanmean(np.array(fixation_duration)) if len(fixation_duration) > 0 else None
        mean_fixation_duration_pre_cue = (
            np.nanmean(np.array(fixation_duration_pre_cue)) if len(fixation_duration_pre_cue) > 0 else None
        )
        mean_fixation_duration_post_cue = (
            np.nanmean(np.array(fixation_duration_post_cue)) if len(fixation_duration_post_cue) > 0 else None
        )

        search_rate = nb_fixations / mean_fixation_duration if mean_fixation_duration is not None else None
        search_rate_pre_cue = (
            nb_fixations_pre_cue / mean_fixation_duration_pre_cue
            if mean_fixation_duration_pre_cue is not None
            else None
        )
        search_rate_post_cue = (
            nb_fixations_post_cue / mean_fixation_duration_post_cue
            if mean_fixation_duration_post_cue is not None
            else None
        )

        nb_blinks = len(blink_sequences)
        nb_blinks_pre_cue = len(blink_sequences_pre_cue)
        nb_blinks_post_cue = len(blink_sequences_post_cue)

        nb_saccades = len(saccade_sequences)
        nb_saccades_pre_cue = len(saccade_sequences_pre_cue)
        nb_saccades_post_cue = len(saccade_sequences_post_cue)

        mean_saccade_duration = np.nanmean(np.array(saccade_duration)) if len(saccade_duration) > 0 else None
        mean_saccade_duration_pre_cue = (
            np.nanmean(np.array(saccade_duration_pre_cue)) if len(saccade_duration_pre_cue) > 0 else None
        )
        mean_saccade_duration_post_cue = (
            np.nanmean(np.array(saccade_duration_post_cue)) if len(saccade_duration_post_cue) > 0 else None
        )

        max_saccade_amplitude = np.nanmax(np.array(saccade_amplitudes)) if len(saccade_amplitudes) > 0 else None

        mean_saccade_amplitude = np.nanmean(np.array(saccade_amplitudes)) if len(saccade_amplitudes) > 0 else None
        saccade_amplitudes_pre_cue = []
        saccade_amplitudes_post_cue = []
        for i, i_sequence in enumerate(saccade_sequences):
            if i_sequence[0] < post_cue_timing_idx:
                saccade_amplitudes_pre_cue.append(saccade_amplitudes[i])
            # If the saccade is happening during the 2s transition, we skip it
            elif i_sequence[0] > post_cue_timing_idx:
                saccade_amplitudes_post_cue.append(saccade_amplitudes[i])
        mean_saccade_amplitude_pre_cue = (
            np.nanmean(np.array(saccade_amplitudes_pre_cue)) if len(saccade_amplitudes_pre_cue) > 0 else None
        )
        mean_saccade_amplitude_post_cue = (
            np.nanmean(np.array(saccade_amplitudes_post_cue)) if len(saccade_amplitudes_post_cue) > 0 else None
        )

        nb_smooth_pursuit = len(smooth_pursuit_sequences)
        nb_smooth_pursuit_pre_cue = len(smooth_pursuit_sequences_pre_cue)
        nb_smooth_pursuit_post_cue = len(smooth_pursuit_sequences_post_cue)

        mean_smooth_pursuit_duration = (
            np.nanmean(np.array(smooth_pursuit_duration)) if len(smooth_pursuit_duration) > 0 else None
        )
        mean_smooth_pursuit_duration_pre_cue = (
            np.nanmean(np.array(smooth_pursuit_duration_pre_cue)) if len(smooth_pursuit_duration_pre_cue) > 0 else None
        )
        mean_smooth_pursuit_duration_post_cue = (
            np.nanmean(np.array(smooth_pursuit_duration_post_cue))
            if len(smooth_pursuit_duration_post_cue) > 0
            else None
        )

        max_smooth_pursuit_trajectory = (
            np.nanmax(np.array(smooth_pursuit_trajectories)) if len(smooth_pursuit_trajectories) > 0 else None
        )

        mean_smooth_pursuit_trajectory = (
            np.nanmean(np.array(smooth_pursuit_trajectories)) if len(smooth_pursuit_trajectories) > 0 else None
        )
        mean_smooth_pursuit_trajectory_pre_cue = (
            np.nanmean(np.array(smooth_pursuit_trajectories_pre_cue))
            if len(smooth_pursuit_trajectories_pre_cue) > 0
            else None
        )
        mean_smooth_pursuit_trajectory_post_cue = (
            np.nanmean(np.array(smooth_pursuit_trajectories_post_cue))
            if len(smooth_pursuit_trajectories_post_cue) > 0
            else None
        )

        nb_visual_scanning = len(visual_scanning_sequences)
        nb_visual_scanning_pre_cue = len(visual_scanning_sequences_pre_cue)
        nb_visual_scanning_post_cue = len(visual_scanning_sequences_post_cue)

        mean_visual_scanning_duration = (
            np.nanmean(np.array(visual_scanning_duration)) if len(visual_scanning_duration) > 0 else None
        )
        mean_visual_scanning_duration_pre_cue = (
            np.nanmean(np.array(visual_scanning_duration_pre_cue))
            if len(visual_scanning_duration_pre_cue) > 0
            else None
        )
        mean_visual_scanning_duration_post_cue = (
            np.nanmean(np.array(visual_scanning_duration_post_cue))
            if len(visual_scanning_duration_post_cue) > 0
            else None
        )

        fixation_ratio = total_fixation_duration / time_vector[-1]
        fixation_ratio_pre_cue = total_fixation_duration_pre_cue / (time_vector[-1] - duration_after_cue)
        fixation_ratio_post_cue = total_fixation_duration_post_cue / duration_after_cue

        smooth_pursuit_ratio = total_smooth_pursuit_duration / time_vector[-1]
        smooth_pursuit_ratio_pre_cue = total_smooth_pursuit_duration_pre_cue / (time_vector[-1] - duration_after_cue)
        smooth_pursuit_ratio_post_cue = total_smooth_pursuit_duration_post_cue / duration_after_cue

        blinking_ratio = total_blink_duration / time_vector[-1]
        blinking_ratio_pre_cue = total_blink_duration_pre_cue / (time_vector[-1] - duration_after_cue)
        blinking_ratio_post_cue = total_blink_duration_post_cue / duration_after_cue

        saccade_ratio = total_saccade_duration / time_vector[-1]
        saccade_ratio_pre_cue = total_saccade_duration_pre_cue / (time_vector[-1] - duration_after_cue)
        saccade_ratio_post_cue = total_saccade_duration_post_cue / duration_after_cue

        visual_scanning_ratio = total_visual_scanning_duration / time_vector[-1]
        visual_scanning_ratio_pre_cue = total_visual_scanning_duration_pre_cue / (time_vector[-1] - duration_after_cue)
        visual_scanning_ratio_post_cue = total_visual_scanning_duration_post_cue / duration_after_cue

        not_classified_ratio = 1 - (
            fixation_ratio + smooth_pursuit_ratio + blinking_ratio + saccade_ratio + visual_scanning_ratio
        )
        if not_classified_ratio < -dt:
            raise ValueError("Problem: The sum of the ratios is greater than 1")

        invalid_ratio = np.sum(np.logical_or(data["eye_valid_L"] != 31, data["eye_valid_R"] != 31)) / len(
            data["eye_valid_L"]
        )

        output = pd.DataFrame(
            {
                "File name": [file_name],
                "Figure name": [file_name],
                # "Participant ID": [file.split("_")[4]],
                # "Mode": [file.split("_")[7]],
                # "Trial name": [file.split("_")[8]],
                # "Trial number": [file.split("_")[9][:-4]],
                "Number of fixations full trial": [nb_fixations],
                "Number of fixations pre cue": [nb_fixations_pre_cue],
                "Number of fixations post cue": [nb_fixations_post_cue],
                "Mean fixation duration full trial [s]": [mean_fixation_duration],
                "Mean fixation duration pre cue [s]": [mean_fixation_duration_pre_cue],
                "Mean fixation duration post cue [s]": [mean_fixation_duration_post_cue],
                "Search rate full trial": [search_rate],
                "Search rate pre cue": [search_rate_pre_cue],
                "Search rate post cue": [search_rate_post_cue],
                "Number of blinks full trial": [nb_blinks],
                "Number of blinks pre cue": [nb_blinks_pre_cue],
                "Number of blinks post cue": [nb_blinks_post_cue],
                "Number of saccades full trial": [nb_saccades],
                "Number of saccades pre cue": [nb_saccades_pre_cue],
                "Number of saccades post cue": [nb_saccades_post_cue],
                "Mean saccade duration full trial [s]": [mean_saccade_duration],
                "Mean saccade duration pre cue [s]": [mean_saccade_duration_pre_cue],
                "Mean saccade duration post cue [s]": [mean_saccade_duration_post_cue],
                "Max saccade amplitude full trial [deg]": [max_saccade_amplitude],
                "Mean saccade amplitude full trial [deg]": [mean_saccade_amplitude],
                "Mean saccade amplitude pre cue [deg]": [mean_saccade_amplitude_pre_cue],
                "Mean saccade amplitude post cue [deg]": [mean_saccade_amplitude_post_cue],
                "Number of smooth pursuit full trial": [nb_smooth_pursuit],
                "Number of smooth pursuit pre cue": [nb_smooth_pursuit_pre_cue],
                "Number of smooth pursuit post cue": [nb_smooth_pursuit_post_cue],
                "Mean smooth pursuit duration full trial [s]": [mean_smooth_pursuit_duration],
                "Mean smooth pursuit duration pre cue [s]": [mean_smooth_pursuit_duration_pre_cue],
                "Mean smooth pursuit duration post cue [s]": [mean_smooth_pursuit_duration_post_cue],
                "Max smooth pursuit trajectory full trial [deg]": [max_smooth_pursuit_trajectory],
                "Mean smooth pursuit trajectory full trial [deg]": [mean_smooth_pursuit_trajectory],
                "Mean smooth pursuit trajectory pre cue [deg]": [mean_smooth_pursuit_trajectory_pre_cue],
                "Mean smooth pursuit trajectory post cue [deg]": [mean_smooth_pursuit_trajectory_post_cue],
                "Number of visual scanning full trial": [nb_visual_scanning],
                "Number of visual scanning pre cue": [nb_visual_scanning_pre_cue],
                "Number of visual scanning post cue": [nb_visual_scanning_post_cue],
                "Mean visual scanning duration full trial [s]": [mean_visual_scanning_duration],
                "Mean visual scanning duration pre cue [s]": [mean_visual_scanning_duration_pre_cue],
                "Mean visual scanning duration post cue [s]": [mean_visual_scanning_duration_post_cue],
                "Fixation ratio full trial": [fixation_ratio],
                "Fixation ratio pre cue": [fixation_ratio_pre_cue],
                "Fixation ratio post cue": [fixation_ratio_post_cue],
                "Smooth pursuit ratio full trial": [smooth_pursuit_ratio],
                "Smooth pursuit ratio pre cue": [smooth_pursuit_ratio_pre_cue],
                "Smooth pursuit ratio post cue": [smooth_pursuit_ratio_post_cue],
                "Blinking ratio full trial": [blinking_ratio],
                "Blinking ratio pre cue": [blinking_ratio_pre_cue],
                "Blinking ratio post cue": [blinking_ratio_post_cue],
                "Saccade ratio full trial": [saccade_ratio],
                "Saccade ratio pre cue": [saccade_ratio_pre_cue],
                "Saccade ratio post cue": [saccade_ratio_post_cue],
                "Visual scanning ratio full trial": [visual_scanning_ratio],
                "Visual scanning ratio pre cue": [visual_scanning_ratio_pre_cue],
                "Visual scanning ratio post cue": [visual_scanning_ratio_post_cue],
                "Not classified ratio full trial": [not_classified_ratio],
                "Invalid ratio full trial": [invalid_ratio],
                "Mean head angular velocity full trial": [mean_head_angular_velocity_deg],
                "Mean head angular velocity pre cue": [mean_head_angular_velocity_deg_pre_cue],
                "Mean head angular velocity post cue": [mean_head_angular_velocity_deg_post_cue],
                "Length of the full trial [s]": [time_vector[-1]],
            }
        )

        # Generate the data for tests
        with open(data_path + "/../../results/HTC_Vive_Pro/" + file_name + ".pkl", "wb") as result_file:
            pickle.dump(output, result_file)

        output_metrics_dataframe = (
            output
            if output_metrics_dataframe is None
            else pd.concat([output_metrics_dataframe, output], ignore_index=True)
        )

    with open("output_metrics.pkl", "wb") as f:
        pickle.dump(output_metrics_dataframe, f)

    with open("output_log.txt", "w") as f:
        f.write(f"Variable: Mean [Min ; Max] \n\n")
        for key in output_metrics_dataframe.keys():
            if key in ["File name", "Figure name", "Participant ID", "Mode", "Trial name", "Trial number"]:
                continue
            f.write(
                f"{key}: {output_metrics_dataframe[key].mean()} [{output_metrics_dataframe[key].min()} ; {output_metrics_dataframe[key].max()}] \n"
            )

    # Write a csv file that can be sed for statistical analysis later
    output_metrics_dataframe.to_csv("output_metrics.csv", index=False)

    bad_data_file.close()


if __name__ == "__main__":
    main()
