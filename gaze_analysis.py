"""
This code aims to identify visual behavior sequences, namely fixation, saccades, blink, and smooth pursuite.
We consider that when the head rotates, the image in the VR helmet (eye-tracker) rotates by the same amount, making it
as if the head was rotating around the subjects eyes instead of the neck joint center.
"""

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import biorbd
import pingouin as pg

"""
-> 'time(100ns)' = time stamps of the recorded frames
'time_stamp(ms)' = time stamps of the recorded frames in milliseconds
'time(UnityVideoPlayer)' = time stamps of the recorded frames in UnityVideoPlayer
'frame' = frame number
'eye_valid_L' = if the data is valid for the left eye
'eye_valid_R' = if the data is valid for the right eye
-> 'openness_L' = openness of the left eye [0, 1]
-> 'openness_R' = openness of the right eye [0, 1]
'pupil_diameter_L(mm)' = pupil diameter of the left eye in mm
'pupil_diameter_R(mm)' = pupil diameter of the right eye in mm
'pos_sensor_L.x' = gaze position in the lentil reference frame for the left eye [0, 1]
'pos_sensor_L.y' = gaze position in the lentil reference frame for the left eye [0, 1]
'pos_sensor_R.x' = gaze position in the lentil reference frame for the right eye [0, 1]
'pos_sensor_R.y' = gaze position in the lentil reference frame for the right eye [0, 1]
'gaze_origin_L.x(mm)' = mean position of the eyes in the helmet reference frame for the left eye in mm
'gaze_origin_L.y(mm)' = mean position of the eyes in the helmet reference frame for the left eye in mm
'gaze_origin_L.z(mm)' = mean position of the eyes in the helmet reference frame for the left eye in mm
'gaze_origin_R.x(mm)' = mean position of the eyes in the helmet reference frame for the right eye in mm
'gaze_origin_R.y(mm)' = mean position of the eyes in the helmet reference frame for the right eye in mm
'gaze_origin_R.z(mm)' = mean position of the eyes in the helmet reference frame for the right eye in mm
'gaze_direct_L.x' = gaze direction vector in the helmet reference frame for the left eye
'gaze_direct_L.y' = gaze direction vector in the helmet reference frame for the left eye
'gaze_direct_L.z' = gaze direction vector in the helmet reference frame for the left eye
'gaze_direct_R.x' = gaze direction vector in the helmet reference frame for the right eye
'gaze_direct_R.y' = gaze direction vector in the helmet reference frame for the right eye
'gaze_direct_R.z' = gaze direction vector in the helmet reference frame for the right eye
'gaze_sensitive' = ?
'frown_L' = ?
'frown_R' = ?
'squeeze_L' = ? 
'squeeze_R' = ?
'wide_L' = ? 
'wide_R' = ?
'distance_valid_C' = if the gaze focus point distance is valid
'distance_C(mm)' = distance of the gaze focus point in mm
'track_imp_cnt' = ?
'helmet_pos_x' = position of the helmet in which reference frame ?
'helmet_pos_y' = position of the helmet in which reference frame ?
'helmet_pos_z' = position of the helmet in which reference frame ?
'helmet_rot_x' = rotation of the helmet in degrees (downward rotation is positive)
'helmet_rot_y' = rotation of the helmet in degrees (leftward rotation is positive)
'helmet_rot_z' = rotation of the helmet in degrees (right tilt rotation is positive)
"""

# Define variables od interest
"""
time_vector in seconds
gaze_origin in meters
eye_direction is a unit vector
gaze_distance in meters
"""

# Plot flags ---------------------------------------
PLOT_BAD_DATA_FLAG = False
PLOT_SACCADES_FLAG = False
PLOT_CLASSIFICATION_FLAG = True  # Only one that should be True not when debugging
PLOT_INTERSACCADIC_FLAG = False
PLOT_CRITERIA_FLAG = False
SKIP_LONG_TRIALS = False
PLOT_ROTATION_VELOCITIES_FLAG = False
quiet_eye_duration_threshold = 2  # Seconds
# ------------------------------------------------------------


def detect_invalid_data(time_vector, eye_direction):
    """
    Identify as invalid data sequences where the eye-tracker did not detect any eye movements.
    """

    # Find where the data does not change
    zero_diffs_x = np.where(np.abs(eye_direction[0, 1:] - eye_direction[0, :-1]) < 1e-8)[0]
    zero_diffs_y = np.where(np.abs(eye_direction[1, 1:] - eye_direction[1, :-1]) < 1e-8)[0]
    zero_diffs_z = np.where(np.abs(eye_direction[2, 1:] - eye_direction[2, :-1]) < 1e-8)[0]

    # Find the common indices
    zero_diffs = np.intersect1d(np.intersect1d(zero_diffs_x, zero_diffs_y), zero_diffs_z)

    # Add 1 to zero_diffs to get the actual positions in the original array
    zero_diffs += 1

    # Group the indices into sequences
    invalid_sequences = np.array_split(zero_diffs, np.flatnonzero(np.diff(zero_diffs) > 1) + 1)

    return invalid_sequences


def detect_blinks(time_vector, data):
    """
    Blinks are detected when both eye openness drops bellow 0.5 il line with
    https://ieeexplore.ieee.org/abstract/document/9483841
    """
    blink_threshold = 0.5
    blink_timing_both = np.where((data["openness_R"] < blink_threshold) & (data["openness_L"] < blink_threshold))[0]

    # Group the indices into sequences
    blink_sequences = np.array_split(blink_timing_both, np.flatnonzero(np.diff(blink_timing_both) > 1) + 1)

    return blink_sequences


def detect_saccades(time_vector, eye_direction):
    """
    Detecting sequences where the eye angular velocity is larger than 100 degrees per second (and acceleration).
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

    velocity_threshold = 5 * np.nanmedian(np.abs(eye_angular_velocity_rad) * 180 / np.pi)
    acceleration_threshold = 4000  # deg/s²

    if PLOT_SACCADES_FLAG:
        plt.figure()
        plt.plot(time_vector, np.abs(eye_angular_velocity_rad * 180 / np.pi), label="Angular Velocity")
        plt.plot(
            np.array([time_vector[0], time_vector[-1]]), np.array([100, 100]), "k--", label=r"Threshold 100$\^circ/s$"
        )
        plt.plot(
            np.array([time_vector[0], time_vector[-1]]),
            np.array([velocity_threshold, velocity_threshold]),
            "b--",
            label=r"Threshold 5 medians",
        )
        plt.legend()
        plt.savefig("figures/saccade_detection_test.png")
        plt.show()

    # thomasromeas : Please confirm the method to use, so that I can delete the other ones
    # # Velocity only classification
    # saccade_timing = np.where(np.abs(gaze_angular_velocity_rad * 180 / np.pi) > velocity_threshold)[0]
    # saccade_sequences = np.array_split(saccade_timing, np.flatnonzero(np.diff(saccade_timing) > 1) + 1)

    # # Velocity + acceleration classification
    # saccade_timing = np.where(np.abs(gaze_angular_velocity_rad * 180 / np.pi) > velocity_threshold)[0]
    # saccade_sequences_tempo = np.array_split(saccade_timing, np.flatnonzero(np.diff(saccade_timing) > 1) + 1)
    # saccade_sequences = []
    # for i in saccade_sequences_tempo:
    #     if any(np.abs(eye_angular_acceleration_rad[i[0]-1: i[-1]+1] * 180 / np.pi) > acceleration_threshold):
    #         saccade_sequences += [i]

    # Velocity + 2 frames acceleration classification
    saccade_timing = np.where(np.abs(eye_angular_velocity_rad * 180 / np.pi) > velocity_threshold)[0]
    saccade_sequences_tempo = np.array_split(saccade_timing, np.flatnonzero(np.diff(saccade_timing) > 1) + 1)
    saccade_sequences = []

    if saccade_sequences_tempo[0].shape != (0,):
        for i in saccade_sequences_tempo:
            acceleration_above_threshold = np.where(
                np.abs(eye_angular_acceleration_rad[i[0] - 1 : i[-1] + 1] * 180 / np.pi) > acceleration_threshold
            )[0]
            if len(acceleration_above_threshold) > 1:
                saccade_sequences += [i]

    # # Velocity 100 + 3 frames acceleration classification
    # saccade_timing = np.where(np.abs(gaze_angular_velocity_rad * 180 / np.pi) > 100)[0]
    # saccade_sequences_tempo = np.array_split(saccade_timing, np.flatnonzero(np.diff(saccade_timing) > 1) + 1)
    # saccade_sequences = []
    # for i in saccade_sequences_tempo:
    #     acceleration_above_threshold = np.where(np.abs(eye_angular_acceleration_rad[i[0]-1: i[-1]+1] * 180 / np.pi) > acceleration_threshold)[0]
    #     if len(acceleration_above_threshold) > 2:
    #         saccade_sequences += [i]

    # Saccade amplitude
    # Defined as the angle between the beginning and end of the saccade,
    # note that there is no check made to detect if there is a larger amplitude reached during the saccade.
    saccade_amplitudes = []
    for sequence in saccade_sequences:
        vector_before = eye_direction[:, sequence[0]]
        vector_after = eye_direction[:, sequence[-1]]
        angle = np.arccos(
            np.dot(vector_before, vector_after) / np.linalg.norm(vector_before) / np.linalg.norm(vector_after)
        )
        saccade_amplitudes += [angle * 180 / np.pi]

    return saccade_sequences, eye_angular_velocity_rad, eye_angular_acceleration_rad, saccade_amplitudes


def get_gaze_direction(helmet_rotation_unwrapped_deg, eye_direction):
    helmet_rotation_in_rad = helmet_rotation_unwrapped_deg * np.pi / 180

    gaze_direction = np.zeros(eye_direction.shape)
    for i_frame in range(helmet_rotation_in_rad.shape[1]):
        rotation_matrix = biorbd.Rotation.fromEulerAngles(helmet_rotation_in_rad[:, i_frame], "xyz").to_array()
        gaze_direction[:, i_frame] = rotation_matrix @ eye_direction[:, i_frame]

    return gaze_direction


def detect_visual_scanning(time_vector, gaze_direction, saccade_sequences):
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

    # velocity_threshold = 100  # deg/s
    velocity_threshold = np.nanmedian(gaze_angular_velocity_rad) * 3

    saccade_sequences_timing = (
        np.hstack(saccade_sequences) if len(saccade_sequences) > 1 else np.array(saccade_sequences)
    )
    visual_scanning_candidates = np.where(np.abs(velocity_threshold * 180 / np.pi) > 100)[0]
    visual_scanning_timing = np.array([i for i in visual_scanning_candidates if i not in saccade_sequences_timing])

    # Group the indices into sequences
    visual_scanning_sequences = np.array_split(
        visual_scanning_timing, np.flatnonzero(np.diff(visual_scanning_timing) > 1) + 1
    )

    return visual_scanning_sequences, gaze_angular_velocity_rad


def sliding_window(time_vector, intersaccadic_sequences, gaze_direction):
    """
    Parameters t_wind (22000 micros), t_overlap (6000 micros) and eta_p (0.1) taken from the original paper
    https://doi.org/10.1016/j.bspc.2014.12.008
    """
    # @thomasromeas : These parameter values are still to be adjusted
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

    # plt.figure()
    # plt.plot(time_vector, mean_p_values)
    # plt.plot(np.array([0, 10]), np.array([eta_p, eta_p]), '--k')
    # plt.xlim(0, 10)
    # plt.show()

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


def discriminate_fixations_and_smooth_pursuite(gaze_direction):

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


def detect_fixations_and_smooth_pursuite(
    time_vector, gaze_direction, intersaccadic_gouped_sequences, fig_name, PLOT_CRITERIA_FLAG
):
    """
    This gaze behavior classification is based on the algorithm described in Larsson et al. (2015).
    https://doi.org/10.1016/j.bspc.2014.12.008
    """
    # Parameters to define ---------------------------------------
    # @thomasromeas : These parameter values are still to be adjusted
    eta_D = 0.45  #  is the threshold for dispersion (without units)
    eta_CD = 0.5  # is the threshold for consistency of direction (without units)
    eta_PD = 0.5  # is the threshold for position displacement (without units)
    eta_maxFix = 1.9 * np.pi / 180  # is the threshold for spacial range (in degrees)
    phi = 45 * np.pi / 180  # is the threshold for similar angular range (in degrees)
    eta_minSmp = 1.7 * np.pi / 180  # is the threshold for merged segments spacial range (in degrees)

    if PLOT_CRITERIA_FLAG:
        fig, axs = plt.subplots(4, 1, figsize=(15, 10))

    # Classify the obvious timings
    fixation_timing = []
    smooth_pursuite_timing = []
    uncertain_timing = []
    if len(intersaccadic_gouped_sequences) == 1 and intersaccadic_gouped_sequences[0].shape == (0,):
        raise RuntimeError("No intersaccadic interval! There should be at least one even if there is no saccades.")
    else:
        for i_sequence, sequence in enumerate(intersaccadic_gouped_sequences):
            parameter_D, parameter_CD, parameter_PD, parameter_R = discriminate_fixations_and_smooth_pursuite(
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
                smooth_pursuite_timing += list(sequence)
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
        parameter_D, parameter_CD, parameter_PD, parameter_R = discriminate_fixations_and_smooth_pursuite(
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
                parameter_D, parameter_CD, parameter_PD, parameter_R = discriminate_fixations_and_smooth_pursuite(
                    gaze_direction[:, range(before_idx, after_idx + 1)]
                )
                if parameter_R > eta_minSmp:
                    smooth_pursuite_timing += list(range(before_idx, after_idx))
                else:
                    fixation_timing += list(range(before_idx, after_idx))

                for i_uncertain_sequences, uncertain in enumerate(uncertain_sequences_tempo):
                    if any(item in uncertain for item in list(range(before_idx, after_idx))):
                        uncertain_sequences_to_remove += [i_uncertain_sequences]
        else:
            # Fixation like segment
            if criteria_4:
                smooth_pursuite_timing += list(sequence)
            else:
                fixation_timing += list(sequence)
            uncertain_sequences_to_remove += [i_sequence]

    uncertain_sequences = []
    for i_sequence in range(len(uncertain_sequences_tempo)):
        if i_sequence not in uncertain_sequences_to_remove:
            uncertain_sequences += [uncertain_sequences_tempo[i_sequence]]

    smooth_pursuite_timing = np.sort(smooth_pursuite_timing)
    fixation_timing = np.sort(fixation_timing)
    fixation_sequences = np.array_split(
        np.array(fixation_timing), np.flatnonzero(np.diff(np.array(fixation_timing)) > 1) + 1
    )
    smooth_pursuite_sequences = np.array_split(
        np.array(smooth_pursuite_timing), np.flatnonzero(np.diff(np.array(smooth_pursuite_timing)) > 1) + 1
    )

    return fixation_sequences, smooth_pursuite_sequences, uncertain_sequences


def measure_smooth_pursuite_trajectory(time_vector, smooth_pursuite_sequences, gaze_angular_velocity_rad, dt):
    """
    The length of the smooth pursuite trajectory is computed as the sum of the angle between two frames in degrees.
    It can be seen as the integral of the angular velocity.
    """
    smooth_pursuite_trajectories = []
    for sequence in smooth_pursuite_sequences:
        trajectory_this_time = 0
        for idx in sequence:
            time_beginning = time_vector[idx]
            time_end = time_vector[idx + 1] if idx + 1 < len(time_vector) else time_vector[-1] + dt
            d_trajectory = np.abs(gaze_angular_velocity_rad[idx] * 180 / np.pi) * (time_end - time_beginning)
            trajectory_this_time += 0 if np.isnan(d_trajectory) else d_trajectory
        smooth_pursuite_trajectories += [trajectory_this_time]
    return smooth_pursuite_trajectories


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
    smooth_pursuite_sequences,
    eyetracker_invalid_sequences,
    visual_scanning_sequences,
    quiet_eye_duration_threshold,
    gaze_angular_velocity_rad,
    eye_angular_acceleration_rad,
    dt,
    figname,
):
    """
    Plot the final gaze classification
    """
    time_vector_step = np.hstack((time_vector, time_vector[-1] + dt))
    fig, axs = plt.subplots(3, 1, figsize=(15, 20), gridspec_kw={"height_ratios": [3, 1, 1]})
    axs[0].plot(time_vector, gaze_direction[0, :], "-k", label="Gaze x (head + eye)")
    axs[0].plot(time_vector, gaze_direction[1, :], "--k", label="Gaze y (head + eye)")
    axs[0].plot(time_vector, gaze_direction[2, :], ":k", label="Gaze z (head + eye)")

    velocity_threshold = 5 * np.nanmedian(gaze_angular_velocity_rad * 180 / np.pi)
    axs[1].plot(time_vector, np.abs(gaze_angular_velocity_rad * 180 / np.pi), "r", label="Gaze velocity norm")
    axs[1].plot(
        np.array([time_vector[0], time_vector[-1]]),
        np.array([velocity_threshold, velocity_threshold]),
        "--r",
        label="5 medians (not exactly)",
    )

    velocity_threshold_3 = 3 * np.nanmedian(gaze_angular_velocity_rad * 180 / np.pi)
    axs[1].plot(np.array([time_vector[0], time_vector[-1]]), np.array([velocity_threshold_3, velocity_threshold_3]), ":r", label="3 medians gaze velocity")

    acceleration_threshold = 4000  # deg/s²
    axs[2].plot(time_vector, np.abs(eye_angular_acceleration_rad * 180 / np.pi), "g", label="Eye acceleration norm")
    axs[2].plot(
        np.array([time_vector[0], time_vector[-1]]),
        np.array([acceleration_threshold, acceleration_threshold]),
        ":g",
        label="4000 deg/s²",
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
                label="Blink Sequences",
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
                label="Saccade Sequences",
            )
            label_flag = False
        else:
            axs[0].axvspan(time_vector[i[0]], time_vector_step[i[-1] + 1], edgecolor=None, color="tab:blue", alpha=0.5)
    label_flag = True
    for i in fixation_sequences:
        if i.shape == (1, 0) or i.shape == (0,) or len(i) < 1:
            continue
        if label_flag:
            axs[0].axvspan(
                time_vector[i[0]],
                time_vector_step[i[-1] + 1],
                edgecolor=None,
                color="tab:purple",
                alpha=0.5,
                label="Fixation Sequences",
            )
            label_flag = False
        else:
            axs[0].axvspan(
                time_vector[i[0]], time_vector_step[i[-1] + 1], edgecolor=None, color="tab:purple", alpha=0.5
            )
    label_flag = True
    for i in smooth_pursuite_sequences:
        if i.shape == (1, 0) or i.shape == (0,) or len(i) < 1:
            continue
        if label_flag:
            axs[0].axvspan(
                time_vector[i[0]],
                time_vector_step[i[-1] + 1],
                edgecolor=None,
                color="tab:orange",
                alpha=0.5,
                label="Smooth Pursuite Sequences",
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
                label="Eye-tracker invalid Sequences",
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
                label="Visual Scanning Sequences",
            )
            label_flag = False
        else:
            axs[0].axvspan(time_vector[i[0]], time_vector_step[i[-1] + 1], edgecolor=None, color="tab:pink", alpha=0.5)

    axs[0].plot(
        np.array([time_vector[-1] - quiet_eye_duration_threshold, time_vector[-1]]),
        np.array([np.nanmax(gaze_direction) + 0.1, np.nanmax(gaze_direction) + 0.1]),
        "k",
    )
    axs[0].plot(
        np.array([time_vector[-1] - quiet_eye_duration_threshold, time_vector[-1] - quiet_eye_duration_threshold]),
        np.array([np.nanmax(gaze_direction) + 0.09, np.nanmax(gaze_direction) + 0.1]),
        "k",
    )
    axs[0].plot(
        np.array([time_vector[-1], time_vector[-1]]),
        np.array([np.nanmax(gaze_direction) + 0.09, np.nanmax(gaze_direction) + 0.1]),
        "k",
    )
    axs[0].text(
        time_vector[-1] - quiet_eye_duration_threshold / 2,
        np.nanmax(gaze_direction) + 0.11,
        f"last {quiet_eye_duration_threshold} sec",
    )

    axs[0].legend(bbox_to_anchor=(1.02, 0.7))
    axs[1].legend(bbox_to_anchor=(1.02, 0.7))
    axs[2].legend(bbox_to_anchor=(1.27, 0.7))
    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8)
    plt.savefig(f"figures/gaze_classification_{figname}.png")
    # plt.show()
    plt.close()
    return


def plot_head_eye_rotation(
    eye_direction,
    helmet_rotation_unwrapped_deg,
    head_angular_velocity_deg,
    gaze_angular_velocity_rad,
    time_vector,
    figname,
):

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

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(time_vector, eye_angular_velocity_rad * 180 / np.pi, "r-", label="Eye velocity [deg/s]")
    axs[0].plot(time_vector, head_angular_velocity_deg, "g-", label="Head velocity [deg/s]")
    axs[0].plot(time_vector, gaze_angular_velocity_rad * 180 / np.pi, "b-", label="Gaze velocity [deg/s]")
    axs[0].legend()
    axs[1].plot(time_vector, helmet_rotation_unwrapped_deg[0], "k-", label="Head rotation x [deg]")
    axs[1].plot(time_vector, helmet_rotation_unwrapped_deg[1], "k--", label="Head rotation y [deg]")
    axs[1].plot(time_vector, helmet_rotation_unwrapped_deg[2], "k:", label="Head rotation z [deg]")
    axs[1].legend()
    plt.savefig(f"figures/head_eye_rotation_{figname}.png")
    # plt.show()

    return


def fix_helmet_rotation(time_vector, helmet_rotation):

    # Unwrap the helmet rotation to avoid 360 jumps
    helmet_rotation_unwrapped_deg = np.unwrap(helmet_rotation, period=360, axis=1)

    # Interpolate to avoid frames being repeated, which will mess up with the velocities thresholds
    i = 0
    while i < len(time_vector) - 2:
        j = i + 1
        if helmet_rotation_unwrapped_deg[0, j] == helmet_rotation_unwrapped_deg[0, i]:
            while helmet_rotation_unwrapped_deg[0, j] == helmet_rotation_unwrapped_deg[0, i]:
                if j + 1 < len(time_vector) - 2:
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

    # Head angular velecity
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

    # plt.figure()
    # plt.plot(helmet_rotation[0, :])
    # plt.plot(helmet_rotation[1, :])
    # plt.plot(helmet_rotation[2, :])
    # plt.plot(helmet_rotation_unwrapped_deg[0, :])
    # plt.plot(helmet_rotation_unwrapped_deg[1, :])
    # plt.plot(helmet_rotation_unwrapped_deg[2, :])
    # plt.plot(head_angular_velocity_deg, 'k-')
    # plt.savefig("figures/test.png")
    # plt.show()

    return helmet_rotation_unwrapped_deg, head_angular_velocity_deg


def compute_intermediary_metrics(
    time_vector,
    smooth_pursuite_sequences,
    fixation_sequences,
    blink_sequences,
    saccade_sequences,
    visual_scanning_sequences,
    gaze_angular_velocity_rad,
    dt,
    quiet_eye_duration_threshold,
):

    def split_sequences_before_and_after_quiet_eye(last_2s_timing_idx, sequences):
        sequences_before_quiet_eye = []
        sequences_last_2s = []
        if len(sequences) == 0 or sequences[0].shape == (0,) or sequences[0].shape == (1, 0):
            return sequences, sequences  # returning two empty sequences
        for i_sequence in sequences:
            if i_sequence[-1] < last_2s_timing_idx:
                sequences_before_quiet_eye.append(i_sequence)
            elif last_2s_timing_idx in i_sequence:
                last_2s_idx_this_sequence = np.where(i_sequence == last_2s_timing_idx)[0][0]
                sequences_before_quiet_eye.append(i_sequence[: last_2s_idx_this_sequence + 1])
                sequences_last_2s.append(i_sequence[last_2s_idx_this_sequence + 1 :])
            elif i_sequence[0] > last_2s_timing_idx:
                sequences_last_2s.append(i_sequence)
            else:
                raise RuntimeError("The sequence is not properly classified.")
        return sequences_before_quiet_eye, sequences_last_2s

    def split_durations_before_and_after_quiet_eye(
        sequences, last_2s_timing_idx, dt, time_vector, duration_threshold=0
    ):
        durations = []
        durations_before_quiet_eye = []
        durations_last_2s = []
        for i in sequences:
            if len(i) > 0:
                duration = time_vector[i[-1]] - time_vector[i[0]] + dt
                if duration > duration_threshold:
                    durations.append(duration)
                    if i[-1] < last_2s_timing_idx:
                        durations_before_quiet_eye.append(time_vector[i[-1]] - time_vector[i[0]] + dt)
                    elif last_2s_timing_idx in i:
                        durations_before_quiet_eye.append(time_vector[last_2s_timing_idx + 1] - time_vector[i[0]])
                        durations_last_2s.append(time_vector[i[-1]] - time_vector[last_2s_timing_idx + 1] + dt)
                    elif i[0] > last_2s_timing_idx:
                        durations_last_2s.append(time_vector[i[-1]] - time_vector[i[0]] + dt)
        return durations, durations_before_quiet_eye, durations_last_2s

    # Instant at which the "last 2s switch" is happening
    last_2s_timing_idx = np.where(time_vector > time_vector[-1] - quiet_eye_duration_threshold)[0][0]
    smooth_pursuite_sequences_before_quiet_eye, smooth_pursuite_sequences_last_2s = (
        split_sequences_before_and_after_quiet_eye(last_2s_timing_idx, smooth_pursuite_sequences)
    )
    fixation_sequences_before_quiet_eye, fixation_sequences_last_2s = split_sequences_before_and_after_quiet_eye(
        last_2s_timing_idx, fixation_sequences
    )
    blink_sequences_before_quiet_eye, blink_sequences_last_2s = split_sequences_before_and_after_quiet_eye(
        last_2s_timing_idx, blink_sequences
    )
    saccade_sequences_before_quiet_eye, saccade_sequences_last_2s = split_sequences_before_and_after_quiet_eye(
        last_2s_timing_idx, saccade_sequences
    )
    visual_scanning_sequences_before_quiet_eye, visual_scanning_sequences_last_2s = (
        split_sequences_before_and_after_quiet_eye(last_2s_timing_idx, visual_scanning_sequences)
    )

    # Intermediary metrics
    smooth_pursuite_trajectories = measure_smooth_pursuite_trajectory(
        time_vector, smooth_pursuite_sequences, gaze_angular_velocity_rad, dt
    )
    smooth_pursuite_trajectories_before_quiet_eye = measure_smooth_pursuite_trajectory(
        time_vector, smooth_pursuite_sequences_before_quiet_eye, gaze_angular_velocity_rad, dt
    )
    smooth_pursuite_trajectories_last_2s = measure_smooth_pursuite_trajectory(
        time_vector, smooth_pursuite_sequences_last_2s, gaze_angular_velocity_rad, dt
    )

    # Total time spent in fixations
    fixation_duration, fixation_duration_before_quiet_eye, fixation_duration_last_2s = (
        split_durations_before_and_after_quiet_eye(
            fixation_sequences, last_2s_timing_idx, dt, time_vector, duration_threshold=0.1
        )
    )
    total_fixation_duration = np.sum(np.array(fixation_duration))
    total_fixation_duration_before_quiet_eye = np.sum(np.array(fixation_duration_before_quiet_eye))
    total_fixation_duration_last_2s = np.sum(np.array(fixation_duration_last_2s))

    # Total time spent in smooth pursuite
    smooth_pursuite_duration, smooth_pursuite_duration_before_quiet_eye, smooth_pursuite_duration_last_2s = (
        split_durations_before_and_after_quiet_eye(smooth_pursuite_sequences, last_2s_timing_idx, dt, time_vector)
    )
    total_smooth_pursuite_duration = np.sum(np.array(smooth_pursuite_duration))
    total_smooth_pursuite_duration_before_quiet_eye = np.sum(np.array(smooth_pursuite_duration_before_quiet_eye))
    total_smooth_pursuite_duration_last_2s = np.sum(np.array(smooth_pursuite_duration_last_2s))

    # Total time spent in blinks
    blink_duration, blink_duration_before_quiet_eye, blink_duration_last_2s = (
        split_durations_before_and_after_quiet_eye(blink_sequences, last_2s_timing_idx, dt, time_vector)
    )
    total_blink_duration = np.sum(np.array(blink_duration))
    total_blink_duration_before_quiet_eye = np.sum(np.array(blink_duration_before_quiet_eye))
    total_blink_duration_last_2s = np.sum(np.array(blink_duration_last_2s))

    # Total time spent in saccades
    saccade_duration, saccade_duration_before_quiet_eye, saccade_duration_last_2s = (
        split_durations_before_and_after_quiet_eye(saccade_sequences, last_2s_timing_idx, dt, time_vector)
    )
    total_saccade_duration = np.sum(np.array(saccade_duration))
    total_saccade_duration_before_quiet_eye = np.sum(np.array(saccade_duration_before_quiet_eye))
    total_saccade_duration_last_2s = np.sum(np.array(saccade_duration_last_2s))

    # Total time spent in visual scanning
    visual_scanning_duration, visual_scanning_duration_before_quiet_eye, visual_scanning_duration_last_2s = (
        split_durations_before_and_after_quiet_eye(visual_scanning_sequences, last_2s_timing_idx, dt, time_vector)
    )
    total_visual_scanning_duration = np.sum(np.array(visual_scanning_duration))
    total_visual_scanning_duration_before_quiet_eye = np.sum(np.array(visual_scanning_duration_before_quiet_eye))
    total_visual_scanning_duration_last_2s = np.sum(np.array(visual_scanning_duration_last_2s))

    return (
        smooth_pursuite_sequences_before_quiet_eye,
        smooth_pursuite_sequences_last_2s,
        fixation_sequences_before_quiet_eye,
        fixation_sequences_last_2s,
        blink_sequences_before_quiet_eye,
        blink_sequences_last_2s,
        saccade_sequences_before_quiet_eye,
        saccade_sequences_last_2s,
        visual_scanning_sequences_before_quiet_eye,
        visual_scanning_sequences_last_2s,
        fixation_duration,
        fixation_duration_before_quiet_eye,
        fixation_duration_last_2s,
        smooth_pursuite_duration,
        smooth_pursuite_duration_before_quiet_eye,
        smooth_pursuite_duration_last_2s,
        blink_duration,
        blink_duration_before_quiet_eye,
        blink_duration_last_2s,
        saccade_duration,
        saccade_duration_before_quiet_eye,
        saccade_duration_last_2s,
        visual_scanning_duration,
        visual_scanning_duration_before_quiet_eye,
        visual_scanning_duration_last_2s,
        smooth_pursuite_trajectories,
        smooth_pursuite_trajectories_before_quiet_eye,
        smooth_pursuite_trajectories_last_2s,
        total_fixation_duration,
        total_fixation_duration_before_quiet_eye,
        total_fixation_duration_last_2s,
        total_smooth_pursuite_duration,
        total_smooth_pursuite_duration_before_quiet_eye,
        total_smooth_pursuite_duration_last_2s,
        total_blink_duration,
        total_blink_duration_before_quiet_eye,
        total_blink_duration_last_2s,
        total_saccade_duration,
        total_saccade_duration_before_quiet_eye,
        total_saccade_duration_last_2s,
        total_visual_scanning_duration,
        total_visual_scanning_duration_before_quiet_eye,
        total_visual_scanning_duration_last_2s,
        last_2s_timing_idx,
    )


if "figures" not in os.listdir():
    os.mkdir("figures")
# ------------------------------------------------------------
# Define the path to the data
datapath = "AllData/"
black_screen_timing_file_path = "length_before_black_screen.xlsx"
black_screen_timing_data = pd.read_excel(datapath + black_screen_timing_file_path)
trial_names = list(black_screen_timing_data["Video Name"])


# ----------------------------------------------------------------
# Create a file with the name of the data files that were excluded for poor quality
bad_data_file = open("bad_data_files.txt", "w")
bad_data_file.write(
    "The following files were excluded because more than 50% of the points were excluded by the eye-tracker : \n\n"
)

# ----------------------------------------------------------------
long_trials = ["20240222172036_eye_tracking_VideoListOne_TESTVA13_Experiment_Mode_360VR_Pen28_012.csv"]
output_metrics_dataframe = None
for path, folders, files in os.walk(datapath):
    for file in files:
        if file.endswith(".csv"):

            if SKIP_LONG_TRIALS:
                # skip trials that take too long to compute, useful for debugging only
                if file in long_trials:
                    continue

            # Get the data from the file
            this_trial_name = file.split("_")[-2]
            if "Experiment" not in file:
                continue
            else:
                if this_trial_name not in trial_names:
                    raise RuntimeError(
                        f"This trial {this_trial_name} ({file}) is not in the list of lengths of trials ('durée vidéos.xlsx')"
                    )
                    length_trial = np.nan
                else:
                    length_trial = black_screen_timing_data["Lenght before black screen (s)"][
                        trial_names.index(this_trial_name)
                    ]

            figname = (
                file.split("_")[4] + "_" + file.split("_")[7] + "_" + file.split("_")[8] + "_" + file.split("_")[9][:-4]
            )
            print(f"Treating the data from file : {file}")
            data = pd.read_csv(path + "/" + file, sep=";")

            if len(data["time(100ns)"]) == 0:
                print("\n\n ****************************************************************************** \n")
                print(f"Data from file {file} is empty")
                print("\n ****************************************************************************** \n\n")
                bad_data_file.write(f"{file} \n")
                continue

            if (
                np.sum(np.logical_or(data["eye_valid_L"] != 31, data["eye_valid_R"] != 31))
                > len(data["eye_valid_L"]) / 2
            ):
                print("\n\n ****************************************************************************** \n")
                print(f"More than 50% of the data from file {file} is declared invalid by the eye-tracker")
                print("\n ****************************************************************************** \n\n")
                bad_data_file.write(f"{file} \n")
                continue

            time_vector = np.array((data["time(100ns)"] - data["time(100ns)"][0]) / 10000000)
            length_trial = time_vector[-1] if np.isnan(length_trial) else length_trial

            # cut the data after the black screen
            black_screen_index = (
                np.where(time_vector > length_trial)[0][0] if length_trial < time_vector[-1] else len(time_vector)
            )
            time_vector = time_vector[:black_screen_index]
            data = data.iloc[:black_screen_index, :]

            # Remove the duplicated timestamps in the data
            bad_timestamps_index = list(np.where((time_vector[1:] - time_vector[:-1]) == 0)[0])
            good_timestamps_index = [i for i in range(len(time_vector)) if i not in bad_timestamps_index]
            time_vector = time_vector[good_timestamps_index]
            data = data.iloc[good_timestamps_index, :]

            eye_direction = np.array([data["gaze_direct_L.x"], data["gaze_direct_L.y"], data["gaze_direct_L.z"]])
            helmet_rotation = np.array([data["helmet_rot_x"], data["helmet_rot_y"], data["helmet_rot_z"]])
            helmet_rotation_unwrapped_deg, head_angular_velocity_deg = fix_helmet_rotation(time_vector, helmet_rotation)

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
            blink_sequences = detect_blinks(time_vector, data)

            if PLOT_BAD_DATA_FLAG:
                plot_bad_data_timing(time_vector, eye_direction, figname)

            # Remove blink sequences from the variable vectors
            for blink in blink_sequences:
                eye_direction[:, blink] = np.nan
            # Remove the data that the eye-tracker declares invalid
            if eyetracker_invalid_data_index.shape != (0,):
                eye_direction[:, eyetracker_invalid_data_index] = np.nan

            # Detect saccades
            saccade_sequences, eye_angular_velocity_rad, eye_angular_acceleration_rad, saccade_amplitudes = (
                detect_saccades(time_vector, eye_direction)
            )
            gaze_direction = get_gaze_direction(helmet_rotation_unwrapped_deg, eye_direction)

            # Detect visual scanning
            visual_scanning_sequences, gaze_angular_velocity_rad = detect_visual_scanning(
                time_vector, gaze_direction, saccade_sequences
            )

            if PLOT_ROTATION_VELOCITIES_FLAG:
                plot_head_eye_rotation(
                    eye_direction,
                    helmet_rotation_unwrapped_deg,
                    head_angular_velocity_deg,
                    gaze_angular_velocity_rad,
                    time_vector,
                    figname,
                )

            # Detect fixations
            intersaccadic_interval = np.zeros((len(time_vector),))
            all_index = np.arange(len(time_vector))
            for i in all_index:
                i_in_saccades = True if any(i in sequence for sequence in saccade_sequences) else False
                i_in_visual_scanning = True if any(i in sequence for sequence in visual_scanning_sequences) else False
                i_in_blinks = True if any(i in sequence for sequence in blink_sequences) else False
                i_in_eyetracker_invalid = True if i in eyetracker_invalid_data_index else False
                if i_in_saccades or i_in_visual_scanning or i_in_blinks or i_in_eyetracker_invalid:
                    continue
                else:
                    intersaccadic_interval[i] = 1
            intersaccadic_timing = np.where(intersaccadic_interval == 1)[0]
            intersaccadic_sequences_temporary = np.array_split(
                intersaccadic_timing, np.flatnonzero(np.diff(intersaccadic_timing) > 1) + 1
            )
            intersaccadic_sequences = [
                np.hstack((intersaccadic_sequences_temporary[i], intersaccadic_sequences_temporary[i][-1] + 1))
                for i in range(len(intersaccadic_sequences_temporary))
                if len(intersaccadic_sequences_temporary[i]) > 2
            ]
            intersaccadic_gouped_sequences, intersaccadic_coherent_sequences, intersaccadic_incoherent_sequences = (
                sliding_window(time_vector, intersaccadic_sequences, gaze_direction)
            )
            if PLOT_INTERSACCADIC_FLAG:
                plot_intersaccadic_interval_subinterval_cut(
                    time_vector, intersaccadic_coherent_sequences, intersaccadic_incoherent_sequences, figname
                )
            fixation_sequences, smooth_pursuite_sequences, uncertain_sequences = detect_fixations_and_smooth_pursuite(
                time_vector, gaze_direction, intersaccadic_gouped_sequences, figname, PLOT_CRITERIA_FLAG
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
                    smooth_pursuite_sequences,
                    eyetracker_invalid_sequences,
                    visual_scanning_sequences,
                    quiet_eye_duration_threshold,
                    gaze_angular_velocity_rad,
                    eye_angular_acceleration_rad,
                    dt,
                    figname,
                )

            (
                smooth_pursuite_sequences_before_quiet_eye,
                smooth_pursuite_sequences_last_2s,
                fixation_sequences_before_quiet_eye,
                fixation_sequences_last_2s,
                blink_sequences_before_quiet_eye,
                blink_sequences_last_2s,
                saccade_sequences_before_quiet_eye,
                saccade_sequences_last_2s,
                visual_scanning_sequences_before_quiet_eye,
                visual_scanning_sequences_last_2s,
                fixation_duration,
                fixation_duration_before_quiet_eye,
                fixation_duration_last_2s,
                smooth_pursuite_duration,
                smooth_pursuite_duration_before_quiet_eye,
                smooth_pursuite_duration_last_2s,
                blink_duration,
                blink_duration_before_quiet_eye,
                blink_duration_last_2s,
                saccade_duration,
                saccade_duration_before_quiet_eye,
                saccade_duration_last_2s,
                visual_scanning_duration,
                visual_scanning_duration_before_quiet_eye,
                visual_scanning_duration_last_2s,
                smooth_pursuite_trajectories,
                smooth_pursuite_trajectories_before_quiet_eye,
                smooth_pursuite_trajectories_last_2s,
                total_fixation_duration,
                total_fixation_duration_before_quiet_eye,
                total_fixation_duration_last_2s,
                total_smooth_pursuite_duration,
                total_smooth_pursuite_duration_before_quiet_eye,
                total_smooth_pursuite_duration_last_2s,
                total_blink_duration,
                total_blink_duration_before_quiet_eye,
                total_blink_duration_last_2s,
                total_saccade_duration,
                total_saccade_duration_before_quiet_eye,
                total_saccade_duration_last_2s,
                total_visual_scanning_duration,
                total_visual_scanning_duration_before_quiet_eye,
                total_visual_scanning_duration_last_2s,
                last_2s_timing_idx,
            ) = compute_intermediary_metrics(
                time_vector,
                smooth_pursuite_sequences,
                fixation_sequences,
                blink_sequences,
                saccade_sequences,
                visual_scanning_sequences,
                gaze_angular_velocity_rad,
                dt,
                quiet_eye_duration_threshold,
            )

            # Metrics
            nb_fixations = len(fixation_duration)
            nb_fixations_before_quiet_eye = len(fixation_duration_before_quiet_eye)
            nb_fixations_last_2s = len(fixation_duration_last_2s)

            mean_fixation_duration = np.nanmean(np.array(fixation_duration)) if len(fixation_duration) > 0 else None
            mean_fixation_duration_before_quiet_eye = (
                np.nanmean(np.array(fixation_duration_before_quiet_eye))
                if len(fixation_duration_before_quiet_eye) > 0
                else None
            )
            mean_fixation_duration_last_2s = (
                np.nanmean(np.array(fixation_duration_last_2s)) if len(fixation_duration_last_2s) > 0 else None
            )

            search_rate = nb_fixations / mean_fixation_duration if mean_fixation_duration is not None else None
            search_rate_before_quiet_eye = (
                nb_fixations_before_quiet_eye / mean_fixation_duration_before_quiet_eye
                if mean_fixation_duration_before_quiet_eye is not None
                else None
            )
            search_rate_last_2s = (
                nb_fixations_last_2s / mean_fixation_duration_last_2s
                if mean_fixation_duration_last_2s is not None
                else None
            )

            nb_blinks = len(blink_sequences)
            nb_blinks_before_quiet_eye = len(blink_sequences_before_quiet_eye)
            nb_blinks_last_2s = len(blink_sequences_last_2s)

            nb_saccades = len(saccade_sequences)
            nb_saccades_before_quiet_eye = len(saccade_sequences_before_quiet_eye)
            nb_saccades_last_2s = len(saccade_sequences_last_2s)

            mean_saccade_duration = np.nanmean(np.array(saccade_duration)) if len(saccade_duration) > 0 else None
            mean_saccade_duration_before_quiet_eye = (
                np.nanmean(np.array(saccade_duration_before_quiet_eye))
                if len(saccade_duration_before_quiet_eye) > 0
                else None
            )
            mean_saccade_duration_last_2s = (
                np.nanmean(np.array(saccade_duration_last_2s)) if len(saccade_duration_last_2s) > 0 else None
            )

            max_saccade_amplitude = np.nanmax(np.array(saccade_amplitudes)) if len(saccade_amplitudes) > 0 else None

            mean_saccade_amplitude = np.nanmean(np.array(saccade_amplitudes)) if len(saccade_amplitudes) > 0 else None
            saccade_amplitudes_before_quiet_eye = []
            saccade_amplitudes_last_2s = []
            for i, i_sequence in enumerate(saccade_sequences):
                if i_sequence[0] < last_2s_timing_idx:
                    saccade_amplitudes_before_quiet_eye.append(saccade_amplitudes[i])
                # If the saccade is happening during the 2s transition, we skip it
                elif i_sequence[0] > last_2s_timing_idx:
                    saccade_amplitudes_last_2s.append(saccade_amplitudes[i])
            mean_saccade_amplitude_before_quiet_eye = (
                np.nanmean(np.array(saccade_amplitudes_before_quiet_eye))
                if len(saccade_amplitudes_before_quiet_eye) > 0
                else None
            )
            mean_saccade_amplitude_last_2s = (
                np.nanmean(np.array(saccade_amplitudes_last_2s)) if len(saccade_amplitudes_last_2s) > 0 else None
            )

            nb_smooth_pursuite = len(smooth_pursuite_sequences)
            nb_smooth_pursuite_before_quiet_eye = len(smooth_pursuite_sequences_before_quiet_eye)
            nb_smooth_pursuite_last_2s = len(smooth_pursuite_sequences_last_2s)

            mean_smooth_pursuite_duration = (
                np.nanmean(np.array(smooth_pursuite_duration)) if len(smooth_pursuite_duration) > 0 else None
            )
            mean_smooth_pursuite_duration_before_quiet_eye = (
                np.nanmean(np.array(smooth_pursuite_duration_before_quiet_eye))
                if len(smooth_pursuite_duration_before_quiet_eye) > 0
                else None
            )
            mean_smooth_pursuite_duration_last_2s = (
                np.nanmean(np.array(smooth_pursuite_duration_last_2s))
                if len(smooth_pursuite_duration_last_2s) > 0
                else None
            )

            max_smooth_pursuite_trajectory = (
                np.nanmax(np.array(smooth_pursuite_trajectories)) if len(smooth_pursuite_trajectories) > 0 else None
            )

            mean_smooth_pursuite_trajectory = (
                np.nanmean(np.array(smooth_pursuite_trajectories)) if len(smooth_pursuite_trajectories) > 0 else None
            )
            mean_smooth_pursuite_trajectory_before_quiet_eye = (
                np.nanmean(np.array(smooth_pursuite_trajectories_before_quiet_eye))
                if len(smooth_pursuite_trajectories_before_quiet_eye) > 0
                else None
            )
            mean_smooth_pursuite_trajectory_last_2s = (
                np.nanmean(np.array(smooth_pursuite_trajectories_last_2s))
                if len(smooth_pursuite_trajectories_last_2s) > 0
                else None
            )

            nb_visual_scanning = len(visual_scanning_sequences)
            nb_visual_scanning_before_quiet_eye = len(visual_scanning_sequences_before_quiet_eye)
            nb_visual_scanning_last_2s = len(visual_scanning_sequences_last_2s)

            mean_visual_scanning_duration = (
                np.nanmean(np.array(visual_scanning_duration)) if len(visual_scanning_duration) > 0 else None
            )
            mean_visual_scanning_duration_before_quiet_eye = (
                np.nanmean(np.array(visual_scanning_duration_before_quiet_eye))
                if len(visual_scanning_duration_before_quiet_eye) > 0
                else None
            )
            mean_visual_scanning_duration_last_2s = (
                np.nanmean(np.array(visual_scanning_duration_last_2s))
                if len(visual_scanning_duration_last_2s) > 0
                else None
            )

            fixation_ratio = total_fixation_duration / time_vector[-1]
            fixation_ratio_before_quiet_eye = total_fixation_duration_before_quiet_eye / (
                time_vector[-1] - quiet_eye_duration_threshold
            )
            fixation_ratio_last_2s = total_fixation_duration_last_2s / quiet_eye_duration_threshold

            smooth_pursuite_ratio = total_smooth_pursuite_duration / time_vector[-1]
            smooth_pursuite_ratio_before_quiet_eye = total_smooth_pursuite_duration_before_quiet_eye / (
                time_vector[-1] - quiet_eye_duration_threshold
            )
            smooth_pursuite_ratio_last_2s = total_smooth_pursuite_duration_last_2s / quiet_eye_duration_threshold

            blinking_ratio = total_blink_duration / time_vector[-1]
            blinking_ratio_before_quiet_eye = total_blink_duration_before_quiet_eye / (
                time_vector[-1] - quiet_eye_duration_threshold
            )
            blinking_ratio_last_2s = total_blink_duration_last_2s / quiet_eye_duration_threshold

            saccade_ratio = total_saccade_duration / time_vector[-1]
            saccade_ratio_before_quiet_eye = total_saccade_duration_before_quiet_eye / (
                time_vector[-1] - quiet_eye_duration_threshold
            )
            saccade_ratio_last_2s = total_saccade_duration_last_2s / quiet_eye_duration_threshold

            visual_scanning_ratio = total_visual_scanning_duration / time_vector[-1]
            visual_scanning_ratio_before_quiet_eye = total_visual_scanning_duration_before_quiet_eye / (
                time_vector[-1] - quiet_eye_duration_threshold
            )
            visual_scanning_ratio_last_2s = total_visual_scanning_duration_last_2s / quiet_eye_duration_threshold

            not_classified_ratio = 1 - (
                fixation_ratio + smooth_pursuite_ratio + blinking_ratio + saccade_ratio + visual_scanning_ratio
            )
            invalid_ratio = np.sum(np.logical_or(data["eye_valid_L"] != 31, data["eye_valid_R"] != 31)) / len(
                data["eye_valid_L"]
            )

            output = pd.DataFrame(
                {
                    "File name": [file],
                    "Figure name": [figname],
                    "Participant ID": [file.split("_")[4]],
                    "Mode": [file.split("_")[7]],
                    "Trial name": [file.split("_")[8]],
                    "Trial number": [file.split("_")[9][:-4]],
                    "Number of fixations": [nb_fixations],
                    "Number of fixations before quiet eye": [nb_fixations_before_quiet_eye],
                    "Number of fixations last 2s": [nb_fixations_last_2s],
                    "Mean fixation duration [s]": [mean_fixation_duration],
                    "Mean fixation duration before quiet eye [s]": [mean_fixation_duration_before_quiet_eye],
                    "Mean fixation duration last 2s [s]": [mean_fixation_duration_last_2s],
                    "Search rate": [search_rate],
                    "Search rate before quiet eye": [search_rate_before_quiet_eye],
                    "Search rate last 2s": [search_rate_last_2s],
                    "Number of blinks": [nb_blinks],
                    "Number of blinks before quiet eye": [nb_blinks_before_quiet_eye],
                    "Number of blinks last 2s": [nb_blinks_last_2s],
                    "Number of saccades": [nb_saccades],
                    "Number of saccades before quiet eye": [nb_saccades_before_quiet_eye],
                    "Number of saccades last 2s": [nb_saccades_last_2s],
                    "Mean saccade duration [s]": [mean_saccade_duration],
                    "Mean saccade duration before quiet eye [s]": [mean_saccade_duration_before_quiet_eye],
                    "Mean saccade duration last 2s [s]": [mean_saccade_duration_last_2s],
                    "Max saccade amplitude [deg]": [max_saccade_amplitude],
                    "Mean saccade amplitude [deg]": [mean_saccade_amplitude],
                    "Mean saccade amplitude before quiet eye [deg]": [mean_saccade_amplitude_before_quiet_eye],
                    "Mean saccade amplitude last 2s [deg]": [mean_saccade_amplitude_last_2s],
                    "Number of smooth pursuite": [nb_smooth_pursuite],
                    "Number of smooth pursuite before quiet eye": [nb_smooth_pursuite_before_quiet_eye],
                    "Number of smooth pursuite last 2s": [nb_smooth_pursuite_last_2s],
                    "Mean smooth pursuite duration [s]": [mean_smooth_pursuite_duration],
                    "Mean smooth pursuite duration before quiet eye [s]": [
                        mean_smooth_pursuite_duration_before_quiet_eye
                    ],
                    "Mean smooth pursuite duration last 2s [s]": [mean_smooth_pursuite_duration_last_2s],
                    "Max smooth pursuite trajectory [deg]": [max_smooth_pursuite_trajectory],
                    "Mean smooth pursuite trajectory [deg]": [mean_smooth_pursuite_trajectory],
                    "Mean smooth pursuite trajectory before quiet eye [deg]": [
                        mean_smooth_pursuite_trajectory_before_quiet_eye
                    ],
                    "Mean smooth pursuite trajectory last 2s [deg]": [mean_smooth_pursuite_trajectory_last_2s],
                    "Number of visual scanning": [nb_visual_scanning],
                    "Number of visual scanning before quiet eye": [nb_visual_scanning_before_quiet_eye],
                    "Number of visual scanning last 2s": [nb_visual_scanning_last_2s],
                    "Mean visual scanning duration [s]": [mean_visual_scanning_duration],
                    "Mean visual scanning duration before quiet eye [s]": [
                        mean_visual_scanning_duration_before_quiet_eye
                    ],
                    "Mean visual scanning duration last 2s [s]": [mean_visual_scanning_duration_last_2s],
                    "Fixation ratio": [fixation_ratio],
                    "Fixation ratio before quiet eye": [fixation_ratio_before_quiet_eye],
                    "Fixation ratio last 2s": [fixation_ratio_last_2s],
                    "Smooth pursuite ratio": [smooth_pursuite_ratio],
                    "Smooth pursuite ratio before quiet eye": [smooth_pursuite_ratio_before_quiet_eye],
                    "Smooth pursuite ratio last 2s": [smooth_pursuite_ratio_last_2s],
                    "Blinking ratio": [blinking_ratio],
                    "Blinking ratio before quiet eye": [blinking_ratio_before_quiet_eye],
                    "Blinking ratio last 2s": [blinking_ratio_last_2s],
                    "Saccade ratio": [saccade_ratio],
                    "Saccade ratio before quiet eye": [saccade_ratio_before_quiet_eye],
                    "Saccade ratio last 2s": [saccade_ratio_last_2s],
                    "Visual scanning ratio": [visual_scanning_ratio],
                    "Visual scanning ratio before quiet eye": [visual_scanning_ratio_before_quiet_eye],
                    "Visual scanning ratio last 2s": [visual_scanning_ratio_last_2s],
                    "Not classified ratio": [not_classified_ratio],
                    "Invalid ratio": [invalid_ratio],
                    "Length of the trial [s]": [time_vector[-1]],
                }
            )
            output_metrics_dataframe = (
                output
                if output_metrics_dataframe is None
                else pd.concat([output_metrics_dataframe, output], ignore_index=True)
            )

with open("output_metrics.pkl", "wb") as f:
    pickle.dump(output_metrics_dataframe, f)

# Write a csv file that can be sed for statistical analysis later
output_metrics_dataframe.to_csv("output_metrics.csv", index=False)

bad_data_file.close()
