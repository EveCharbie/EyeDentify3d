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
PLOT_CLASSIFICATION_FLAG = True
SKIP_LONG_TRIALS = True
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

def detect_saccades(time_vector, eye_direction, helmet_rotation):
    """
    Detecting sequences where the gaze angular velocity is larger than 100 degrees per second.
    """

    helmet_rotation_in_rad = helmet_rotation * np.pi / 180

    gaze_direction = np.zeros(eye_direction.shape)
    for i_frame in range(helmet_rotation_in_rad.shape[1]):
        rotation_matrix = biorbd.Rotation.fromEulerAngles(helmet_rotation_in_rad[:, i_frame], 'xyz').to_array()
        gaze_direction[:, i_frame] = rotation_matrix @ eye_direction[:, i_frame]

    gaze_angular_velocity_rad = np.zeros((gaze_direction.shape[1], ))
    for i_frame in range(1, gaze_direction.shape[1] - 1):  # Skipping the first and last frames
        vector_before = gaze_direction[:, i_frame - 1]
        vector_after = gaze_direction[:, i_frame + 1]
        gaze_angular_velocity_rad[i_frame] = np.arccos(np.dot(vector_before, vector_after) / np.linalg.norm(vector_before) / np.linalg.norm(vector_after)) / (time_vector[i_frame + 1] - time_vector[i_frame - 1])
        if np.isnan(gaze_angular_velocity_rad[i_frame]) and not (any(np.isnan(vector_before)) or any(np.isnan(vector_after))):
            print("nan")
    gaze_angular_velocity_rad[0] = np.arccos(
        np.dot(gaze_direction[:, 0], gaze_direction[:, 1]) / np.linalg.norm(gaze_direction[:, 0]) / np.linalg.norm(gaze_direction[:, 1])) / (
                                                     time_vector[1] - time_vector[0])
    gaze_angular_velocity_rad[-1] = np.arccos(
        np.dot(gaze_direction[:, -2], gaze_direction[:, -1]) / np.linalg.norm(gaze_direction[:, -2]) / np.linalg.norm(
            gaze_direction[:, -1])) / (
                                           time_vector[-1] - time_vector[-2])

    gaze_angular_acceleration_rad = np.zeros((gaze_direction.shape[1], ))
    gaze_angular_acceleration_rad[:-1] = (gaze_angular_velocity_rad[1:] - gaze_angular_velocity_rad[:-1]) / (time_vector[1:] - time_vector[:-1])

    velocity_threshold = 5 * np.nanmedian(gaze_angular_velocity_rad * 180 / np.pi)
    # velocity_threshold = 100  # deg/s
    acceleration_threshold = 4000  # deg/s²

    if PLOT_SACCADES_FLAG:
        plt.figure()
        plt.plot(time_vector, np.abs(gaze_angular_velocity_rad * 180 / np.pi), label='Angular Velocity')
        plt.plot(np.array([time_vector[0], time_vector[-1]]), np.array([100, 100]), 'k--', label=r'Threshold 100$\^circ/s$')
        plt.plot(np.array([time_vector[0], time_vector[-1]]), np.array([velocity_threshold, velocity_threshold]), 'b--', label=r'Threshold 5 medians')
        plt.legend()
        plt.savefig("figures/saccade_detection_test.png")
        plt.show()

    # # Velocity only classification
    # saccade_timing = np.where(np.abs(gaze_angular_velocity_rad * 180 / np.pi) > velocity_threshold)[0]
    # saccade_sequences = np.array_split(saccade_timing, np.flatnonzero(np.diff(saccade_timing) > 1) + 1)

    # # Velocity + acceleration classification
    # saccade_timing = np.where(np.abs(gaze_angular_velocity_rad * 180 / np.pi) > velocity_threshold)[0]
    # saccade_sequences_tempo = np.array_split(saccade_timing, np.flatnonzero(np.diff(saccade_timing) > 1) + 1)
    # saccade_sequences = []
    # for i in saccade_sequences_tempo:
    #     if any(np.abs(gaze_angular_acceleration_rad[i[0]-1: i[-1]+1] * 180 / np.pi) > acceleration_threshold):
    #         saccade_sequences += [i]

    # Velocity + 2 frames acceleration classification
    saccade_timing = np.where(np.abs(gaze_angular_velocity_rad * 180 / np.pi) > velocity_threshold)[0]
    saccade_sequences_tempo = np.array_split(saccade_timing, np.flatnonzero(np.diff(saccade_timing) > 1) + 1)
    saccade_sequences = []
    for i in saccade_sequences_tempo:
        acceleration_above_threshold = np.where(np.abs(gaze_angular_acceleration_rad[i[0]-1: i[-1]+1] * 180 / np.pi) > acceleration_threshold)[0]
        if len(acceleration_above_threshold) > 1:
            saccade_sequences += [i]

    # # Velocity 100 + 3 frames acceleration classification
    # saccade_timing = np.where(np.abs(gaze_angular_velocity_rad * 180 / np.pi) > 100)[0]
    # saccade_sequences_tempo = np.array_split(saccade_timing, np.flatnonzero(np.diff(saccade_timing) > 1) + 1)
    # saccade_sequences = []
    # for i in saccade_sequences_tempo:
    #     acceleration_above_threshold = np.where(np.abs(gaze_angular_acceleration_rad[i[0]-1: i[-1]+1] * 180 / np.pi) > acceleration_threshold)[0]
    #     if len(acceleration_above_threshold) > 2:
    #         saccade_sequences += [i]

    # Saccade amplitude
    # Defined as the angle between the beginning and end of the saccade,
    # note that there is no check made to detect if there is a larger amplitude reached during the saccade.
    saccade_amplitudes = []
    for sequence in saccade_sequences:
        vector_before = gaze_direction[:, sequence[0]]
        vector_after = gaze_direction[:, sequence[-1]]
        angle = np.arccos(np.dot(vector_before, vector_after) / np.linalg.norm(vector_before) / np.linalg.norm(vector_after))
        saccade_amplitudes += [angle * 180 / np.pi]

    return saccade_sequences, gaze_direction, gaze_angular_velocity_rad, gaze_angular_acceleration_rad, saccade_amplitudes


def sliding_window(time_vector, intersaccadic_sequences, gaze_direction):
    """
    Parameters t_wind (22000 micros), t_overlap (6000 micros) and eta_p (0.1) taken from the original paper
    https://doi.org/10.1016/j.bspc.2014.12.008
    """
    # @thomasromeas : These parameter values are still to be adjusted
    t_wind = 0.022 * 2  # Window size in ms
    t_overlap = 0.006  # Window overlap in ms
    eta_p = 0.01  # Threshold for the p-value of the Rayleigh test

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

    intersaccadic_coherent_sequences = np.array_split(coherent_windows, np.flatnonzero(np.diff(coherent_windows) > 1) + 1)
    intersaccadic_incoherent_sequences = np.array_split(incoherent_windows, np.flatnonzero(np.diff(incoherent_windows) > 1) + 1)

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
                    elif i_sequence[0] > intersaccadic_gouped_sequences[j][0] and i_sequence[0] < intersaccadic_gouped_sequences[j+1][0]:
                        intersaccadic_gouped_sequences.insert(j+1, i_sequence)

    return intersaccadic_gouped_sequences, intersaccadic_coherent_sequences, intersaccadic_incoherent_sequences


def detect_directionality_coherence(gaze_direction):

    gaze_displacement_rad = gaze_direction[:, 1:] - gaze_direction[:, :-1]

    horizontal_vector_on_the_sphere = np.zeros(gaze_displacement_rad.shape)
    horizontal_vector_on_the_sphere[:2, :] = gaze_displacement_rad[:2, :]
    alpha = np.zeros((horizontal_vector_on_the_sphere.shape[1], ))
    for i_frame in range(gaze_direction.shape[1] - 1):
        alpha[i_frame] = np.arccos(np.dot(gaze_displacement_rad[:, i_frame], horizontal_vector_on_the_sphere[:, i_frame]) / np.linalg.norm(gaze_displacement_rad[:, i_frame]) / np.linalg.norm(horizontal_vector_on_the_sphere[:, i_frame]))

    # Test that the gaze displacement and orientation are coherent inside the window
    z_value_alpha, p_value_alpha = pg.circ_rayleigh(alpha)
    return p_value_alpha



def discriminate_fixations_and_smooth_pursuite(gaze_direction):

    mean_gaze_direction = np.nanmean(gaze_direction, axis=1)
    zeros_mean_gaze_direction = gaze_direction - mean_gaze_direction[:, np.newaxis]
    cov = np.ma.cov(np.ma.masked_invalid(zeros_mean_gaze_direction)).data
    eigen_values_decomposition = np.linalg.eig(cov)
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
    mean_radius_range_gaze_direction = np.sqrt((np.max(gaze_direction[0, :]) - np.min(gaze_direction[0, :])) ** 2 +
                                               (np.max(gaze_direction[1, :]) - np.min(gaze_direction[1, :])) ** 2 +
                                               (np.max(gaze_direction[2, :]) - np.min(gaze_direction[2, :])) ** 2)

    parameter_D = d_pc2 / d_pc1
    parameter_CD = gaze_distance_parcourrue / d_pc1
    parameter_PD = gaze_distance_parcourrue / trajectory_length
    parameter_R = np.arctan(mean_radius_range_gaze_direction / 1)

    return parameter_D, parameter_CD, parameter_PD, parameter_R


def detect_fixations_and_smooth_pursuite(time_vector, gaze_direction, intersaccadic_gouped_sequences):
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

    # Classify the obvious timings
    fixation_timing = []
    smooth_pursuite_timing = []
    uncertain_timing = []
    if len(intersaccadic_gouped_sequences) == 1 and intersaccadic_gouped_sequences[0].shape == (0, ):
        raise RuntimeError("No intersaccadic interval! There should be at least one even if there is no saccades.")
    else:
        for i_sequence, sequence in enumerate(intersaccadic_gouped_sequences):
            parameter_D, parameter_CD, parameter_PD, parameter_R = discriminate_fixations_and_smooth_pursuite(gaze_direction[:, sequence])
            criteria_1 = parameter_D < eta_D
            criteria_2 = parameter_CD > eta_CD
            criteria_3 = parameter_PD > eta_PD
            criteria_4 = parameter_R > eta_maxFix
            sum_criteria = int(criteria_1) + int(criteria_2) + int(criteria_3) + int(criteria_4)
            if sum_criteria == 0:
                fixation_timing += list(sequence)
            elif sum_criteria == 4:
                smooth_pursuite_timing += list(sequence)
            else:
                uncertain_timing += list(sequence)

    # Classify the ambiguous timings
    uncertain_sequences_tempo = np.array_split(uncertain_timing, np.flatnonzero(np.diff(uncertain_timing) > 1) + 1)
    uncertain_sequences_to_remove = []
    for i_sequence, sequence in enumerate(uncertain_sequences_tempo):
        parameter_D, parameter_CD, parameter_PD, parameter_R = discriminate_fixations_and_smooth_pursuite(
            gaze_direction[:, sequence])
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
                        current_mean = np.nanmean(gaze_direction[:, intersaccadic_gouped_sequences[current_i_sequence]], axis=1)
                        angle = np.arccos(np.dot(uncertain_mean, current_mean) / np.linalg.norm(uncertain_mean) / np.linalg.norm(current_mean))
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
                if after_idx in intersaccadic_gouped_sequences[current_i_sequence]:
                    if after_idx in fixation_timing:
                        same_segment_forward = False
                    else:
                        uncertain_mean = np.nanmean(gaze_direction[:, sequence], axis=1)
                        current_mean = np.nanmean(gaze_direction[:, intersaccadic_gouped_sequences[current_i_sequence]], axis=1)
                        angle = np.arccos(np.dot(uncertain_mean, current_mean) / np.linalg.norm(uncertain_mean) / np.linalg.norm(current_mean))
                        if np.abs(angle) < phi:
                            after_idx = intersaccadic_gouped_sequences[current_i_sequence][-1]
                        else:
                            same_segment_forward = False
                else:
                    same_segment_forward = False
            if len(range(before_idx, after_idx)) > 2:
                parameter_D, parameter_CD, parameter_PD, parameter_R = discriminate_fixations_and_smooth_pursuite(
                    gaze_direction[:, range(before_idx, after_idx+1)])
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
    fixation_sequences = np.array_split(np.array(fixation_timing), np.flatnonzero(np.diff(np.array(fixation_timing)) > 1) + 1)
    smooth_pursuite_sequences = np.array_split(np.array(smooth_pursuite_timing), np.flatnonzero(np.diff(np.array(smooth_pursuite_timing)) > 1) + 1)

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
            time_end = time_vector[idx + 1] if idx+1 < len(time_vector) else time_vector[-1] + dt
            d_trajectory = np.abs(gaze_angular_velocity_rad[idx] * 180 / np.pi) * (time_end - time_beginning)
            trajectory_this_time += 0 if np.isnan(d_trajectory) else d_trajectory
        smooth_pursuite_trajectories += [trajectory_this_time]
    return smooth_pursuite_trajectories

def plot_bad_data_timing(time_vector, eye_direction, file):
    """
    Plot the timing of the data to remove either because of blinks or because of invalid data
    """
    plt.figure()
    plt.plot(time_vector, eye_direction[0], label='eye_direction_x')
    plt.plot(time_vector, eye_direction[1], label='eye_direction_y')
    plt.plot(time_vector, eye_direction[2], label='eye_direction_z')
    label_flag = True
    for i in blink_sequences:
        if len(i) < 1:
            continue
        if label_flag:
            plt.axvspan(time_vector[i[0]], time_vector[i[-1]+1], color='g', alpha=0.5, label='Blink Sequences')
            label_flag = False
        else:
            plt.axvspan(time_vector[i[0]], time_vector[i[-1]+1], color='g', alpha=0.5)
    plt.legend()
    plt.savefig(f"figures/bad_data_{file[:-4]}.png")
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
        quiet_eye_duration_threshold,
        gaze_angular_velocity_rad,
        gaze_angular_acceleration_rad,
        dt,
        file):
    """
    Plot the final gaze classification
    """
    time_vector_step = np.hstack((time_vector, time_vector[-1] + dt))
    fig, axs = plt.subplots(3, 1, figsize=(15, 20), gridspec_kw={'height_ratios': [3, 1, 1]})
    axs[0].plot(time_vector, gaze_direction[0, :], 'k', label='Gaze x (head + eye)')
    axs[0].plot(time_vector, gaze_direction[1, :], 'k', label='Gaze y (head + eye)')
    axs[0].plot(time_vector, gaze_direction[2, :], 'k', label='Gaze z (head + eye)')

    velocity_threshold = 5 * np.nanmedian(gaze_angular_velocity_rad * 180 / np.pi)
    axs[1].plot(time_vector, np.abs(gaze_angular_velocity_rad * 180 / np.pi), 'r', label='Gaze velocity norm')
    axs[1].plot(np.array([time_vector[0], time_vector[-1]]),
                np.array([velocity_threshold, velocity_threshold]), '--r', label='5 medians')
    axs[1].plot(np.array([time_vector[0], time_vector[-1]]),
                np.array([100, 100]), ':r', label='100 deg/s')

    acceleration_threshold = 4000  # deg/s²
    axs[2].plot(time_vector, np.abs(gaze_angular_acceleration_rad * 180 / np.pi), 'g', label='Gaze acceleration norm')
    axs[2].plot(np.array([time_vector[0], time_vector[-1]]),
                np.array([acceleration_threshold, acceleration_threshold]), ':g', label='4000 deg/s²')

    label_flag = True
    for i in blink_sequences:
        if i.shape == (1, 0) or i.shape == (0, ) or len(i) < 1:
            continue
        if label_flag:
            axs[0].axvspan(time_vector[i[0]], time_vector_step[i[-1]+1], edgecolor=None, color='tab:green', alpha=0.5,
                        label='Blink Sequences')
            label_flag = False
        else:
            axs[0].axvspan(time_vector[i[0]], time_vector_step[i[-1]+1], edgecolor=None, color='tab:green', alpha=0.5)
    label_flag = True
    for i in saccade_sequences:
        if i.shape == (1, 0) or i.shape == (0, ) or len(i) < 1:
            continue
        if label_flag:
            axs[0].axvspan(time_vector[i[0]], time_vector_step[i[-1]+1], edgecolor=None, color='tab:blue', alpha=0.5,
                        label='Saccade Sequences')
            label_flag = False
        else:
            axs[0].axvspan(time_vector[i[0]], time_vector_step[i[-1]+1], edgecolor=None, color='tab:blue', alpha=0.5)
    label_flag = True
    for i in fixation_sequences:
        if i.shape == (1, 0) or i.shape == (0, ) or len(i) < 1:
            continue
        if label_flag:
            axs[0].axvspan(time_vector[i[0]], time_vector_step[i[-1]+1], edgecolor=None, color='tab:purple', alpha=0.5,
                        label='Fixation Sequences')
            label_flag = False
        else:
            axs[0].axvspan(time_vector[i[0]], time_vector_step[i[-1]+1], edgecolor=None, color='tab:purple', alpha=0.5)
    label_flag = True
    for i in smooth_pursuite_sequences:
        if i.shape == (1, 0) or i.shape == (0, ) or len(i) < 1:
            continue
        if label_flag:
            axs[0].axvspan(time_vector[i[0]], time_vector_step[i[-1]+1], edgecolor=None, color='tab:orange', alpha=0.5,
                        label='Smooth Pursuite Sequences')
            label_flag = False
        else:
            axs[0].axvspan(time_vector[i[0]], time_vector_step[i[-1]+1], edgecolor=None, color='tab:orange', alpha=0.5)
    label_flag = True
    for i in eyetracker_invalid_sequences:
        if i.shape == (1, 0) or i.shape == (0, ) or len(i) < 1:
            continue
        if label_flag:
            axs[0].axvspan(time_vector[i[0]], time_vector_step[i[-1]+1], edgecolor=None, color='tab:red', alpha=0.5,
                        label='Eye-tracker invalid Sequences')
            label_flag = False
        else:
            axs[0].axvspan(time_vector[i[0]], time_vector_step[i[-1]+1], edgecolor=None, color='tab:red', alpha=0.5)
    axs[0].plot(np.array([time_vector[-1] - quiet_eye_duration_threshold, time_vector[-1]]),
             np.array([np.nanmax(gaze_direction) + 0.1, np.nanmax(gaze_direction) + 0.1]), 'k')
    axs[0].plot(np.array([time_vector[-1] - quiet_eye_duration_threshold, time_vector[-1] - quiet_eye_duration_threshold]),
             np.array([np.nanmax(gaze_direction) + 0.09, np.nanmax(gaze_direction) + 0.1]), 'k')
    axs[0].plot(np.array([time_vector[-1], time_vector[-1]]),
             np.array([np.nanmax(gaze_direction) + 0.09, np.nanmax(gaze_direction) + 0.1]), 'k')
    axs[0].text(time_vector[-1] - quiet_eye_duration_threshold/2, np.nanmax(gaze_direction) + 0.11, f"last {quiet_eye_duration_threshold} sec")

    axs[0].legend(bbox_to_anchor=(1.02, 0.7))
    axs[1].legend(bbox_to_anchor=(1.02, 0.7))
    axs[2].legend(bbox_to_anchor=(1.27, 0.7))
    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8)
    plt.savefig(f"figures/gaze_classification_{file[:-4]}.png")
    plt.show()
    return

if "figures" not in os.listdir():
    os.mkdir("figures")
# ------------------------------------------------------------
# Define the path to the data
datapath = "AllData/"
black_screen_timing_file_path = "length_before_black_screen.xlsx"
black_screen_timing_data = pd.read_excel(datapath+black_screen_timing_file_path)
trial_names = list(black_screen_timing_data['Video Name'])


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

            print(f"Treating the data from file : {file}")
            data = pd.read_csv(path+'/'+file, sep=';')
    
            # Get the data from the file
            this_trial_name = file.split('_')[-2]
            if "Experiment" not in file:
                continue
            else:
                if this_trial_name not in trial_names:
                    raise RuntimeError("This trial is not in the list of lengths of trials ('durée vidéos.xlsx')")
                    length_trial = np.nan
                else:
                    length_trial = black_screen_timing_data['Lenght before black screen (s)'][trial_names.index(this_trial_name)]

            time_vector = np.array((data["time_stamp(ms)"] - data["time_stamp(ms)"][0]) / 1000)
            length_trial = time_vector[-1] if np.isnan(length_trial) else length_trial

            # cut the data after the black screen
            black_screen_index = np.where(time_vector > length_trial)[0][0] if length_trial < time_vector[-1] else len(time_vector)
            time_vector = time_vector[:black_screen_index]
            data = data.iloc[:black_screen_index, :]

            # Remove the duplicated timestamps in the data
            bad_timestamps_index = list(np.where((time_vector[1:] - time_vector[:-1]) == 0)[0])
            good_timestamps_index = [i for i in range(len(time_vector)) if i not in bad_timestamps_index]
            time_vector = time_vector[good_timestamps_index]
            data = data.iloc[good_timestamps_index, :]

            eye_direction = np.array(
                [data["gaze_direct_L.x"], data["gaze_direct_L.y"], data["gaze_direct_L.z"]])
            helmet_rotation = np.array([data["helmet_rot_x"], data["helmet_rot_y"], data["helmet_rot_z"]])

            eyetracker_invalid_data_index = np.array([])
            if np.sum(data['eye_valid_L']) != 31 * len(data['eye_valid_L']) or np.sum(data['eye_valid_R']) != 31 * len(data['eye_valid_R']):
                if PLOT_BAD_DATA_FLAG:
                    plt.figure()
                    plt.plot(data['eye_valid_L'] / 31, label='eye_valid_L')
                    plt.plot(data['eye_valid_R'] / 31, label='eye_valid_R')
                    plt.plot(data['openness_L'], label='openness_L')
                    plt.plot(data['openness_R'], label='openness_R')
                    plt.legend()
                    plt.show()
                eyetracker_invalid_data_index = np.where(np.logical_or(data['eye_valid_L'] != 31, data['eye_valid_R'] != 31))[0]
            eyetracker_invalid_sequences = np.array_split(np.array(eyetracker_invalid_data_index), np.flatnonzero(np.diff(np.array(eyetracker_invalid_data_index)) > 1) + 1)

            # Remove blinks
            blink_sequences = detect_blinks(time_vector, data)
            
            if PLOT_BAD_DATA_FLAG:
                plot_bad_data_timing(time_vector, eye_direction, file)

            # Remove blink sequences from the variable vectors
            for blink in blink_sequences:
                eye_direction[:, blink] = np.nan
            # Remove the data that the eye-tracker declares invalid
            if eyetracker_invalid_data_index.shape != (0, ):
                eye_direction[:, eyetracker_invalid_data_index] = np.nan
            
            
            # Detect saccades
            saccade_sequences, gaze_direction, gaze_angular_velocity_rad, gaze_angular_acceleration_rad, saccade_amplitudes = detect_saccades(time_vector, eye_direction, helmet_rotation)
            
            # Detect fixations
            intersaccadic_interval = np.zeros((len(time_vector), ))
            all_index = np.arange(len(time_vector))
            for i in all_index:
                i_in_saccades = True if any(i in sequence for sequence in saccade_sequences) else False
                i_in_saccades = True if any(i in sequence for sequence in saccade_sequences) else False
                i_in_blinks = True if any(i in sequence for sequence in blink_sequences) else False
                i_in_eyetracker_invalid = True if i in eyetracker_invalid_data_index else False
                if i_in_saccades or i_in_blinks or i_in_eyetracker_invalid:
                    continue
                else:
                    intersaccadic_interval[i] = 1
            intersaccadic_timing = np.where(intersaccadic_interval == 1)[0]
            intersaccadic_sequences_temporary = np.array_split(intersaccadic_timing, np.flatnonzero(np.diff(intersaccadic_timing) > 1) + 1)
            intersaccadic_sequences = [np.hstack((intersaccadic_sequences_temporary[i], intersaccadic_sequences_temporary[i][-1] + 1)) for i in range(len(intersaccadic_sequences_temporary)) if len(intersaccadic_sequences_temporary[i]) > 2]
            intersaccadic_gouped_sequences, intersaccadic_coherent_sequences, intersaccadic_incoherent_sequences = sliding_window(time_vector, intersaccadic_sequences, gaze_direction)
            fixation_sequences, smooth_pursuite_sequences, uncertain_sequences = detect_fixations_and_smooth_pursuite(time_vector, gaze_direction, intersaccadic_gouped_sequences)

            if PLOT_CLASSIFICATION_FLAG:
                # The mean duration of a frame because we only have data at the frame and the duration of the events
                # should be computed as a step function, thus we have to add a step after the last index.
                dt = np.mean(time_vector[1:] - time_vector[
                                               :-1])
                plot_gaze_classification(time_vector,
                                         gaze_direction,
                                         blink_sequences,
                                         saccade_sequences, 
                                         fixation_sequences, 
                                         smooth_pursuite_sequences,
                                         eyetracker_invalid_sequences,
                                         quiet_eye_duration_threshold,
                                         gaze_angular_velocity_rad,
                                         gaze_angular_acceleration_rad,
                                         dt,
                                         file)

            # Intermediary metrics
            smooth_pursuite_trajectories = measure_smooth_pursuite_trajectory(time_vector, smooth_pursuite_sequences, gaze_angular_velocity_rad, dt)
            fixation_duration = []
            for i in fixation_sequences:
                if len(i) > 0:
                    duration = time_vector[i[-1]] - time_vector[i[0]] + dt
                    fixation_duration_threshold = 0.1  # minimum of 100 ms for a fixation
                    if duration > fixation_duration_threshold:
                        fixation_duration.append(duration)
            total_fixation_duration = np.sum(np.array(fixation_duration))
            smooth_pursuite_duration = []
            for i in smooth_pursuite_sequences:
                if len(i) > 0:
                    smooth_pursuite_duration.append(time_vector[i[-1]] - time_vector[i[0]])
            total_smooth_pursuite_duration = np.sum(np.array(smooth_pursuite_duration))
            blink_duration = []
            for i in blink_sequences:
                if len(i) > 0:
                    blink_duration.append(time_vector[i[-1]] - time_vector[i[0]] + dt)
            total_blink_duration = np.sum(np.array(blink_duration))
            saccade_duration = []
            for i in saccade_sequences:
                if len(i) > 0:
                    saccade_duration.append(time_vector[i[-1]] - time_vector[i[0]] + dt)
            total_saccade_duration = np.sum(np.array(saccade_duration))
            
            # Metrics
            nb_fixations = len(fixation_sequences)
            mean_fixation_duration = np.mean(np.array(fixation_duration))
            search_rate = nb_fixations / mean_fixation_duration
            nb_blinks = len(blink_sequences)
            nb_saccades = len(saccade_sequences)
            mean_saccade_duration = np.mean(np.array(saccade_duration))
            max_saccade_amplitude = np.max(np.array(saccade_amplitudes))
            mean_saccade_amplitude = np.mean(np.array(saccade_amplitudes))
            nb_smooth_pursuite = len(smooth_pursuite_sequences)
            mean_smooth_pursuite_duration = np.mean(np.array(smooth_pursuite_duration))
            max_smooth_pursuite_trajectory = np.max(np.array(smooth_pursuite_trajectories))
            mean_smooth_pursuite_trajectory = np.mean(np.array(smooth_pursuite_trajectories))
            fixation_ratio = total_fixation_duration / time_vector[-1]
            smooth_pursuite_ratio = total_smooth_pursuite_duration / time_vector[-1]
            blinking_ratio = total_blink_duration / time_vector[-1]
            saccade_ratio = total_saccade_duration / time_vector[-1]
            not_classified_ratio = 1 - (fixation_ratio + smooth_pursuite_ratio + blinking_ratio + saccade_ratio)

            if output_metrics_dataframe is None:
                output_metrics_dataframe = pd.DataFrame({
                    'File name': [file],
                    'Participant ID': [file.split('_')[4]],
                    'Mode': [file.split('_')[7]],
                    'Trial name': [file.split('_')[8]],
                    'Trial number': [file.split('_')[9][:-4]],
                    'Number of fixations': [nb_fixations],
                    'Mean fixation duration [s]': [mean_fixation_duration],
                    'Search rate': [search_rate],
                    'Number of blinks': [nb_blinks],
                    'Number of saccades': [nb_saccades],
                    'Mean saccade duration [s]': [mean_saccade_duration],
                    'Max saccade amplitude [deg]': [max_saccade_amplitude],
                    'Mean saccade amplitude [deg]': [mean_saccade_amplitude],
                    'Number of smooth pursuite': [nb_smooth_pursuite],
                    'Mean smooth pursuite duration [s]': [mean_smooth_pursuite_duration],
                    'Max smooth pursuite trajectory [deg]': [max_smooth_pursuite_trajectory],
                    'Mean smooth pursuite trajectory [deg]': [mean_smooth_pursuite_trajectory],
                    'Fixation ratio': [fixation_ratio],
                    'Smooth pursuite ratio': [smooth_pursuite_ratio],
                    'Blinking ratio': [blinking_ratio],
                    'Saccade ratio': [saccade_ratio],
                    'Not classified ratio': [not_classified_ratio],
                })
            else:
                output_metrics_dataframe = pd.concat([output_metrics_dataframe, pd.DataFrame({
                    'File name': [file],
                    'Participant ID': [file.split('_')[4]],
                    'Mode': [file.split('_')[7]],
                    'Trial name': [file.split('_')[8]],
                    'Trial number': [file.split('_')[9][:-4]],
                    'Number of fixations': [nb_fixations],
                    'Mean fixation duration [s]': [mean_fixation_duration],
                    'Search rate': [search_rate],
                    'Number of blinks': [nb_blinks],
                    'Number of saccades': [nb_saccades],
                    'Mean saccade duration [s]': [mean_saccade_duration],
                    'Max saccade amplitude [deg]': [max_saccade_amplitude],
                    'Mean saccade amplitude [deg]': [mean_saccade_amplitude],
                    'Number of smooth pursuite': [nb_smooth_pursuite],
                    'Mean smooth pursuite duration [s]': [mean_smooth_pursuite_duration],
                    'Max smooth pursuite trajectory [deg]': [max_smooth_pursuite_trajectory],
                    'Mean smooth pursuite trajectory [deg]': [mean_smooth_pursuite_trajectory],
                    'Fixation ratio': [fixation_ratio],
                    'Smooth pursuite ratio': [smooth_pursuite_ratio],
                    'Blinking ratio': [blinking_ratio],
                    'Saccade ratio': [saccade_ratio],
                    'Not classified ratio': [not_classified_ratio],
                })], ignore_index=True)


with open("output_metrics.pkl", 'wb') as f:
    pickle.dump(output_metrics_dataframe, f)

# Write a csv file that can be sed for statistical analysis later
output_metrics_dataframe.to_csv('output_metrics.csv', index=False)

