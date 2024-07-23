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


# Define the path to the data
datapath = "AllData/"
test_path = "TESTNA01/VideoListOne/20231030161004_eye_tracking_VideoListOne_TESTNA01_Demo_Mode_2D_Pen1_000.csv"
black_screen_timing_file_path = "length_before_black_screen.xlsx"
test_data = pd.read_csv(datapath+test_path, sep=';')
black_screen_timing_data = pd.read_excel(datapath+black_screen_timing_file_path)

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

# Parameters to define ---------------------------------------
blink_threshold = 0.5
gaze_distance_fixed = 7
PLOT_BAD_DATA_FLAG = False
PLOT_SACCADES_FLAG = False
# ------------------------------------------------------------

trial_names = list(black_screen_timing_data['Video Name'])
this_trial_name = test_path.split('_')[-2]
if this_trial_name not in trial_names:
    # @thomasromeas : this happens a lot!
    print("This trial is not in the list of lengths of trials ('durée vidéos.xlsx'), what does it mean?")
    length_trial = np.nan
else:
    length_trial = black_screen_timing_data['Lenght before black screen (s)'][trial_names.index(this_trial_name)]

time_vector = np.array((test_data["time_stamp(ms)"] - test_data["time_stamp(ms)"][0]) / 1000)
length_trial = time_vector[-1] if np.isnan(length_trial) else length_trial

# cut the data after the black screen
black_screen_index = np.where(time_vector > length_trial)[0][0] if length_trial < time_vector[-1] else len(time_vector)
time_vector = time_vector[:black_screen_index]
test_data = test_data.iloc[:black_screen_index, :]

eye_origin = np.array(
    [test_data["gaze_origin_L.x(mm)"] / 1000, test_data["gaze_origin_L.y(mm)"] / 1000, test_data["gaze_origin_L.z(mm)"] / 1000])
eye_direction = np.array(
    [test_data["gaze_direct_L.x"], test_data["gaze_direct_L.y"], test_data["gaze_direct_L.z"]])
gaze_distance = np.ones(test_data["distance_C(mm)"].shape) * gaze_distance_fixed
# np.array(test_data["distance_C(mm)"] / 1000)  # It should be recorded, but the data seems invalid
helmet_rotation = np.array([test_data["helmet_rot_x"], test_data["helmet_rot_y"], test_data["helmet_rot_z"]])

if np.sum(test_data['eye_valid_L']) != 31 * len(test_data['eye_valid_L']) or np.sum(test_data['eye_valid_R']) != 31 * len(test_data['eye_valid_R']):
    plt.figure()
    plt.plot(test_data['eye_valid_L'] / 31, label='eye_valid_L')
    plt.plot(test_data['eye_valid_R'] / 31, label='eye_valid_R')
    plt.plot(test_data['openness_L'], label='openness_L')
    plt.plot(test_data['openness_R'], label='openness_R')
    plt.legend()
    plt.show()
    raise ValueError("The eye_valid data is not valid, please see graph for more information.")

def detect_invalid_data(time_vector, eye_direction):

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

def detect_blinks(time_vector, test_data, blink_threshold):
    blink_timing_right = np.where(test_data["openness_R"] < blink_threshold)[0]
    blink_timing_left = np.where(test_data["openness_L"] < blink_threshold)[0]
    blink_timing_both = np.where((test_data["openness_R"] < blink_threshold) & (test_data["openness_L"] < blink_threshold))[0]
    blink_timing_missmatch = np.where(((test_data["openness_R"] < blink_threshold) & (test_data["openness_L"] > blink_threshold)) | (
                (test_data["openness_R"] > blink_threshold) & (test_data["openness_L"] < blink_threshold)))[0]

    # plt.figure()
    # plt.plot(time_vector, test_data["openness_R"], color='m', label='Openness Right')
    # plt.plot(time_vector, test_data["openness_L"], color='c', label='Openness Left')
    # if len(blink_timing_right) > 0 or len(blink_timing_left > 0):
    #     for i in blink_timing_both:
    #         plt.axvspan(time_vector[i], time_vector[i + 1], color='g', alpha=0.5)
    #     for i in blink_timing_missmatch:
    #         plt.axvspan(time_vector[i], time_vector[i + 1], color='r', alpha=0.5)
    # plt.plot(np.array([0, time_vector[-1]]), np.array([blink_threshold, blink_threshold]), 'k--', label='Blink Threshold')
    # plt.legend()
    # plt.savefig("figures/blink_detection_test.png")
    # plt.show()

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
    for i_frame in range(1, gaze_direction.shape[1]):  # Skipping the first frame
        vector_before = gaze_direction[:, i_frame - 1]
        vector_after = gaze_direction[:, i_frame]
        gaze_angular_velocity_rad[i_frame] = np.arccos(np.dot(vector_before, vector_after) / np.linalg.norm(vector_before) / np.linalg.norm(vector_after)) / (time_vector[i_frame] - time_vector[i_frame - 1])

    threshold_5sigma = 5 * np.nanstd(gaze_angular_velocity_rad * 180 / np.pi)

    if PLOT_SACCADES_FLAG:
        plt.figure()
        plt.plot(time_vector, gaze_angular_velocity_rad * 180 / np.pi, label='Angular Velocity')
        plt.plot(np.array([time_vector[0], time_vector[-1]]), np.array([100, 100]), 'k--', label=r'Threshold 100$\^circ/s$')
        plt.plot(np.array([time_vector[0], time_vector[-1]]), np.array([threshold_5sigma, threshold_5sigma]), 'b--', label=r'Threshold 5$\sigma$')
        plt.legend()
        plt.savefig("figures/saccade_detection_test.png")
        plt.show()

    saccade_timing = np.where(gaze_angular_velocity_rad * 180 / np.pi > 100)[0]
    saccade_sequences = np.array_split(saccade_timing, np.flatnonzero(np.diff(saccade_timing) > 1) + 1)

    return saccade_sequences, gaze_direction


def sliding_window(time_vector, intersaccadic_sequences, gaze_direction):
    """
    Parameters t_wind (22000 micros), t_overlap (6000 micros) and eta_p (0.1) taken from the original paper.
    """
    # @thomasromeas : please confirm the values to use
    t_wind = 0.022 * 2  # Window size in ms
    t_overlap = 0.006  # Window overlap in ms
    eta_p = 0.01 * 2  # Threshold for the p-value of the Rayleigh test

    intersaccadic_window_idx = []
    for i_intersaccadic_gap in intersaccadic_sequences:

        # Index of the windows
        window_start_idx = i_intersaccadic_gap[0]
        window_end_idx = 0
        end_of_intersaccadic_gap = i_intersaccadic_gap[-1]
        while window_end_idx < end_of_intersaccadic_gap:
            window_end_idx = np.where(time_vector > time_vector[window_start_idx] + t_wind)[0]
            if len(window_end_idx) != 0 and window_end_idx[0] < end_of_intersaccadic_gap:
                window_end_idx = window_end_idx[0]
            else:
                # @thomasromeas : do we want to do this?
                window_end_idx = end_of_intersaccadic_gap
            if window_end_idx - window_start_idx > 1:
                intersaccadic_window_idx.append(np.arange(window_start_idx, window_end_idx))

            window_start_idx = np.where(time_vector < time_vector[window_end_idx] - t_overlap)[0][-1]

    coherent_windows = []
    incoherent_windows = []
    for i_window in intersaccadic_window_idx:
        if detect_directionality_coherence(gaze_direction[:, i_window], eta_p):
            coherent_windows += list(i_window)
        else:
            incoherent_windows += list(i_window)

    # @thomasromeas : Here I remove the duplicates,
    # but in the original article the mean p-value is used
    # (which does not make sens since the coherence is determined with the neighbors....)
    coherent_windows_tempo = []
    for x in coherent_windows:
        if x not in coherent_windows_tempo:
            coherent_windows_tempo.append(x)
    coherent_windows_array = np.array(coherent_windows_tempo)
    incorherent_windows_tempo = []
    for x in incoherent_windows:
        if x not in incorherent_windows_tempo:
            incorherent_windows_tempo.append(x)
    incoherent_windows_array = np.array(incorherent_windows_tempo)
    intersaccadic_coherent_sequences = np.array_split(coherent_windows_array, np.flatnonzero(np.diff(coherent_windows_array) > 1) + 1)
    intersaccadic_incoherent_sequences = np.array_split(incoherent_windows_array, np.flatnonzero(np.diff(incoherent_windows_array) > 1) + 1)

    intersaccadic_gouped_sequences = intersaccadic_coherent_sequences[:]
    for i_sequence in intersaccadic_incoherent_sequences:
        for j in range(len(intersaccadic_gouped_sequences)):
            if i_sequence[0] > intersaccadic_gouped_sequences[j][0]:
                if j == len(intersaccadic_gouped_sequences) - 1:
                    intersaccadic_gouped_sequences.append(i_sequence)
                elif i_sequence[0] > intersaccadic_gouped_sequences[j][0] and i_sequence[0] < intersaccadic_gouped_sequences[j+1][0]:
                    intersaccadic_gouped_sequences.insert(j+1, i_sequence)

    return intersaccadic_gouped_sequences, intersaccadic_coherent_sequences, intersaccadic_incoherent_sequences


def detect_directionality_coherence(gaze_direction, eta_p):

    gaze_displacement_rad = gaze_direction[:, 1:] - gaze_direction[:, :-1]

    horizontal_vector_on_the_sphere = np.zeros(gaze_displacement_rad.shape)
    horizontal_vector_on_the_sphere[:2, :] = gaze_displacement_rad[:2, :]
    alpha = np.zeros((horizontal_vector_on_the_sphere.shape[1], ))
    for i_frame in range(gaze_direction.shape[1] - 1):
        alpha[i_frame] = np.arccos(np.dot(gaze_displacement_rad[:, i_frame], horizontal_vector_on_the_sphere[:, i_frame]) / np.linalg.norm(gaze_displacement_rad[:, i_frame]) / np.linalg.norm(horizontal_vector_on_the_sphere[:, i_frame]))

    # Test that the gaze diplacement and orientation are coherent inside the window
    z_value_alpha, p_value_alpha = pg.circ_rayleigh(alpha)
    if p_value_alpha <= eta_p:
        return False  # The data is directionnaly incoherent (uniformally distributed)
    else:
        return True  # The data is directionnaly coherent (the data goes in a particular direction)



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
    mean_radius_range_gaze_direction = np.sqrt((np.max(gaze_direction[0]) - np.min(gaze_direction[0])) ** 2 +
                                               (np.max(gaze_direction[1]) - np.min(gaze_direction[1])) ** 2 +
                                               (np.max(gaze_direction[2]) - np.min(gaze_direction[2])) ** 2)

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
    # @thomasromeas : please confirm the values to use
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
    uncertain_sequences = np.array_split(uncertain_timing, np.flatnonzero(np.diff(uncertain_timing) > 1) + 1)
    for i_sequence, sequence in enumerate(uncertain_sequences):
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
                        uncertain_mean = np.mean(gaze_direction[sequence])
                        current_mean = np.mean(gaze_direction[intersaccadic_gouped_sequences[current_i_sequence]])
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
                        uncertain_mean = np.mean(gaze_direction[sequence])
                        current_mean = np.mean(gaze_direction[intersaccadic_gouped_sequences[current_i_sequence]])
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
                for i_uncertain_sequences, uncertain in enumerate(uncertain_sequences):
                    if before_idx in uncertain:
                        uncertain_sequences.pop(i_uncertain_sequences)
                    if after_idx in uncertain:
                        uncertain_sequences.pop(i_uncertain_sequences)
        else:
            # Fixation like segment
            if criteria_4:
                smooth_pursuite_timing += list(sequence)
            else:
                fixation_timing += list(sequence)
            uncertain_sequences.pop(i_sequence)

    fixation_sequences = np.array_split(np.array(fixation_timing), np.flatnonzero(np.diff(np.array(fixation_timing)) > 1) + 1)
    smooth_pursuite_sequences = np.array_split(np.array(smooth_pursuite_timing), np.flatnonzero(np.diff(np.array(smooth_pursuite_timing)) > 1) + 1)
    # @thomasromeas : Do we want the last fixation or the longest or something else?
    quiet_eye_sequences = fixation_sequences[-1]
    longest_fixation_sequence = fixation_sequences[np.argmax([len(fixation) for fixation in fixation_sequences])]

    return fixation_sequences, smooth_pursuite_sequences, quiet_eye_sequences, uncertain_sequences


# Remove invalid sequences where the eye-tracker did not detect any eye movement
invalid_sequences = detect_invalid_data(time_vector, eye_direction)
# Remove blinks
blink_sequences = detect_blinks(time_vector, test_data, blink_threshold)

if PLOT_BAD_DATA_FLAG:
    # Plot the timing of the bad data
    plt.figure()
    plt.plot(time_vector, eye_direction[0], label='eye_direction_x')
    plt.plot(time_vector, eye_direction[1], label='eye_direction_y')
    plt.plot(time_vector, eye_direction[2], label='eye_direction_z')
    label_flag = True
    for i in invalid_sequences:
        if label_flag:
            plt.axvspan(time_vector[i[0]], time_vector[i[-1]], color='r', alpha=0.5, label='Invalid Sequences')
            label_flag = False
        else:
            plt.axvspan(time_vector[i[0]], time_vector[i[-1]], color='r', alpha=0.5)
    label_flag = True
    for i in blink_sequences:
        if len(i) < 1:
            continue
        if label_flag:
            plt.axvspan(time_vector[i[0]], time_vector[i[-1]], color='g', alpha=0.5, label='Blink Sequences')
            label_flag = False
        else:
            plt.axvspan(time_vector[i[0]], time_vector[i[-1]], color='g', alpha=0.5)
    plt.legend()
    plt.savefig("figures/gaze_classification_bad.png")
    plt.show()

# Remove invalid sequences from the variable vectors
for invalid in invalid_sequences:
    eye_origin[:, invalid] = np.nan
    eye_direction[:, invalid] = np.nan
    gaze_distance[invalid] = np.nan

# Remove blink sequences from the variable vectors
for blink in blink_sequences:
    eye_origin[:, blink] = np.nan
    eye_direction[:, blink] = np.nan
    gaze_distance[blink] = np.nan


# Detect saccades
saccade_sequences, gaze_direction = detect_saccades(time_vector, eye_direction, helmet_rotation)

# Detect fixations
intersaccadic_interval = np.zeros((len(time_vector), ))
all_index = np.arange(len(time_vector))
for i in all_index:
    i_in_saccades = True if any(i in sequence for sequence in saccade_sequences) else False
    i_in_blinks = True if any(i in sequence for sequence in blink_sequences) else False
    i_in_invalid = True if any(i in sequence for sequence in invalid_sequences) else False
    if i_in_saccades or i_in_blinks or i_in_invalid:
        continue
    else:
        intersaccadic_interval[i] = 1
intersaccadic_timing = np.where(intersaccadic_interval == 1)[0]
intersaccadic_sequences = np.array_split(intersaccadic_timing, np.flatnonzero(np.diff(intersaccadic_timing) > 1) + 1)
intersaccadic_gouped_sequences, intersaccadic_coherent_sequences, intersaccadic_incoherent_sequences = sliding_window(time_vector, intersaccadic_sequences, gaze_direction)
fixation_sequences, smooth_pursuite_sequences, quiet_eye_sequences, uncertain_sequences = detect_fixations_and_smooth_pursuite(time_vector, gaze_direction, intersaccadic_gouped_sequences)

# Plot the classification of gaze data
plt.figure(figsize=(15, 15))
plt.plot(time_vector, gaze_direction[0], 'k', label='Gaze x (head + eye)')
plt.plot(time_vector, gaze_direction[1], 'k', label='Gaze y (head + eye)')
plt.plot(time_vector, gaze_direction[2], 'k', label='Gaze z (head + eye)')
label_flag = True
for i in invalid_sequences:
    if label_flag:
        plt.axvspan(time_vector[i[0]], time_vector[i[-1]], edgecolor=None, color='tab:red', alpha=0.5, label='Invalid Sequences')
        label_flag = False
    else:
        plt.axvspan(time_vector[i[0]], time_vector[i[-1]], edgecolor=None, color='tab:red', alpha=0.5)
label_flag = True
for i in blink_sequences:
    if len(i) < 1:
        continue
    if label_flag:
        plt.axvspan(time_vector[i[0]], time_vector[i[-1]], edgecolor=None, color='tab:green', alpha=0.5, label='Blink Sequences')
        label_flag = False
    else:
        plt.axvspan(time_vector[i[0]], time_vector[i[-1]], edgecolor=None, color='tab:green', alpha=0.5)
label_flag = True
for i in saccade_sequences:
    if len(i) < 1:
        continue
    if label_flag:
        plt.axvspan(time_vector[i[0]], time_vector[i[-1]], edgecolor=None, color='tab:blue', alpha=0.5, label='Saccade Sequences')
        label_flag = False
    else:
        plt.axvspan(time_vector[i[0]], time_vector[i[-1]], edgecolor=None, color='tab:blue', alpha=0.5)
label_flag = True
for i in fixation_sequences:
    if len(i) < 1:
        continue
    if label_flag:
        plt.axvspan(time_vector[i[0]], time_vector[i[-1]], edgecolor=None, color='tab:purple', alpha=0.5, label='Fixation Sequences')
        label_flag = False
    else:
        plt.axvspan(time_vector[i[0]], time_vector[i[-1]], edgecolor=None, color='tab:purple', alpha=0.5)
label_flag = True
for i in smooth_pursuite_sequences:
    if len(i) < 1:
        continue
    if label_flag:
        plt.axvspan(time_vector[i[0]], time_vector[i[-1]], edgecolor=None, color='tab:orange', alpha=0.5, label='Smooth Pursuite Sequences')
        label_flag = False
    else:
        plt.axvspan(time_vector[i[0]], time_vector[i[-1]], edgecolor=None, color='tab:orange', alpha=0.5)
for i in quiet_eye_sequences:
    if len(i) < 1:
        continue
    plt.axvspan(time_vector[i[0]], time_vector[i[-1]], edgecolor=None, color='tab:pink', alpha=0.5, label='Quiet Eye Sequence')
plt.legend()
plt.savefig("figures/gaze_classification_test.png")
plt.show()

# Intermediary metrics
fixation_duration = []
for i in fixation_sequences:
    fixation_duration.append(time_vector[i[-1]] - time_vector[i[0]])
total_fixation_duration = np.sum(np.array(fixation_duration))
smooth_pursuite_duration = []
for i in smooth_pursuite_sequences:
    smooth_pursuite_duration.append(time_vector[i[-1]] - time_vector[i[0]])
total_smooth_pursuite_duration = np.sum(np.array(smooth_pursuite_duration))
quiet_eye_duration = time_vector[quiet_eye_sequences[0][-1]] - time_vector[quiet_eye_sequences[0][0]]
blink_duration = []
for i in blink_sequences:
    blink_duration.append(time_vector[i[-1]] - time_vector[i[0]])
total_blink_duration = np.sum(np.array(blink_duration))

# Metrics
# @thomasromeas: Please confirm which metrics we want to use
nb_fixations = len(fixation_sequences)
mean_fixation_duration = np.mean(np.array(fixation_duration))
search_rate = nb_fixations / mean_fixation_duration
nb_blinks = len(blink_sequences)
nb_saccades = len(saccade_sequences)
# mean_saccade_amplitude?
# max_smooth_pursuite_trajectory?
fixation_ratio = total_fixation_duration / time_vector[-1]
smooth_pursuite_ratio = total_smooth_pursuite_duration / time_vector[-1]
quiet_eye_ratio = quiet_eye_duration / time_vector[-1]
blinking_ration = total_blink_duration / time_vector[-1]

