
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython import embed
import pandas as pd
import biorbd


# Define the path to the data
datapath = "AllData/"
test_path = "TESTNA01/VideoListOne/20231030161004_eye_tracking_VideoListOne_TESTNA01_Demo_Mode_2D_Pen1_000.csv"

test_data = pd.read_csv(datapath+test_path, sep=';')

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
gaze_direction is a unit vector
gaze_distance in meters
gaze_endpoint in meters
"""

# Parameters to define ---------------------------------------
blink_threshold = 0.5
gaze_distance_fixed = 7
PLOT_BAD_DATA_FLAG = True
# ------------------------------------------------------------

time_vector = np.array((test_data["time_stamp(ms)"] - test_data["time_stamp(ms)"][0]) / 1000)
gaze_origin = np.array(
    [test_data["gaze_origin_L.x(mm)"] / 1000, test_data["gaze_origin_L.y(mm)"] / 1000, test_data["gaze_origin_L.z(mm)"] / 1000])
gaze_direction = np.array(
    [test_data["gaze_direct_L.x"], test_data["gaze_direct_L.y"], test_data["gaze_direct_L.z"]])
gaze_distance = np.array(test_data["distance_C(mm)"] / 1000)
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

def detect_valid_data(time_vector, gaze_direction):

    # Find where the data does not change
    zero_diffs_x = np.where(np.abs(gaze_direction[0, 1:] - gaze_direction[0, :-1]) < 1e-8)[0]
    zero_diffs_y = np.where(np.abs(gaze_direction[1, 1:] - gaze_direction[1, :-1]) < 1e-8)[0]
    zero_diffs_z = np.where(np.abs(gaze_direction[2, 1:] - gaze_direction[2, :-1]) < 1e-8)[0]

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

def detect_saccades(time_vector, gaze_origin, gaze_direction, gaze_distance, helmet_rotation):
    """
    I arbitrarily decided that the system origin is positioned 10 cm away from the neck joint center.
    """
    neck_system_origin_shift = np.array([0, 0, 0.1])
    gaze_endpoint_system_origin = gaze_origin + gaze_direction * gaze_distance
    helmet_rotation_in_rad = helmet_rotation * np.pi / 180

    gaze_endpoint_word_origin = np.zeros(gaze_endpoint_system_origin.shape)
    for i_frame in range(helmet_rotation_in_rad.shape[1]):
        rotation_matrix = biorbd.Rotation.fromEulerAngles(helmet_rotation_in_rad[:, i_frame], 'xyz').to_array()
        gaze_endpoint_word_origin[:, i_frame] = rotation_matrix @ (neck_system_origin_shift + gaze_endpoint_system_origin[:, i_frame])

    plt.figure()
    plt.plot(time_vector, gaze_distance)
    plt.savefig("figures/distance.png")
    plt.show()

    gaze_endpoint_angular_velocity_rad = np.zeros((gaze_endpoint_word_origin.shape[1], ))
    for i_frame in range(1, gaze_endpoint_word_origin.shape[1]):  # Skipping the first frame
        vector_before = gaze_endpoint_word_origin[:, i_frame - 1]
        vector_after = gaze_endpoint_word_origin[:, i_frame]
        gaze_endpoint_angular_velocity_rad[i_frame] = np.arccos(np.dot(vector_before, vector_after) / np.linalg.norm(vector_before) / np.linalg.norm(vector_after)) / (time_vector[i_frame] - time_vector[i_frame - 1])

    threshold_5sigma = 5 * np.nanstd(gaze_endpoint_angular_velocity_rad * 180 / np.pi)
    plt.figure()
    plt.plot(time_vector, gaze_endpoint_angular_velocity_rad * 180 / np.pi, label='Angular Velocity')
    plt.plot(np.array([time_vector[0], time_vector[-1]]), np.array([100, 100]), 'k--', label=r'Threshold 100$\^circ/s$')
    plt.plot(np.array([time_vector[0], time_vector[-1]]), np.array([threshold_5sigma, threshold_5sigma]), 'b--', label=r'Threshold 5$\sigma$')
    plt.legend()
    plt.savefig("figures/saccade_detection_test.png")
    plt.show()

    saccade_timing = np.where(gaze_endpoint_angular_velocity_rad * 180 / np.pi > threshold_5sigma)[0]
    saccade_sequences = np.array_split(saccade_timing, np.flatnonzero(np.diff(saccade_timing) > 1) + 1)

    return saccade_sequences, gaze_endpoint_word_origin


def detect_fixations(test_data):

    gaze_endpoint = gaze_origin + gaze_direction * gaze_distance

    gaze_displacement_angle = np.zeros(len(gaze_origin[0]))
    for i in range(len(gaze_origin[0])):
        gaze_displacement_angle[i] = np.arccos(np.dot(gaze_direction[:, i], gaze_direction[:, i+1]) / (np.linalg.norm(gaze_direction[:, i]) * np.linalg.norm(gaze_direction[:, i+1])))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(gaze_origin[0, :], gaze_origin[1, :], gaze_origin[2, :], ".g", label='gaze_origin')
    ax.plot(gaze_endpoint[0, :], gaze_endpoint[1, :], gaze_endpoint[2, :], ".r", label='gaze_endpoint')
    plt.savefig("figures/gaze_3D_test.png")
    plt.show()
    return

# Remove invalid sequences where the eye-tracker did not detect any eye movement
invalid_sequences = detect_valid_data(time_vector, gaze_direction)
# Remove blinks
blink_sequences = detect_blinks(time_vector, test_data, blink_threshold)

if PLOT_BAD_DATA_FLAG:
    # Plot the timing of the bad data
    plt.figure()
    plt.plot(time_vector, gaze_direction[0], label='gaze_direction_x')
    plt.plot(time_vector, gaze_direction[1], label='gaze_direction_y')
    plt.plot(time_vector, gaze_direction[2], label='gaze_direction_z')
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
    gaze_origin[:, invalid] = np.nan
    gaze_direction[:, invalid] = np.nan
    gaze_distance[invalid] = np.nan

# Remove blink sequences from the variable vectors
for blink in blink_sequences:
    gaze_origin[:, blink] = np.nan
    gaze_direction[:, blink] = np.nan
    gaze_distance[blink] = np.nan


# Detect saccades
saccade_sequences, gaze_endpoint_word_origin = detect_saccades(time_vector, gaze_origin, gaze_direction, gaze_distance, helmet_rotation)


# Plot the classification of gaze data
plt.figure()
plt.plot(time_vector, gaze_direction[0], label='gaze_direction_x')
plt.plot(time_vector, gaze_direction[1], label='gaze_direction_y')
plt.plot(time_vector, gaze_direction[2], label='gaze_direction_z')
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
label_flag = True
for i in saccade_sequences:
    if len(i) < 1:
        continue
    if label_flag:
        plt.axvspan(time_vector[i[0]], time_vector[i[-1]], color='b', alpha=0.5, label='Saccade Sequences')
        label_flag = False
    else:
        plt.axvspan(time_vector[i[0]], time_vector[i[-1]], color='b', alpha=0.5)

plt.legend()
plt.savefig("figures/gaze_classification_test.png")
plt.show()







fig, axs = plt.subplots(3, 1)
axs[0].plot(test_data["time_stamp(ms)"], np.unwrap(test_data["helmet_rot_x"]), label='helmet_rot_x')
axs[1].plot(test_data["time_stamp(ms)"], np.unwrap(test_data["helmet_rot_y"]), label='helmet_rot_y')
axs[2].plot(test_data["time_stamp(ms)"], np.unwrap(test_data["helmet_rot_z"]), label='helmet_rot_z')
plt.savefig("figures/head_rotation_test.png")
plt.show()

fig, axs = plt.subplots(3, 1)
axs[0].plot(test_data["time_stamp(ms)"], test_data["gaze_direct_L.x"], label='gaze_direct_L.x')
axs[1].plot(test_data["time_stamp(ms)"], test_data["gaze_direct_L.y"], label='gaze_direct_L.y')
axs[2].plot(test_data["time_stamp(ms)"], test_data["gaze_direct_L.z"], label='gaze_direct_L.z')
plt.savefig("figures/gaze_direction_test.png")
plt.show()


gaze_origin = np.array([test_data["gaze_origin_L.x(mm)"], test_data["gaze_origin_L.y(mm)"], test_data["gaze_origin_L.z(mm)"]])
gaze_direction = np.array([test_data["gaze_direct_L.x"], test_data["gaze_direct_L.y"], test_data["gaze_direct_L.z"]])
gaze_distance = np.array(test_data["distance_C(mm)"])
gaze_endpoint = gaze_origin + gaze_direction * gaze_distance
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(gaze_origin[0, :], gaze_origin[1, :], gaze_origin[2, :], ".g", label='gaze_origin')
ax.plot(gaze_endpoint[0, :], gaze_endpoint[1, :], gaze_endpoint[2, :], ".r", label='gaze_endpoint')
plt.savefig("figures/gaze_3D_test.png")
plt.show()







# Get the list of CSV files
files_names = list(datapath.rglob("*.csv"))
nb_files = len(files_names)

# Create a list for data names
data_names = [file.stem for file in files_names]

# Read all datasets into a list
dataset_list = [pd.read_csv(file, sep=';') for file in files_names]

# Add a column with the file name
for i, df in enumerate(dataset_list):
    df['idname'] = data_names[i]

# Combine all datasets into one DataFrame
dataout = pd.concat(dataset_list, ignore_index=True)

# Remove spaces in the 'idname' column
dataout['idname'] = dataout['idname'].str.replace(" ", "")

# Create new columns from 'idname'
dataout['idname1'] = dataout['idname'].str.replace("_", " ")
dataout['participant'] = dataout['idname1'].apply(lambda x: x.split()[6])
dataout['mode'] = dataout['idname1'].apply(lambda x: x.split()[9])
dataout['Video.Name'] = dataout['idname1'].apply(lambda x: x.split()[10])
dataout['Moment'] = dataout['idname1'].apply(lambda x: x.split()[7])
dataout['Groupe'] = dataout['participant'].str[4:6]

# Filter for 'Experiment' mode
data = dataout[dataout['Moment'] == 'Experiment']

# Read MinMax.xlsx
min_max_path = "Results/MinMax.xlsx"
MinMax1 = pd.read_excel(min_max_path)

# Merge data with MinMax
data = pd.merge(data, MinMax1, on=["Video.Name", "mode"])

# Filter based on time.UnityVideoPlayer
data['time.UnityVideoPlayer.'] = pd.to_numeric(data['time.UnityVideoPlayer.'], errors='coerce')
data = data[data['time.UnityVideoPlayer.'] <= data['DureeVideo']]

# Calculate velocity
data['combinaisonDGx'] = data[['gaze_origin_L.x.mm.', 'gaze_origin_R.x.mm.']].mean(axis=1)
data['combinaisonDGy'] = data[['gaze_origin_L.y.mm.', 'gaze_origin_R.y.mm.']].mean(axis=1)
data['combinaisonDGz'] = data[['gaze_origin_L.z.mm.', 'gaze_origin_R.z.mm.']].mean(axis=1)

# Filter for '2D' and '360VR' modes
event2D = data[data['mode'] == '2D']
event360VR = data[data['mode'] == '360VR']

# Function to calculate velocity and filter data
def calculate_velocity(data):
    data['distancex'] = data['combinaisonDGx'].diff()
    data['distancey'] = data['combinaisonDGy'].diff()
    data['distance'] = np.sqrt(data['distancex']**2 + data['distancey']**2)
    data['temps0'] = data['time_stamp.ms.'].diff()
    data['temps'] = data['temps0'] * 10**-3
    data['velocity'] = data['distance'] / data['temps']
    data = data[(data['temps0'] >= 8) & (data['temps0'] <= 9)]
    data = data[data['velocity'] > 0]
    data = data[(data['pupil_diameter_L.mm.'] >= 0) & (data['pupil_diameter_R.mm.'] >= 0)]
    return data

data_velocity2D = calculate_velocity(event2D)
data_velocity360VR = calculate_velocity(event360VR)

# Function to calculate fixations and saccades
def calculate_fixations(data_velocity):
    data_velocity['seuil'] = np.where(data_velocity['velocity'] > 5 * data_velocity['velocity'].median(), 5 * data_velocity['velocity'].median(), 0)
    data_velocity['duree'] = np.where(data_velocity['seuil'] == 0, 1, 0)
    data_velocity['duree2'] = data_velocity.groupby((data_velocity['duree'] == 0).cumsum()).cumcount() + 1
    data_velocity['dureefix'] = data_velocity['duree2'] * 8.33
    data_velocity_fixations = data_velocity[data_velocity['dureefix'] > 0]
    data_velocity_fixations = data_velocity_fixations[data_velocity_fixations['dureefix'] >= 100]
    data_velocity['duree3'] = np.where(data_velocity['duree'] == 0, 1, 0)
    data_velocity['duree4'] = data_velocity.groupby((data_velocity['duree3'] == 0).cumsum()).cumcount() + 1
    data_velocity['dureesacc'] = data_velocity['duree4'] * 8.33
    data_velocity_saccades = data_velocity[data_velocity['dureesacc'] > 0]
    data_velocity_saccades = data_velocity_saccades[(data_velocity_saccades['dureesacc'] <= 50) & (data_velocity_saccades['dureesacc'] >= 20)]
    return data_velocity_fixations, data_velocity_saccades

data_velocity2D_fixations, data_velocity2D_saccades = calculate_fixations(data_velocity2D)
data_velocity360VR_fixations, data_velocity360VR_saccades = calculate_fixations(data_velocity360VR)

# Normalize data based on the duration of each video sequence
def normalize_data(data_fixations, data_saccades, MinMax):
    fixation_mean = data_fixations.groupby(['participant', 'mode', 'Video.Name'])['dureefix'].mean().reset_index()
    fixation_nb = data_fixations.groupby(['participant', 'mode', 'Video.Name']).size().reset_index(name='nbfixation')
    saccade_nb = data_saccades.groupby(['participant', 'mode', 'Video.Name']).size().reset_index(name='nbsaccade')
    fixation_nb = pd.merge(fixation_nb, MinMax, on=['Video.Name', 'mode'])
    fixation_nb['Nbfix_normalized'] = fixation_nb['nbfixation'] / fixation_nb['DureeVideo']
    saccade_nb = pd.merge(saccade_nb, MinMax, on=['Video.Name', 'mode'])
    saccade_nb['Nbsaccade_normalized'] = saccade_nb['nbsaccade'] / saccade_nb['DureeVideo']
    return fixation_mean, fixation_nb, saccade_nb

fixation_mean_2D, fixation_nb_2D, saccade_nb_2D = normalize_data(data_velocity2D_fixations, data_velocity2D_saccades, MinMax1)
fixation_mean_360VR, fixation_nb_360VR, saccade_nb_360VR = normalize_data(data_velocity360VR_fixations, data_velocity360VR_saccades, MinMax1)

# Further processing for fixations, saccades, and search rate
def process_fixations(fixation_mean, fixation_nb, saccade_nb):
    nb_fixation_participant = fixation_nb.groupby(['participant', 'mode'])['Nbfix_normalized'].mean().reset_index()
    duree_fixation_participant = fixation_mean.groupby(['participant', 'mode'])['dureefix'].mean().reset_index()
    search_rate = pd.merge(nb_fixation_participant, duree_fixation_participant, on=['participant', 'mode'])
    search_rate['searchrate'] = search_rate['Nbfix_normalized'] / search_rate['dureefix']
    nb_saccade_participant = saccade_nb.groupby(['participant', 'mode'])['Nbsaccade_normalized'].mean().reset_index()
    return nb_fixation_participant, duree_fixation_participant, search_rate, nb_saccade_participant

nb_fixation2D_participant, duree_fixation2D_participant, search_rate_2D, nb_saccade2D_participant = process_fixations(fixation_mean_2D, fixation_nb_2D, saccade_nb_2D)
nb_fixation360VR_participant, duree_fixation360VR_participant, search_rate_360VR, nb_saccade360VR_participant = process_fixations(fixation_mean_360VR, fixation_nb_360VR, saccade_nb_360VR)

# Combine data for Excel output
Nbfixation = pd.concat([nb_fixation2D_participant, nb_fixation360VR_participant])
Dureefixation = pd.concat([duree_fixation2D_participant, duree_fixation360VR_participant])
Searchrate = pd.concat([search_rate_2D, search_rate_360VR])
Nbsaccade = pd.concat([nb_saccade2D_participant, nb_saccade360VR_participant])

GB = Nbfixation.merge(Dureefixation).merge(Searchrate).merge(Nbsaccade)
GB.to_excel("/Users/MildredTaupin/Desktop/Pro/PostDoc/Canada/Code_Bishop_Basket/Results/GB.xlsx", index=False)

# Process head rotation
dataout['helmet_rot_x'] = pd.to_numeric(dataout['helmet_rot_x'], errors='coerce')
dataout['helmet_rot_y'] = pd.to_numeric(dataout['helmet_rot_y'], errors='coerce')
dataout['helmet_rot_z'] = pd.to_numeric(dataout['helmet_rot_z'], errors='coerce')

dataout['helmet_rot_x'] = np.where(dataout['helmet_rot_x'] > 180, dataout['helmet_rot_x'] - 360, dataout['helmet_rot_x'])
dataout['helmet_rot_y'] = np.where(dataout['helmet_rot_y'] > 180, dataout['helmet_rot_y'] - 360, dataout['helmet_rot_y'])
dataout['helmet_rot_z'] = np.where(dataout['helmet_rot_z'] > 180, dataout['helmet_rot_z'] - 360, dataout['helmet_rot_z'])

dataout['HRot_x'] = dataout.groupby('Video.Name')['helmet_rot_x'].diff().abs()
dataout['HRot_y'] = dataout.groupby('Video.Name')['helmet_rot_y'].diff().abs()
dataout['HRot_z'] = dataout.groupby('Video.Name')['helmet_rot_z'].diff().abs()

HeadRot_Results = dataout.groupby(['Video.Name', 'participant', 'mode']).agg({'HRot_x': 'sum', 'HRot_y': 'sum', 'HRot_z': 'sum'}).reset_index()

HeadRot_Results = pd.merge(HeadRot_Results, MinMax1, on=['Video.Name', 'mode'])
HeadRot_Results['HRot_x_normalized'] = HeadRot_Results['HRot_x'] / HeadRot_Results['DureeVideo']
HeadRot_Results['HRot_y_normalized'] = HeadRot_Results['HRot_y'] / HeadRot_Results['DureeVideo']
HeadRot_Results['HRot_z_normalized'] = HeadRot_Results['HRot_z'] / HeadRot_Results['DureeVideo']

HeadRot_Final = HeadRot_Results.groupby(['participant', 'mode']).agg({'HRot_x_normalized': 'mean', 'HRot_y_normalized': 'mean', 'HRot_z_normalized': 'mean'}).reset_index()
HeadRot_Final.to_excel("/Users/MildredTaupin/Desktop/Pro/PostDoc/Canada/Code_Bishop_Basket/Results/TeteRotation.xlsx", index=False)
