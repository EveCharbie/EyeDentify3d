
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython import embed
import pandas as pd


# Define the path to the data
datapath = "AllData/"
test_path = "TESTNA01/VideoListOne/20231030161004_eye_tracking_VideoListOne_TESTNA01_Demo_Mode_2D_Pen1_000.csv"

test_data = pd.read_csv(datapath+test_path, sep=';')

fig, axs = plt.subplots(3, 1)
axs[0].plot(test_data["time_stamp(ms)"], test_data["helmet_rot_x"], label='helmet_rot_x')
axs[1].plot(test_data["time_stamp(ms)"], test_data["helmet_rot_y"], label='helmet_rot_y')
axs[2].plot(test_data["time_stamp(ms)"], test_data["helmet_rot_z"], label='helmet_rot_z')
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
