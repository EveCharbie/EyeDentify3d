"""
This example shows how to process all data from a study.
For each trial from each participant, the data is loaded, the gaze behaviors are detected, and the results are appended
to a data frame. This data frame is then exported to a CSV file for future statistical analysis.
In this example, we are using an HTC Vive Pro eye-tracker and are dividing each trial in two sub trials (pre-cue and
post-cue).
"""

import os
from copy import deepcopy
from pathlib import Path
import pandas as pd
from eyedentify3d import (
    TimeRange,
    HtcViveProData,
    ErrorType,
    GazeBehaviorIdentifier,
)


def perform_one_file(
    data_file_path: str,
    min_time: float,
    max_time: float,
    participant_id: str,
    trial_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # Load the data
    time_range = TimeRange(min_time=min_time, max_time=max_time)
    data_object = HtcViveProData(data_file_path, error_type=ErrorType.FILE, time_range=time_range)

    # Check that the trial is valid (either _validity_flag=False or data_object.time_vector is None can be used)
    if data_object.time_vector is None:
        # This trial was problematic and an error was raised
        return None, None, None

    # Create a GazeBehaviorIdentifier object
    gaze_behavior_identifier = GazeBehaviorIdentifier(data_object)

    # Detect gaze behaviors (in order)
    gaze_behavior_identifier.detect_blink_sequences(eye_openness_threshold=0.5)
    gaze_behavior_identifier.detect_invalid_sequences()
    gaze_behavior_identifier.detect_saccade_sequences(
        min_acceleration_threshold=4000,
        velocity_window_size=0.52,
        velocity_factor=5.0,
    )
    gaze_behavior_identifier.detect_visual_scanning_sequences(
        min_velocity_threshold=100,
        minimal_duration=0.040,  # 5 frames
    )
    gaze_behavior_identifier.detect_fixation_and_smooth_pursuit_sequences(
        inter_saccade_minimal_duration=0.04,  # 5 frames
        fixation_minimal_duration=0.1,  # 100 ms
        smooth_pursuit_minimal_duration=0.1,  # 100 ms
        window_duration=0.022 * 5,
        window_overlap=0.006 * 5,
        eta_p=0.001,
        eta_d=0.45,
        eta_cd=0.5,
        eta_pd=0.5,
        eta_max_fixation=3,
        eta_min_smooth_pursuit=2,
        phi=45,
    )
    gaze_behavior_identifier.finalize()  # This is mandatory

    # See other examples to see how to plot and animate the results

    # Split the gaze behavior identifier into pre-cue and post-cue
    time_between_cue_and_trial_end = 2  # seconds
    split_timings = [data_object.time_vector[-1] - time_between_cue_and_trial_end]
    gaze_behavior_identifiers = gaze_behavior_identifier.split(split_timings, event_at_split_handling=ErrorType.FILE)
    gaze_behavior_identifier_pre_cue = gaze_behavior_identifiers[0]
    gaze_behavior_identifier_post_cue = gaze_behavior_identifiers[1]

    # Get the results for the full trial, pre-cue and post-cue as dataframes
    # You can add any additional information to the result dataframe by passing keyword arguments. For example,
    # here we add the participant_id and trial_id to each dataframe.
    full_trial_results = gaze_behavior_identifier.get_results(participant_id=participant_id, trial_id=trial_id)
    pre_cue_results = gaze_behavior_identifier_pre_cue.get_results(participant_id=participant_id, trial_id=trial_id)
    post_cue_results = gaze_behavior_identifier_post_cue.get_results(participant_id=participant_id, trial_id=trial_id)

    return full_trial_results, pre_cue_results, post_cue_results


def perform_all_files():

    # Define the path to the data
    current_path_file = Path(__file__).parent
    data_path = f"{current_path_file}/../examples/data/HTC_Vive_Pro/"

    # Selecting error_type=ErrorType.FILE, adds logs to bad_data_file.txt for all the files that could not be processed
    # and the event happening at the cut-off time when using gaze_behavior_identifier.split.
    # So we better make sure the file does not exist before starting the processing
    if os.path.exists("bad_data_file.txt"):
        os.remove("bad_data_file.txt")

    # For each trial, define the interesting time range
    time_range = {
        "TESTNA01_2D_Fist3": [0, 7.180],
        "TESTNA01_360VR_Fist3": [0, 7.180],
        "TESTNA05_2D_Spread7": [0, 5.060],
        "TESTNA05_360VR_Spread7": [0, 5.060],
        "TESTNA15_2D_Pen3": [0, 4.230],
        "TESTNA15_360VR_Pen3": [0, 4.230],
        "TESTVA03_2D_Spread9": [0, 6.150],  # Bad data (no data)
        "TESTNA10_360VR_Fist3": [0, 7.180],  # Bad data (more than 50% of the data is invalid)
    }

    # Perform the data treatment for each file in the folder
    full_trial_list_of_dataframe = []
    pre_cue_list_of_dataframe = []
    post_cue_list_of_dataframe = []
    for path, folders, files in os.walk(data_path):
        for file_name in files:
            if file_name.endswith(".csv"):

                file = data_path + file_name
                name = file_name.replace(".csv", "")

                # Get the participant ID and trial ID from the file name (optional step, but useful for statistics)
                participant_id = name.split("_")[0]  # e.g., "TESTNA01"
                trial_id = "_".join(name.split("_")[1:])  # e.g., "2D_Fist3"

                min_time, max_time = time_range[name]
                full_trial_results, pre_cue_results, post_cue_results = perform_one_file(
                    file,
                    min_time,
                    max_time,
                    participant_id,
                    trial_id,
                )

                print(f"Processed file: {file_name}")
                if full_trial_results is not None:
                    full_trial_list_of_dataframe += [full_trial_results]
                    pre_cue_list_of_dataframe += [pre_cue_results]
                    post_cue_list_of_dataframe += [post_cue_results]

    # Save the complete data frame containing the data of each trial from each participant
    if not os.path.exists(f"{current_path_file}/../examples/results"):
        os.makedirs(f"{current_path_file}/../examples/results")
    if not os.path.exists(f"{current_path_file}/../examples/results/HTC_Vive_Pro"):
        os.makedirs(f"{current_path_file}/../examples/results/HTC_Vive_Pro")

    full_trial_dataframe = pd.concat(full_trial_list_of_dataframe, ignore_index=True)
    full_trial_dataframe.to_csv(
        f"{current_path_file}/../examples/results/HTC_Vive_Pro/full_trial_results.csv", index=False
    )

    pre_cue_dataframe = pd.concat(pre_cue_list_of_dataframe, ignore_index=True)
    pre_cue_dataframe.to_csv(f"{current_path_file}/../examples/results/HTC_Vive_Pro/pre_cue_results.csv", index=False)

    post_cue_dataframe = pd.concat(post_cue_list_of_dataframe, ignore_index=True)
    post_cue_dataframe.to_csv(f"{current_path_file}/../examples/results/HTC_Vive_Pro/post_cue_results.csv", index=False)


if __name__ == "__main__":
    perform_all_files()
