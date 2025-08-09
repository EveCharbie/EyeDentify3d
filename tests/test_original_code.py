from pathlib import Path
import pandas as pd
import numpy as np
import pandas.testing as pdt
import pickle
import sys
import io
from copy import deepcopy


from eyedentify3d import (
    TimeRange,
    HtcViveProData,
    ErrorType,
    detect_visual_scanning,
    apply_minimal_duration,
    sliding_window,
    detect_fixations_and_smooth_pursuit,
    compute_intermediary_metrics,
    check_if_there_is_sequence_overlap,
    GazeBehaviorIdentifier,
)


def perform_one_file(
    file_name,
    data_file_path,
    length_before_black_screen,
    fixation_duration_threshold,
    smooth_pursuit_duration_threshold,
):

    # --- new version (start) --- #
    # Cut the data after the end of the trial (black screen)
    black_screen_time = length_before_black_screen[file_name]
    time_range = TimeRange(min_time=0, max_time=black_screen_time)

    # Load the data from the HTC Vive Pro
    original_data_object = HtcViveProData(data_file_path, error_type=ErrorType.PRINT, time_range=time_range)

    if original_data_object.time_vector is None:
        # This trial was problematic and an error was raised
        return

    # Create a GazeBehaviorIdentifier object
    gaze_behavior_identifier = GazeBehaviorIdentifier(deepcopy(original_data_object))
    gaze_behavior_identifier.detect_blink_sequences()
    gaze_behavior_identifier.detect_invalid_sequences()
    gaze_behavior_identifier.detect_saccade_sequences()
    gaze_behavior_identifier.detect_visual_scanning_sequences()
    data_object = gaze_behavior_identifier.data_object
    # --- new version (end) --- #

    blink_sequences = gaze_behavior_identifier.blink.sequences
    eyetracker_invalid_data_index = gaze_behavior_identifier.invalid.frame_indices
    eyetracker_invalid_sequences = gaze_behavior_identifier.invalid.sequences
    gaze_direction = data_object.gaze_direction
    saccade_sequences = gaze_behavior_identifier.saccade.sequences
    saccade_amplitudes = gaze_behavior_identifier.saccade.saccade_amplitudes
    visual_scanning_sequences = gaze_behavior_identifier.visual_scanning.sequences
    gaze_angular_velocity_rad = gaze_behavior_identifier.visual_scanning.gaze_angular_velocity * np.pi / 180  # Convert deg/s to rad/s
    identified_indices = gaze_behavior_identifier.identified_indices

    # Detect fixations
    intersaccadic_interval = np.zeros((len(data_object.time_vector),))
    all_index = np.arange(len(data_object.time_vector))
    for i in all_index:
        i_in_saccades = True if any(i in sequence for sequence in saccade_sequences) else False
        i_in_visual_scanning = True if any(i in sequence for sequence in visual_scanning_sequences) else False
        i_in_blinks = True if any(i in sequence for sequence in blink_sequences) else False
        i_in_eyetracker_invalid = True if i in eyetracker_invalid_data_index else False
        gaze_velocity_criteria = True if (gaze_angular_velocity_rad[i] * 180 / np.pi) > 100 else False
        if i_in_saccades or i_in_visual_scanning or i_in_blinks or i_in_eyetracker_invalid or gaze_velocity_criteria:
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
        sliding_window(original_data_object.time_vector, intersaccadic_sequences, gaze_direction)
    )
    fixation_sequences, smooth_pursuit_sequences, uncertain_sequences = detect_fixations_and_smooth_pursuit(
        data_object.time_vector, gaze_direction, intersaccadic_gouped_sequences, identified_indices, file_name, False
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
        original_data_object.time_vector,
        smooth_pursuit_sequences,
        fixation_sequences,
        blink_sequences,
        saccade_sequences,
        visual_scanning_sequences,
        gaze_angular_velocity_rad,
        original_data_object.dt,
        2,
        None,
        fixation_duration_threshold,
        smooth_pursuit_duration_threshold,
        original_data_object.head_velocity_norm,
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
        nb_fixations_pre_cue / mean_fixation_duration_pre_cue if mean_fixation_duration_pre_cue is not None else None
    )
    search_rate_post_cue = (
        nb_fixations_post_cue / mean_fixation_duration_post_cue if mean_fixation_duration_post_cue is not None else None
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
        np.nanmean(np.array(smooth_pursuit_duration_post_cue)) if len(smooth_pursuit_duration_post_cue) > 0 else None
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
        np.nanmean(np.array(visual_scanning_duration_pre_cue)) if len(visual_scanning_duration_pre_cue) > 0 else None
    )
    mean_visual_scanning_duration_post_cue = (
        np.nanmean(np.array(visual_scanning_duration_post_cue)) if len(visual_scanning_duration_post_cue) > 0 else None
    )

    fixation_ratio = total_fixation_duration / original_data_object.time_vector[-1]
    fixation_ratio_pre_cue = total_fixation_duration_pre_cue / (original_data_object.time_vector[-1] - 2)
    fixation_ratio_post_cue = total_fixation_duration_post_cue / 2

    smooth_pursuit_ratio = total_smooth_pursuit_duration / original_data_object.time_vector[-1]
    smooth_pursuit_ratio_pre_cue = total_smooth_pursuit_duration_pre_cue / (original_data_object.time_vector[-1] - 2)
    smooth_pursuit_ratio_post_cue = total_smooth_pursuit_duration_post_cue / 2

    blinking_ratio = total_blink_duration / original_data_object.time_vector[-1]
    blinking_ratio_pre_cue = total_blink_duration_pre_cue / (original_data_object.time_vector[-1] - 2)
    blinking_ratio_post_cue = total_blink_duration_post_cue / 2

    saccade_ratio = total_saccade_duration / original_data_object.time_vector[-1]
    saccade_ratio_pre_cue = total_saccade_duration_pre_cue / (original_data_object.time_vector[-1] - 2)
    saccade_ratio_post_cue = total_saccade_duration_post_cue / 2

    visual_scanning_ratio = total_visual_scanning_duration / original_data_object.time_vector[-1]
    visual_scanning_ratio_pre_cue = total_visual_scanning_duration_pre_cue / (original_data_object.time_vector[-1] - 2)
    visual_scanning_ratio_post_cue = total_visual_scanning_duration_post_cue / 2

    not_classified_ratio = 1 - (
        fixation_ratio + smooth_pursuit_ratio + blinking_ratio + saccade_ratio + visual_scanning_ratio
    )
    if not_classified_ratio < -original_data_object.dt:
        raise ValueError("Problem: The sum of the ratios is greater than 1")

    invalid_ratio = np.sum(
        np.logical_or(
            original_data_object.csv_data["eye_valid_L"] != 31, original_data_object.csv_data["eye_valid_R"] != 31
        )
    ) / len(original_data_object.csv_data["eye_valid_L"])

    output = pd.DataFrame(
        {
            "File name": [file_name],
            "Figure name": [file_name],
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
            "Length of the full trial [s]": [original_data_object.time_vector[-1]],
        }
    )

    return output


def test_original_code():

    # Define the path to the data
    current_path_file = Path(__file__).parent
    data_path = f"{current_path_file}/../examples/data/HTC_Vive_Pro/"
    length_before_black_screen = {
        "TESTNA01_2D_Fist3": 7.180,  # s
        "TESTNA01_360VR_Fist3": 7.180,
        "TESTNA05_2D_Spread7": 5.060,
        "TESTNA05_360VR_Spread7": 5.060,
        "TESTNA15_2D_Pen3": 4.230,
        "TESTNA15_360VR_Pen3": 4.230,
        "TESTVA03_2D_Spread9": 6.150,  # Bad data (no data)
        "TESTNA10_360VR_Fist3": 7.180,  # Bad data (more than 50% of the data is invalid)
    }

    # Define some constants
    fixation_duration_threshold = 0.1  # 100 ms
    smooth_pursuit_duration_threshold = 0.1  # 100 ms

    # Perform the data treatment
    for file_name in length_before_black_screen.keys():
        file = data_path + file_name + ".csv"

        # Redirect print output
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.

        output = perform_one_file(
            file_name, file, length_before_black_screen, fixation_duration_threshold, smooth_pursuit_duration_threshold
        )

        # Reset print output
        sys.stdout = sys.__stdout__  # Reset redirect.
        print(file_name)

        if file_name == "TESTNA01_2D_Fist3":
            assert captured_output.getvalue() == r"Smooth pursuit : 1.24955 s ----"
        elif file_name == "TESTNA01_360VR_Fist3":
            assert captured_output.getvalue() == "Smooth pursuit : 0.10806 s ----"
        elif file_name == "TESTNA05_2D_Spread7":
            assert captured_output.getvalue() == "Fixation : 0.95033 s ----"
        elif file_name == "TESTNA05_360VR_Spread7":
            assert captured_output.getvalue() == "Smooth pursuit : 0.96577 s ----"
        elif file_name == "TESTNA15_2D_Pen3":
            assert captured_output.getvalue() == "Fixation : 0.21578 s ----"
        elif file_name == "TESTNA15_360VR_Pen3":
            assert captured_output.getvalue() == "Smooth pursuit : 0.15893 s ----"
        elif file_name == "TESTVA03_2D_Spread9":
            assert (
                captured_output.getvalue()
                == "The file TESTVA03_2D_Spread9.csv is empty. There is no element in the field 'time(100ns)'. Please check the file.\n"
            )
        elif file_name == "TESTNA10_360VR_Fist3":
            assert (
                captured_output.getvalue()
                == "More than 50% of the data from file TESTNA10_360VR_Fist3.csv is declared invalid by the eye-tracker, skipping this file.\n"
            )

        # Compare the data with reference
        if file_name not in ["TESTNA10_360VR_Fist3", "TESTVA03_2D_Spread9"]:
            with open(data_path + "/../../results/HTC_Vive_Pro/" + file_name + ".pkl", "rb") as result_file:
                output_reference = pickle.load(result_file)

            pdt.assert_frame_equal(output, output_reference, check_exact=False, rtol=1e-5)


"""
TODO:
    1. unit-tests
    2. test the whole pipeline so that it stays the same as the old code
    3. test the plots pixels
"""
