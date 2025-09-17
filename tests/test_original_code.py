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
    GazeBehaviorIdentifier,
)


def perform_one_file(
    file_name,
    data_file_path,
    length_before_black_screen,
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

    # Split the gaze behavior identifier into pre-cue and post-cue
    time_between_cue_and_trial_end = 2  # seconds
    split_timings = [original_data_object.time_vector[-1] - time_between_cue_and_trial_end]
    gaze_behavior_identifiers = gaze_behavior_identifier.split(split_timings)

    # --- new version (end) --- #
    blink_sequences = gaze_behavior_identifier.blink.sequences
    saccade_sequences = gaze_behavior_identifier.saccade.sequences
    visual_scanning_sequences = gaze_behavior_identifier.visual_scanning.sequences
    gaze_angular_velocity_rad = (
        gaze_behavior_identifier.data_object.gaze_angular_velocity * np.pi / 180
    )  # Convert deg/s to rad/s
    fixation_sequences = gaze_behavior_identifier.fixation.sequences
    smooth_pursuit_sequences = gaze_behavior_identifier.smooth_pursuit.sequences

    # Pre-cue and post-cue sequences
    pre_cue_gaze_behavior_identifier, post_cue_gaze_behavior_identifier = (
        gaze_behavior_identifiers[0],
        gaze_behavior_identifiers[1],
    )
    smooth_pursuit_sequences_pre_cue = pre_cue_gaze_behavior_identifier.smooth_pursuit.sequences
    smooth_pursuit_sequences_post_cue = post_cue_gaze_behavior_identifier.smooth_pursuit.sequences
    fixation_sequences_pre_cue = pre_cue_gaze_behavior_identifier.fixation.sequences
    fixation_sequences_post_cue = post_cue_gaze_behavior_identifier.fixation.sequences
    blink_sequences_pre_cue = pre_cue_gaze_behavior_identifier.blink.sequences
    blink_sequences_post_cue = post_cue_gaze_behavior_identifier.blink.sequences
    saccade_sequences_pre_cue = pre_cue_gaze_behavior_identifier.saccade.sequences
    saccade_sequences_post_cue = post_cue_gaze_behavior_identifier.saccade.sequences
    visual_scanning_sequences_pre_cue = pre_cue_gaze_behavior_identifier.visual_scanning.sequences
    visual_scanning_sequences_post_cue = post_cue_gaze_behavior_identifier.visual_scanning.sequences

    # Intermediary metrics
    # Number of events
    nb_blinks = gaze_behavior_identifier.blink.nb_events()
    nb_blinks_pre_cue = pre_cue_gaze_behavior_identifier.blink.nb_events()
    nb_blinks_post_cue = post_cue_gaze_behavior_identifier.blink.nb_events()

    nb_saccades = gaze_behavior_identifier.saccade.nb_events()
    nb_saccades_pre_cue = pre_cue_gaze_behavior_identifier.saccade.nb_events()
    nb_saccades_post_cue = post_cue_gaze_behavior_identifier.saccade.nb_events()

    nb_visual_scanning = gaze_behavior_identifier.visual_scanning.nb_events()
    nb_visual_scanning_pre_cue = pre_cue_gaze_behavior_identifier.visual_scanning.nb_events()
    nb_visual_scanning_post_cue = post_cue_gaze_behavior_identifier.visual_scanning.nb_events()

    nb_fixations = gaze_behavior_identifier.fixation.nb_events()
    nb_fixations_pre_cue = pre_cue_gaze_behavior_identifier.fixation.nb_events()
    nb_fixations_post_cue = post_cue_gaze_behavior_identifier.fixation.nb_events()

    nb_smooth_pursuit = gaze_behavior_identifier.smooth_pursuit.nb_events()
    nb_smooth_pursuit_pre_cue = pre_cue_gaze_behavior_identifier.smooth_pursuit.nb_events()
    nb_smooth_pursuit_post_cue = post_cue_gaze_behavior_identifier.smooth_pursuit.nb_events()

    # Duration
    blink_duration = gaze_behavior_identifier.blink.duration()
    blink_duration_pre_cue = pre_cue_gaze_behavior_identifier.blink.duration()
    blink_duration_post_cue = gaze_behavior_identifier.blink.duration()
    total_blink_duration = gaze_behavior_identifier.blink.total_duration()
    total_blink_duration_pre_cue = pre_cue_gaze_behavior_identifier.blink.total_duration()
    total_blink_duration_post_cue = gaze_behavior_identifier.blink.total_duration()

    saccade_duration = gaze_behavior_identifier.saccade.duration()
    saccade_duration_pre_cue = pre_cue_gaze_behavior_identifier.saccade.duration()
    saccade_duration_post_cue = gaze_behavior_identifier.saccade.duration()
    total_saccade_duration = gaze_behavior_identifier.saccade.total_duration()
    total_saccade_duration_pre_cue = pre_cue_gaze_behavior_identifier.saccade.total_duration()
    total_saccade_duration_post_cue = gaze_behavior_identifier.saccade.total_duration()

    visual_scanning_duration = gaze_behavior_identifier.visual_scanning.duration()
    visual_scanning_pre_cue = pre_cue_gaze_behavior_identifier.visual_scanning.duration()
    visual_scanning_post_cue = gaze_behavior_identifier.visual_scanning.duration()
    total_visual_scanning_duration = gaze_behavior_identifier.visual_scanning.total_duration()
    total_visual_scanning_pre_cue = pre_cue_gaze_behavior_identifier.visual_scanning.total_duration()
    total_visual_scanning_post_cue = gaze_behavior_identifier.visual_scanning.total_duration()

    fixation_duration = gaze_behavior_identifier.fixation.duration()
    fixation_duration_pre_cue = pre_cue_gaze_behavior_identifier.fixation.duration()
    fixation_duration_post_cue = gaze_behavior_identifier.fixation.duration()
    total_fixation_duration = gaze_behavior_identifier.fixation.total_duration()
    total_fixation_duration_pre_cue = pre_cue_gaze_behavior_identifier.fixation.total_duration()
    total_fixation_duration_post_cue = gaze_behavior_identifier.fixation.total_duration()

    smooth_pursuit_duration = gaze_behavior_identifier.smooth_pursuit.duration()
    smooth_pursuit_duration_pre_cue = pre_cue_gaze_behavior_identifier.smooth_pursuit.duration()
    smooth_pursuit_duration_post_cue = gaze_behavior_identifier.smooth_pursuit.duration()
    total_smooth_pursuit_duration = gaze_behavior_identifier.smooth_pursuit.total_duration()
    total_smooth_pursuit_duration_pre_cue = pre_cue_gaze_behavior_identifier.smooth_pursuit.total_duration()
    total_smooth_pursuit_duration_post_cue = gaze_behavior_identifier.smooth_pursuit.total_duration()

    # Ratios

    invalid_ratio = gaze_behavior_identifier.invalid.ratio()

    blinking_ratio = gaze_behavior_identifier.blink.ratio()
    blinking_ratio_pre_cue = pre_cue_gaze_behavior_identifier.blink.ratio()
    blinking_ratio_post_cue = post_cue_gaze_behavior_identifier.blink.ratio()

    fixation_ratio = gaze_behavior_identifier.fixation.ratio()
    fixation_ratio_pre_cue = pre_cue_gaze_behavior_identifier.fixation.ratio()
    fixation_ratio_post_cue = post_cue_gaze_behavior_identifier.fixation.ratio()

    smooth_pursuit_ratio = gaze_behavior_identifier.smooth_pursuit.ratio()
    smooth_pursuit_ratio_pre_cue = pre_cue_gaze_behavior_identifier.smooth_pursuit.ratio()
    smooth_pursuit_ratio_post_cue = post_cue_gaze_behavior_identifier.smooth_pursuit.ratio()

    saccade_ratio = gaze_behavior_identifier.saccade.ratio()
    saccade_ratio_pre_cue = pre_cue_gaze_behavior_identifier.saccade.ratio()
    saccade_ratio_post_cue = post_cue_gaze_behavior_identifier.saccade.ratio()

    visual_scanning_ratio = gaze_behavior_identifier.visual_scanning.ratio()
    visual_scanning_ratio_pre_cue = pre_cue_gaze_behavior_identifier.visual_scanning.ratio()
    visual_scanning_ratio_post_cue = post_cue_gaze_behavior_identifier.visual_scanning.ratio()

    not_classified_ratio = 1 - (
        fixation_ratio + smooth_pursuit_ratio + blinking_ratio + saccade_ratio + visual_scanning_ratio
    )
    if not_classified_ratio < -original_data_object.dt:
        raise ValueError("Problem: The sum of the ratios is greater than 1")

    # Other specific metrics
    mean_head_angular_velocity = np.nanmean(gaze_behavior_identifier.data_object.head_velocity_norm)
    mean_head_angular_velocity_pre_cue = np.nanmean(pre_cue_gaze_behavior_identifier.data_object.head_velocity_norm)
    mean_head_angular_velocity_post_cue = np.nanmean(post_cue_gaze_behavior_identifier.data_object.head_velocity_norm)

    mean_saccade_duration = gaze_behavior_identifier.saccade.mean_duration()
    mean_saccade_duration_pre_cue = pre_cue_gaze_behavior_identifier.saccade.mean_duration()
    mean_saccade_duration_post_cue = post_cue_gaze_behavior_identifier.saccade.mean_duration()
    saccade_amplitudes = gaze_behavior_identifier.saccade.saccade_amplitudes
    max_saccade_amplitude = np.nanmax(gaze_behavior_identifier.saccade.saccade_amplitudes) if nb_saccades > 0 else None
    mean_saccade_amplitude = (
        np.nanmean(gaze_behavior_identifier.saccade.saccade_amplitudes) if nb_saccades > 0 else None
    )
    mean_saccade_amplitude_pre_cue = (
        np.nanmean(pre_cue_gaze_behavior_identifier.saccade.saccade_amplitudes) if nb_saccades_pre_cue > 0 else None
    )
    mean_saccade_amplitude_post_cue = (
        np.nanmean(post_cue_gaze_behavior_identifier.saccade.saccade_amplitudes) if nb_saccades_post_cue > 0 else None
    )

    mean_visual_scanning_duration = gaze_behavior_identifier.visual_scanning.mean_duration()
    mean_visual_scanning_duration_pre_cue = pre_cue_gaze_behavior_identifier.visual_scanning.mean_duration()
    mean_visual_scanning_duration_post_cue = post_cue_gaze_behavior_identifier.visual_scanning.mean_duration()

    mean_fixation_duration = gaze_behavior_identifier.fixation.mean_duration()
    mean_fixation_duration_pre_cue = pre_cue_gaze_behavior_identifier.fixation.mean_duration()
    mean_fixation_duration_post_cue = post_cue_gaze_behavior_identifier.fixation.mean_duration()
    search_rate = gaze_behavior_identifier.fixation.search_rate
    search_rate_pre_cue = pre_cue_gaze_behavior_identifier.fixation.search_rate
    search_rate_post_cue = post_cue_gaze_behavior_identifier.fixation.search_rate

    mean_smooth_pursuit_duration = gaze_behavior_identifier.smooth_pursuit.mean_duration()
    mean_smooth_pursuit_duration_pre_cue = pre_cue_gaze_behavior_identifier.smooth_pursuit.mean_duration()
    mean_smooth_pursuit_duration_post_cue = post_cue_gaze_behavior_identifier.smooth_pursuit.mean_duration()
    smooth_pursuit_trajectories = gaze_behavior_identifier.smooth_pursuit.smooth_pursuit_trajectories
    smooth_pursuit_trajectories_pre_cue = pre_cue_gaze_behavior_identifier.smooth_pursuit.smooth_pursuit_trajectories
    smooth_pursuit_trajectories_post_cue = post_cue_gaze_behavior_identifier.smooth_pursuit.smooth_pursuit_trajectories
    mean_smooth_pursuit_trajectory = (
        np.nanmean(smooth_pursuit_trajectories) if len(smooth_pursuit_trajectories) > 0 else None
    )
    mean_smooth_pursuit_trajectory_pre_cue = (
        np.nanmean(smooth_pursuit_trajectories_pre_cue) if len(smooth_pursuit_trajectories_pre_cue) > 0 else None
    )
    mean_smooth_pursuit_trajectory_post_cue = (
        np.nanmean(smooth_pursuit_trajectories_post_cue) if len(smooth_pursuit_trajectories_post_cue) > 0 else None
    )
    max_smooth_pursuit_trajectory = (
        np.nanmax(smooth_pursuit_trajectories) if len(smooth_pursuit_trajectories) > 0 else None
    )

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
            "Mean head angular velocity full trial": [mean_head_angular_velocity],
            "Mean head angular velocity pre cue": [mean_head_angular_velocity_pre_cue],
            "Mean head angular velocity post cue": [mean_head_angular_velocity_post_cue],
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

    # Perform the data treatment
    for file_name in length_before_black_screen.keys():
        file = data_path + file_name + ".csv"

        # Redirect print output
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.

        output = perform_one_file(file_name, file, length_before_black_screen)

        # Reset print output
        sys.stdout = sys.__stdout__  # Reset redirect.
        print(file_name)

        if file_name == "TESTNA01_2D_Fist3":
            assert captured_output.getvalue() == r"Smooth pursuit : 0.33311 s ----"
        elif file_name == "TESTNA01_360VR_Fist3":
            assert captured_output.getvalue() == ""
        elif file_name == "TESTNA05_2D_Spread7":
            assert captured_output.getvalue() == "Fixation : 0.64128 s ----"
        elif file_name == "TESTNA05_360VR_Spread7":
            assert captured_output.getvalue() == "Smooth pursuit : 0.55851 s ----"
        elif file_name == "TESTNA15_2D_Pen3":
            assert captured_output.getvalue() == "Fixation : 0.20745 s ----"
        elif file_name == "TESTNA15_360VR_Pen3":
            assert captured_output.getvalue() == ""
        elif file_name == "TESTVA03_2D_Spread9":
            assert (
                captured_output.getvalue()
                == "The file TESTVA03_2D_Spread9.csv is empty. There is no element in the field 'time(100ns)'. Please check the file."
            )
        elif file_name == "TESTNA10_360VR_Fist3":
            assert (
                captured_output.getvalue()
                == "More than 50% of the data from file TESTNA10_360VR_Fist3.csv is declared invalid by the eye-tracker, skipping this file."
            )

        # Compare the data with reference
        if file_name not in ["TESTNA10_360VR_Fist3", "TESTVA03_2D_Spread9"]:
            with open(f"{current_path_file}/original_results/" + file_name + ".pkl", "rb") as result_file:
                output_reference = pickle.load(result_file)

            pdt.assert_frame_equal(output, output_reference, check_exact=False, rtol=1e-5)


"""
TODO: test the plots pixels
"""
