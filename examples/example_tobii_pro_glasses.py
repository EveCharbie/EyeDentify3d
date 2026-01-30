"""
In this example, we will load the data from the Tobii Pro Glasses 3 and extract gaze behavior from the data.
"""

import os

from eyedentify3d import TobiiProGlassesData, TimeRange, ErrorType, GazeBehaviorIdentifier


def main():

    time_range = TimeRange()

    # Load the data from the Tobii Pro glasses 3
    data_folder_path = "data/Tobii_Pro_Glasses_3/calibration_1/"
    data_object = TobiiProGlassesData(data_folder_path, error_type=ErrorType.PRINT, time_range=time_range)

    # Create a GazeBehaviorIdentifier object
    gaze_behavior_identifier = GazeBehaviorIdentifier(data_object)

    # Detect gaze behaviors (must be performed in the desired order)
    gaze_behavior_identifier.detect_blink_sequences(eye_openness_threshold=0.5)
    gaze_behavior_identifier.detect_invalid_sequences()
    gaze_behavior_identifier.detect_saccade_sequences(
        min_acceleration_threshold=4000,
        nb_acceleration_frames=2,
        velocity_window_size=0.52,
        velocity_factor=5.0,
    )
    gaze_behavior_identifier.detect_visual_scanning_sequences(
        min_velocity_threshold=100,
        minimal_duration=0.040,
    )
    gaze_behavior_identifier.detect_fixation_and_smooth_pursuit_sequences(
        inter_saccade_minimal_duration=0.04,
        fixation_minimal_duration=0.1,
        smooth_pursuit_minimal_duration=0.1,
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

    # Plot the results
    gaze_behavior_identifier.blink.plot()
    gaze_behavior_identifier.invalid.plot()
    gaze_behavior_identifier.saccade.plot()
    gaze_behavior_identifier.visual_scanning.plot()
    gaze_behavior_identifier.inter_saccadic_sequences.plot()
    gaze_behavior_identifier.fixation.plot()
    gaze_behavior_identifier.smooth_pursuit.plot()
    gaze_behavior_identifier.plot()

    # Animate the results
    gaze_behavior_identifier.animate()

    # For this example, we will remove all files generated, but in a real case, they should be kept
    if os.path.exists("bad_data_files.txt"):
        os.remove("bad_data_files.txt")


if __name__ == "__main__":
    main()
