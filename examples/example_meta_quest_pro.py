"""
In this example, we will load the data from the Meta Quest Pro and extract gaze behavior from the data.
"""

import os

from eyedentify3d import MetaQuestProData, ErrorType, GazeBehaviorIdentifier


def main():

    # Load the data from the HTC Vive Pro
    data_file_path = "data/Meta_Quest_Pro/baseball_360VR.csv"
    data_object = MetaQuestProData(data_file_path, error_type=ErrorType.PRINT)

    # Create a GazeBehaviorIdentifier object
    gaze_behavior_identifier = GazeBehaviorIdentifier(data_object)

    # Detect gaze behaviors (must be performed in the desired order)
    gaze_behavior_identifier.detect_blink_sequences(eye_openness_threshold=0.5)
    gaze_behavior_identifier.detect_invalid_sequences()
    gaze_behavior_identifier.detect_saccade_sequences(
        min_acceleration_threshold=4000,
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


if __name__ == "__main__":
    main()
