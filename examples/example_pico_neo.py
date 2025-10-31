"""
In this example, we will load the data from the Pico Noe and extract gaze behavior from the data.
Note: The blink detection is not possible yet with the Pico Neo data, since the eye openness information was not available.
If you have a set of data where this information is available, we would appreciate if you could contact us :)
"""

import os

from eyedentify3d import PicoNeoData, ErrorType, GazeBehaviorIdentifier


def main():

    # Load the data from the HTC Vive Pro
    data_file_path = "data/Pico_Neo_3_Pro/boxing_VR.csv"
    data_object = PicoNeoData(data_file_path, error_type=ErrorType.PRINT)

    # Create a GazeBehaviorIdentifier object
    gaze_behavior_identifier = GazeBehaviorIdentifier(data_object)

    # Detect gaze behaviors (must be performed in the desired order)
    gaze_behavior_identifier.detect_blink_sequences(eye_openness_threshold=0.5)
    gaze_behavior_identifier.detect_invalid_sequences()
    gaze_behavior_identifier.detect_saccade_sequences(
        min_acceleration_threshold=1000,
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
    if not os.path.exists("figures"):
        os.makedirs("figures")
    gaze_behavior_identifier.blink.plot(save_name="figures/blink_detection.png")
    gaze_behavior_identifier.invalid.plot(save_name="figures/invalid_detection.png")
    gaze_behavior_identifier.saccade.plot(save_name="figures/saccade_detection.png")
    gaze_behavior_identifier.visual_scanning.plot(save_name="figures/visual_scanning_detection.png")
    gaze_behavior_identifier.inter_saccadic_sequences.plot(save_name="figures/inter_saccadic_detection.png")
    gaze_behavior_identifier.fixation.plot(save_name="figures/fixation_detection.png")
    gaze_behavior_identifier.smooth_pursuit.plot(save_name="figures/smooth_pursuit_detection.png")
    gaze_behavior_identifier.plot(save_name="figures/all_gaze_behaviors.png")

    # Animate the results
    gaze_behavior_identifier.animate()

    # For this example, we will remove all files generated, but in a real case, they should be kept
    if os.path.exists("bad_data_files.txt"):
        os.remove("bad_data_files.txt")
    if os.path.exists("figures/blink_detection.png"):
        os.remove("figures/blink_detection.png")
    if os.path.exists("figures/invalid_detection.png"):
        os.remove("figures/invalid_detection.png")
    if os.path.exists("figures/saccade_detection.png"):
        os.remove("figures/saccade_detection.png")
    if os.path.exists("figures/visual_scanning_detection.png"):
        os.remove("figures/visual_scanning_detection.png")
    if os.path.exists("figures/inter_saccadic_detection.png"):
        os.remove("figures/inter_saccadic_detection.png")
    if os.path.exists("figures/fixation_detection.png"):
        os.remove("figures/fixation_detection.png")
    if os.path.exists("figures/smooth_pursuit_detection.png"):
        os.remove("figures/smooth_pursuit_detection.png")
    if os.path.exists("figures/all_gaze_behaviors.png"):
        os.remove("figures/all_gaze_behaviors.png")


if __name__ == "__main__":
    main()
