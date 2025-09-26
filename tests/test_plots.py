from pathlib import Path
import pytest
import platform

from eyedentify3d import (
    TimeRange,
    HtcViveProData,
    ErrorType,
    GazeBehaviorIdentifier,
)


@pytest.fixture(scope="module")
def initialize_plots():

    if platform.system() != "Darwin":
        # Skip the tests on other systems because the reference images were generated with Mac
        pytest.skip("Skipping tests other than Darwin", allow_module_level=True)

    # Cut the data after the end of the trial (black screen happening at 7.180 seconds)
    time_range = TimeRange(min_time=0, max_time=5.06)

    # Load the data from the HTC Vive Pro
    current_path_file = Path(__file__).parent
    data_file_path = f"{current_path_file}/../examples/data/HTC_Vive_Pro/TESTNA05_360VR_Spread7.csv"
    data_object = HtcViveProData(data_file_path, error_type=ErrorType.PRINT, time_range=time_range)

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
    return gaze_behavior_identifier


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test_blink_plotting(initialize_plots):
    gaze_behavior_identifier = initialize_plots
    return gaze_behavior_identifier.blink.plot(live_show=False)


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test_invalid_plotting(initialize_plots):
    gaze_behavior_identifier = initialize_plots
    return gaze_behavior_identifier.invalid.plot(live_show=False)


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test_saccade_plotting(initialize_plots):
    gaze_behavior_identifier = initialize_plots
    return gaze_behavior_identifier.saccade.plot(live_show=False)


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test_visual_scanning_plotting(initialize_plots):
    gaze_behavior_identifier = initialize_plots
    return gaze_behavior_identifier.visual_scanning.plot(live_show=False)


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test_inter_saccadic_sequences_plotting(initialize_plots):
    gaze_behavior_identifier = initialize_plots
    return gaze_behavior_identifier.inter_saccadic_sequences.plot(live_show=False)


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test_fixation_plotting(initialize_plots):
    gaze_behavior_identifier = initialize_plots
    return gaze_behavior_identifier.fixation.plot(live_show=False)


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test_smooth_pursuit_plotting(initialize_plots):
    gaze_behavior_identifier = initialize_plots
    return gaze_behavior_identifier.smooth_pursuit.plot(live_show=False)


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 100})
def test_all_gaze_behaviors_plotting(initialize_plots):
    gaze_behavior_identifier = initialize_plots
    return gaze_behavior_identifier.plot(live_show=False)


# Testing only that the plot constructs so that codecov stays high (although plot pixels are tested above)
def test_plot_build():

    # Cut the data after the end of the trial (black screen happening at 7.180 seconds)
    time_range = TimeRange(min_time=0, max_time=5.06)

    # Load the data from the HTC Vive Pro
    current_path_file = Path(__file__).parent
    data_file_path = f"{current_path_file}/../examples/data/HTC_Vive_Pro/TESTNA05_360VR_Spread7.csv"
    data_object = HtcViveProData(data_file_path, error_type=ErrorType.PRINT, time_range=time_range)

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
    gaze_behavior_identifier.finalize()

    # Build the plots
    gaze_behavior_identifier.blink.plot(live_show=False)
    gaze_behavior_identifier.invalid.plot(live_show=False)
    gaze_behavior_identifier.saccade.plot(live_show=False)
    gaze_behavior_identifier.visual_scanning.plot(live_show=False)
    gaze_behavior_identifier.inter_saccadic_sequences.plot(live_show=False)
    gaze_behavior_identifier.fixation.plot(live_show=False)
    gaze_behavior_identifier.smooth_pursuit.plot(live_show=False)
    gaze_behavior_identifier.plot(live_show=False)
