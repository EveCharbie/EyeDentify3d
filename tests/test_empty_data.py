from pathlib import Path

from matplotlib import pyplot as plt

from eyedentify3d import (
    TimeRange,
    HtcViveProData,
    ErrorType,
    GazeBehaviorIdentifier,
)
from eyedentify3d.data_parsers.abstract_data import EmptyData
from eyedentify3d.data_parsers.reduced_data import ReducedData



# ---------------------------------------------------------------------------
# Unit tests for the EmptyData data object
# ---------------------------------------------------------------------------
def test_empty_data_initialization():
    """Test that EmptyData initializes with all data attributes set to None."""
    data = EmptyData()

    assert data.time_vector is None
    assert data.right_eye_openness is None
    assert data.left_eye_openness is None
    assert data.eye_direction is None
    assert data.head_angles is None
    assert data.gaze_direction is None
    assert data.head_angular_velocity is None
    assert data.head_velocity_norm is None


def test_empty_data_sentinel_properties():
    """Test that the EmptyData properties return the sentinel values used downstream."""
    data = EmptyData()

    assert data.time_range is None
    assert data.trial_duration == 0
    assert data.file_name is None
    assert data.nb_frames == 0


def test_empty_data_time_range_setter_is_noop():
    """Test that setting the time_range on EmptyData does nothing (stays None)."""
    data = EmptyData()
    data.time_range = TimeRange(0.0, 1.0)
    assert data.time_range is None


def test_empty_data_noop_methods_do_not_raise():
    """Test that the (no-op) processing methods of EmptyData can be called safely."""
    data = EmptyData()

    # None of these should raise, and none should populate any data attribute.
    data._set_dt()
    data._set_time_vector()
    data._set_gaze_direction()
    data._set_eye_openness()
    data._set_eye_direction()
    data._set_head_angles()
    data._set_head_angular_velocity()
    data._set_data_invalidity()
    data._check_validity()
    data._discard_data_out_of_range()
    data.set_gaze_angular_velocity()

    assert data.time_vector is None
    assert data.gaze_direction is None


def test_empty_data_finalize():
    """Test that finalize sets the is_finalized flag."""
    data = EmptyData()
    assert data.is_finalized is False
    data.finalize()
    assert data.is_finalized is True


def test_empty_data_plot_gaze_vector_is_noop():
    """Test that plotting the gaze vector of EmptyData is a no-op that does not raise."""
    data = EmptyData()
    _, ax = plt.subplots(1, 1)
    # Should not raise even though there is no data to plot.
    data.plot_gaze_vector(ax)
    plt.close("all")


# ---------------------------------------------------------------------------
# GazeBehaviorIdentifier built directly on an EmptyData object
# ---------------------------------------------------------------------------
def test_identifier_accepts_empty_data():
    """Test that GazeBehaviorIdentifier accepts an EmptyData object."""
    identifier = GazeBehaviorIdentifier(EmptyData())
    assert isinstance(identifier.data_object, EmptyData)


def test_identifier_empty_data_unavailable_indices():
    """Test that the unavailable indices are an empty array for EmptyData."""
    identifier = GazeBehaviorIdentifier(EmptyData())
    assert identifier.unavailable_indices.shape == (0,)
    assert identifier.unavailable_indices.dtype == bool


def test_identifier_empty_data_unidentified_indices():
    """Test that the unidentified indices are an empty array for EmptyData."""
    identifier = GazeBehaviorIdentifier(EmptyData())
    assert identifier.unidentified_indices.shape == (0,)


def test_identifier_empty_data_get_results():
    """Test that get_results works on an EmptyData identifier and reports empty metrics."""
    identifier = GazeBehaviorIdentifier(EmptyData())
    identifier.is_finalized = True  # No events were detected, so we finalize manually.

    results = identifier.get_results(participant_id="P", trial_id="T")

    # With no data, the mean head velocity norm cannot be computed.
    assert results["mean_head_velocity_norm"].iloc[0] is None
    assert results["total_trial_duration"].iloc[0] == 0
    assert results["total_identified_ratio"].iloc[0] == 0.0
    assert results["total_unidentified_ratio"].iloc[0] == 1.0
    # The extra keyword arguments are still propagated to the results.
    assert results["participant_id"].iloc[0] == "P"
    assert results["trial_id"].iloc[0] == "T"


# ---------------------------------------------------------------------------
# End-to-end: splitting close to the end of the file yields an EmptyData segment
# ---------------------------------------------------------------------------
def _build_finalized_identifier(data_object: HtcViveProData) -> GazeBehaviorIdentifier:
    """Run the full detection pipeline (as in the complete example) and finalize."""
    identifier = GazeBehaviorIdentifier(data_object)
    identifier.detect_blink_sequences(eye_openness_threshold=0.5)
    identifier.detect_invalid_sequences()
    identifier.detect_saccade_sequences(
        min_acceleration_threshold=4000,
        nb_acceleration_frames=2,
        velocity_window_size=0.52,
        velocity_factor=5.0,
    )
    identifier.detect_visual_scanning_sequences(
        min_velocity_threshold=100,
        minimal_duration=0.040,
    )
    identifier.detect_fixation_and_smooth_pursuit_sequences(
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
    identifier.finalize()
    return identifier


def test_split_close_to_end_produces_empty_data_segment():
    """Test that a split timing very close to the end of the file produces an EmptyData segment.

    When the last sub-trial contains fewer than 3 frames, GazeBehaviorIdentifier builds an
    EmptyData object instead of a ReducedData one (nb_frames < 3 branch), so that the results
    can still be collected without crashing.
    """
    current_path_file = Path(__file__).parent
    data_file_path = f"{current_path_file}/../examples/data/HTC_Vive_Pro/TESTNA01_2D_Fist3.csv"

    data_object = HtcViveProData(
        data_file_path, error_type=ErrorType.PRINT, time_range=TimeRange(min_time=0, max_time=7.180)
    )
    identifier = _build_finalized_identifier(data_object)

    # Split half a frame before the very last sample: the last segment then spans less than
    # 3 frames, which triggers the EmptyData branch.
    split_timing = data_object.time_vector[-1] - 0.5 * data_object.dt
    segments = identifier.split([split_timing], event_at_split_handling=ErrorType.PRINT)

    assert len(segments) == 2
    pre_cue, post_cue = segments

    # The first segment holds the bulk of the trial and is a regular (non-empty) segment.
    assert isinstance(pre_cue.data_object, ReducedData)

    # The last segment has fewer than 3 frames and is therefore backed by EmptyData.
    assert isinstance(post_cue.data_object, EmptyData)
    assert post_cue.data_object.nb_frames == 0

    # Collecting the results on the empty segment must work and report empty metrics.
    results = post_cue.get_results(participant_id="TESTNA01", trial_id="2D_Fist3")
    assert results["mean_head_velocity_norm"].iloc[0] is None
    assert results["total_trial_duration"].iloc[0] == 0
    assert results["saccade_number"].iloc[0] == 0
    assert results["smooth_pursuit_number"].iloc[0] == 0
