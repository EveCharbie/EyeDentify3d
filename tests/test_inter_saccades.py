import numpy as np
import pytest
import numpy.testing as npt
from unittest.mock import patch

from eyedentify3d.identification.inter_saccades import InterSaccadicEvent
from eyedentify3d import ReducedData


def mock_data_object():
    """Create a mock data object for testing."""
    np.random.seed(42)  # For reproducibility

    # Create time vector
    dt = 0.01
    time_vector = np.arange(0, 1, dt)
    n_samples = len(time_vector)

    # Create gaze direction data with some patterns
    gaze_direction = np.zeros((3, n_samples))
    gaze_direction[2, :] = 1.0  # Default looking forward
    gaze_direction += np.random.normal(0, 0.1, gaze_direction.shape)  # Add some noise
    gaze_direction /= np.linalg.norm(gaze_direction, axis=0)  # Normalize to unit vectors

    # Create a coherent movement (smooth pursuit) around frame 30-40
    for i in range(30, 40):
        angle = (i - 30) * 2  # Small increasing angle
        # Rotate around y-axis
        gaze_direction[0, i] = np.sin(np.radians(angle))
        gaze_direction[2, i] = np.cos(np.radians(angle))

    # Create a fixation (incoherent movement) around frame 60-70
    for i in range(60, 70):
        # Small random movements
        gaze_direction[0, i] = np.sin(np.radians(np.random.uniform(-1, 1)))
        gaze_direction[1, i] = np.sin(np.radians(np.random.uniform(-1, 1)))
        gaze_direction[2, i] = np.sqrt(1 - gaze_direction[0, i] ** 2 - gaze_direction[1, i] ** 2)

    # Create sample eye openness data
    right_eye_openness = np.ones(n_samples) * 0.8
    left_eye_openness = np.ones(n_samples) * 0.7

    # Create sample direction vectors (3D)
    eye_direction = np.zeros((3, n_samples))
    eye_direction[2, :] = 1.0  # Looking forward along z-axis

    head_angles = np.zeros((3, n_samples))
    head_angles[0, :] = np.linspace(0, 0.1, n_samples)  # Small rotation around x-axis

    gaze_angular_velocity = np.zeros((n_samples,))
    gaze_angular_velocity[30:35] = 150  # Large angular velocity
    gaze_angular_velocity[70:75] = 150  # Large angular velocity
    gaze_angular_velocity += np.random.normal(0, 5, n_samples)  # Add some noise

    head_angular_velocity = np.zeros((3, n_samples))
    head_angular_velocity[0, :] = 0.1  # Constant angular velocity around x-axis

    head_velocity_norm = np.ones(n_samples) * 0.1

    data_invalidity = np.zeros(n_samples, dtype=bool)
    data_invalidity[80:85] = True  # Some invalid data

    data_object = ReducedData(
        original_dt=dt,
        original_time_vector=time_vector,
        original_right_eye_openness=right_eye_openness,
        original_left_eye_openness=left_eye_openness,
        original_eye_direction=eye_direction,
        original_head_angles=head_angles,
        original_gaze_direction=gaze_direction,
        original_head_angular_velocity=head_angular_velocity,
        original_head_velocity_norm=head_velocity_norm,
        original_data_invalidity=data_invalidity,
    )

    # Create identified_indices (none identified yet)
    identified_indices = np.zeros(n_samples, dtype=bool)

    return data_object, identified_indices


def test_inter_saccadic_event_initialization():
    """Test that InterSaccadicEvent initializes correctly."""
    mock_data, mock_indices = mock_data_object()
    identified_indices = np.zeros(10, dtype=bool)

    event = InterSaccadicEvent(
        mock_data,
        identified_indices,
        minimal_duration=0.05,
        window_duration=0.1,
        window_overlap=0.02,
        eta_p=0.05,
        eta_d=0.5,
        eta_cd=0.7,
        eta_pd=0.7,
        eta_max_fixation=2.0,
        eta_min_smooth_pursuit=5.0,
        phi=30.0,
    )

    assert event.data_object is mock_data
    assert event.identified_indices is identified_indices
    assert event.minimal_duration == 0.05
    assert event.window_duration == 0.1
    assert event.window_overlap == 0.02
    assert event.eta_p == 0.05
    assert event.eta_d == 0.5
    assert event.eta_cd == 0.7
    assert event.eta_pd == 0.7
    assert event.eta_max_fixation == 2.0
    assert event.eta_min_smooth_pursuit == 5.0
    assert event.phi == 30.0
    assert event.coherent_sequences is None
    assert event.incoherent_sequences is None
    assert event.fixation_indices is None
    assert event.smooth_pursuit_indices is None
    assert event.uncertain_sequences is None


def test_window_duration_validation():
    """Test that initialization validates window_duration and window_overlap."""
    mock_data, mock_indices = mock_data_object()
    identified_indices = np.zeros(10, dtype=bool)

    # window_duration must be at least twice window_overlap
    with pytest.raises(ValueError, match="The window_duration .* must be at least twice the window_overlap"):
        InterSaccadicEvent(
            mock_data, identified_indices, window_duration=0.1, window_overlap=0.06  # More than half of window_duration
        )


def test_detect_intersaccadic_indices():
    """Test that detect_intersaccadic_indices correctly identifies non-identified frames."""
    mock_data, identified_indices = mock_data_object()

    # Mark some frames as already identified
    identified_indices[20:30] = True
    identified_indices[50:60] = True

    event = InterSaccadicEvent(mock_data, identified_indices)
    event.detect_intersaccadic_indices()

    # Check that frame_indices contains frames where identified_indices is False
    expected_indices = np.concatenate([np.arange(0, 20), np.arange(30, 50), np.arange(60, 100)])
    np.testing.assert_array_equal(event.frame_indices, expected_indices)


def test_detect_directionality_coherence_on_axis():
    """Test that detect_directionality_coherence_on_axis correctly computes coherence."""
    # Create a coherent movement (all in one direction)
    n_samples = 10
    gaze_direction = np.zeros((3, n_samples))
    gaze_direction[2, :] = 1.0  # Looking forward

    # Add a consistent movement in the x direction
    for i in range(1, n_samples):
        gaze_direction[0, i] = 0.1 * i  # Increasing x component

    # Normalize to maintain unit vectors
    for i in range(n_samples):
        gaze_direction[:, i] = gaze_direction[:, i] / np.linalg.norm(gaze_direction[:, i])

    # Test coherence on x-axis (should be coherent, low p-value)
    p_value, gaze_displacement_angle = InterSaccadicEvent.detect_directionality_coherence_on_axis(
        gaze_direction, component_to_keep=0
    )
    assert p_value < 1e-4  # Coherent movement should have low p-value

    # Create an incoherent movement
    gaze_direction_incoherent = np.zeros((3, n_samples))
    gaze_direction_incoherent[2, :] = 1.0  # Looking forward

    # Add random movements
    np.random.seed(42)  # For reproducibility
    for i in range(1, n_samples):
        gaze_direction_incoherent[0, i] = np.random.uniform(-0.01, 0.01)
        gaze_direction_incoherent[1, i] = np.random.uniform(-0.01, 0.01)

    # Normalize to maintain unit vectors
    for i in range(n_samples):
        gaze_direction_incoherent[:, i] = gaze_direction_incoherent[:, i] / np.linalg.norm(
            gaze_direction_incoherent[:, i]
        )

    # Test coherence on x-axis (should be incoherent, high p-value)
    p_value, gaze_displacement_angle = InterSaccadicEvent.detect_directionality_coherence_on_axis(
        gaze_direction_incoherent, component_to_keep=0
    )
    assert p_value > 0.008  # Incoherent movement should have high p-value


def test_detect_directionality_coherence_invalid_component():
    """Test that detect_directionality_coherence_on_axis validates component_to_keep."""
    gaze_direction = np.zeros((3, 10))

    with pytest.raises(ValueError, match="component_to_keep must be 0, 1, or 2"):
        InterSaccadicEvent.detect_directionality_coherence_on_axis(gaze_direction, component_to_keep=3)


def test_variability_decomposition():
    """Test that variability_decomposition correctly computes principal components."""
    # Create a gaze direction with variation primarily along one axis
    n_samples = 10
    gaze_direction = np.zeros((3, n_samples))
    gaze_direction[2, :] = 1.0  # Looking forward

    # Add variation primarily along x-axis
    for i in range(n_samples):
        gaze_direction[0, i] = 0.1 * np.sin(i)  # Larger variation in x
        gaze_direction[1, i] = 0.01 * np.sin(i)  # Smaller variation in y

    # Normalize to maintain unit vectors
    for i in range(n_samples):
        gaze_direction[:, i] = gaze_direction[:, i] / np.linalg.norm(gaze_direction[:, i])

    # Compute variability decomposition
    length_principal, length_second = InterSaccadicEvent.variability_decomposition(gaze_direction)

    # Principal component should be larger than second component
    assert length_principal > length_second
    npt.assert_almost_equal(length_principal, 0.19486684295650236, decimal=5)
    npt.assert_almost_equal(length_second, 0.005072940256090891, decimal=5)

    # Test with too few frames
    with pytest.raises(ValueError, match="The gaze direction must contain at least 3 frames"):
        InterSaccadicEvent.variability_decomposition(gaze_direction[:, 0:2])


def test_compute_gaze_travel_distance():
    """Test that compute_gaze_travel_distance correctly computes distance."""
    # Create a gaze direction with known start and end points
    n_samples = 10
    gaze_direction = np.zeros((3, n_samples))

    # Start point: [0, 0, 1]
    gaze_direction[:, 0] = np.array([0, 0, 1])

    # End point: [0, 0.5, 0.866] (30 degrees from start)
    gaze_direction[:, -1] = np.array([0, 0.5, 0.866])

    # Compute travel distance
    distance = InterSaccadicEvent.compute_gaze_travel_distance(gaze_direction)

    # Expected distance is sqrt((0-0)^2 + (0.5-0)^2 + (0.866-1)^2)
    expected_distance = np.sqrt(0.5**2 + (0.866 - 1) ** 2)
    np.testing.assert_almost_equal(distance, expected_distance, decimal=5)


def test_compute_gaze_trajectory_length():
    """Test that compute_gaze_trajectory_length correctly computes length."""
    # Create a gaze direction with known trajectory
    n_samples = 4
    gaze_direction = np.zeros((3, n_samples))

    # Points forming a square path
    gaze_direction[:, 0] = np.array([0, 0, 1])
    gaze_direction[:, 1] = np.array([1, 0, 0])
    gaze_direction[:, 2] = np.array([0, 1, 0])
    gaze_direction[:, 3] = np.array([0, 0, 1])

    # Compute trajectory length
    length = InterSaccadicEvent.compute_gaze_trajectory_length(gaze_direction)

    # Expected length is sum of distances between consecutive points
    expected_length = np.sqrt(2) + np.sqrt(2) + np.sqrt(2)
    np.testing.assert_almost_equal(length, expected_length, decimal=5)


def test_compute_mean_gaze_direction_radius_range():
    """Test that compute_mean_gaze_direction_radius_range correctly computes range."""
    # Create a gaze direction with known range
    n_samples = 3
    gaze_direction = np.zeros((3, n_samples))

    # Points with known min/max values
    gaze_direction[:, 0] = np.array([0, 0, 1])
    gaze_direction[:, 1] = np.array([1, 0, 0])
    gaze_direction[:, 2] = np.array([0, 1, 0])

    # Compute radius range
    radius_range = InterSaccadicEvent.compute_mean_gaze_direction_radius_range(gaze_direction)

    # Expected range is sqrt((1-0)^2 + (1-0)^2 + (1-0)^2)
    expected_range = np.sqrt(3)
    np.testing.assert_almost_equal(radius_range, expected_range, decimal=5)


def test_compute_larsson_parameters():
    """Test that compute_larsson_parameters correctly computes parameters."""
    # Create a mock InterSaccadicEvent
    mock_data, identified_indices = mock_data_object()
    event = InterSaccadicEvent(mock_data, np.zeros(10, dtype=bool))

    # Compute Larsson parameters
    parameter_D, parameter_CD, parameter_PD, parameter_R = event.compute_larsson_parameters(mock_data.gaze_direction)

    # Check parameters
    npt.assert_almost_equal(parameter_D, 0.7117307016032802)
    npt.assert_almost_equal(parameter_CD, 0.12052426548425645)
    npt.assert_almost_equal(parameter_PD, 0.004991590634662587)
    npt.assert_almost_equal(parameter_R, 0.6368709148754008 * 180 / np.pi)

    # TODO: test the actual values against known good values or expected ranges


@patch("eyedentify3d.identification.inter_saccades.find_time_index")
def test_get_window_sequences(mock_find_time_index):
    """Test that get_window_sequences correctly splits sequences into windows."""
    mock_data, identified_indices = mock_data_object()

    # Mock find_time_index to return predictable values
    mock_find_time_index.side_effect = lambda time_vector, target_time, method: int(target_time * 100)

    event = InterSaccadicEvent(mock_data, identified_indices, window_duration=0.1, window_overlap=0.02)

    # Set sequences
    event.sequences = [np.arange(10, 30), np.arange(50, 70)]

    # Get window sequences
    window_sequences = event.get_window_sequences(mock_data.time_vector)

    # Check that window_sequences contains the expected windows
    assert len(window_sequences) > 0

    # Check that each window has the correct duration
    for window in window_sequences:
        assert len(window) > 0


def test_set_coherent_and_incoherent_sequences():
    """Test that set_coherent_and_incoherent_sequences correctly classifies sequences."""
    mock_data, identified_indices = mock_data_object()

    event = InterSaccadicEvent(
        mock_data, identified_indices, minimal_duration=0.05, window_duration=0.1, window_overlap=0.02, eta_p=0.05
    )

    # Set frame_indices and sequences
    event.frame_indices = np.arange(100)
    event.sequences = [np.arange(10, 30), np.arange(50, 70)]

    # Mock get_window_sequences to return predictable values
    with (
        patch.object(
            event,
            "get_window_sequences",
            return_value=[np.arange(10, 20), np.arange(20, 30), np.arange(50, 60), np.arange(60, 70)],
        ),
        patch.object(
            event,
            "detect_directionality_coherence_on_axis",
            side_effect=[[0.01, 0.01], [0.1, 0.1], [0.01, 0.01], [0.1, 0.1]],
        ),
    ):

        event.set_coherent_and_incoherent_sequences()

        # Check that coherent_sequences and incoherent_sequences are set
        assert event.coherent_sequences is not None
        assert event.incoherent_sequences is not None


def test_classify_obvious_sequences():
    """Test that classify_obvious_sequences correctly classifies obvious sequences."""
    mock_data, identified_indices = mock_data_object()

    event = InterSaccadicEvent(
        mock_data,
        identified_indices,
        minimal_duration=0.0001,
        window_duration=0.05,
        window_overlap=0.01,
        eta_d=0.5,
        eta_cd=0.7,
        eta_pd=0.7,
        eta_max_fixation=2.0,
    )

    # Mock compute_larsson_parameters to return predictable values
    # First sequence: all criteria false (fixation)
    # Second sequence: all criteria true (smooth pursuit)
    # Third sequence: mixed criteria (ambiguous)
    with patch.object(
        event,
        "compute_larsson_parameters",
        side_effect=[
            (0.6, 0.6, 0.6, 1.0),  # Fixation
            (0.4, 0.8, 0.8, 3.0),  # Smooth pursuit
            (0.4, 0.6, 0.8, 1.0),  # Ambiguous
        ],
    ):

        fixation_indices, smooth_pursuit_indices, ambiguous_indices = event.classify_obvious_sequences(
            mock_data, [np.arange(10, 20), np.arange(30, 40), np.arange(50, 60)]
        )

        # Check that indices are classified correctly
        np.testing.assert_array_equal(fixation_indices, np.arange(10, 20))
        np.testing.assert_array_equal(smooth_pursuit_indices, np.arange(30, 40))
        np.testing.assert_array_equal(ambiguous_indices, np.arange(50, 60))


def test_classify_ambiguous_sequences():
    """Test that classify_ambiguous_sequences correctly classifies ambiguous sequences."""
    mock_data, identified_indices = mock_data_object()

    event = InterSaccadicEvent(
        mock_data, identified_indices, eta_pd=0.7, eta_max_fixation=2.0, eta_min_smooth_pursuit=5.0, phi=30.0
    )

    # Set up test data
    all_sequences = [np.arange(10, 20), np.arange(30, 40), np.arange(50, 60)]
    ambiguous_indices = np.arange(50, 60)
    fixation_indices = np.arange(10, 20)
    smooth_pursuit_indices = np.arange(30, 40)

    # Mock methods used in classify_ambiguous_sequences
    # Fixation criteria_3 false and criteria_4 false
    with (
        patch.object(event, "compute_larsson_parameters", return_value=(0.6, 0.6, 0.3, 1.0)),
        patch.object(event, "_find_mergeable_segment_range", return_value=None),
    ):

        new_fixation_indices, new_smooth_pursuit_indices, uncertain_sequences = event.classify_ambiguous_sequences(
            mock_data, all_sequences, ambiguous_indices, fixation_indices, smooth_pursuit_indices
        )

        # Check that indices are classified correctly
        assert len(new_fixation_indices) > len(fixation_indices)
        assert len(new_fixation_indices) == 20
        assert len(new_smooth_pursuit_indices) == len(smooth_pursuit_indices)
        assert len(uncertain_sequences) == 0

    # Fixation criteria_3 false and criteria_4 false
    with (
        patch.object(event, "compute_larsson_parameters", return_value=(0.6, 0.6, 0.3, 30.0)),
        patch.object(event, "_find_mergeable_segment_range", return_value=None),
    ):
        new_fixation_indices, new_smooth_pursuit_indices, uncertain_sequences = event.classify_ambiguous_sequences(
            mock_data, all_sequences, ambiguous_indices, fixation_indices, smooth_pursuit_indices
        )

        # Check that indices are classified correctly
        assert len(new_fixation_indices) == len(fixation_indices)
        assert len(new_smooth_pursuit_indices) > len(smooth_pursuit_indices)
        assert len(new_smooth_pursuit_indices) == 20
        assert len(uncertain_sequences) == 0

    # Fixation criteria_3 true and the mergable criteria si not met, so the sequence is unidentified
    with (
        patch.object(event, "compute_larsson_parameters", return_value=(0.6, 0.6, 0.8, 30.0)),
        patch.object(event, "_find_mergeable_segment_range", return_value=None),
    ):
        new_fixation_indices, new_smooth_pursuit_indices, uncertain_sequences = event.classify_ambiguous_sequences(
            mock_data, all_sequences, ambiguous_indices, fixation_indices, smooth_pursuit_indices
        )

        # Check that indices are classified correctly
        assert len(new_fixation_indices) == len(fixation_indices)
        assert len(new_smooth_pursuit_indices) == len(smooth_pursuit_indices)
        assert len(uncertain_sequences) == 1
        assert len(uncertain_sequences[0]) == 10


def test_initialize():
    """Test that initialize correctly sets up the event."""
    mock_data, identified_indices = mock_data_object()

    # Create an InterSaccadicEvent with mocked methods
    event = InterSaccadicEvent(mock_data, identified_indices)

    # Mock all the methods called in initialize
    with (
        patch.object(event, "detect_intersaccadic_indices") as mock_detect,
        patch.object(event, "split_sequences") as mock_split,
        patch.object(event, "keep_only_sequences_long_enough") as mock_keep,
        patch.object(event, "set_coherent_and_incoherent_sequences") as mock_set_coherent,
        patch.object(event, "set_intersaccadic_sequences") as mock_set_intersaccadic,
        patch.object(event, "classify_sequences") as mock_classify,
    ):

        event.initialize()

        # Check that all methods were called in the correct order
        mock_detect.assert_called_once()
        mock_split.assert_called_once()
        mock_keep.assert_called_once()
        mock_set_coherent.assert_called_once()
        mock_set_intersaccadic.assert_called_once()
        mock_classify.assert_called_once()


def test_finalize():
    """Test that finalize correctly sets up the event."""
    mock_data, identified_indices = mock_data_object()
    event = InterSaccadicEvent(mock_data, np.zeros(10, dtype=bool))

    # Set up test data
    fixation_sequences = [np.array([1, 2, 3]), np.array([7, 8])]
    smooth_pursuit_sequences = [np.array([4, 5, 6])]

    # Call finalize
    event.finalize(fixation_sequences, smooth_pursuit_sequences)

    # Check that sequences and frame_indices are set correctly
    assert len(event.sequences) == 3
    np.testing.assert_array_equal(event.frame_indices, np.array([1, 2, 3, 4, 5, 6, 7, 8]))
