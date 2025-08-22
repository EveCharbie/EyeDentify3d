import numpy as np
import pytest
from numpy import testing as npt

from eyedentify3d import ReducedData, ErrorType, TimeRange


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create sample data
    dt = 0.01
    time_vector = np.arange(0, 1, dt)
    n_samples = len(time_vector)

    # Create sample eye openness data
    right_eye_openness = np.ones(n_samples) * 0.8
    left_eye_openness = np.ones(n_samples) * 0.7

    # Create sample direction vectors (3D)
    eye_direction = np.zeros((3, n_samples))
    eye_direction[2, :] = 1.0  # Looking forward along z-axis

    head_angles = np.zeros((3, n_samples))
    head_angles[0, :] = np.linspace(0, 0.1, n_samples)  # Small rotation around x-axis

    gaze_direction = np.zeros((3, n_samples))
    gaze_direction[2, :] = 1.0  # Looking forward along z-axis

    head_angular_velocity = np.zeros((3, n_samples))
    head_angular_velocity[0, :] = 0.1  # Constant angular velocity around x-axis

    head_velocity_norm = np.ones(n_samples) * 0.1

    data_invalidity = np.zeros(n_samples, dtype=bool)
    data_invalidity[80:85] = True  # Some invalid data

    return {
        "dt": dt,
        "time_vector": time_vector,
        "right_eye_openness": right_eye_openness,
        "left_eye_openness": left_eye_openness,
        "eye_direction": eye_direction,
        "head_angles": head_angles,
        "gaze_direction": gaze_direction,
        "head_angular_velocity": head_angular_velocity,
        "head_velocity_norm": head_velocity_norm,
        "data_invalidity": data_invalidity,
    }


def test_reduced_data_initialization(sample_data):
    """Test that ReducedData initializes correctly."""
    # Create a ReducedData object with default time range
    data = ReducedData(
        original_dt=sample_data["dt"],
        original_time_vector=sample_data["time_vector"],
        original_right_eye_openness=sample_data["right_eye_openness"],
        original_left_eye_openness=sample_data["left_eye_openness"],
        original_eye_direction=sample_data["eye_direction"],
        original_head_angles=sample_data["head_angles"],
        original_gaze_direction=sample_data["gaze_direction"],
        original_head_angular_velocity=sample_data["head_angular_velocity"],
        original_head_velocity_norm=sample_data["head_velocity_norm"],
        original_data_invalidity=sample_data["data_invalidity"],
    )

    # Check that all attributes are set correctly
    assert data.dt == sample_data["dt"]
    npt.assert_array_equal(data.time_vector, sample_data["time_vector"])
    npt.assert_array_equal(data.right_eye_openness, sample_data["right_eye_openness"])
    npt.assert_array_equal(data.left_eye_openness, sample_data["left_eye_openness"])
    npt.assert_array_equal(data.eye_direction, sample_data["eye_direction"])
    npt.assert_array_equal(data.head_angles, sample_data["head_angles"])
    npt.assert_array_equal(data.gaze_direction, sample_data["gaze_direction"])
    npt.assert_array_equal(data.head_angular_velocity, sample_data["head_angular_velocity"])
    npt.assert_array_equal(data.head_velocity_norm, sample_data["head_velocity_norm"])
    npt.assert_array_equal(data.data_invalidity, sample_data["data_invalidity"])


def test_reduced_data_with_time_range(sample_data):
    """Test that ReducedData correctly applies time range filtering."""
    # Create a time range that only includes part of the data
    time_range = TimeRange(min_time=0.3, max_time=0.7)

    # Create a ReducedData object with the time range
    data = ReducedData(
        original_dt=sample_data["dt"],
        original_time_vector=sample_data["time_vector"],
        original_right_eye_openness=sample_data["right_eye_openness"],
        original_left_eye_openness=sample_data["left_eye_openness"],
        original_eye_direction=sample_data["eye_direction"],
        original_head_angles=sample_data["head_angles"],
        original_gaze_direction=sample_data["gaze_direction"],
        original_head_angular_velocity=sample_data["head_angular_velocity"],
        original_head_velocity_norm=sample_data["head_velocity_norm"],
        original_data_invalidity=sample_data["data_invalidity"],
        time_range=time_range,
    )

    # Get the expected indices
    expected_indices = np.arange(30, 70)

    # Check that all attributes are filtered correctly
    npt.assert_array_equal(data.time_vector, sample_data["time_vector"][expected_indices])
    npt.assert_array_equal(data.right_eye_openness, sample_data["right_eye_openness"][expected_indices])
    npt.assert_array_equal(data.left_eye_openness, sample_data["left_eye_openness"][expected_indices])
    npt.assert_array_equal(data.eye_direction, sample_data["eye_direction"][:, expected_indices])
    npt.assert_array_equal(data.head_angles, sample_data["head_angles"][:, expected_indices])
    npt.assert_array_equal(data.gaze_direction, sample_data["gaze_direction"][:, expected_indices])
    npt.assert_array_equal(data.head_angular_velocity, sample_data["head_angular_velocity"][:, expected_indices])
    npt.assert_array_equal(data.head_velocity_norm, sample_data["head_velocity_norm"][expected_indices])
    npt.assert_array_equal(data.data_invalidity, sample_data["data_invalidity"][expected_indices])


def test_set_indices_with_none_time_vector():
    """Test that _set_indices raises ValueError when time_vector is None."""
    data = ReducedData.__new__(ReducedData)  # Create instance without calling __init__
    data.time_range = TimeRange()

    with pytest.raises(ValueError, match="The time vector must be provided."):
        data._set_indices(None)


def test_property_setters_with_none_values(sample_data):
    """Test that property setters handle None values correctly."""
    # Create a ReducedData object
    data = ReducedData(
        original_dt=sample_data["dt"],
        original_time_vector=sample_data["time_vector"],
        original_right_eye_openness=sample_data["right_eye_openness"],
        original_left_eye_openness=sample_data["left_eye_openness"],
        original_eye_direction=sample_data["eye_direction"],
        original_head_angles=sample_data["head_angles"],
        original_gaze_direction=sample_data["gaze_direction"],
        original_head_angular_velocity=sample_data["head_angular_velocity"],
        original_head_velocity_norm=sample_data["head_velocity_norm"],
        original_data_invalidity=sample_data["data_invalidity"],
    )

    # Test setting properties to None
    data.time_vector = None
    assert data._time_vector is None

    data.right_eye_openness = None
    assert data._time_vector is None

    data.left_eye_openness = None
    assert data._time_vector is None

    data.eye_direction = None
    assert data._time_vector is None

    data.head_angles = None
    assert data._time_vector is None

    data.gaze_direction = None
    assert data._time_vector is None

    data.head_angular_velocity = None
    assert data._time_vector is None

    data.head_velocity_norm = None
    assert data._time_vector is None

    data.data_invalidity = None
    assert data._time_vector is None


def test_property_setters_without_indices():
    """Test that property setters raise RuntimeError when indices is None."""
    data = ReducedData.__new__(ReducedData)  # Create instance without calling __init__
    data.indices = None

    with pytest.raises(
        RuntimeError, match="The time vector can only be set once the indices are initialized using _set_indices."
    ):
        data.time_vector = np.array([0, 1, 2])


def test_abstract_methods_implementation():
    """Test that abstract methods are implemented (even as pass-through)."""
    # Create a minimal ReducedData object
    time_vector = np.array([0, 1, 2])
    data = ReducedData(
        original_dt=1.0,
        original_time_vector=time_vector,
        original_right_eye_openness=np.ones_like(time_vector),
        original_left_eye_openness=np.ones_like(time_vector),
        original_eye_direction=np.ones((3, len(time_vector))),
        original_head_angles=np.ones((3, len(time_vector))),
        original_gaze_direction=np.ones((3, len(time_vector))),
        original_head_angular_velocity=np.ones((3, len(time_vector))),
        original_head_velocity_norm=np.ones_like(time_vector),
        original_data_invalidity=np.zeros_like(time_vector, dtype=bool),
    )

    # Call abstract methods to ensure they don't raise exceptions
    data._check_validity()
    data._set_time_vector()
    data._discard_data_out_of_range()
    data._set_eye_openness()
    data._set_eye_direction()
    data._set_head_angles()
    data._set_head_angular_velocity()
    data._set_data_invalidity()


def test_error_type_property():
    """Test the error_type property and setter."""
    # Create a minimal ReducedData object
    time_vector = np.array([0, 1, 2])
    data = ReducedData(
        original_dt=1.0,
        original_time_vector=time_vector,
        original_right_eye_openness=np.ones_like(time_vector),
        original_left_eye_openness=np.ones_like(time_vector),
        original_eye_direction=np.ones((3, len(time_vector))),
        original_head_angles=np.ones((3, len(time_vector))),
        original_gaze_direction=np.ones((3, len(time_vector))),
        original_head_angular_velocity=np.ones((3, len(time_vector))),
        original_head_velocity_norm=np.ones_like(time_vector),
        original_data_invalidity=np.zeros_like(time_vector, dtype=bool),
    )

    # Check default error_type
    assert data.error_type == ErrorType.PRINT

    # Set error_type and check
    data.error_type = ErrorType.RAISE
    assert data.error_type == ErrorType.RAISE
