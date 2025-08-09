import pytest
import numpy as np
import numpy.testing as npt
import pandas as pd
from unittest.mock import patch

from eyedentify3d import HtcViveProData, ErrorType, TimeRange


@pytest.fixture
def mock_csv_data():
    """Create mock CSV data for testing"""
    # Create a DataFrame with minimal required columns
    data = {
        "time(100ns)": np.linspace(1, 2, 200),
        "eye_valid_L": [31] * 200,
        "eye_valid_R": [31] * 200,
        "openness_L": [0.9] * 200,
        "openness_R": [0.9] * 200,
        "gaze_direct_L.x": [0.1, 0.2, 0.3, 0.4, 0.5] * 40,
        "gaze_direct_L.y": [0.1, 0.2, 0.3, 0.4, 0.5] * 40,
        "gaze_direct_L.z": [0.9, 0.8, 0.7, 0.6, 0.5] * 40,
        "helmet_rot_x": [10, 11, 12, 13, 14] * 40,
        "helmet_rot_y": [20, 21, 22, 23, 24] * 40,
        "helmet_rot_z": [30, 31, 32, 33, 34] * 40,
    }
    return pd.DataFrame(data)


@patch("pandas.read_csv")
def test_htc_vive_pro_data_init(mock_read_csv, mock_csv_data):
    """Test initialization of HtcViveProData"""
    mock_read_csv.return_value = mock_csv_data

    data = HtcViveProData("test.csv")

    assert data.data_file_path == "test.csv"
    assert data._validity_flag is True
    assert data.dt is not None
    assert data.time_vector is not None
    assert data.right_eye_openness is not None
    assert data.left_eye_openness is not None
    assert data.eye_direction is not None
    assert data.head_angles is not None
    assert data.head_angular_velocity is not None
    assert data.head_velocity_norm is not None
    assert data.data_invalidity is not None


def test_data_file_path_setter_valid():
    """Test setting a valid data file path"""
    data = HtcViveProData.__new__(HtcViveProData)  # Create instance without calling __init__
    data.data_file_path = "valid_file.csv"
    assert data.data_file_path == "valid_file.csv"


def test_data_file_path_setter_invalid_type():
    """Test setting an invalid data file path type"""
    data = HtcViveProData.__new__(HtcViveProData)  # Create instance without calling __init__
    with pytest.raises(ValueError, match="The data_file_path must be a string, got 123."):
        data.data_file_path = 123


def test_data_file_path_setter_invalid_extension():
    """Test setting an invalid data file path extension"""
    data = HtcViveProData.__new__(HtcViveProData)  # Create instance without calling __init__
    with pytest.raises(ValueError, match="The HTC Vive Pro data file must be a .csv file, got invalid_file.txt."):
        data.data_file_path = "invalid_file.txt"


def test_check_validity_empty_file():
    """Test _check_validity with empty file"""
    # Create a mock data object
    data = HtcViveProData.__new__(HtcViveProData)
    data.error_type = ErrorType.SKIP
    data.csv_data = pd.DataFrame({"time(100ns)": [], "eye_valid_L": [], "eye_valid_R": []})
    data.data_file_path = "test.csv"
    data._validity_flag = True

    # Check that the validity flag is modified by _check_validity
    data._check_validity()
    assert data._validity_flag is False


def test_check_validity_invalid_data():
    """Test _check_validity with mostly invalid data"""
    # Create a mock data object
    data = HtcViveProData.__new__(HtcViveProData)
    data.error_type = ErrorType.SKIP
    data_dict = {
        "time(100ns)": list(range(10)),
        "eye_valid_L": [0] * 8 + [31] * 2,  # 80% invalid
        "eye_valid_R": [31] * 10,
    }
    data.csv_data = pd.DataFrame(data_dict)
    data.data_file_path = "test.csv"
    data._validity_flag = True

    # Check that the validity flag is modified by _check_validity
    data._check_validity()
    assert data._validity_flag is False


def test_check_validity_non_increasing_time():
    """Test _check_validity with non-increasing time vector"""

    # Create a mock data object
    data = HtcViveProData.__new__(HtcViveProData)
    data.error_type = ErrorType.SKIP
    data_dict = {
        "time(100ns)": [1, 2, 3, 2, 5],  # Time goes backwards
        "eye_valid_L": [31] * 5,
        "eye_valid_R": [31] * 5,
    }
    data.csv_data = pd.DataFrame(data_dict)
    data.data_file_path = "test.csv"
    data._validity_flag = True

    # Check that the validity flag is modified by _check_validity
    data._check_validity()
    assert data._validity_flag is False


def test_set_time_vector():
    """Test _set_time_vector method"""
    data = HtcViveProData.__new__(HtcViveProData)
    data._validity_flag = True
    data.csv_data = pd.DataFrame({"time(100ns)": [1000, 2000, 3000, 4000]})

    data._set_time_vector()

    assert data.time_vector is not None
    assert len(data.time_vector) == 4
    assert data.time_vector[0] == 0.0  # First value should be 0
    assert np.isclose(data.time_vector[1], 0.0001)  # 1000/10^7 seconds


def test_remove_duplicates():
    """Test _remove_duplicates method"""
    data = HtcViveProData.__new__(HtcViveProData)
    data._validity_flag = True
    data.time_vector = np.array([0.0, 0.1, 0.1, 0.2, 0.3])  # Duplicate at index 2
    data.csv_data = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})

    data._remove_duplicates()

    assert len(data.time_vector) == 4  # One duplicate removed
    assert np.array_equal(data.time_vector, np.array([0.0, 0.1, 0.2, 0.3]))
    assert len(data.csv_data) == 4


def test_discard_data_out_of_range():
    """Test _discard_data_out_of_range method"""
    data = HtcViveProData.__new__(HtcViveProData)
    data._validity_flag = True
    data.time_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    data.csv_data = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})
    data.time_range = TimeRange(0.15, 0.35)

    data._discard_data_out_of_range()

    assert len(data.time_vector) == 2
    assert np.array_equal(data.time_vector, np.array([0.2, 0.3]))
    assert len(data.csv_data) == 2
    assert np.array_equal(data.csv_data["col1"].values, np.array([3, 4]))


def test_set_eye_openness():
    """Test _set_eye_openness method"""
    data = HtcViveProData.__new__(HtcViveProData)
    data._validity_flag = True
    data.csv_data = pd.DataFrame({"openness_R": [0.8, 0.9, 1.0], "openness_L": [0.7, 0.8, 0.9]})

    data._set_eye_openness()

    assert np.array_equal(data.right_eye_openness, np.array([0.8, 0.9, 1.0]))
    assert np.array_equal(data.left_eye_openness, np.array([0.7, 0.8, 0.9]))


def test_set_eye_direction():
    """Test _set_eye_direction method"""
    data = HtcViveProData.__new__(HtcViveProData)
    data._validity_flag = True
    data.data_file_path = "test.csv"
    data.error_type = ErrorType.SKIP

    # Create normalized vectors
    x = [0.1, 0.2, 0.3, 0.1]
    y = [0.1, 0.2, 0.3, -0.1]
    z = [0.9, 0.8, 0.7, 0.9]

    # Ensure they're unit vectors
    for i in range(4):
        norm = np.sqrt(x[i] ** 2 + y[i] ** 2 + z[i] ** 2)
        x[i] /= norm
        y[i] /= norm
        z[i] /= norm

    data.csv_data = pd.DataFrame({"gaze_direct_L.x": x, "gaze_direct_L.y": y, "gaze_direct_L.z": z})

    data._set_eye_direction()

    assert data.eye_direction.shape == (3, 4)
    # Check that vectors are normalized
    norms = np.linalg.norm(data.eye_direction, axis=0)
    assert np.allclose(norms, 1.0)
    npt.assert_almost_equal(
        data.eye_direction,
        np.array(
            [
                [0.10976426, 0.23570226, 0.36650833, 0.10976426],
                [0.10976426, 0.23570226, 0.36650833, -0.10976426],
                [0.98787834, 0.94280904, 0.85518611, 0.98787834],
            ]
        ),
    )


def test_set_eye_direction_invalid_norm():
    """Test _set_eye_direction method with invalid norm"""
    data = HtcViveProData.__new__(HtcViveProData)
    data._validity_flag = True
    data.data_file_path = "test.csv"
    data.error_type = ErrorType.SKIP

    # Create vectors with invalid norms
    data.csv_data = pd.DataFrame(
        {
            "gaze_direct_L.x": [2.0, 0.1, 0.1],  # First vector has norm > 1.2
            "gaze_direct_L.y": [0.0, 0.1, 0.1],
            "gaze_direct_L.z": [0.0, 0.1, 0.1],
        }
    )

    data._set_eye_direction()
    assert data._validity_flag is False


def test_interpolate_repeated_frames():
    """Test interpolate_repeated_frames method"""
    data = HtcViveProData.__new__(HtcViveProData)

    # Create data with repeated frames
    test_data = np.array(
        [
            [1.0, 1.0, 1.0, 4.0, 5.0],  # First 3 frames are identical
            [2.0, 2.0, 2.0, 8.0, 10.0],
            [3.0, 3.0, 3.0, 12.0, 15.0],
        ]
    )

    result = data.interpolate_repeated_frames(test_data)

    # Check shape is preserved
    assert result.shape == test_data.shape

    # Check that repeated frames are interpolated, but unique frames are preserved
    npt.assert_almost_equal(
        result, np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0], [3.0, 6.0, 9.0, 12.0, 15.0]])
    )


def test_interpolate_repeated_frames_invalid_shape():
    """Test interpolate_repeated_frames method with invalid shape"""
    data = HtcViveProData.__new__(HtcViveProData)

    # Create data with invalid shape
    test_data = np.array([1.0, 2.0, 3.0])

    with pytest.raises(
        NotImplementedError, match=r"This function was designed for matrix data of shape \(3, n_frames\). "
    ):
        data.interpolate_repeated_frames(test_data)


def test_set_head_angles():
    """Test _set_head_angles method"""
    data = HtcViveProData.__new__(HtcViveProData)
    data._validity_flag = True
    data.csv_data = pd.DataFrame(
        {"helmet_rot_x": [10, 11, 12, 13], "helmet_rot_y": [20, 21, 22, 23], "helmet_rot_z": [30, 31, 32, 33]}
    )

    # Mock the interpolate_repeated_frames method
    original_interpolate = data.interpolate_repeated_frames
    data.interpolate_repeated_frames = lambda x: x

    data._set_head_angles()

    # Restore original method
    data.interpolate_repeated_frames = original_interpolate

    assert data.head_angles.shape == (3, 4)
    npt.assert_almost_equal(data.head_angles, np.array([[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]))


def test_set_head_angular_velocity():
    """Test _set_head_angular_velocity method"""
    data = HtcViveProData.__new__(HtcViveProData)
    data._validity_flag = True
    data.time_vector = np.linspace(0, 1, 200)
    data.head_angles = np.zeros((3, 200))
    for i in range(3):
        data.head_angles[i, :] = np.linspace(0, 2 * np.pi, 200)

    data._set_head_angular_velocity()

    assert data.head_angular_velocity is not None
    assert data.head_angular_velocity.shape == (3, 200)
    # All values should be 2pi
    npt.assert_almost_equal(data.head_angular_velocity[0, 0], 2 * np.pi)
    npt.assert_almost_equal(data.head_angular_velocity[1, 100], 2 * np.pi)
    npt.assert_almost_equal(data.head_angular_velocity[2, 50], 2 * np.pi)
    npt.assert_almost_equal(data.head_angular_velocity[0, 150], 2 * np.pi)
    assert data.head_velocity_norm is not None

    norm = np.sqrt(3 * (2 * np.pi) ** 2)
    npt.assert_almost_equal(norm, 10.88279619)
    npt.assert_almost_equal(data.head_velocity_norm[0], norm)
    npt.assert_almost_equal(data.head_velocity_norm[50], norm)
    npt.assert_almost_equal(data.head_velocity_norm[100], norm)
    npt.assert_almost_equal(data.head_velocity_norm[150], norm)
    assert data.head_velocity_norm.shape == (200,)


def test_set_data_invalidity():
    """Test _set_data_invalidity method"""
    data = HtcViveProData.__new__(HtcViveProData)
    data._validity_flag = True
    data.csv_data = pd.DataFrame(
        {"eye_valid_L": [31, 30, 31], "eye_valid_R": [31, 31, 30]}  # Second frame is invalid  # Third frame is invalid
    )

    data._set_data_invalidity()

    assert data.data_invalidity is not None
    assert np.array_equal(data.data_invalidity, np.array([False, True, True]))
