import pytest
import numpy as np
import numpy.testing as npt
import pandas as pd
from unittest.mock import patch

from eyedentify3d import PicoNeoData, ErrorType, TimeRange


@pytest.fixture
def mock_csv_data():
    """Create mock CSV data for testing"""
    # Create DataFrames with minimal required columns
    csv_data = {
        "Timeline": np.linspace(10, 11, 200),
        "Left Eye Pose Status": [52] * 200,
        "Right Eye Pose Status": [52] * 200,
        "Right Eye Gaze Openness": [1] * 200,
        "Left Eye Gaze Openness": [1] * 200,
        "Combine Eye Gaze Vector. x": [0] * 200,
        "Combine Eye Gaze Vector. y": [0] * 200,
        "Combine Eye Gaze Vector. z": [1] * 200,
        "Head Rotation. x": [0] * 200,
        "Head Rotation. y": [0] * 200,
        "Head Rotation. z": [0] * 200,
    }
    return pd.DataFrame(csv_data)


@pytest.fixture
def mock_empty_csv_data():
    """Create mock empty CSV data for testing"""
    # Create DataFrames with minimal required columns
    csv_data = {
        "Timeline": [],
        "Left Eye Pose Status": [],
        "Right Eye Pose Status": [],
        "Right Eye Gaze Openness": [],
        "Left Eye Gaze Openness": [],
        "Combine Eye Gaze Vector. x": [],
        "Combine Eye Gaze Vector. y": [],
        "Combine Eye Gaze Vector. z": [],
        "Head Rotation. x": [],
        "Head Rotation. y": [],
        "Head Rotation. z": [],
    }
    return pd.DataFrame(csv_data)


@patch("pandas.read_csv")
def test_pico_neo_data_init(mock_read_csv, mock_csv_data):
    """Test initialization of PupilInvisibleData"""
    mock_read_csv.return_value = mock_csv_data

    data = PicoNeoData("test.csv")

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
    data = PicoNeoData.__new__(PicoNeoData)  # Create instance without calling __init__
    data.data_file_path = "valid_file.csv"
    assert data.data_file_path == "valid_file.csv"


def test_data_file_path_setter_invalid_type():
    """Test setting an invalid data file path type"""
    data = PicoNeoData.__new__(PicoNeoData)  # Create instance without calling __init__
    with pytest.raises(ValueError, match="The data_file_path must be a string, got 123."):
        data.data_file_path = 123


def test_data_file_path_setter_invalid_extension():
    """Test setting an invalid data file path extension"""
    data = PicoNeoData.__new__(PicoNeoData)  # Create instance without calling __init__
    with pytest.raises(ValueError, match="The Pico Neo data file must be a .csv file, got invalid_file.txt."):
        data.data_file_path = "invalid_file.txt"


def test_check_validity_empty_file():
    """Test _check_validity with empty file"""
    # Create a mock data object
    data = PicoNeoData.__new__(PicoNeoData)
    data.error_type = ErrorType.SKIP
    data.csv_data = pd.DataFrame({"Timeline": [], "Left Eye Pose Status": [], "Right Eye Pose Status": []})
    data.data_file_path = "test.csv"
    data._validity_flag = True

    # Check that the validity flag is modified by _check_validity
    data._check_validity()
    assert data._validity_flag is False


def test_check_validity_invalid_data():
    """Test _check_validity with mostly invalid data"""
    # Create a mock data object
    data = PicoNeoData.__new__(PicoNeoData)
    data.error_type = ErrorType.SKIP
    data_dict = {
        "Timeline": list(range(10)),
        "Left Eye Pose Status": [0] * 8 + [31] * 2,  # 80% invalid
        "Right Eye Pose Status": [31] * 10,
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
    data = PicoNeoData.__new__(PicoNeoData)
    data.error_type = ErrorType.SKIP
    data_dict = {
        "Timeline": [1, 2, 3, 2, 5],  # Time goes backwards
        "Left Eye Pose Status": [31] * 5,
        "Right Eye Pose Status": [31] * 5,
    }
    data.csv_data = pd.DataFrame(data_dict)
    data.data_file_path = "test.csv"
    data._validity_flag = True

    # Check that the validity flag is modified by _check_validity
    data._check_validity()
    assert data._validity_flag is False


def test_set_time_vector():
    """Test _set_time_vector method"""
    data = PicoNeoData.__new__(PicoNeoData)
    data._validity_flag = True
    data.csv_data = pd.DataFrame({"Timeline": [1, 2, 3, 4]})

    data._set_time_vector()

    assert data.time_vector is not None
    assert len(data.time_vector) == 4
    assert data.time_vector[0] == 0.0  # First value should be 0
    assert np.isclose(data.time_vector[1], 1)


def test_remove_duplicates():
    """Test _remove_duplicates method"""
    data = PicoNeoData.__new__(PicoNeoData)
    data._validity_flag = True
    data.time_vector = np.array([0.0, 0.1, 0.1, 0.2, 0.3])  # Duplicate at index 2
    data.csv_data = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})

    with pytest.raises(
        RuntimeError,
        match="The time vector has duplicated frames, which never happened with this eye-tracker. Please notify the developer.",
    ):
        data._remove_duplicates()


def test_discard_data_out_of_range():
    """Test _discard_data_out_of_range method"""
    data = PicoNeoData.__new__(PicoNeoData)
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
    data = PicoNeoData.__new__(PicoNeoData)
    data._validity_flag = True
    data.csv_data = pd.DataFrame({"Right Eye Gaze Openness": [0, 0, 1], "Left Eye Gaze Openness": [1, 0, 0]})

    data._set_eye_openness()

    assert np.array_equal(data.right_eye_openness, np.array([0, 0, 1]))
    assert np.array_equal(data.left_eye_openness, np.array([1, 0, 0]))


def test_set_eye_direction():
    """Test _set_eye_direction method"""
    data = PicoNeoData.__new__(PicoNeoData)
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

    data.csv_data = pd.DataFrame(
        {"Combine Eye Gaze Vector. x": x, "Combine Eye Gaze Vector. y": y, "Combine Eye Gaze Vector. z": z}
    )

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
    data = PicoNeoData.__new__(PicoNeoData)
    data._validity_flag = True
    data.data_file_path = "test.csv"
    data.error_type = ErrorType.SKIP

    # Create vectors with invalid norms
    data.csv_data = pd.DataFrame(
        {
            "Combine Eye Gaze Vector. x": [2.0, 0.1, 0.1],  # First vector has norm > 1.2
            "Combine Eye Gaze Vector. y": [0.0, 0.1, 0.1],
            "Combine Eye Gaze Vector. z": [0.0, 0.1, 0.1],
        }
    )

    data._set_eye_direction()
    assert data._validity_flag is False


def test_set_head_angles():
    """Test _set_head_angles method"""
    data = PicoNeoData.__new__(PicoNeoData)
    data._validity_flag = True
    data.csv_data = pd.DataFrame(
        {
            "Head Rotation. x": [10, 11, 12, 13],
            "Head Rotation. y": [20, 21, 22, 23],
            "Head Rotation. z": [30, 31, 32, 33],
        }
    )

    data._set_head_angles()

    assert data.head_angles.shape == (3, 4)
    npt.assert_almost_equal(data.head_angles, np.array([[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]))


def test_set_head_angular_velocity():
    """Test _set_head_angular_velocity method"""
    data = PicoNeoData.__new__(PicoNeoData)
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
    data = PicoNeoData.__new__(PicoNeoData)
    data._validity_flag = True
    data.csv_data = pd.DataFrame(
        {
            "Left Eye Pose Status": [52, 52, 0],
            "Right Eye Pose Status": [52, 1, 52],
        }  # Second frame is invalid  # Third frame is invalid
    )

    data._set_data_invalidity()

    assert data.data_invalidity is not None
    assert np.array_equal(data.data_invalidity, np.array([False, True, True]))
