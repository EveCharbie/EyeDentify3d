import pytest
import numpy as np
import numpy.testing as npt
import json
from unittest.mock import patch, MagicMock

from eyedentify3d import TobiiProGlassesData, ErrorType, TimeRange


@pytest.fixture
def mock_gz_data():
    """Create mock gz data for testing"""

    gaze_lines = []
    for i in range(200):
        t = i / 100  # 0 to 2 seconds
        gaze3d = [
            -10 + 20 * i / 199,  # azimuth from -10 to 10
            -5 + 10 * i / 199,  # elevation from -5 to 5
            0.5,  # z component (will be normalized anyway)
        ]
        pupil_left = np.nan if 100 <= i < 110 else 1.0
        pupil_right = np.nan if 50 <= i < 60 else 1.0

        line_data = {
            "timestamp": t,
            "data": {
                "gaze3d": gaze3d,
                "eyeleft": {"pupildiameter": pupil_left},
                "eyeright": {"pupildiameter": pupil_right},
            },
        }
        # Format as the weird JSON format with key: instead of "key":
        line = json.dumps(line_data).replace('"timestamp":', "timestamp:").replace('"data":', "data:")
        gaze_lines.append(line)

    imu_lines = []
    for i in range(200):
        t = 0.01 + i / 100  # 0.01 to 2.01 seconds

        # 100Hz data (accelerometer + gyroscope)
        if i % 1 == 0:  # Every sample for 100Hz
            line_data = {
                "timestamp": t,
                "data": {
                    "accelerometer": [-10 + 20 * i / 199, -5 + 10 * i / 199, 0.5],  # x -10 to 10  # y -5 to 5
                    "gyroscope": [-20 + 20 * i / 199, -15 + 10 * i / 199, -5 + 5 * i / 199],
                },
            }
            line = json.dumps(line_data).replace('"timestamp":', "timestamp:").replace('"data":', "data:")
            imu_lines.append(line)

        # 10Hz data (magnetometer)
        if i % 10 == 0:  # Every 10th sample for 10Hz
            line_data = {
                "timestamp": t,
                "data": {"magnetometer": [-100 + 100 * i / 199, -50 + 50 * i / 199, -75 + 75 * i / 199]},
            }
            line = json.dumps(line_data).replace('"timestamp":', "timestamp:").replace('"data":', "data:")
            imu_lines.append(line)

    return {"gazedata": gaze_lines, "imudata": imu_lines}


@patch("eyedentify3d.data_parsers.tobii_pro_glasses_data.gzip.open")
def test_tobii_pro_data_init(mock_gzip_open, mock_gz_data):
    """Test initialization of TobiiProGlassesData"""

    # Configure the mock to return different file contents for different file paths
    def side_effect(path, mode="rt"):
        mock_file = MagicMock()
        if path.endswith("gazedata.gz"):
            # Create a CSV reader-like iterator
            mock_file.__enter__.return_value = mock_gz_data["gazedata"]
            mock_file.__exit__.return_value = None
        elif path.endswith("imudata.gz"):
            mock_file.__enter__.return_value = mock_gz_data["imudata"]
            mock_file.__exit__.return_value = None
        return mock_file

    mock_gzip_open.side_effect = side_effect

    data = TobiiProGlassesData("test_folder/")

    assert data.data_folder_path == "test_folder/"
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


def test_data_folder_path_setter_valid():
    """Test setting a valid data folder path"""
    data = TobiiProGlassesData.__new__(TobiiProGlassesData)  # Create instance without calling __init__
    data.data_folder_path = "valid_folder"
    assert data.data_folder_path == "valid_folder/"  # Should add trailing slash


def test_data_folder_path_setter_with_slash():
    """Test setting a valid data folder path with trailing slash"""
    data = TobiiProGlassesData.__new__(TobiiProGlassesData)  # Create instance without calling __init__
    data.data_folder_path = "valid_folder/"
    assert data.data_folder_path == "valid_folder/"  # Should keep trailing slash


def test_data_folder_path_setter_invalid_type():
    """Test setting an invalid data folder path type"""
    data = TobiiProGlassesData.__new__(TobiiProGlassesData)  # Create instance without calling __init__
    with pytest.raises(ValueError, match="The data_folder_path must be a string, got 123."):
        data.data_folder_path = 123


@patch("eyedentify3d.data_parsers.tobii_pro_glasses_data.gzip.open")
def test_check_validity_empty_file(mock_gzip_open, mock_gz_data):
    """Test _check_validity with empty file"""

    # Configure the mock to return different file contents for different file paths
    def side_effect(path, mode="rt"):
        mock_file = MagicMock()
        if path.endswith("gazedata.gz"):
            # Create a CSV reader-like iterator
            mock_file.__enter__.return_value = mock_gz_data["gazedata"]
            mock_file.__exit__.return_value = None
        elif path.endswith("imudata.gz"):
            mock_file.__enter__.return_value = mock_gz_data["imudata"]
            mock_file.__exit__.return_value = None
        return mock_file

    mock_gzip_open.side_effect = side_effect
    data = TobiiProGlassesData("test_folder/", error_type=ErrorType.SKIP)

    data._validity_flag = True

    # Check that the validity flag is modified by _check_validity
    data.gaze_data_dict["timestamp"] = []
    data._check_validity()
    assert data._validity_flag is False


@patch("eyedentify3d.data_parsers.tobii_pro_glasses_data.gzip.open")
def test_check_validity_invalid_data(mock_gzip_open, mock_gz_data):
    """Test _check_validity with mostly invalid data"""

    # Configure the mock to return different file contents for different file paths
    def side_effect(path, mode="rt"):
        mock_file = MagicMock()
        if path.endswith("gazedata.gz"):
            # Create a CSV reader-like iterator
            mock_file.__enter__.return_value = mock_gz_data["gazedata"]
            mock_file.__exit__.return_value = None
        elif path.endswith("imudata.gz"):
            mock_file.__enter__.return_value = mock_gz_data["imudata"]
            mock_file.__exit__.return_value = None
        return mock_file

    mock_gzip_open.side_effect = side_effect
    data = TobiiProGlassesData("test_folder/", error_type=ErrorType.SKIP)
    data.gaze_data_dict["pupil_diameter_left"] = [np.nan] * 180 + [1] * 20  # 90% invalid
    data.gaze_data_dict["pupil_diameter_right"] = [np.nan] * 180 + [1] * 20  # 90% invalid

    assert data._validity_flag is True

    # Check that the validity flag is modified by _check_validity
    data._check_validity()
    assert data._validity_flag is False


@patch("eyedentify3d.data_parsers.tobii_pro_glasses_data.gzip.open")
def test_check_validity_non_increasing_time(mock_gzip_open, mock_gz_data):
    """Test _check_validity with non-increasing time vector"""

    # Configure the mock to return different file contents for different file paths
    def side_effect(path, mode="rt"):
        mock_file = MagicMock()
        if path.endswith("gazedata.gz"):
            # Create a CSV reader-like iterator
            mock_file.__enter__.return_value = mock_gz_data["gazedata"]
            mock_file.__exit__.return_value = None
        elif path.endswith("imudata.gz"):
            mock_file.__enter__.return_value = mock_gz_data["imudata"]
            mock_file.__exit__.return_value = None
        return mock_file

    mock_gzip_open.side_effect = side_effect

    data = TobiiProGlassesData("test_folder/", error_type=ErrorType.SKIP)
    data.gaze_data_dict["timestamp"][:] = np.array(data.gaze_data_dict["timestamp"])[
        ::-1
    ]  # Reverse to make non-increasing

    data._validity_flag = True

    # Check that the validity flag is modified by _check_validity
    data._check_validity()
    assert data._validity_flag is False


def test_set_time_vector():
    """Test _set_time_vector method"""
    data = TobiiProGlassesData.__new__(TobiiProGlassesData)
    data._validity_flag = True
    data.gaze_data_dict = {}
    data.gaze_data_dict["timestamp"] = [1, 2, 3, 4]
    data.imu_data_dict = {}
    data.imu_data_dict["timestamp_100Hz"] = [1, 2, 3, 4]
    data.imu_data_dict["timestamp_10Hz"] = [1, 4]

    # Before unchanged
    npt.assert_almost_equal(data.imu_data_dict["timestamp_100Hz"][0], 1)
    npt.assert_almost_equal(data.imu_data_dict["timestamp_10Hz"][1], 4)

    data._set_time_vector()

    assert data.time_vector is not None
    assert len(data.time_vector) == 4
    assert data.time_vector[0] == 0.0  # First value should be 0
    npt.assert_almost_equal(data.time_vector[1], 1.0)  # 1 second difference

    # Check that blink and imu timestamps are also transformed
    npt.assert_almost_equal(data.imu_data_dict["timestamp_10Hz"][1], 3)


def test_remove_duplicates():
    """Test _remove_duplicates method"""
    data = TobiiProGlassesData.__new__(TobiiProGlassesData)
    data._validity_flag = True
    data.time_vector = np.array([0.0, 0.1, 0.2, 0.3])  # No duplicates

    # This should not raise an exception
    data._remove_duplicates()

    # Test with duplicates
    data.time_vector = np.array([0.0, 0.1, 0.1, 0.3])  # Duplicate at index 2

    with pytest.raises(
        RuntimeError,
        match="The time vector has duplicated frames, which never happened with this eye-tracker. Please notify the developer.",
    ):
        data._remove_duplicates()


def test_discard_data_out_of_range():
    """Test _discard_data_out_of_range method"""
    data = TobiiProGlassesData.__new__(TobiiProGlassesData)
    data._validity_flag = True
    data.gaze_data_dict = {}
    data.gaze_data_dict["timestamp"] = np.array([0.1 * i for i in range(200)])
    data.gaze_data_dict["gaze_vector"] = np.array([[1, 0, 0] for _ in range(200)])
    data.gaze_data_dict["pupil_diameter_right"] = np.ones((200,))
    data.gaze_data_dict["pupil_diameter_left"] = np.ones((200,))
    data.imu_data_dict = {}
    data.imu_data_dict["timestamp_100Hz"] = np.array([0.1 * i for i in range(200)])
    data.imu_data_dict["accelerometer"] = np.array([[0.1 * i, 0.2 * i, 0.3 * i] for i in range(1, 201)])
    data.imu_data_dict["gyroscope"] = np.array([[0.1 * i, 0.2 * i, 0.3 * i] for i in range(1, 201)])
    data.imu_data_dict["timestamp_10Hz"] = np.array([0.1 * i for i in range(200)])
    data.imu_data_dict["magnetometer"] = np.array([[0.1 * i, 0.2 * i, 0.3 * i] for i in range(1, 201)])
    data.time_vector = np.array([0.1 * i for i in range(200)])
    data.time_range = TimeRange(0.15, 15.25)
    data.dt = 0.1

    data._discard_data_out_of_range()

    # Finalize initialization
    data._set_eye_openness()
    data._set_eye_direction()
    data._set_head_angles()
    data._set_gaze_direction()
    data._set_head_angular_velocity()
    data._set_data_invalidity()
    data.finalize()

    assert len(data.time_vector) == 151
    npt.assert_almost_equal(data.time_vector[0], 0.2)
    npt.assert_almost_equal(data.time_vector[-1], 15.2)
    assert len(data.right_eye_openness) == 151
    assert data.eye_direction.shape == (3, 151)
    assert data.head_angles.shape == (3, 151)
    assert data.gaze_direction.shape == (3, 151)
    assert data.head_angular_velocity.shape == (3, 151)
    assert len(data.head_velocity_norm) == 151
    assert len(data.data_invalidity) == 151


def test_set_eye_openness():
    """Test _set_eye_openness method"""
    data = TobiiProGlassesData.__new__(TobiiProGlassesData)
    data._validity_flag = True
    data.time_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    data.gaze_data_dict = {}
    data.gaze_data_dict["pupil_diameter_right"] = np.array(
        [1.0, np.nan, np.nan, np.nan, 1.0, 1.0, np.nan, np.nan, np.nan, 1.0]
    )
    data.gaze_data_dict["pupil_diameter_left"] = np.array(
        [1.0, np.nan, np.nan, np.nan, 1.0, 1.0, np.nan, np.nan, np.nan, 1.0]
    )

    data._set_eye_openness()

    npt.assert_almost_equal(data.right_eye_openness, np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]))
    npt.assert_almost_equal(data.left_eye_openness, np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]))


def test_set_eye_direction():
    """Test _set_eye_direction method"""
    data = TobiiProGlassesData.__new__(TobiiProGlassesData)
    data._validity_flag = True
    data.time_vector = np.array([0.0, 0.1, 0.2])

    # Create normalized vectors using azimuth and elevation
    data.gaze_data_dict = {}
    data.gaze_data_dict["gaze_vector"] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    data._set_eye_direction()

    assert data.eye_direction.shape == (3, 3)
    # Check that vectors are normalized
    norms = np.linalg.norm(data.eye_direction, axis=0)
    assert np.allclose(norms, 1.0)

    npt.assert_almost_equal(
        data.eye_direction,
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    )


def test_set_eye_direction_invalid_norm():
    """Test _set_eye_direction method with invalid norm"""
    data = TobiiProGlassesData.__new__(TobiiProGlassesData)
    data._validity_flag = True
    data.time_vector = np.array([0.0, 0.1, 0.2])

    # Create normalized vectors using azimuth and elevation
    data.gaze_data_dict = {}
    data.gaze_data_dict["gaze_vector"] = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    with pytest.raises(RuntimeError, match="The gaze direction norm should be 1.0, please check the data."):
        data._set_eye_direction()


def test_interpolate_to_eye_timestamps():
    """Test interpolate_to_eye_timestamps method"""
    data = TobiiProGlassesData.__new__(TobiiProGlassesData)
    data.time_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4])

    data.gaze_data_dict = {}
    data.gaze_data_dict["timestamp"] = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    data.gaze_data_dict["gaze_vector"] = np.array([[1, 0, 0] for _ in range(5)])
    data.gaze_data_dict["pupil_diameter_right"] = np.ones((5,))
    data.gaze_data_dict["pupil_diameter_left"] = np.ones((5,))
    data.imu_data_dict = {}
    data.imu_data_dict["timestamp_100Hz"] = np.array([0.0, 0.2, 0.4])
    data.imu_data_dict["accelerometer"] = np.array([[0.1 * i, 0.2 * i, 0.3 * i] for i in range(1, 4)])
    data.imu_data_dict["gyroscope"] = np.array([[0.1 * i, 0.2 * i, 0.3 * i] for i in range(1, 4)])
    data.imu_data_dict["timestamp_10Hz"] = np.array([0.2, 0.4])
    data.imu_data_dict["magnetometer"] = np.array([[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]])

    interpolated_imu = data.interpolate_to_eye_timestamps()

    # Check interpolated values
    npt.assert_almost_equal(interpolated_imu["accelerometer"][:, 0], np.array([0.1, 0.2, 0.3]))  # t=0.0, exact match
    npt.assert_almost_equal(
        interpolated_imu["accelerometer"][:, 1], np.array([0.15, 0.3, 0.45])
    )  # t=0.1, interpolation between
    npt.assert_almost_equal(interpolated_imu["accelerometer"][:, 2], np.array([0.2, 0.4, 0.6]))  # t=0.2, exact match
    npt.assert_almost_equal(interpolated_imu["gyroscope"][:, 0], np.array([0.1, 0.2, 0.3]))  # t=0.0, exact match
    npt.assert_almost_equal(
        interpolated_imu["gyroscope"][:, 1], np.array([0.15, 0.3, 0.45])
    )  # t=0.1, interpolation between
    npt.assert_almost_equal(interpolated_imu["gyroscope"][:, 2], np.array([0.2, 0.4, 0.6]))  # t=0.2, exact match
    npt.assert_almost_equal(interpolated_imu["magnetometer"][:, 2], np.array([0.1, 0.2, 0.3]))  # t=0.2, exact match
    npt.assert_almost_equal(
        interpolated_imu["magnetometer"][:, 3], np.array([0.15, 0.3, 0.45])
    )  # t=0.3, interpolation between
    npt.assert_almost_equal(interpolated_imu["magnetometer"][:, 4], np.array([0.2, 0.4, 0.6]))  # t=0.4, exact match


def test_set_head_angles_without_tags():
    """Test _set_head_angles method without tags in experiment"""
    data = TobiiProGlassesData.__new__(TobiiProGlassesData)
    data._validity_flag = True
    data.time_vector = np.array([0.0, 0.1, 0.2])

    data.gaze_data_dict = {}
    data.gaze_data_dict["timestamp"] = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    data.gaze_data_dict["gaze_vector"] = np.array([[1, 0, 0] for _ in range(5)])
    data.gaze_data_dict["pupil_diameter_right"] = np.ones((5,))
    data.gaze_data_dict["pupil_diameter_left"] = np.ones((5,))
    data.imu_data_dict = {}
    data.imu_data_dict["timestamp_100Hz"] = np.array([0.0, 0.2, 0.4])
    data.imu_data_dict["accelerometer"] = np.array([[0.1 * i, 0.2 * i, 0.3 * i] for i in range(1, 4)])
    data.imu_data_dict["gyroscope"] = np.array([[0.1 * i, 0.2 * i, 0.3 * i] for i in range(1, 4)])
    data.imu_data_dict["timestamp_10Hz"] = np.array([0.2, 0.4])
    data.imu_data_dict["magnetometer"] = np.array([[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]])

    data._set_head_angles()

    npt.assert_almost_equal(
        data.head_angles,
        np.array([[0.0, 7.75185744, 15.40319576], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )  # time_vector_imu


def test_set_head_angular_velocity():
    """Test _set_head_angular_velocity method"""
    data = TobiiProGlassesData.__new__(TobiiProGlassesData)
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
    data = TobiiProGlassesData.__new__(TobiiProGlassesData)
    data._validity_flag = True

    data.gaze_data_dict = {}
    data.gaze_data_dict["gaze_vector"] = np.array([[1, 0, 0] for _ in range(10)])
    data.gaze_data_dict["pupil_diameter_right"] = np.array(
        [1.0, 1.0, np.nan, np.nan, 1.0, 1.0, np.nan, np.nan, np.nan, 1.0]
    )
    data.gaze_data_dict["pupil_diameter_left"] = np.array(
        [1.0, np.nan, np.nan, np.nan, 1.0, 1.0, 1.0, np.nan, np.nan, 1.0]
    )

    data._set_data_invalidity()

    assert data.data_invalidity is not None
    npt.assert_almost_equal(
        data.data_invalidity, np.array([False, True, True, True, False, False, True, True, True, False])
    )
