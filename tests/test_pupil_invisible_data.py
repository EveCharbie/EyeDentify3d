import pytest
import numpy as np
import numpy.testing as npt
import pandas as pd
from unittest.mock import patch, MagicMock

from eyedentify3d import PupilInvisibleData, ErrorType, TimeRange


@pytest.fixture
def mock_csv_data():
    """Create mock CSV data for testing"""
    # Create DataFrames with minimal required columns
    gaze_data = {
        "timestamp [ns]": np.linspace(1e9, 2e9, 200),
        "worn": [1] * 200,
        "azimuth [deg]": [10, 11, 12, 13, 14] * 40,
        "elevation [deg]": [20, 21, 22, 23, 24] * 40,
    }
    
    imu_data = {
        "timestamp [ns]": np.linspace(1e9, 2e9, 200),
        "roll [deg]": [1, 2, 3, 4, 5] * 40,
        "pitch [deg]": [6, 7, 8, 9, 10] * 40,
        "yaw [deg]": [11, 12, 13, 14, 15] * 40,
        "acceleration x [g]": [0.1] * 200,
        "acceleration y [g]": [0.2] * 200,
        "acceleration z [g]": [0.9] * 200,
        "gyro x [deg/s]": [1.0] * 200,
        "gyro y [deg/s]": [2.0] * 200,
        "gyro z [deg/s]": [3.0] * 200,
    }
    
    blink_data = {
        "start timestamp [ns]": [1.02010050e9, 1.29145729e9],
        "end timestamp [ns]": [1.07035176e9, 1.55778894e9],
    }
    
    return {
        "gaze": pd.DataFrame(gaze_data),
        "imu": pd.DataFrame(imu_data),
        "blinks": pd.DataFrame(blink_data)
    }


@patch("pandas.read_csv")
def test_pupil_invisible_data_init(mock_read_csv, mock_csv_data):
    """Test initialization of PupilInvisibleData"""
    # Configure the mock to return different DataFrames for different file paths
    def side_effect(path):
        if path.endswith("gaze.csv"):
            return mock_csv_data["gaze"]
        elif path.endswith("imu.csv"):
            return mock_csv_data["imu"]
        elif path.endswith("blinks.csv"):
            return mock_csv_data["blinks"]
        return None
    
    mock_read_csv.side_effect = side_effect

    data = PupilInvisibleData("test_folder/")

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
    data = PupilInvisibleData.__new__(PupilInvisibleData)  # Create instance without calling __init__
    data.data_folder_path = "valid_folder"
    assert data.data_folder_path == "valid_folder/"  # Should add trailing slash


def test_data_folder_path_setter_with_slash():
    """Test setting a valid data folder path with trailing slash"""
    data = PupilInvisibleData.__new__(PupilInvisibleData)  # Create instance without calling __init__
    data.data_folder_path = "valid_folder/"
    assert data.data_folder_path == "valid_folder/"  # Should keep trailing slash


def test_data_folder_path_setter_invalid_type():
    """Test setting an invalid data folder path type"""
    data = PupilInvisibleData.__new__(PupilInvisibleData)  # Create instance without calling __init__
    with pytest.raises(ValueError, match="The data_folder_path must be a string, got 123."):
        data.data_folder_path = 123


def test_check_validity_empty_file():
    """Test _check_validity with empty file"""
    # Create a mock data object
    data = PupilInvisibleData.__new__(PupilInvisibleData)
    data.error_type = ErrorType.SKIP
    data.gaze_csv_data = pd.DataFrame({"timestamp [ns]": []})
    data.file_name = "test_folder"
    data._validity_flag = True

    # Check that the validity flag is modified by _check_validity
    data._check_validity()
    assert data._validity_flag is False


def test_check_validity_invalid_data():
    """Test _check_validity with mostly invalid data"""
    # Create a mock data object
    data = PupilInvisibleData.__new__(PupilInvisibleData)
    data.error_type = ErrorType.SKIP
    data_dict = {
        "timestamp [ns]": list(range(10)),
        "worn": [0] * 8 + [1] * 2,  # 80% invalid
    }
    data.gaze_csv_data = pd.DataFrame(data_dict)
    data.file_name = "test_folder"
    data._validity_flag = True

    # Check that the validity flag is modified by _check_validity
    data._check_validity()
    assert data._validity_flag is False


def test_check_validity_non_increasing_time():
    """Test _check_validity with non-increasing time vector"""
    # Create a mock data object
    data = PupilInvisibleData.__new__(PupilInvisibleData)
    data.error_type = ErrorType.SKIP
    data_dict = {
        "timestamp [ns]": [1, 2, 3, 2, 5],  # Time goes backwards
        "worn": [1] * 5,
    }
    data.gaze_csv_data = pd.DataFrame(data_dict)
    data.file_name = "test_folder"
    data._validity_flag = True

    # Check that the validity flag is modified by _check_validity
    data._check_validity()
    assert data._validity_flag is False


def test_set_time_vector():
    """Test _set_time_vector method"""
    data = PupilInvisibleData.__new__(PupilInvisibleData)
    data._validity_flag = True
    data.gaze_csv_data = pd.DataFrame({"timestamp [ns]": [1000000000, 2000000000, 3000000000, 4000000000]})
    data.blink_csv_data = pd.DataFrame({"start timestamp [ns]": [1500000000], "end timestamp [ns]": [2500000000]})
    data.imu_csv_data = pd.DataFrame({"timestamp [ns]": [1000000000, 2000000000, 3000000000, 4000000000]})

    # Before unchanged
    npt.assert_almost_equal(data.blink_csv_data["start timestamp [ns]"].iloc[0], 1500000000)
    npt.assert_almost_equal(data.blink_csv_data["end timestamp [ns]"].iloc[0], 2500000000)
    npt.assert_almost_equal(data.imu_csv_data["timestamp [ns]"].iloc[1], 2000000000)

    data._set_time_vector()

    assert data.time_vector is not None
    assert len(data.time_vector) == 4
    assert data.time_vector[0] == 0.0  # First value should be 0
    npt.assert_almost_equal(data.time_vector[1], 1.0)  # 1 second difference
    
    # Check that blink and imu timestamps are also transformed
    npt.assert_almost_equal(data.blink_csv_data["start timestamp [ns]"].iloc[0], 0.5)
    npt.assert_almost_equal(data.blink_csv_data["end timestamp [ns]"].iloc[0], 1.5)
    npt.assert_almost_equal(data.imu_csv_data["timestamp [ns]"].iloc[1], 1.0)


def test_remove_duplicates():
    """Test _remove_duplicates method"""
    data = PupilInvisibleData.__new__(PupilInvisibleData)
    data._validity_flag = True
    data.time_vector = np.array([0.0, 0.1, 0.2, 0.3])  # No duplicates

    # This should not raise an exception
    data._remove_duplicates()
    
    # Test with duplicates
    data.time_vector = np.array([0.0, 0.1, 0.1, 0.3])  # Duplicate at index 2
    
    with pytest.raises(RuntimeError, match="The time vector has duplicated frames, which never happened with this eye-tracker. Please notify the developer."):
        data._remove_duplicates()


def test_discard_data_out_of_range():
    """Test _discard_data_out_of_range method"""
    data = PupilInvisibleData.__new__(PupilInvisibleData)
    data._validity_flag = True
    data.time_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    data.dt = 0.1
    data.right_eye_openness = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    data.left_eye_openness = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    data.eye_direction = np.ones((3, 5))
    data.head_angles = np.ones((3, 5))
    data.gaze_direction = np.ones((3, 5))
    data.head_angular_velocity = np.ones((3, 5))
    data.head_velocity_norm = np.ones(5)
    data.data_invalidity = np.zeros(5, dtype=bool)
    data.time_range = TimeRange(0.15, 0.35)

    data._discard_data_out_of_range()

    assert len(data.time_vector) == 2
    npt.assert_almost_equal(data.time_vector, np.array([0.2, 0.3]))
    assert len(data.right_eye_openness) == 2
    assert data.eye_direction.shape == (3, 2)
    assert data.head_angles.shape == (3, 2)
    assert data.gaze_direction.shape == (3, 2)
    assert data.head_angular_velocity.shape == (3, 2)
    assert len(data.head_velocity_norm) == 2
    assert len(data.data_invalidity) == 2


def test_set_eye_openness():
    """Test _set_eye_openness method"""
    data = PupilInvisibleData.__new__(PupilInvisibleData)
    data._validity_flag = True
    data.time_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    data.blink_csv_data = pd.DataFrame({
        "start timestamp [ns]": [0.1, 0.6],
        "end timestamp [ns]": [0.3, 0.8]
    })

    data._set_eye_openness()

    npt.assert_almost_equal(data.right_eye_openness, np.array([1., 0., 0., 0., 1., 1., 0., 0., 0., 1.]))
    npt.assert_almost_equal(data.left_eye_openness, np.array([1., 0., 0., 0., 1., 1., 0., 0., 0., 1.]))


def test_set_eye_openness_blink_not_in_time_vector():
    """Test _set_eye_openness method with blink times not in time vector"""
    data = PupilInvisibleData.__new__(PupilInvisibleData)
    data._validity_flag = True
    data.time_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    data.blink_csv_data = pd.DataFrame({
        "start timestamp [ns]": [0.15],  # Not in time vector
        "end timestamp [ns]": [0.25]     # Not in time vector
    })

    with pytest.raises(RuntimeError, match="The blink start or end times are not in the time vector. This should not happen, please notify the developer."):
        data._set_eye_openness()


def test_set_eye_direction():
    """Test _set_eye_direction method"""
    data = PupilInvisibleData.__new__(PupilInvisibleData)
    data._validity_flag = True
    data.time_vector = np.array([0.0, 0.1, 0.2])
    
    # Create normalized vectors using azimuth and elevation
    azimuth = np.array([0.0, 10.0, 20.0])  # degrees
    elevation = np.array([0.0, 5.0, 10.0])  # degrees
    
    data.gaze_csv_data = pd.DataFrame({
        "azimuth [deg]": azimuth,
        "elevation [deg]": elevation
    })

    data._set_eye_direction()

    assert data.eye_direction.shape == (3, 3)
    # Check that vectors are normalized
    norms = np.linalg.norm(data.eye_direction, axis=0)
    assert np.allclose(norms, 1.0)
    
    # First vector should be [0, 0, 1] (forward)
    npt.assert_almost_equal(data.eye_direction, np.array([[ 0.        ,  0.08715574,  0.17364818],
                           [ 0.        , -0.17298739, -0.33682409],
                           [ 1.        ,  0.98106026,  0.92541658]]))


def test_set_eye_direction_invalid_norm():
    """Test _set_eye_direction method with invalid norm"""
    data = PupilInvisibleData.__new__(PupilInvisibleData)
    data._validity_flag = True
    data.time_vector = np.array([0.0])
    
    # Create a mock that will cause the norm check to fail
    with patch('eyedentify3d.data_parsers.pupil_invisible_data.rotation_matrix_from_euler_angles') as mock_rotation:
        mock_rotation.return_value = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])  # Will result in small norm
        
        data.gaze_csv_data = pd.DataFrame({
            "azimuth [deg]": [0.0],
            "elevation [deg]": [0.0]
        })
        
        with pytest.raises(RuntimeError, match="There was an issue with the eye direction computation"):
            data._set_eye_direction()


def test_interpolate_to_eye_timestamps():
    """Test interpolate_to_eye_timestamps method"""
    data = PupilInvisibleData.__new__(PupilInvisibleData)
    data.time_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    
    # Create test data
    time_vector_imu = np.array([0.0, 0.2, 0.4])
    unwrapped_head_angles = np.array([
        [10.0, 20.0, 30.0],  # roll
        [15.0, 25.0, 35.0],  # pitch
        [20.0, 30.0, 40.0]   # yaw
    ])
    
    result = data.interpolate_to_eye_timestamps(time_vector_imu, unwrapped_head_angles)
    
    assert result.shape == (3, 5)
    # Check interpolated values
    npt.assert_almost_equal(result[:, 0], np.array([10.0, 15.0, 20.0]))  # t=0.0, exact match
    npt.assert_almost_equal(result[:, 1], np.array([15.0, 20.0, 25.0]))  # t=0.1, interpolated
    npt.assert_almost_equal(result[:, 2], np.array([20.0, 25.0, 30.0]))  # t=0.2, exact match
    npt.assert_almost_equal(result[:, 3], np.array([25.0, 30.0, 35.0]))  # t=0.3, interpolated
    npt.assert_almost_equal(result[:, 4], np.array([30.0, 35.0, 40.0]))  # t=0.4, exact match


def test_interpolate_to_eye_timestamps_invalid_shape():
    """Test interpolate_to_eye_timestamps method with invalid shape"""
    data = PupilInvisibleData.__new__(PupilInvisibleData)
    data.time_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    
    # Create test data with invalid shape
    time_vector_imu = np.array([0.0, 0.2, 0.4])
    unwrapped_head_angles = np.array([10.0, 20.0, 30.0])  # Wrong shape, should be (3, n)
    
    with pytest.raises(NotImplementedError, match=r"This function was designed for head angles of shape \(3, n_frames\)."):
        data.interpolate_to_eye_timestamps(time_vector_imu, unwrapped_head_angles)


def test_interpolate_to_eye_timestamps_duplicated_frames():
    """Test interpolate_to_eye_timestamps method with duplicated frames"""
    data = PupilInvisibleData.__new__(PupilInvisibleData)
    data.time_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    
    # Create test data with duplicated frames
    time_vector_imu = np.array([0.0, 0.2, 0.4])
    unwrapped_head_angles = np.array([
        [10.0, 10.0, 30.0],  # Duplicated frame at indices 0 and 1
        [15.0, 15.0, 35.0],
        [20.0, 20.0, 40.0]
    ])
    
    with pytest.raises(RuntimeError, match="There were repeated frames in the imu data, which never happened with this eye-tracker. Please notify the developer."):
        data.interpolate_to_eye_timestamps(time_vector_imu, unwrapped_head_angles)


def test_interpolate_to_eye_timestamps_out_of_range():
    """Test interpolate_to_eye_timestamps method with timestamps out of range"""
    data = PupilInvisibleData.__new__(PupilInvisibleData)
    data.time_vector = np.array([-0.1, 0.0, 0.2, 0.4, 0.5])  # First and last are out of range
    
    # Create test data
    time_vector_imu = np.array([0.0, 0.2, 0.4])
    unwrapped_head_angles = np.array([
        [10.0, 20.0, 30.0],
        [15.0, 25.0, 35.0],
        [20.0, 30.0, 40.0]
    ])
    
    result = data.interpolate_to_eye_timestamps(time_vector_imu, unwrapped_head_angles)
    
    assert result.shape == (3, 5)
    # Check out of range values are NaN
    assert np.all(np.isnan(result[:, 0]))  # t=-0.1, out of range
    assert np.all(np.isnan(result[:, 4]))  # t=0.5, out of range


@patch('eyedentify3d.data_parsers.pupil_invisible_data.unwrap_rotation')
@patch('eyedentify3d.data_parsers.pupil_invisible_data.angles_from_imu_fusion')
def test_set_head_angles_with_tags(mock_angles_from_imu, mock_unwrap):
    """Test _set_head_angles method with tags in experiment"""
    data = PupilInvisibleData.__new__(PupilInvisibleData)
    data._validity_flag = True
    data.time_vector = np.array([0.0, 0.1, 0.2])
    
    # Mock data with valid yaw values (tags present)
    data.imu_csv_data = pd.DataFrame({
        "timestamp [ns]": [0.0, 0.1, 0.2],
        "roll [deg]": [1.0, 2.0, 3.0],
        "pitch [deg]": [4.0, 5.0, 6.0],
        "yaw [deg]": [7.0, 8.0, 9.0],  # Not NaN, so tags are present
        "acceleration x [g]": [0.1, 0.1, 0.1],
        "acceleration y [g]": [0.2, 0.2, 0.2],
        "acceleration z [g]": [0.9, 0.9, 0.9],
        "gyro x [deg/s]": [1.0, 1.0, 1.0],
        "gyro y [deg/s]": [2.0, 2.0, 2.0],
        "gyro z [deg/s]": [3.0, 3.0, 3.0]
    })
    
    # Mock the unwrap_rotation function
    mock_unwrap.return_value = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    
    # Mock the interpolate_to_eye_timestamps method
    data.interpolate_to_eye_timestamps = MagicMock(return_value=np.array([
        [1.5, 2.5, 3.5],
        [4.5, 5.5, 6.5],
        [7.5, 8.5, 9.5]
    ]))
    
    data.time_vector = np.array([0.05, 0.15, 0.25])
    
    data._set_head_angles()
    
    # Check that angles_from_imu_fusion was not called (since tags are present)
    mock_angles_from_imu.assert_not_called()
    
    # Check that unwrap_rotation was called with the right arguments
    mock_unwrap.assert_called_once()
    args = mock_unwrap.call_args[0][0]
    npt.assert_almost_equal(args[0], np.array([1.0, 2.0, 3.0]))  # roll
    npt.assert_almost_equal(args[1], np.array([4.0, 5.0, 6.0]))  # pitch
    npt.assert_almost_equal(args[2], np.array([7.0, 8.0, 9.0]))  # yaw
    
    # Check that interpolate_to_eye_timestamps was called
    data.interpolate_to_eye_timestamps.assert_called_once()
    
    # Check the result
    npt.assert_almost_equal(data.head_angles, np.array([
        [1.5, 2.5, 3.5],
        [4.5, 5.5, 6.5],
        [7.5, 8.5, 9.5]
    ]))


@patch('eyedentify3d.data_parsers.pupil_invisible_data.unwrap_rotation')
@patch('eyedentify3d.data_parsers.pupil_invisible_data.angles_from_imu_fusion')
def test_set_head_angles_without_tags(mock_angles_from_imu, mock_unwrap):
    """Test _set_head_angles method without tags in experiment"""
    data = PupilInvisibleData.__new__(PupilInvisibleData)
    data._validity_flag = True
    data.time_vector = np.array([0.0, 0.1, 0.2])
    
    # Mock data with NaN yaw values (no tags)
    data.imu_csv_data = pd.DataFrame({
        "timestamp [ns]": [0.0, 0.1, 0.2],
        "roll [deg]": [1.0, 2.0, 3.0],
        "pitch [deg]": [4.0, 5.0, 6.0],
        "yaw [deg]": [np.nan, np.nan, np.nan],  # NaN, so no tags
        "acceleration x [g]": [0.1, 0.1, 0.1],
        "acceleration y [g]": [0.2, 0.2, 0.2],
        "acceleration z [g]": [0.9, 0.9, 0.9],
        "gyro x [deg/s]": [1.0, 1.0, 1.0],
        "gyro y [deg/s]": [2.0, 2.0, 2.0],
        "gyro z [deg/s]": [3.0, 3.0, 3.0]
    })
    
    # Mock the angles_from_imu_fusion function
    mock_angles_from_imu.return_value = (
        np.array([1.1, 2.1, 3.1]),  # roll
        np.array([4.1, 5.1, 6.1]),  # pitch
        np.array([7.1, 8.1, 9.1])   # yaw
    )
    
    # Mock the unwrap_rotation function
    mock_unwrap.return_value = np.array([
        [1.1, 2.1, 3.1],
        [4.1, 5.1, 6.1],
        [7.1, 8.1, 9.1]
    ])
    
    # Mock the interpolate_to_eye_timestamps method
    data.interpolate_to_eye_timestamps = MagicMock(return_value=np.array([
        [1.5, 2.5, 3.5],
        [4.5, 5.5, 6.5],
        [7.5, 8.5, 9.5]
    ]))
    
    data.time_vector = np.array([0.05, 0.15, 0.25])
    
    data._set_head_angles()
    
    # Check that angles_from_imu_fusion was called with the right arguments
    mock_angles_from_imu.assert_called_once()
    args = mock_angles_from_imu.call_args[0]
    npt.assert_almost_equal(args[0], np.array([0.0, 0.1, 0.2]))  # time_vector_imu
    npt.assert_almost_equal(args[1], np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.9, 0.9, 0.9]]))  # acceleration
    npt.assert_almost_equal(args[2], np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]))  # gyroscope
    
    # Check that unwrap_rotation was called with the right arguments
    mock_unwrap.assert_called_once()
    args = mock_unwrap.call_args[0][0]
    npt.assert_almost_equal(args[0], np.array([1.1, 2.1, 3.1]))  # roll
    npt.assert_almost_equal(args[1], np.array([4.1, 5.1, 6.1]))  # pitch
    npt.assert_almost_equal(args[2], np.array([7.1, 8.1, 9.1]))  # yaw
    
    # Check that interpolate_to_eye_timestamps was called
    data.interpolate_to_eye_timestamps.assert_called_once()
    
    # Check the result
    npt.assert_almost_equal(data.head_angles, np.array([
        [1.5, 2.5, 3.5],
        [4.5, 5.5, 6.5],
        [7.5, 8.5, 9.5]
    ]))


def test_set_head_angular_velocity():
    """Test _set_head_angular_velocity method"""
    data = PupilInvisibleData.__new__(PupilInvisibleData)
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
    data = PupilInvisibleData.__new__(PupilInvisibleData)
    data._validity_flag = True
    data.gaze_csv_data = pd.DataFrame({
        "worn": [1, 0, 1, 0, 1]  # Second and fourth frames are invalid
    })

    data._set_data_invalidity()

    assert data.data_invalidity is not None
    npt.assert_almost_equal(data.data_invalidity, np.array([False, True, False, True, False]))
