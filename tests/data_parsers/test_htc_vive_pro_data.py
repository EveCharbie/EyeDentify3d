import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from eyedentify3d.data_parsers.htc_vive_pro_data import HtcViveProData
from eyedentify3d.error_type import ErrorType
from eyedentify3d.time_range import TimeRange


@pytest.fixture
def mock_csv_data():
    """Create mock CSV data for testing"""
    # Create a DataFrame with minimal required columns
    data = {
        "time(100ns)": [1000, 2000, 3000, 4000, 5000],
        "eye_valid_L": [31, 31, 31, 31, 31],
        "eye_valid_R": [31, 31, 31, 31, 31],
        "openness_L": [0.9, 0.9, 0.9, 0.9, 0.9],
        "openness_R": [0.9, 0.9, 0.9, 0.9, 0.9],
        "gaze_direct_L.x": [0.1, 0.2, 0.3, 0.4, 0.5],
        "gaze_direct_L.y": [0.1, 0.2, 0.3, 0.4, 0.5],
        "gaze_direct_L.z": [0.9, 0.8, 0.7, 0.6, 0.5],
        "helmet_rot_x": [10, 11, 12, 13, 14],
        "helmet_rot_y": [20, 21, 22, 23, 24],
        "helmet_rot_z": [30, 31, 32, 33, 34],
    }
    return pd.DataFrame(data)


@patch('pandas.read_csv')
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
    assert data.data_validity is not None


def test_data_file_path_setter_valid():
    """Test setting a valid data file path"""
    data = HtcViveProData.__new__(HtcViveProData)  # Create instance without calling __init__
    data.data_file_path = "valid_file.csv"
    assert data.data_file_path == "valid_file.csv"


def test_data_file_path_setter_invalid_type():
    """Test setting an invalid data file path type"""
    data = HtcViveProData.__new__(HtcViveProData)  # Create instance without calling __init__
    with pytest.raises(ValueError):
        data.data_file_path = 123


def test_data_file_path_setter_invalid_extension():
    """Test setting an invalid data file path extension"""
    data = HtcViveProData.__new__(HtcViveProData)  # Create instance without calling __init__
    with pytest.raises(ValueError):
        data.data_file_path = "invalid_file.txt"


@patch('pandas.read_csv')
def test_check_validity_empty_file(mock_read_csv):
    """Test _check_validity with empty file"""
    mock_read_csv.return_value = pd.DataFrame({"time(100ns)": []})
    
    with patch.object(ErrorType, 'SKIP', return_value=None) as mock_error_handler:
        data = HtcViveProData.__new__(HtcViveProData)
        data.error_type = ErrorType.SKIP
        data.csv_data = pd.DataFrame({"time(100ns)": []})
        data.file_name = "test.csv"
        data._validity_flag = True
        
        data._check_validity()
        
        assert data._validity_flag is False
        mock_error_handler.assert_called_once()


@patch('pandas.read_csv')
def test_check_validity_invalid_data(mock_read_csv):
    """Test _check_validity with mostly invalid data"""
    # Create data where most frames are invalid
    data_dict = {
        "time(100ns)": list(range(10)),
        "eye_valid_L": [0] * 8 + [31] * 2,  # 80% invalid
        "eye_valid_R": [31] * 10
    }
    mock_read_csv.return_value = pd.DataFrame(data_dict)
    
    with patch.object(ErrorType, 'SKIP', return_value=None) as mock_error_handler:
        data = HtcViveProData.__new__(HtcViveProData)
        data.error_type = ErrorType.SKIP
        data.csv_data = pd.DataFrame(data_dict)
        data.file_name = "test.csv"
        data._validity_flag = True
        
        data._check_validity()
        
        assert data._validity_flag is False
        mock_error_handler.assert_called_once()


@patch('pandas.read_csv')
def test_check_validity_non_increasing_time(mock_read_csv):
    """Test _check_validity with non-increasing time vector"""
    # Create data with non-increasing time
    data_dict = {
        "time(100ns)": [1, 2, 3, 2, 5],  # Time goes backwards
        "eye_valid_L": [31] * 5,
        "eye_valid_R": [31] * 5
    }
    mock_read_csv.return_value = pd.DataFrame(data_dict)
    
    with patch.object(ErrorType, 'SKIP', return_value=None) as mock_error_handler:
        data = HtcViveProData.__new__(HtcViveProData)
        data.error_type = ErrorType.SKIP
        data.csv_data = pd.DataFrame(data_dict)
        data.file_name = "test.csv"
        data._validity_flag = True
        
        data._check_validity()
        
        assert data._validity_flag is False
        mock_error_handler.assert_called_once()


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
    data.csv_data = pd.DataFrame({
        "openness_R": [0.8, 0.9, 1.0],
        "openness_L": [0.7, 0.8, 0.9]
    })
    
    data._set_eye_openness()
    
    assert np.array_equal(data.right_eye_openness, np.array([0.8, 0.9, 1.0]))
    assert np.array_equal(data.left_eye_openness, np.array([0.7, 0.8, 0.9]))


def test_set_eye_direction():
    """Test _set_eye_direction method"""
    data = HtcViveProData.__new__(HtcViveProData)
    data._validity_flag = True
    data.file_name = "test.csv"
    data.error_type = ErrorType.SKIP
    
    # Create normalized vectors
    x = [0.1, 0.2, 0.3]
    y = [0.1, 0.2, 0.3]
    z = [0.9, 0.8, 0.7]
    
    # Ensure they're unit vectors
    for i in range(3):
        norm = np.sqrt(x[i]**2 + y[i]**2 + z[i]**2)
        x[i] /= norm
        y[i] /= norm
        z[i] /= norm
    
    data.csv_data = pd.DataFrame({
        "gaze_direct_L.x": x,
        "gaze_direct_L.y": y,
        "gaze_direct_L.z": z
    })
    
    data._set_eye_direction()
    
    assert data.eye_direction.shape == (3, 3)
    # Check that vectors are normalized
    norms = np.linalg.norm(data.eye_direction, axis=0)
    assert np.allclose(norms, 1.0)


def test_set_eye_direction_invalid_norm():
    """Test _set_eye_direction method with invalid norm"""
    data = HtcViveProData.__new__(HtcViveProData)
    data._validity_flag = True
    data.file_name = "test.csv"
    
    with patch.object(ErrorType, 'SKIP', return_value=None) as mock_error_handler:
        data.error_type = ErrorType.SKIP
        
        # Create vectors with invalid norms
        data.csv_data = pd.DataFrame({
            "gaze_direct_L.x": [2.0, 0.1, 0.1],  # First vector has norm > 1.2
            "gaze_direct_L.y": [0.0, 0.1, 0.1],
            "gaze_direct_L.z": [0.0, 0.1, 0.1]
        })
        
        data._set_eye_direction()
        
        assert data._validity_flag is False
        mock_error_handler.assert_called_once()


def test_interpolate_repeated_frames():
    """Test interpolate_repeated_frames method"""
    data = HtcViveProData.__new__(HtcViveProData)
    
    # Create data with repeated frames
    test_data = np.array([
        [1.0, 1.0, 1.0, 4.0, 5.0],  # First 3 frames are identical
        [2.0, 2.0, 2.0, 8.0, 10.0],
        [3.0, 3.0, 3.0, 12.0, 15.0]
    ])
    
    result = data.interpolate_repeated_frames(test_data)
    
    # Check shape is preserved
    assert result.shape == test_data.shape
    
    # Check that repeated frames are interpolated
    assert np.allclose(result[:, 1], [2.0, 4.0, 6.0])  # Interpolated values
    assert np.allclose(result[:, 2], [3.0, 6.0, 9.0])  # Interpolated values
    
    # Check that unique frames are preserved
    assert np.allclose(result[:, 0], test_data[:, 0])
    assert np.allclose(result[:, 3], test_data[:, 3])
    assert np.allclose(result[:, 4], test_data[:, 4])


def test_interpolate_repeated_frames_invalid_shape():
    """Test interpolate_repeated_frames method with invalid shape"""
    data = HtcViveProData.__new__(HtcViveProData)
    
    # Create data with invalid shape
    test_data = np.array([1.0, 2.0, 3.0])
    
    with pytest.raises(NotImplementedError):
        data.interpolate_repeated_frames(test_data)


def test_set_head_angles():
    """Test _set_head_angles method"""
    data = HtcViveProData.__new__(HtcViveProData)
    data._validity_flag = True
    data.csv_data = pd.DataFrame({
        "helmet_rot_x": [10, 11, 12],
        "helmet_rot_y": [20, 21, 22],
        "helmet_rot_z": [30, 31, 32]
    })
    
    # Mock the interpolate_repeated_frames method
    original_interpolate = data.interpolate_repeated_frames
    data.interpolate_repeated_frames = lambda x: x
    
    data._set_head_angles()
    
    # Restore original method
    data.interpolate_repeated_frames = original_interpolate
    
    assert data.head_angles.shape == (3, 3)
    assert np.array_equal(data.head_angles[0, :], np.array([10, 11, 12]))
    assert np.array_equal(data.head_angles[1, :], np.array([20, 21, 22]))
    assert np.array_equal(data.head_angles[2, :], np.array([30, 31, 32]))


@patch('eyedentify3d.utils.signal_utils.centered_finite_difference')
@patch('eyedentify3d.utils.signal_utils.filter_data')
def test_set_head_angular_velocity(mock_filter_data, mock_centered_diff):
    """Test _set_head_angular_velocity method"""
    data = HtcViveProData.__new__(HtcViveProData)
    data._validity_flag = True
    data.time_vector = np.array([0.0, 0.1, 0.2])
    data.head_angles = np.array([
        [10, 11, 12],
        [20, 21, 22],
        [30, 31, 32]
    ])
    
    # Mock the centered_finite_difference to return a known array
    mock_velocity = np.array([
        [10, 10, 10],
        [10, 10, 10],
        [10, 10, 10]
    ])
    mock_centered_diff.return_value = mock_velocity
    
    # Mock filter_data to return the input
    mock_filter_data.return_value = np.array([[17.32, 17.32, 17.32]])
    
    data._set_head_angular_velocity()
    
    assert data.head_angular_velocity is not None
    assert np.array_equal(data.head_angular_velocity, mock_velocity)
    assert data.head_velocity_norm is not None
    assert len(data.head_velocity_norm) == 3


def test_set_data_validity():
    """Test _set_data_validity method"""
    data = HtcViveProData.__new__(HtcViveProData)
    data._validity_flag = True
    data.csv_data = pd.DataFrame({
        "eye_valid_L": [31, 30, 31],  # Second frame is invalid
        "eye_valid_R": [31, 31, 30]   # Third frame is invalid
    })
    
    data._set_data_validity()
    
    assert data.data_validity is not None
    assert np.array_equal(data.data_validity, np.array([False, True, True]))
