import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from eyedentify3d.data_parsers.abstract_data import Data, destroy_on_fail
from eyedentify3d import ErrorType, TimeRange


class MockData(Data):
    """Mock implementation of the abstract Data class for testing"""

    def _check_validity(self):
        pass

    def _set_time_vector(self):
        self.time_vector = np.array([0.0, 0.1, 0.2, 0.3, 0.4])

    def _discard_data_out_of_range(self):
        pass

    def _set_eye_openness(self):
        pass

    def _set_eye_direction(self):
        pass

    def _set_head_angles(self):
        pass

    def _set_head_angular_velocity(self):
        pass

    def _set_data_invalidity(self):
        pass


def test_data_init():
    """Test initialization of Data class with default parameters"""
    data = MockData()
    assert data.error_type == ErrorType.PRINT
    assert isinstance(data.time_range, TimeRange)
    assert data._validity_flag is True
    assert data.dt is None


def test_data_init_with_params():
    """Test initialization of Data class with custom parameters"""
    error_type = ErrorType.SKIP
    time_range = TimeRange(1.0, 5.0)
    data = MockData(error_type=error_type, time_range=time_range)

    assert data.error_type == error_type
    assert data.time_range == time_range


def test_error_type_setter_valid():
    """Test setting a valid error type"""
    data = MockData()
    data.error_type = ErrorType.SKIP
    assert data.error_type == ErrorType.SKIP


def test_error_type_setter_invalid():
    """Test setting an invalid error type"""
    data = MockData()
    with pytest.raises(ValueError, match="The error type must be an ErrorType, got INVALID."):
        data.error_type = "INVALID"


@patch("builtins.open", new_callable=MagicMock)
def test_error_type_setter_file(mock_open):
    """Test setting error type to FILE creates a file"""
    data = MockData()
    data.error_type = ErrorType.FILE

    mock_open.assert_called_once()
    assert "bad_data_files.txt" in mock_open.call_args[0][0]
    assert "w" in mock_open.call_args[0][1]


def test_time_range_setter_valid():
    """Test setting a valid time range"""
    data = MockData()
    time_range = TimeRange(2.0, 8.0)
    data.time_range = time_range
    assert data.time_range == time_range


def test_time_range_setter_invalid():
    """Test setting an invalid time range"""
    data = MockData()
    with pytest.raises(ValueError, match="The time range must be an TimeRange, got INVALID."):
        data.time_range = "INVALID"


def test_trial_duration():
    """Test getting trial duration"""
    data = MockData()
    data._set_time_vector()
    assert data.trial_duration == 0.4  # 0.4 - 0.0


def test_trial_duration_no_time_vector():
    """Test getting trial duration without time vector"""
    data = MockData()
    with pytest.raises(
        RuntimeError,
        match=r"The trial_duration property can only be called after the time_vector has been set \(i.e., after the data objects has been instantiated\).",
    ):
        _ = data.trial_duration


def test_set_dt():
    """Test setting dt"""
    data = MockData()
    data._set_time_vector()
    data._set_dt()
    assert data.dt == 0.1  # (0.1 - 0.0 + 0.2 - 0.1 + 0.3 - 0.2 + 0.4 - 0.3) / 4


def test_set_dt_no_time_vector():
    """Test setting dt without time vector"""
    data = MockData()
    with pytest.raises(
        RuntimeError,
        match=r"The dt property can only be called after the time_vector has been set \(i.e., after the data objects has been instantiated\).",
    ):
        data._set_dt()


def test_set_dt_twice():
    """Test setting dt twice"""
    data = MockData()
    data._set_time_vector()
    data._set_dt()
    with pytest.raises(
        RuntimeError,
        match="dt can only be set once at the very beginning of the data processing, because the time vector will be modified later.",
    ):
        data._set_dt()


def test_file_name():
    """Test getting file name"""
    data = MockData()
    data.data_file_path = "/path/to/file.csv"
    assert data.file_name == "file.csv"


def test_file_name_no_path():
    """Test getting file name without path"""
    data = MockData()
    with pytest.raises(AttributeError, match="The data file or folder path is not set."):
        _ = data.file_name


def test_destroy_on_fail_decorator():
    """Test the destroy_on_fail decorator"""

    class TestData(MockData):
        @destroy_on_fail
        def failing_method(self):
            self._validity_flag = False

    data = TestData()
    assert data._validity_flag is True

    data.failing_method()

    assert data.time_vector is None
    assert data.right_eye_openness is None
    assert data.left_eye_openness is None
    assert data.eye_direction is None
    assert data.head_angles is None
    assert data.head_angular_velocity is None
    assert data.head_velocity_norm is None


def test_destroy_on_error():
    """Test the destroy_on_error method"""
    data = MockData()
    data._set_time_vector()  # Set time_vector to non-None

    data.destroy_on_error()

    assert data.time_vector is None
    assert data.right_eye_openness is None
    assert data.left_eye_openness is None
    assert data.eye_direction is None
    assert data.head_angles is None
    assert data.head_angular_velocity is None
    assert data.head_velocity_norm is None


def test_nb_frames():
    """Test getting the number of frames"""
    data = MockData()
    data._set_time_vector()
    assert data.nb_frames == 5  # 5 time points in the time vector


def test_nb_frames_error():
    """Test getting the number of frames when time_vector is not set"""
    data = MockData()
    with pytest.raises(
        RuntimeError,
        match=r"The nb_frames property can only be called after the time_vector has been set \(i.e., after the data objects has been instantiated\).",
    ):
        data.nb_frames
