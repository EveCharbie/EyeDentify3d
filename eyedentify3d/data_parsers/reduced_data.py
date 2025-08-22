import numpy as np

from ..time_range import TimeRange
from ..error_type import ErrorType
from .abstract_data import Data


class ReducedData(Data):
    """
    Create a simple data object.
    """

    def __init__(
        self,
        original_dt: float,
        original_time_vector: np.ndarray[float],
        original_right_eye_openness: np.ndarray[float],
        original_left_eye_openness: np.ndarray[float],
        original_eye_direction: np.ndarray[float],
        original_head_angles: np.ndarray[float],
        original_gaze_direction: np.ndarray[float],
        original_head_angular_velocity: np.ndarray[float],
        original_head_velocity_norm: np.ndarray[float],
        original_data_invalidity: np.ndarray[bool],
        time_range: TimeRange = TimeRange(),
        error_type: ErrorType = ErrorType.PRINT,
    ):

        super().__init__(error_type=error_type)  # Dot not put time_range here, as it does not have the same meaning

        # Original attributes
        self.time_range = time_range
        self.dt = original_dt

        # Extended attributes
        self.indices = None
        self._set_indices(original_time_vector)
        self.time_vector = original_time_vector

        self.right_eye_openness = original_right_eye_openness
        self.left_eye_openness = original_left_eye_openness
        self.eye_direction = original_eye_direction
        self.head_angles = original_head_angles
        self.gaze_direction = original_gaze_direction
        self.head_angular_velocity = original_head_angular_velocity
        self.head_velocity_norm = original_head_velocity_norm
        self.data_invalidity = original_data_invalidity

        # Finalize
        self.finalize()

    def _set_indices(self, time_vector):
        """
        Set the indices of the data object based on the time vector and the time range.
        """
        if time_vector is None:
            raise ValueError("The time vector must be provided.")
        self.indices = self.time_range.get_indices(time_vector)

    # Bypass some abstract methods to skip in this edgy context
    def _check_validity(self):
        pass

    def _set_time_vector(self):
        pass

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

    @property
    def time_vector(self):
        return self._time_vector

    @time_vector.setter
    def time_vector(self, value: np.ndarray[float]):
        if value is None:
            self._time_vector = None
        else:
            if self.indices is None:
                raise RuntimeError(
                    "The time vector can only be set once the indices are initialized using _set_indices."
                )
            self._time_vector = value[self.indices]

    @property
    def right_eye_openness(self):
        return self._right_eye_openness

    @right_eye_openness.setter
    def right_eye_openness(self, value: np.ndarray[float]):
        if value is None:
            self._time_vector = None
        else:
            self._right_eye_openness = value[self.indices]

    @property
    def left_eye_openness(self):
        return self._left_eye_openness

    @left_eye_openness.setter
    def left_eye_openness(self, value: np.ndarray[float]):
        if value is None:
            self._time_vector = None
        else:
            self._left_eye_openness = value[self.indices]

    @property
    def eye_direction(self):
        return self._eye_direction

    @eye_direction.setter
    def eye_direction(self, value: np.ndarray[float]):
        if value is None:
            self._time_vector = None
        else:
            self._eye_direction = value[:, self.indices]

    @property
    def head_angles(self):
        return self._head_angles

    @head_angles.setter
    def head_angles(self, value: np.ndarray[float]):
        if value is None:
            self._time_vector = None
        else:
            self._head_angles = value[:, self.indices]

    @property
    def gaze_direction(self):
        return self._gaze_direction

    @gaze_direction.setter
    def gaze_direction(self, value: np.ndarray[float]):
        if value is None:
            self._time_vector = None
        else:
            self._gaze_direction = value[:, self.indices]

    @property
    def head_angular_velocity(self):
        return self._head_angular_velocity

    @head_angular_velocity.setter
    def head_angular_velocity(self, value: np.ndarray[float]):
        if value is None:
            self._time_vector = None
        else:
            self._head_angular_velocity = value[:, self.indices]

    @property
    def head_velocity_norm(self):
        return self._head_velocity_norm

    @head_velocity_norm.setter
    def head_velocity_norm(self, value: np.ndarray[float]):
        if value is None:
            self._time_vector = None
        else:
            self._head_velocity_norm = value[self.indices]

    @property
    def data_invalidity(self):
        return self._data_invalidity

    @data_invalidity.setter
    def data_invalidity(self, value: np.ndarray[bool]):
        if value is None:
            self._time_vector = None
        else:
            self._data_invalidity = value[self.indices]
