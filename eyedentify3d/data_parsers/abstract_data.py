from abc import ABC, abstractmethod
from functools import wraps
from datetime import datetime
import numpy as np

from ..error_type import ErrorType
from ..time_range import TimeRange


def destroy_on_fail(method):
    """Decorator to exit initialization automatically"""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self._validity_flag:
            # Call the original function
            method(self, *args, **kwargs)

            # Check if method failed
            if not self._validity_flag:
                self.destroy_on_error()

    return wrapper


class Data(ABC):
    """
    Load the data from a HTC Vive Pro file.
    """

    def __init__(self, error_type: ErrorType = ErrorType.PRINT, time_range: TimeRange = TimeRange()):
        """
        Parameters
        ----------
        error_type: How to handle the errors.
        time_range: The time range to consider in the trial.
        """
        # Original attributes
        self.error_type = error_type
        self.time_range = time_range

        # Extended attributes
        self._validity_flag = True
        # These will be set by the subclass
        self.time_vector: np.ndarray[float] | None = None
        self.eye_direction: np.ndarray[float] | None = None
        self.head_angles: np.ndarray[float] | None = None
        self.head_angular_velocity: np.ndarray[float] | None = None
        self.head_velocity_norm: np.ndarray[float] | None = None
        self.data_validity: np.ndarray[bool] | None = None

    @property
    def error_type(self):
        return self._error_type

    @error_type.setter
    def error_type(self, value: ErrorType):
        if not isinstance(value, ErrorType):
            raise ValueError(f"The error type must be an ErrorType, got {value}.")
        if value == ErrorType.FILE:
            with open("bad_data_files.txt", "w") as bad_data_file:
                bad_data_file.write(
                    f"Bad data file created on {bad_data_file.write(f"Bad data file created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \n")} \n"
                )

        self._error_type = value

    @property
    def time_range(self):
        return self._time_range

    @time_range.setter
    def time_range(self, value: TimeRange):
        if not isinstance(value, TimeRange):
            raise ValueError(f"The time range must be an TimeRange, got {value}.")
        self._time_range = value

    @property
    def trial_duration(self):
        if self.time_vector is None:
            raise RuntimeError(
                "The trial_duration property can only be called after the time_vector has been set "
                "(i.e., after the data objects has been instantiated)."
            )
        return self.time_vector[-1] - self.time_vector[0]

    @abstractmethod
    def _check_validity(self):
        """
        Check if the data is valid.
        """
        pass

    @abstractmethod
    def _set_time_vector(self):
        """
        Set the time vector from the data file.
        """
        pass

    @abstractmethod
    def _discard_data_out_of_range(self):
        """
        Discard the data that is out of the time range.
        """
        pass

    @abstractmethod
    def _get_eye_direction(self):
        """
        Get the eye direction from the data file.
        """
        pass

    @abstractmethod
    def _get_head_angles(self):
        """
        Get the head angles from the data file.
        """
        pass

    @abstractmethod
    def _get_head_angular_velocity(self):
        """
        Get the head angular velocity from the data file.
        """
        pass

    @abstractmethod
    def _set_data_validity(self):
        """
        Set the validity of the data.
        """
        pass

    def destroy_on_error(self):
        """
        In case of an error, return an object full of Nones.
        """
        self.time_vector = None
        self.eye_direction = None
        self.head_angles = None
        self.head_angular_velocity = None
        self.head_velocity_norm = None
