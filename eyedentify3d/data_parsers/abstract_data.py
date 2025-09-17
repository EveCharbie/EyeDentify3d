from abc import ABC, abstractmethod
from functools import wraps
from datetime import datetime
import numpy as np
import matplotlib
from matplotlib.axes import Axes

from ..error_type import ErrorType
from ..time_range import TimeRange
from ..utils.rotation_utils import get_gaze_direction, compute_angular_velocity


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
        self.dt: float | None = None
        # These will be set by the subclass
        self.time_vector: np.ndarray[float] | None = None
        self.right_eye_openness: np.ndarray[float] | None = None
        self.left_eye_openness: np.ndarray[float] | None = None
        self.eye_direction: np.ndarray[float] | None = None
        self.head_angles: np.ndarray[float] | None = None
        self.gaze_direction: np.ndarray[float] | None = None
        self.head_angular_velocity: np.ndarray[float] | None = None
        self.head_velocity_norm: np.ndarray[float] | None = None
        self.data_invalidity: np.ndarray[bool] | None = None
        # These will be set by finalize
        self.gaze_angular_velocity: np.ndarray[float] | None = None
        self.is_finalized = False

    @property
    def error_type(self):
        return self._error_type

    @error_type.setter
    def error_type(self, value: ErrorType):
        if not isinstance(value, ErrorType):
            raise ValueError(f"The error type must be an ErrorType, got {value}.")
        if value == ErrorType.FILE:
            with open("bad_data_files.txt", "w") as bad_data_file:
                bad_data_file.write(f"Bad data file created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \n")

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

    @destroy_on_fail
    def _set_dt(self):
        if self.time_vector is None:
            raise RuntimeError(
                "The dt property can only be called after the time_vector has been set "
                "(i.e., after the data objects has been instantiated)."
            )
        if self.dt is not None:
            raise RuntimeError(
                "dt can only be set once at the very beginning of the data processing, because the time vector will be modified later."
            )
        self.dt = np.nanmean(self.time_vector[1:] - self.time_vector[:-1])

    @destroy_on_fail
    def _set_gaze_direction(self):
        """
        Get the gaze direction from the head angles and the eye direction.
        The gaze direction is a unit vector expressed in the global reference frame representing the combined rotations of the head and eyes.
        """
        if self.head_angles is None or self.eye_direction is None:
            raise RuntimeError(
                "The gaze direction can only be set after the head angles and eye direction have been set "
                "(i.e., after the data objects has been instantiated)."
            )
        self.gaze_direction = get_gaze_direction(self.head_angles, self.eye_direction)

    @property
    def file_name(self):
        """
        Get the name of the data file or folder name.
        """
        if hasattr(self, "data_file_path"):
            return self.data_file_path.split("/")[-1]
        elif hasattr(self, "data_folder_path"):
            return self.data_folder_path.split("/")[-1]
        else:
            raise AttributeError("The data file or folder path is not set.")

    @property
    def nb_frames(self):
        """
        Get the number of frames measured in the data file.
        """
        if self.time_vector is None:
            raise RuntimeError(
                "The nb_frames property can only be called after the time_vector has been set "
                "(i.e., after the data objects has been instantiated)."
            )
        return self.time_vector.shape[0]

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
    def _set_eye_openness(self):
        """
        Get the eye openness from the data file.
        """
        pass

    @abstractmethod
    def _set_eye_direction(self):
        """
        Get the eye direction from the data file.
        """
        pass

    @abstractmethod
    def _set_head_angles(self):
        """
        Get the head angles from the data file.
        """
        pass

    @abstractmethod
    def _set_head_angular_velocity(self):
        """
        Get the head angular velocity from the data file.
        """
        pass

    @abstractmethod
    def _set_data_invalidity(self):
        """
        Set the invalidity of the data.
        """
        pass

    def set_gaze_angular_velocity(self):
        """
        Computes the gaze (eye + head) angular velocity in deg/s as the angle difference between two frames divided by
        the time difference between them. It is computed like a centered finite difference, meaning that the frame i+1
        and i-1 are used to set the value for the frame i.
        """
        self.gaze_angular_velocity = compute_angular_velocity(self.time_vector, self.gaze_direction)

    @destroy_on_fail
    def finalize(self):
        """
        Finalize the data object by computing some secondary quantities.
        This method should be called after all the data has been set.
        """
        if self.time_vector is None or self.gaze_direction is None:
            raise RuntimeError(
                "The finalize method can only be called after the time_vector and gaze_direction have been set "
                "(i.e., after the data objects has been instantiated)."
            )
        self.set_gaze_angular_velocity()
        self.is_finalized = True

    def plot_gaze_vector(self, ax: Axes):
        if not self.is_finalized:
            raise RuntimeError("Please call .finalize() before calling plot functions.")

        ax.plot(self.time_vector, self.gaze_direction[0, :], "-k", label="Gaze X")
        ax.plot(self.time_vector, self.gaze_direction[1, :], "--k", label="Gaze Y")
        ax.plot(self.time_vector, self.gaze_direction[2, :], ":k", label="Gaze Z")

    def destroy_on_error(self):
        """
        In case of an error, return an object full of Nones.
        """
        self.time_vector = None
        self.right_eye_openness = None
        self.left_eye_openness = None
        self.eye_direction = None
        self.head_angles = None
        self.gaze_direction = None
        self.head_angular_velocity = None
        self.head_velocity_norm = None
