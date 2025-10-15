import pandas as pd
import numpy as np

from .abstract_data import Data, destroy_on_fail
from ..error_type import ErrorType
from ..time_range import TimeRange
from ..utils.rotation_utils import unwrap_rotation


class PicoNeoData(Data):
    """
    Load the data from a Pico Neo 3 Pro file.

    For the reference frame definition, see ... TODO
    """

    def __init__(
        self,
        data_file_path: str,
        error_type: ErrorType = ErrorType.PRINT,
        time_range: TimeRange = TimeRange(),
    ):
        """
        Parameters
        ----------
        data_file_path: The path to the HTC Vive Pro data file.
        error_type: The error handling method to use.
        time_range: The time range to consider in the trial.
        """
        # Initial attributes
        super().__init__(error_type, time_range)
        self.data_file_path: str = data_file_path

        # Load the data and set the time vector
        self.csv_data: pd.DataFrame = pd.read_csv(self.data_file_path, sep="\t")
        self._check_validity()
        self._set_time_vector()
        self._discard_data_out_of_range()
        self._set_dt()
        self._remove_duplicates()  # There should not be any duplicates

        # Initialize variables
        self._set_eye_openness()
        self._set_eye_direction()
        self._set_head_angles()
        self._set_gaze_direction()
        self._set_head_angular_velocity()
        self._set_data_invalidity()

        # Finalize the data object
        self.finalize()

    @property
    def data_file_path(self):
        return self._data_file_path

    @data_file_path.setter
    def data_file_path(self, value: str):
        if not isinstance(value, str):
            raise ValueError(f"The data_file_path must be a string, got {value}.")
        if not value.endswith(".csv"):
            raise ValueError(f"The HTC Vive Pro data file must be a .csv file, got {value}.")
        self._data_file_path = value

    @destroy_on_fail
    def _check_validity(self):
        """
        Check if the eye-tracker data is valid.
        """
        time_vector = np.array(self.csv_data["Timeline"])
        if len(time_vector) == 0:
            self._validity_flag = False
            error_str = f"The file {self.file_name} is empty. There is no element in the field 'Timeline'. Please check the file."
            self.error_type(error_str)

        if (
            np.sum(
                np.logical_or(self.csv_data["Left Eye Pose Status"] != 52, self.csv_data["Right Eye Pose Status"] != 52)
            )
            > len(self.csv_data["Left Eye Pose Status"]) / 2
        ):
            self._validity_flag = False
            error_str = f"More than 50% of the data from file {self.file_name} is declared invalid by the eye-tracker, skipping this file."
            self.error_type(error_str)
            return

        if np.any((time_vector[1:] - time_vector[:-1]) < 0):
            self._validity_flag = False
            error_str = f"The time vector in file {self.file_name} is not strictly increasing. Please check the file."
            self.error_type(error_str)
            return

        # If we reach this point, the data is valid
        return

    @destroy_on_fail
    def _set_time_vector(self):
        """
        Set the time vector [seconds] from the csv data.
        """
        factor = 1  # Already in seconds
        self.time_vector = np.array((self.csv_data["Timeline"] - self.csv_data["Timeline"][0]) / factor)

    @destroy_on_fail
    def _remove_duplicates(self):
        """
        Check that there are no duplicate time frames in the time vector.
        """
        if len(np.where(np.abs(self.time_vector[1:] - self.time_vector[:-1]) < 1e-10)[0]) > 0:
            raise RuntimeError(
                "The time vector has duplicated frames, which never happened with this eye-tracker. Please notify the developer."
            )

    @destroy_on_fail
    def _discard_data_out_of_range(self):
        """
        Discard the data that is out of the time range specified in the time_range attribute.
        """
        indices_to_keep = self.time_range.get_indices(self.time_vector)
        self.time_vector = self.time_vector[indices_to_keep]
        self.csv_data = self.csv_data.iloc[indices_to_keep, :]

    @destroy_on_fail
    def _set_eye_openness(self) -> None:
        """
        Set the eye openness of both eyes.
        """
        self.right_eye_openness = np.ones_like(self.csv_data["Eye Right Blinking"])
        self.left_eye_openness = np.ones_like(self.csv_data["Eye Left Blinking"])

        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 1)
        # axs[0].plot(self.right_eye_openness, "r", label="Right Eye Gaze Openness")
        # axs[0].plot(self.left_eye_openness, "g", label="Left Eye Gaze Openness")
        axs[0].legend()
        axs[1].plot(self.csv_data["Eye Right Blinking"], "r", label="Eye Right blinking")
        axs[1].plot(self.csv_data["Eye Left Blinking"], "g", label="Eye Left blinking")
        axs[1].legend()
        plt.show()

    @destroy_on_fail
    def _set_eye_direction(self):
        """
        Get the eye direction from the csv data. It is a unit vector in the same direction as the eyes.
        """
        eye_direction = np.array(
            [
                self.csv_data["Combine Eye Gaze Vector. x"],
                self.csv_data["Combine Eye Gaze Vector. y"],
                self.csv_data["Combine Eye Gaze Vector. z"],
            ]
        )

        eye_direction_norm = np.linalg.norm(eye_direction, axis=0)
        # Replace zeros, which are due to bad data
        eye_direction_norm[eye_direction_norm == 0] = np.nan
        if np.any(np.logical_or(eye_direction_norm > 1.2, eye_direction_norm < 0.8)):
            self._validity_flag = False
            error_str = f"The eye direction in file {self.file_name} is not normalized (min = {np.min(eye_direction_norm)}, max = {np.max(eye_direction_norm)}). Please check the file."
            self.error_type(error_str)
            return

        # If the norm is not far from one, still renormalize to avoir issues later on
        self.eye_direction = eye_direction / eye_direction_norm

    @destroy_on_fail
    def _set_head_angles(self):
        """
        Get the head orientation from the csv data. It is expressed as Euler angles in degrees and is measured by the VR helmet.
        """
        head_angles = np.array(
            [self.csv_data["Head Rotation. x"], self.csv_data["Head Rotation. y"], self.csv_data["Head Rotation. z"]]
        )
        self.head_angles = unwrap_rotation(head_angles)

    @destroy_on_fail
    def _set_data_invalidity(self):
        """
        Get a numpy array of bool indicating if the eye-tracker declared this data frame as invalid.
        """
        self.data_invalidity = np.logical_or(
            self.csv_data["Left Eye Pose Status"] != 52, self.csv_data["Right Eye Pose Status"] != 52
        )
