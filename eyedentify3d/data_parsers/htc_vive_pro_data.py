import pandas as pd
import numpy as np

from .abstract_data import Data, destroy_on_fail
from ..error_type import ErrorType
from ..time_range import TimeRange
from ..utils.rotation_utils import unwrap_rotation
from ..utils.signal_utils import filter_data, centered_finite_difference


class HtcViveProData(Data):
    """
    Load the data from a HTC Vive Pro file.

    For the reference frame definition, see image https://www.researchgate.net/figure/Left-Coordinate-system-of-HTC-Vive-Pro-Eye-and-right-a-diagram-showing-gaze-origin_fig4_373699457
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
        self.csv_data: pd.DataFrame = pd.read_csv(self.data_file_path, sep=";")
        self._check_validity()
        self._set_time_vector()
        self._discard_data_out_of_range()
        self._set_dt()
        self._remove_duplicates()  # This method is specific to HTC Vive Pro data, as it has duplicated frames

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
        time_vector = np.array(self.csv_data["time(100ns)"])
        if len(time_vector) == 0:
            self._validity_flag = False
            error_str = f"The file {self.file_name} is empty. There is no element in the field 'time(100ns)'. Please check the file."
            self.error_type(error_str)

        if (
            np.sum(np.logical_or(self.csv_data["eye_valid_L"] != 31, self.csv_data["eye_valid_R"] != 31))
            > len(self.csv_data["eye_valid_L"]) / 2
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
        factor = 10000000  # 100 ns to seconds
        self.time_vector = np.array((self.csv_data["time(100ns)"] - self.csv_data["time(100ns)"][0]) / factor)

    @destroy_on_fail
    def _remove_duplicates(self):
        """
        A few frames are duplicated in the HTC Vive Pro data, which can cause issues later.
        So we completely remove the duplicated frames.
        """
        good_timestamps_indices = np.where(np.abs(self.time_vector[1:] - self.time_vector[:-1]) > 1e-10)[0] + 1
        good_timestamps_indices = np.insert(good_timestamps_indices, 0, 0)  # Include the first frame
        self.time_vector = self.time_vector[good_timestamps_indices]
        self.csv_data = self.csv_data.iloc[good_timestamps_indices, :]

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
        self.right_eye_openness = self.csv_data["openness_R"]
        self.left_eye_openness = self.csv_data["openness_L"]

    @destroy_on_fail
    def _set_eye_direction(self):
        """
        Get the eye direction from the csv data. It is a unit vector in the same direction as the eyes.
        """
        eye_direction = np.array(
            [self.csv_data["gaze_direct_L.x"], self.csv_data["gaze_direct_L.y"], self.csv_data["gaze_direct_L.z"]]
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

    def interpolate_repeated_frames(self, data_to_interpolate: np.ndarray[float]) -> np.ndarray[float]:
        """
        This function detects repeated frames and replace them with a linear interpolation between the last and the nex frame.
        Unfortunately, this step is necessary as the HTC Vive Pro duplicates some frames.
        This is particularly important as the velocities are computed as finite differences.

        Parameters
        ----------
        data_to_interpolate: A numpy array matrix to modify to demove duplicates (3, n_frames)

        Returns
        -------
        The modified numpy array matrix with duplicates removed, and replaced with a linear interpolation (3, n_frames)
        """
        # Check shapes
        if len(data_to_interpolate.shape) != 2 or data_to_interpolate.shape[0] != 3:
            raise NotImplementedError("This function was designed for matrix data of shape (3, n_frames). ")

        # Avoid too small vectors
        n_frames = data_to_interpolate.shape[1]
        if n_frames < 2:
            return data_to_interpolate

        # Find where frames are different from the previous frame
        frame_diffs = np.linalg.norm(data_to_interpolate[:, 1:] - data_to_interpolate[:, :-1], axis=0)
        unique_frame_mask = np.concatenate([[True], frame_diffs > 1e-10])
        unique_indices = np.where(unique_frame_mask)[0]

        # Interpolate between unique frames
        result = data_to_interpolate.copy()
        for i in range(len(unique_indices) - 1):
            start_idx = unique_indices[i]
            end_idx = unique_indices[i + 1]
            if end_idx - start_idx > 1:
                # There are repeated frames to interpolate
                for i_component in range(3):
                    result[i_component, start_idx:end_idx] = np.linspace(
                        data_to_interpolate[i_component, start_idx],
                        data_to_interpolate[i_component, end_idx],
                        end_idx - start_idx + 1,
                    )[:-1]

        return result

    @destroy_on_fail
    def _set_head_angles(self):
        """
        Get the head orientation from the csv data. It is expressed as Euler angles in degrees and is measured by the VR helmet.
        """
        head_angles = np.array(
            [self.csv_data["helmet_rot_x"], self.csv_data["helmet_rot_y"], self.csv_data["helmet_rot_z"]]
        )
        unwrapped_head_angles = unwrap_rotation(head_angles)
        # We interpolate to avoid duplicated frames, which would affect the finite difference computation
        self.head_angles = self.interpolate_repeated_frames(unwrapped_head_angles)

    @destroy_on_fail
    def _set_head_angular_velocity(self):
        """
        Get the head angular velocity from the csv data.
        We keep both the Euler angles derivative in degrees/s and the filtered angular velocity norm.
        """
        self.head_angular_velocity = centered_finite_difference(self.time_vector, self.head_angles)
        head_velocity_norm = np.linalg.norm(self.head_angular_velocity, axis=0)
        self.head_velocity_norm = filter_data(head_velocity_norm[np.newaxis, :])[0, :]

    @destroy_on_fail
    def _set_data_invalidity(self):
        """
        Get a numpy array of bool indicating if the eye-tracker declared this data frame as invalid.
        """
        self.data_invalidity = np.logical_or(self.csv_data["eye_valid_L"] != 31, self.csv_data["eye_valid_R"] != 31)
