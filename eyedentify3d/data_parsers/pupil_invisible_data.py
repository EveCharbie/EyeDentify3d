import pandas as pd
import numpy as np

from .abstract_data import Data, destroy_on_fail
from ..error_type import ErrorType
from ..time_range import TimeRange
from ..utils.rotation_utils import unwrap_rotation, rotation_matrix_from_euler_angles, angles_from_imu_fusion
from ..utils.signal_utils import interpolate_to_specified_timestamps


class PupilInvisibleData(Data):
    """
    Load the data from a Pupil Invisible folder.

    For the reference frame definition, see:
        gaze coordinates: https://framerusercontent.com/images/OXOwlMKDg5fYJd2Vv5kQGvBXJw.jpg
        imu coordinates: https://docs.pupil-labs.com/invisible/assets/pi-imu-diagram.DoPp4CcW.jpg
    """

    def __init__(
        self,
        data_folder_path: str,
        error_type: ErrorType = ErrorType.PRINT,
        time_range: TimeRange = None,
    ):
        """
        Parameters
        ----------
        data_folder_path: The path to the Pupil Invisible data files.
        error_type: The error handling method to use.
        time_range: The time range to consider in the trial.
        """
        # Initial attributes
        super().__init__(error_type, time_range)
        self.data_folder_path: str = data_folder_path

        # Load the data, please note that these entries will not be updated with the time_range
        self.gaze_csv_data: pd.DataFrame = pd.read_csv(self.data_folder_path + "gaze.csv")
        self.imu_csv_data: pd.DataFrame = pd.read_csv(self.data_folder_path + "imu.csv")
        self.blink_csv_data: pd.DataFrame = pd.read_csv(self.data_folder_path + "blinks.csv")

        self._check_validity()
        self._set_time_vector()
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
        self._discard_data_out_of_range()
        self.finalize()

    @property
    def data_folder_path(self):
        return self._data_folder_path

    @data_folder_path.setter
    def data_folder_path(self, value: str):
        if not isinstance(value, str):
            raise ValueError(f"The data_folder_path must be a string, got {value}.")
        if not value.endswith("/"):
            value += "/"
        self._data_folder_path = value

    @destroy_on_fail
    def _check_validity(self):
        """
        Check if the eye-tracker data is valid.
        Note: The Pupil Invisible eye-tracker does not provide a good estimate of the validity of the measured data.
        The best we have is the "worn" field which is typically always set to 1.
        """
        time_vector = np.array(self.gaze_csv_data["timestamp [ns]"])
        if len(time_vector) == 0:
            self._validity_flag = False
            error_str = f"The file {self.file_name} is empty. There is no element in the field 'timestamp [ns]'. Please check the file."
            self.error_type(error_str)
            return

        elif np.sum(self.gaze_csv_data["worn"] != 1) > len(self.gaze_csv_data["worn"]) / 2:
            self._validity_flag = False
            error_str = f"More than 50% of the data from file {self.file_name} is declared invalid by the eye-tracker, skipping this file."
            self.error_type(error_str)
            return

        elif np.any((time_vector[1:] - time_vector[:-1]) < 0):
            self._validity_flag = False
            error_str = f"The time vector in file {self.file_name} is not strictly increasing. Please check the file."
            self.error_type(error_str)
            return

        # If we reach this point, the data is valid
        return

    @destroy_on_fail
    def _set_time_vector(self):
        """
        Set the time vector [seconds] from the gaze csv data.
        Note: We use the eye data timestamps as reference, and the IMU data is then interpolated to match these timestamps.
        """
        factor = 1e9  # ns to seconds
        initial_time = self.gaze_csv_data["timestamp [ns]"][0]

        # Set the time vector
        self.time_vector = np.array(np.array((self.gaze_csv_data["timestamp [ns]"]) - initial_time) / factor)

        # also transform the blink and imu timings
        self.blink_csv_data["start timestamp [ns]"] = (
            self.blink_csv_data["start timestamp [ns]"] - initial_time
        ) / factor
        self.blink_csv_data["end timestamp [ns]"] = (self.blink_csv_data["end timestamp [ns]"] - initial_time) / factor
        self.imu_csv_data["timestamp [ns]"] = (self.imu_csv_data["timestamp [ns]"] - initial_time) / factor

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

        # Update the attributes with the indices to keep
        self.time_vector = self.time_vector[indices_to_keep]
        self.dt = np.nanmean(self.time_vector[1:] - self.time_vector[:-1])
        self.right_eye_openness = self.right_eye_openness[indices_to_keep]
        self.left_eye_openness = self.left_eye_openness[indices_to_keep]
        self.eye_direction = self.eye_direction[:, indices_to_keep]
        self.head_angles = self.head_angles[:, indices_to_keep]
        self.gaze_direction = self.gaze_direction[:, indices_to_keep]
        self.head_angular_velocity = self.head_angular_velocity[:, indices_to_keep]
        self.head_velocity_norm = self.head_velocity_norm[indices_to_keep]
        self.data_invalidity = self.data_invalidity[indices_to_keep]

    @destroy_on_fail
    def _set_eye_openness(self) -> None:
        """
        Pupil Invisible does not provide eye openness, so we set it to 0 when there is a blink and 1 otherwise.
        """
        self.right_eye_openness = np.ones((self.nb_frames,))
        self.left_eye_openness = np.ones((self.nb_frames,))
        for blink_beginning, blink_end in zip(
            self.blink_csv_data["start timestamp [ns]"], self.blink_csv_data["end timestamp [ns]"]
        ):
            start_idx = np.where(np.abs(self.time_vector - blink_beginning) < 1e-6)[0]
            end_idx = np.where(np.abs(self.time_vector - blink_end) < 1e-6)[0]

            if len(start_idx) == 0 or len(end_idx) == 0:
                raise RuntimeError(
                    "The blink start or end times are not in the time vector. This should not happen, please notify the developer."
                )

            # Set the eye openness to 0 during blinks
            start_idx = start_idx[0]
            end_idx = end_idx[0]
            self.right_eye_openness[range(start_idx, end_idx + 1)] = 0.0
            self.left_eye_openness[range(start_idx, end_idx + 1)] = 0.0

    @destroy_on_fail
    def _set_eye_direction(self):
        """
        Get the eye direction by applying the eye rotation angles to a unit vector. It gives a unit vector in the same direction as the eyes.
        """

        # Get the eye angles in radians
        eye_azimuth = self.gaze_csv_data["azimuth [deg]"] * np.pi / 180.0
        eye_elevation = self.gaze_csv_data["elevation [deg]"] * np.pi / 180.0
        eye_angles = np.array([eye_azimuth, eye_elevation])

        # Get the eye rotation matrix and orientation
        forward_vector = np.array([0.0, 0.0, 1.0])
        rotation_matrix = np.zeros((3, 3, self.nb_frames))
        eye_direction = np.zeros((3, self.nb_frames))
        for i_frame in range(self.nb_frames):
            rotation_matrix[:, :, i_frame] = rotation_matrix_from_euler_angles("xy", eye_angles[:, i_frame])
            eye_direction[:, i_frame] = np.reshape(rotation_matrix[:, :, i_frame] @ forward_vector[:, np.newaxis], (3,))

        # Check that the eye direction is normalized
        eye_direction_norm = np.linalg.norm(eye_direction, axis=0)
        if np.any(np.logical_or(eye_direction_norm > 1.2, eye_direction_norm < 0.8)):
            raise RuntimeError("There was an issue with the eye direction computation, please notify the developer.")

        # If the norm is not far from one, still renormalize to avoir issues later on
        self.eye_direction = eye_direction / eye_direction_norm

    def interpolate_to_eye_timestamps(
        self, time_vector_imu: np.ndarray[float], unwrapped_head_angles: np.ndarray[float]
    ) -> np.ndarray[float]:
        """
        This function gets the head orientation at the eye data time stamps by interpolating if necessary.

        Parameters
        ----------
        time_vector_imu: The time vector of the imu data (not the same as the eye data) (n_frames_imu)
        unwrapped_head_angles: The unwrapped head angles (roll, pitch, yaw) in degrees (3, n_frames_imu)

        Returns
        -------
        The modified numpy array of head angles aligned with the eye data timestamps (3, n_frames)
        """
        interpolated_head_angles = interpolate_to_specified_timestamps(
            time_vector_imu,
            self.time_vector,
            unwrapped_head_angles,
        )
        return interpolated_head_angles

    @destroy_on_fail
    def _set_head_angles(self):
        """
        Get the head orientation from the imu csv data. It is expressed as Euler angles in degrees and is measured by
        the glasses IMU containing a gyroscope and accelerometer. If there are no tags in your experimental setup,
        Pupil Invisible does not provide the yaw angle, so we approximate it here. But please note that this
        approximation is less precise since there is no magnetometer in the glasses' IMU. So the yaw angle is prone to
        drifting, but in our cas the effect should be minimal sinc we mainly compare frame through a small time interval.
        """
        # Get the time vector of the imu data (not the same as the eye data)
        time_vector_imu = np.array(self.imu_csv_data["timestamp [ns]"])

        tags_in_exp: bool = not np.all(np.isnan(self.imu_csv_data["yaw [deg]"]))
        if tags_in_exp:
            # The yaw angle is already provided by Pupil as there were tags in the experimental setup
            head_angles = np.array(
                [self.imu_csv_data["roll [deg]"], self.imu_csv_data["pitch [deg]"], self.imu_csv_data["yaw [deg]"]]
            )
        else:
            # No tags were used in the experimental setup, so we approximate the yaw angle using a Madgwick filter
            acceleration = np.array(
                [
                    self.imu_csv_data["acceleration x [g]"],
                    self.imu_csv_data["acceleration y [g]"],
                    self.imu_csv_data["acceleration z [g]"],
                ]
            )
            gyroscope = np.array(
                [
                    self.imu_csv_data["gyro x [deg/s]"],
                    self.imu_csv_data["gyro y [deg/s]"],
                    self.imu_csv_data["gyro z [deg/s]"],
                ]
            )
            roll, pitch, yaw = angles_from_imu_fusion(
                time_vector_imu, acceleration, gyroscope, roll_offset=7, pitch_offset=90
            )
            head_angles = np.array([roll, pitch, yaw])

        unwrapped_head_angles = unwrap_rotation(head_angles)
        # We interpolate to align the head angles with the eye orientation timestamps
        self.head_angles = self.interpolate_to_eye_timestamps(time_vector_imu, unwrapped_head_angles)

    @destroy_on_fail
    def _set_data_invalidity(self):
        """
        Get a numpy array of bool indicating if the eye-tracker declared that the glasses were not worn.
        """
        self.data_invalidity = np.array(self.gaze_csv_data["worn"] != 1)
