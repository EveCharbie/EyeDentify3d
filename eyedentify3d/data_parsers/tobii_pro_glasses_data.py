from typing import Any, Callable
import numpy as np
import gzip
import csv
import json
import re

from .abstract_data import Data, destroy_on_fail
from ..error_type import ErrorType
from ..time_range import TimeRange
from ..utils.rotation_utils import unwrap_rotation


class TobiiProGlassesData(Data):
    """
    Load the data from a Tobii Pro Glasses 3 file.

    For the reference frame definition, see image
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
        data_file_path: The path to the HTC Vive Pro data file.
        error_type: The error handling method to use.
        time_range: The time range to consider in the trial.
        """
        # Initial attributes
        super().__init__(error_type, time_range)
        self.data_folder_path: str = data_folder_path

        gaze_data_dict, imu_data_dict = self.read_data()
        self.gaze_data_dict = gaze_data_dict
        self.imu_data_dict = imu_data_dict

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

    @staticmethod
    def parse_gz(file_path: str, data_dict: dict[str, Any], parsing_function: Callable):
        with gzip.open(file_path, 'rt') as f:
            reader = csv.reader(f)
            for row in reader:
                # Join the row back into a single string
                line = ','.join(row)

                # Replace patterns like 'key:' with '"key":'
                line = re.sub(r'(\w+):', r'"\1":', line)

                # Now transform the data from each line of the json
                data = json.loads(line)
                parsing_function(data, data_dict)
        return

    @staticmethod
    def parse_gaze_data(data: dict[str, Any], gaze_data_dict: dict[str, Any]):
        if "timestamp" in data.keys() and "data" in data.keys():
            gaze_data_dict["timestamp"] += [data["timestamp"]]
            if "gaze3d" in data["data"].keys():
                # Normalize the gaze direction vector
                gaze3d = np.array(data["data"]["gaze3d"])
                gaze_data_dict["gaze_vector"] += [gaze3d / np.linalg.norm(gaze3d)]
                if "eyeleft" in data["data"].keys() and "pupildiameter" in data["data"]["eyeleft"].keys() :
                    gaze_data_dict["pupil_diameter_left"] += [data["data"]["eyeleft"]["pupildiameter"]]
                else:
                    gaze_data_dict["pupil_diameter_left"] += [np.nan]
                if "eyeright" in data["data"].keys() and "pupildiameter" in data["data"]["eyeright"].keys() :
                    gaze_data_dict["pupil_diameter_right"] += [data["data"]["eyeright"]["pupildiameter"]]
                else:
                    gaze_data_dict["pupil_diameter_right"] += [np.nan]
            else:
                gaze_data_dict["gaze_vector"] += [np.nan]
                gaze_data_dict["pupil_diameter_left"] += [np.nan]
                gaze_data_dict["pupil_diameter_right"] += [np.nan]

    @staticmethod
    def parse_imu_data(data: dict[str, Any], imu_data_dict: dict[str, Any]):
        if "timestamp" in data.keys() and "data" in data.keys() and "accelerometer" in data["data"].keys() and "gyroscope" in data["data"].keys():
            imu_data_dict["timestamp"] += [data["timestamp"]]
            imu_data_dict["accelerometer"] += [np.array(data["data"]["accelerometer"])]
            imu_data_dict["gyroscope"] += [np.array(data["data"]["gyroscope"])]

    @staticmethod
    def parse_event_data(data: dict[str, Any], event_data_dict: dict[str, Any]):
        if "timestamp" in data.keys() and "data" in data.keys():
            print(data["type"])
            print(data["data"].keys())
            event_data_dict["timestamp"] += [data["timestamp"]]

    def read_data(self):
        """
        This function reads the Tobii Pro Glasses 3 data file, which is a gzipped CSV file with weird JSON lines.
        It returns a dictionary with `timestamp`and `gaze_vector`.
        """

        gaze_data_file_path = self.data_folder_path + "gazedata.gz"
        imu_data_file_path = self.data_folder_path + "imudata.gz"
        event_data_file_path = self.data_folder_path + "eventdata.gz"

        # Read the gaze data file
        gaze_data_dict = {
            "timestamp": [],  # in seconds
            "gaze_vector": [],  # unit vector of gaze orientation in 3D space
            "pupil_diameter_left": [],  # in mm
            "pupil_diameter_right": [],  # in mm
        }
        self.parse_gz(gaze_data_file_path, gaze_data_dict, self.parse_gaze_data)

        # Read the IMU data file
        imu_data_dict = {
            "timestamp": [],  # in seconds
            "accelerometer": [],  # Euler angles in degrees
            "gyroscope": [],  # in degrees per second
        }
        self.parse_gz(imu_data_file_path, imu_data_dict, self.parse_imu_data)

        return gaze_data_dict, imu_data_dict

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
        """
        time_vector = np.array(self.gaze_data_dict["timestamp"])
        if len(time_vector) == 0:
            self._validity_flag = False
            error_str = f"The file {self.data_folder_path + "gazedata.gz"} is empty. There is no element in the field 'timestamp'. Please check the file."
            self.error_type(error_str)

        if (
            np.sum(np.logical_or(np.isnan(self.gaze_data_dict["gaze_vector"]),
                                 np.isnan(self.gaze_data_dict["pupil_diameter_left"]),
                                 np.isnan(self.gaze_data_dict["pupil_diameter_right"]),
                                 ))
            > len(self.gaze_data_dict["timestamp"]) / 2
        ):
            self._validity_flag = False
            error_str = f"More than 50% of the data from file {self.data_folder_path + "gazedata.gz"} is declared invalid or in a blink by the eye-tracker, skipping this file."
            self.error_type(error_str)
            return

        if np.any((time_vector[1:] - time_vector[:-1]) < 0):
            self._validity_flag = False
            error_str = f"The time vector in file {self.data_folder_path + "gazedata.gz"} is not strictly increasing. Please check the file."
            self.error_type(error_str)
            return

        # If we reach this point, the data is valid
        return

    @destroy_on_fail
    def _set_time_vector(self):
        """
        Set the time vector [seconds] from the gaze data file.
        """
        # Set the time vector (already in seconds)
        time_vector = np.array(self.gaze_data_dict["timestamp"])
        initial_time = time_vector[0]
        self.time_vector = (time_vector - initial_time)

        # also transform imu timings
        self.imu_data_dict["timestamp"] = np.array(self.imu_data_dict["timestamp"]) - initial_time

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
        # Check shapes
        if len(unwrapped_head_angles.shape) != 2 or unwrapped_head_angles.shape[0] != 3:
            raise NotImplementedError("This function was designed for head angles of shape (3, n_frames). ")

        # Check if there is duplicated frames in the imu data
        frame_diffs = np.linalg.norm(unwrapped_head_angles[:, 1:] - unwrapped_head_angles[:, :-1], axis=0)
        if not np.all(frame_diffs > 1e-10):
            raise RuntimeError(
                "There were repeated frames in the imu data, which never happened with this eye-tracker. Please notify the developer."
            )

        # Interpolate the head angles to the eye timestamps
        interpolated_head_angles = np.zeros((3, self.nb_frames))
        for i_time, time in enumerate(self.time_vector):
            if time < time_vector_imu[0] or time > time_vector_imu[-1]:
                interpolated_head_angles[:, i_time] = np.nan
            else:
                if time in time_vector_imu:
                    idx = np.where(time_vector_imu == time)[0][0]
                    interpolated_head_angles[:, i_time] = unwrapped_head_angles[:, idx]
                else:
                    idx_before = np.where(time_vector_imu < time)[0][-1]
                    idx_after = np.where(time_vector_imu > time)[0][0]
                    t_before = time_vector_imu[idx_before]
                    t_after = time_vector_imu[idx_after]
                    angles_before = unwrapped_head_angles[:, idx_before]
                    angles_after = unwrapped_head_angles[:, idx_after]
                    interpolated_head_angles[:, i_time] = angles_before + (time - t_before) * (
                        (angles_after - angles_before) / (t_after - t_before)
                    )
        return interpolated_head_angles

    @destroy_on_fail
    def _set_eye_openness(self) -> None:
        """
        Set the eye openness of both eyes.
        """
        self.right_eye_openness = np.logical_not(np.isnan(self.gaze_data_dict["pupil_diameter_right"]))
        self.left_eye_openness = np.logical_not(np.isnan(self.gaze_data_dict["pupil_diameter_left"]))

    @destroy_on_fail
    def _set_eye_direction(self):
        """
        Get the eye direction from the gaze data. It is a unit vector in the same direction as the eyes.
        """
        self.eye_direction = np.array(self.gaze_data_dict["gaze_vector"]).T

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


        self.imu_data_dict["accelerometer"]
        self.imu_data_dict["gyroscope"]

        unwrapped_head_angles = unwrap_rotation(head_angles)
        # We interpolate to avoid duplicated frames, which would affect the finite difference computation
        self.head_angles = self.interpolate_repeated_frames(unwrapped_head_angles)


        self.head_angles = self.interpolate_to_eye_timestamps(time_vector_imu, unwrapped_head_angles)

    @destroy_on_fail
    def _set_data_invalidity(self):
        """
        Get a numpy array of bool indicating if the eye-tracker declared this data frame as invalid.
        """
        self.data_invalidity = np.logical_or(self.csv_data["eye_valid_L"] != 31, self.csv_data["eye_valid_R"] != 31)
