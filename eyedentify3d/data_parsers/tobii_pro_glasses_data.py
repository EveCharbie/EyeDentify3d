from typing import Any, Callable
import numpy as np
import gzip
import csv
import json
import re

from .abstract_data import Data, destroy_on_fail
from ..error_type import ErrorType
from ..time_range import TimeRange
from ..utils.rotation_utils import unwrap_rotation, angles_from_imu_fusion
from ..utils.signal_utils import interpolate_to_specified_timestamps


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
        self._remove_duplicates()

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
        with gzip.open(file_path, "rt") as f:
            reader = csv.reader(f)
            for row in reader:
                # Join the row back into a single string
                line = ",".join(row)

                # Replace patterns like 'key:' with '"key":'
                line = re.sub(r"(\w+):", r'"\1":', line)

                # Now transform the data from each line of the json
                data = json.loads(line)
                parsing_function(data, data_dict)

        # Put the data back into numpy arrays
        for key in data_dict.keys():
            data_dict[key] = np.array(data_dict[key])

        return

    @staticmethod
    def parse_gaze_data(data: dict[str, Any], gaze_data_dict: dict[str, Any]):
        if "timestamp" in data.keys() and "data" in data.keys():
            gaze_data_dict["timestamp"] += [data["timestamp"]]
            if "gaze3d" in data["data"].keys():
                # Normalize the gaze direction vector
                gaze3d = np.array(data["data"]["gaze3d"])
                gaze_data_dict["gaze_vector"] += [gaze3d / np.linalg.norm(gaze3d)]
                if "eyeleft" in data["data"].keys() and "pupildiameter" in data["data"]["eyeleft"].keys():
                    gaze_data_dict["pupil_diameter_left"] += [data["data"]["eyeleft"]["pupildiameter"]]
                else:
                    gaze_data_dict["pupil_diameter_left"] += [np.nan]
                if "eyeright" in data["data"].keys() and "pupildiameter" in data["data"]["eyeright"].keys():
                    gaze_data_dict["pupil_diameter_right"] += [data["data"]["eyeright"]["pupildiameter"]]
                else:
                    gaze_data_dict["pupil_diameter_right"] += [np.nan]
            else:
                gaze_data_dict["gaze_vector"] += [[np.nan, np.nan, np.nan]]
                gaze_data_dict["pupil_diameter_left"] += [np.nan]
                gaze_data_dict["pupil_diameter_right"] += [np.nan]

    @staticmethod
    def parse_imu_data(data: dict[str, Any], imu_data_dict: dict[str, Any]):
        if "timestamp" in data.keys():
            if "data" in data.keys() and "accelerometer" in data["data"].keys() and "gyroscope" in data["data"].keys():
                imu_data_dict["timestamp_100Hz"] += [data["timestamp"]]
                imu_data_dict["accelerometer"] += [np.array(data["data"]["accelerometer"])]
                imu_data_dict["gyroscope"] += [np.array(data["data"]["gyroscope"])]
            elif "magnetometer" in data["data"].keys():
                imu_data_dict["timestamp_10Hz"] += [data["timestamp"]]
                imu_data_dict["magnetometer"] += [np.array(data["data"]["magnetometer"])]

    @staticmethod
    def parse_event_data(data: dict[str, Any], event_data_dict: dict[str, Any]):
        """
        This function could be implemented if in the future we want to extract data from the event data file.
        """
        pass

    def read_data(self):
        """
        This function reads the Tobii Pro Glasses 3 data file, which is a gzipped CSV file with weird JSON lines.
        It returns a dictionary with `timestamp`and `gaze_vector`.
        """

        gaze_data_file_path = self.data_folder_path + "gazedata.gz"
        imu_data_file_path = self.data_folder_path + "imudata.gz"
        # event_data_file_path = self.data_folder_path + "eventdata.gz"

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
            "timestamp_100Hz": [],  # in seconds
            "timestamp_10Hz": [],  # in seconds
            "accelerometer": [],  # meters per second squared
            "gyroscope": [],  # in degrees per second
            "magnetometer": [],  # in microtesla
        }
        self.parse_gz(imu_data_file_path, imu_data_dict, self.parse_imu_data)

        # # Read the event data file
        # event_data_dict = {
        #     "timestamp": [],  # in seconds
        # }
        # self.parse_gz(event_data_file_path, event_data_dict, self.parse_event_data)

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
            error_str = f"The file {self.data_folder_path}gazedata.gz is empty. There is no element in the field 'timestamp'. Please check the file."
            self.error_type(error_str)

        if (
            np.sum(
                np.logical_or(
                    np.sum(np.isnan(self.gaze_data_dict["gaze_vector"]), axis=1),
                    np.isnan(self.gaze_data_dict["pupil_diameter_left"]),
                    np.isnan(self.gaze_data_dict["pupil_diameter_right"]),
                )
            )
            > len(self.gaze_data_dict["timestamp"]) / 2
        ):
            self._validity_flag = False
            error_str = f"More than 50% of the data from file {self.data_folder_path}gazedata.gz is declared invalid or in a blink by the eye-tracker, skipping this file."
            self.error_type(error_str)
            return

        if np.any((time_vector[1:] - time_vector[:-1]) < 0):
            self._validity_flag = False
            error_str = f"The time vector in file {self.data_folder_path}gazedata.gz is not strictly increasing. Please check the file."
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
        self.time_vector = time_vector - initial_time

        # also transform imu timings
        self.imu_data_dict["timestamp_100Hz"] = np.array(self.imu_data_dict["timestamp_100Hz"]) - initial_time
        self.imu_data_dict["timestamp_10Hz"] = np.array(self.imu_data_dict["timestamp_10Hz"]) - initial_time

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
        # Gaze timestamps
        indices_to_keep_gaze = self.time_range.get_indices(self.time_vector)
        self.time_vector = self.time_vector[indices_to_keep_gaze]
        self.gaze_data_dict["timestamp"] = self.gaze_data_dict["timestamp"][indices_to_keep_gaze]
        self.gaze_data_dict["gaze_vector"] = self.gaze_data_dict["gaze_vector"][indices_to_keep_gaze]
        self.gaze_data_dict["pupil_diameter_right"] = self.gaze_data_dict["pupil_diameter_right"][indices_to_keep_gaze]
        self.gaze_data_dict["pupil_diameter_left"] = self.gaze_data_dict["pupil_diameter_left"][indices_to_keep_gaze]

        # Accelero/gyro timestamps
        indices_to_keep_imu = self.time_range.get_indices(self.imu_data_dict["timestamp_100Hz"])
        self.imu_data_dict["timestamp_100Hz"] = self.imu_data_dict["timestamp_100Hz"][indices_to_keep_imu]
        self.imu_data_dict["accelerometer"] = self.imu_data_dict["accelerometer"][indices_to_keep_imu]
        self.imu_data_dict["gyroscope"] = self.imu_data_dict["gyroscope"][indices_to_keep_imu]

        # Magneto timestamps
        indices_to_keep_magneto = self.time_range.get_indices(self.imu_data_dict["timestamp_10Hz"])
        self.imu_data_dict["timestamp_10Hz"] = self.imu_data_dict["timestamp_10Hz"][indices_to_keep_magneto]
        self.imu_data_dict["magnetometer"] = self.imu_data_dict["magnetometer"][indices_to_keep_magneto]

    def interpolate_to_eye_timestamps(
        self,
    ) -> dict[str, np.ndarray[float]]:
        """
        This function gets the imu data at the eye data time stamps by interpolating if necessary.

        Returns
        -------
        The modified version of the imu data aligned with the eye data timestamps (3, n_frames)
        """
        interpolated_accelerometer = interpolate_to_specified_timestamps(
            self.imu_data_dict["timestamp_100Hz"],
            self.time_vector,
            self.imu_data_dict["accelerometer"].T,
        )
        interpolated_gyroscope = interpolate_to_specified_timestamps(
            self.imu_data_dict["timestamp_100Hz"],
            self.time_vector,
            self.imu_data_dict["gyroscope"].T,
        )
        interpolated_magnetometer = interpolate_to_specified_timestamps(
            self.imu_data_dict["timestamp_10Hz"],
            self.time_vector,
            self.imu_data_dict["magnetometer"].T,
        )
        interpolated_imu = {
            "accelerometer": interpolated_accelerometer,
            "gyroscope": interpolated_gyroscope,
            "magnetometer": interpolated_magnetometer,
        }
        return interpolated_imu

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

    @destroy_on_fail
    def _set_head_angles(self):
        """
        Get the head orientation from the imu data. It is expressed as Euler angles in degrees and is measured by
        the glasses IMU containing a gyroscope, accelerometer, and magnetometer. Please note that this is an
        approximation of the head orientation through sensor fusion, and we do not offer any guarantee that the angles
        are as precise as the Tobii Pro Labs estimate.
        """

        # We interpolate to align the imu data with the eye orientation timestamps
        interpolated_imu = self.interpolate_to_eye_timestamps()

        # Starting from firmware version 1.29, the IMU data is rotated and aligned with the Head Unit
        # coordinate system, so that the data is expressed in the same coordinate system as the gaze data
        roll, pitch, yaw = angles_from_imu_fusion(
            time_vector=self.time_vector,
            acceleration=interpolated_imu["accelerometer"],
            gyroscope=interpolated_imu["gyroscope"],
            magnetometer=interpolated_imu["magnetometer"],
            roll_offset=0,
            pitch_offset=0,
        )
        head_angles = np.array([roll, np.zeros_like(pitch), np.zeros_like(roll)])

        self.head_angles = unwrap_rotation(head_angles)

    @destroy_on_fail
    def _set_data_invalidity(self):
        """
        Get a numpy array of bool indicating if the eye-tracker declared this data frame as invalid.
        """
        self.data_invalidity = np.logical_or(
            np.sum(np.isnan(self.gaze_data_dict["gaze_vector"]), axis=1),
            np.isnan(self.gaze_data_dict["pupil_diameter_left"]),
            np.isnan(self.gaze_data_dict["pupil_diameter_right"]),
        )
