import numpy as np

from .signal_utils import get_largest_non_nan_sequence


def rot_x_matrix(angle):
    """
    Rotation matrix around the x-axis
    """
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )


def rot_y_matrix(angle):
    """
    Rotation matrix around the y-axis
    """
    return np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )


def rot_z_matrix(angle):
    """
    Rotation matrix around the z-axis
    """
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )


def unwrap_rotation(angles: np.ndarray) -> np.ndarray:
    """
    Unwrap rotation to avoid 360 degree jumps

    Parameters
    ----------
    angles: A numpy array of shape (3, n_frames) containing Euler angles expressed in degrees.
    """
    unwrapped_angles = np.zeros_like(angles)
    unwrapped_angles[:, :] = np.nan
    start_idx, end_idx = get_largest_non_nan_sequence(angles)
    unwrapped_angles[:, start_idx:end_idx] = np.unwrap(angles[:, start_idx:end_idx], period=360, axis=1)
    return unwrapped_angles


def rotation_matrix_from_euler_angles(angle_sequence: str, angles: np.ndarray):
    if len(angles.shape) > 1:
        raise ValueError(f"The angles should be of shape (nb_angles, ). You have {angles.shape}")
    if len(angle_sequence) != angles.shape[0]:
        raise ValueError(
            f"The number of angles and the length of the angle_sequence must match. You have {angles.shape} and {angle_sequence}"
        )

    matrix = {
        "x": rot_x_matrix,
        "y": rot_y_matrix,
        "z": rot_z_matrix,
    }

    rotation_matrix = np.identity(3)
    for angle, axis in zip(angles, angle_sequence):
        rotation_matrix = rotation_matrix @ matrix[axis](angle)
    return rotation_matrix


def get_gaze_direction(head_angles: np.ndarray, eye_direction: np.ndarray):
    """
    Get the gaze direction. It is a unit vector expressed in the global reference frame representing the combined
    rotations of the head and eyes.

    Parameters
    ----------
    head_angles: A numpy array of shape (3, n_frames) containing the Euler angles in degrees of the head orientation expressed in
        the global reference frame.
    eye_direction: A numpy array of shape (3, n_frames) containing a unit vector of the eye direction expressed in the
        head reference frame.
    """
    # Convert head angles from degrees to radians for the rotation matrix
    head_angles_rad = head_angles * np.pi / 180

    gaze_direction = np.zeros(eye_direction.shape)
    for i_frame in range(head_angles_rad.shape[1]):
        # Convert Euler angles into a rotation matrix
        rotation_matrix = rotation_matrix_from_euler_angles("xyz", head_angles_rad[:, i_frame])
        # Rotate the eye direction vector using the head rotation matrix
        gaze_direction[:, i_frame] = rotation_matrix @ eye_direction[:, i_frame]

        # Ensure it is a unit vector
        gaze_direction_norm = np.linalg.norm(gaze_direction[:, i_frame])
        if gaze_direction_norm > 1.2 or gaze_direction_norm < 0.8:
            raise RuntimeError(
                "The gaze direction should be a unit vector. This should not happen, please contact the developer."
            )
        gaze_direction[:, i_frame] /= gaze_direction_norm

    return gaze_direction


def get_angle_between_vectors(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Get the angle between two vectors in degrees.

    Parameters
    ----------
    vector1: A numpy array of shape (3, ) representing the first vector.
    vector2: A numpy array of shape (3, ) representing the second vector.

    Returns
    -------
    The angle between the two vectors in radians.
    """
    if vector1.shape != (3,) or vector2.shape != (3,):
        raise ValueError("Both vectors must be of shape (3,).")

    if np.all(vector1 == vector2):
        # Set here because it creates problem later
        angle = 0
    else:
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        if norm1 == 0 or norm2 == 0:
            raise RuntimeError(
                "The gaze vectors should be unitary. This should not happen, please contact the developer."
            )

        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

        if cos_angle > 1 or cos_angle < -1:
            raise RuntimeError("This should not happen, please contact the developer.")

        angle = np.arccos(cos_angle)

    return angle * 180 / np.pi  # Convert to degrees


def compute_angular_velocity(time_vector: np.ndarray, direction_vector: np.ndarray) -> np.ndarray:
    """
    Computes the angular velocity in deg/s as the angle difference between two frames divided by
    the time difference between them. It is computed like a centered finite difference, meaning that the frame i+1
    and i-1 are used to set the value for the frame i.

    Parameters
    ----------
    time_vector: The time vector of the data acquisition, shape (n_frames,).
    direction_vector: A numpy array of shape (3, n_frames) containing the direction vector for which to compute the angular velocity.

    Returns
    -------
    A numpy array of shape (n_frames,) containing the angular velocity in deg/s.
    """
    if direction_vector.shape[0] != 3:
        raise ValueError("The direction vector should be a 3D vector.")

    nb_frames = time_vector.shape[0]
    if nb_frames < 3:
        raise ValueError("The time vector should have at least 3 frames to compute angular velocity.")

    if direction_vector.shape[1] != nb_frames:
        raise ValueError("The time vector should have the same number of frames as the direction vector.")

    angular_velocity = np.zeros((nb_frames,))
    for i_frame in range(1, nb_frames - 1):  # Skipping the first and last frames
        vector_before = direction_vector[:, i_frame - 1]
        vector_after = direction_vector[:, i_frame + 1]
        angle = get_angle_between_vectors(vector_before, vector_after)
        angular_velocity[i_frame] = angle / (time_vector[i_frame + 1] - time_vector[i_frame - 1])

    # Deal with the first and last frames separately
    first_angle = get_angle_between_vectors(direction_vector[:, 0], direction_vector[:, 1])
    angular_velocity[0] = first_angle / (time_vector[1] - time_vector[0])
    last_angle = get_angle_between_vectors(direction_vector[:, -2], direction_vector[:, -1])
    angular_velocity[-1] = last_angle / (time_vector[-1] - time_vector[-2])

    return angular_velocity


def angles_from_imu_fusion(
    time_vector: np.ndarray[float],
    acceleration: np.ndarray[float],
    gyroscope: np.ndarray[float],
    magnetometer: np.ndarray[float] = None,
    roll_offset: float = 0,
    pitch_offset: float = 0,
) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Computes the Euler angles from the accelerometer and gyroscope data using the Madgwick filter algorithm.
    Code adapted from https://github.com/pupil-labs/pupil/blob/7a3cd9e8d2e54ac123ab0ed292d741732db899a2/pupil_src/shared_modules/imu_timeline.py
    Magnetometer information was added to increase accuracy, when it is available.
    Note: The initial head orientation is assumed to be pitch=0, roll=0, yaw=0.

    Parameters
    ----------
    time_vector: A numpy array of shape (n_frames,) containing the time vector of the data acquisition.
    acceleration: A numpy array of shape (3, n_frames) containing the acceleration data in G.
    gyroscope: A numpy array of shape (3, n_frames) containing the gyroscope data in deg/s.
    magnetometer: A numpy array of shape (3, n_frames) containing the magnetometer data in microtesla.
    roll_offset: An offset representing the angle between head and IMU in degrees (to be added to the roll angle).
    pitch_offset: An offset representing the angle between head and IMU in degrees (to be added to the pitch angle).
    """
    # Get the largest sequence where the data is not NaN
    acceleration_start_idx, acceleration_end_idx = get_largest_non_nan_sequence(acceleration)

    # Parameters
    nb_frames = time_vector.shape[0]
    gyroscope_error = 50 * np.pi / 180  # Default in Pupil Invisible code
    beta_acceleration = (
        np.sqrt(3.0 / 4.0) * gyroscope_error
    )  # compute beta (see README in original GitHub page: https://github.com/micropython-IMU/micropython-fusion)
    beta_magnetometer = 0.1 * beta_acceleration  # less trust in magnetometer (but ratio is arbitrary)

    # Initialize orientation
    quaternion = np.zeros((4, nb_frames)) * np.nan
    quaternion[:, 0] = [1.0, 0.0, 0.0, 0.0]
    pitch = np.zeros((nb_frames,)) * np.nan
    roll = np.zeros((nb_frames,)) * np.nan
    yaw = np.zeros((nb_frames,)) * np.nan
    pitch[0] = 0.0
    roll[0] = 0.0
    yaw[0] = 0.0

    for i_frame in range(nb_frames - 1):

        if i_frame < acceleration_start_idx or i_frame >= acceleration_end_idx - 1:
            # Index outside valid range, set angles to NaN
            pitch[i_frame + 1] = np.nan
            roll[i_frame + 1] = np.nan
            yaw[i_frame + 1] = np.nan
            quaternion[:, i_frame + 1] = [1.0, 0.0, 0.0, 0.0]  # Reset quaternion

        else:

            dt = time_vector[i_frame + 1] - time_vector[i_frame]
            ax, ay, az = acceleration[:, i_frame]  # Units G (but later normalised)
            gx, gy, gz = np.radians(gyroscope[:, i_frame])  # Units deg/s
            q1, q2, q3, q4 = quaternion[:, i_frame]  # short name local variable for readability

            # Auxiliary variables to avoid repeated arithmetic
            _2q1 = 2 * q1
            _2q2 = 2 * q2
            _2q3 = 2 * q3
            _2q4 = 2 * q4
            _4q1 = 4 * q1
            _4q2 = 4 * q2
            _4q3 = 4 * q3
            _8q2 = 8 * q2
            _8q3 = 8 * q3
            q1q1 = q1 * q1
            q2q2 = q2 * q2
            q3q3 = q3 * q3
            q4q4 = q4 * q4

            # Normalise accelerometer measurement
            norm_a = np.sqrt(ax * ax + ay * ay + az * az)
            if norm_a == 0:
                # This is highly suspicious of a NaN somewhere
                raise RuntimeError("This should not happen, please contact the developer.")

            # --- Compute corrective step for magnetometer if available --- #
            s_mag_1, s_mag_2, s_mag_3, s_mag_4 = 0.0, 0.0, 0.0, 0.0
            if magnetometer is not None and np.sum(np.isnan(magnetometer[:, i_frame])) == 0:
                mx, my, mz = magnetometer[:, i_frame]  # Units microtesla

                # Normalize magnetometer
                norm_m = 1 / np.sqrt(mx * mx + my * my + mz * mz)
                mx *= norm_m
                my *= norm_m
                mz *= norm_m

                # Reference direction of Earth's magnetic field (in Earth frame)
                # Rotate magnetometer reading to Earth frame and extract horizontal component
                hx = mx * (q1q1 + q2q2 - q3q3 - q4q4) + my * 2 * (q2 * q3 - q1 * q4) + mz * 2 * (q2 * q4 + q1 * q3)
                hy = mx * 2 * (q2 * q3 + q1 * q4) + my * (q1q1 - q2q2 + q3q3 - q4q4) + mz * 2 * (q3 * q4 - q1 * q2)
                hz = mx * 2 * (q2 * q4 - q1 * q3) + my * 2 * (q3 * q4 + q1 * q2) + mz * (q1q1 - q2q2 - q3q3 + q4q4)

                # Compute reference field (horizontal component + vertical)
                bx = np.sqrt(hx * hx + hy * hy)  # Horizontal magnitude
                bz = hz  # Vertical component

                # Magnetometer objective function gradient
                _2bx, _2bz = 2 * bx, 2 * bz
                _4bx, _4bz = 4 * bx, 4 * bz

                # Expected magnetometer reading given current quaternion
                # m_expected = R(q)^T * [bx, 0, bz]^T
                mx_exp = bx * (q1q1 + q2q2 - q3q3 - q4q4) + bz * 2 * (q2 * q4 - q1 * q3)
                my_exp = bx * 2 * (q2 * q3 - q1 * q4) + bz * 2 * (q1 * q2 + q3 * q4)
                mz_exp = bx * 2 * (q1 * q3 + q2 * q4) + bz * (q1q1 - q2q2 - q3q3 + q4q4)

                # Error
                ex_m = mx_exp - mx
                ey_m = my_exp - my
                ez_m = mz_exp - mz

                # Jacobian^T * error (gradient of objective function)
                s_mag_1 = (-_2bz * q3) * ex_m + (-_2bx * q4 + _2bz * q2) * ey_m + (_2bx * q3) * ez_m
                s_mag_2 = (_2bz * q4) * ex_m + (_2bx * q3 + _2bz * q1) * ey_m + (_2bx * q4 - _2bz * 2 * q2) * ez_m
                s_mag_3 = (
                    (-_4bx * q3 - _2bz * q1) * ex_m
                    + (_2bx * q2 + _2bz * q4) * ey_m
                    + (_2bx * q1 - _2bz * 2 * q3) * ez_m
                )
                s_mag_4 = (-_4bx * q4 + _2bz * q2) * ex_m + (-_2bx * q1 + _2bz * q3) * ey_m + (_2bx * q2) * ez_m

                # Normalize magnetometer gradient
                norm_s_mag = 1 / np.sqrt(s_mag_1**2 + s_mag_2**2 + s_mag_3**2 + s_mag_4**2)
                s_mag_1 *= norm_s_mag
                s_mag_2 *= norm_s_mag
                s_mag_3 *= norm_s_mag
                s_mag_4 *= norm_s_mag

            # --- Compute corrective step for accelerometer if available --- #
            norm_a = 1 / norm_a  # use reciprocal for division
            ax *= norm_a
            ay *= norm_a
            az *= norm_a

            # Gradient decent algorithm corrective step for accelerometer
            s1 = _4q1 * q3q3 + _2q3 * ax + _4q1 * q2q2 - _2q2 * ay
            s2 = _4q2 * q4q4 - _2q4 * ax + 4 * q1q1 * q2 - _2q1 * ay - _4q2 + _8q2 * q2q2 + _8q2 * q3q3 + _4q2 * az
            s3 = 4 * q1q1 * q3 + _2q1 * ax + _4q3 * q4q4 - _2q4 * ay - _4q3 + _8q3 * q2q2 + _8q3 * q3q3 + _4q3 * az
            s4 = 4 * q2q2 * q4 - _2q2 * ax + 4 * q3q3 * q4 - _2q3 * ay

            norm_s = 1 / np.sqrt(s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4)  # normalise step magnitude
            s1 *= norm_s
            s2 *= norm_s
            s3 *= norm_s
            s4 *= norm_s

            # Compute rate of change of quaternion
            q_dot1 = 0.5 * (-q2 * gx - q3 * gy - q4 * gz)
            q_dot2 = 0.5 * (q1 * gx + q3 * gz - q4 * gy)
            q_dot3 = 0.5 * (q1 * gy - q2 * gz + q4 * gx)
            q_dot4 = 0.5 * (q1 * gz + q2 * gy - q3 * gx)

            # Apply accelerometer correction (always)
            q_dot1 -= beta_acceleration * s1
            q_dot2 -= beta_acceleration * s2
            q_dot3 -= beta_acceleration * s3
            q_dot4 -= beta_acceleration * s4

            # Apply magnetometer correction (when available)
            q_dot1 -= beta_magnetometer * s_mag_1
            q_dot2 -= beta_magnetometer * s_mag_2
            q_dot3 -= beta_magnetometer * s_mag_3
            q_dot4 -= beta_magnetometer * s_mag_4

            # Integrate to yield quaternion
            q1 += q_dot1 * dt
            q2 += q_dot2 * dt
            q3 += q_dot3 * dt
            q4 += q_dot4 * dt
            norm_q = 1 / np.sqrt(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4)  # normalise quaternion
            this_quaternion = q1 * norm_q, q2 * norm_q, q3 * norm_q, q4 * norm_q
            quaternion[:, i_frame + 1] = this_quaternion

            # These are modified to account for Invisible IMU coordinate system and positioning of
            # the IMU within the invisible headset
            this_roll = (
                np.degrees(
                    -np.arcsin(
                        2.0 * (this_quaternion[1] * this_quaternion[3] - this_quaternion[0] * this_quaternion[2])
                    )
                )
                + roll_offset
            )
            # bring to range [-180, 180]
            roll[i_frame + 1] = ((this_roll + 180) % 360) - 180

            this_pitch = (
                np.degrees(
                    np.arctan2(
                        2.0 * (this_quaternion[0] * this_quaternion[1] + this_quaternion[2] * this_quaternion[3]),
                        this_quaternion[0] * this_quaternion[0]
                        - this_quaternion[1] * this_quaternion[1]
                        - this_quaternion[2] * this_quaternion[2]
                        + this_quaternion[3] * this_quaternion[3],
                    )
                )
                + pitch_offset
            )
            # bring to range [-180, 180]
            pitch[i_frame + 1] = ((this_pitch + 180) % 360) - 180

            this_yaw = np.degrees(
                np.arctan2(
                    2.0 * (this_quaternion[1] * this_quaternion[2] + this_quaternion[0] * this_quaternion[3]),
                    this_quaternion[0] * this_quaternion[0]
                    + this_quaternion[1] * this_quaternion[1]
                    - this_quaternion[2] * this_quaternion[2]
                    - this_quaternion[3] * this_quaternion[3],
                )
            )
            yaw[i_frame + 1] = ((this_yaw + 180) % 360) - 180

    return pitch, roll, yaw
