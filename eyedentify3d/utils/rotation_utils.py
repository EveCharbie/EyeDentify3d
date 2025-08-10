import numpy as np


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
    return np.unwrap(angles, period=360, axis=1)


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
