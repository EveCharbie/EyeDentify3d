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

    return gaze_direction


def get_angle_between_vectors(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Get the angle between two vectors in radians.

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
            raise RuntimeError("The vectors are too far apart to compute a valid angle.")

        angle = np.arccos(cos_angle)

    return angle
