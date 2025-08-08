import numpy as np


def unwrap_rotation(angles: np.ndarray) -> np.ndarray:
    """
    Unwrap rotation to avoid 360 degree jumps

    Parameters
    ----------
    angles: A numpy array of shape (3, n_frames) containing Euler angles expressed in degrees.
    """
    return np.unwrap(angles, period=360, axis=1)
