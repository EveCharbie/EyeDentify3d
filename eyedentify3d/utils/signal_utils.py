import numpy as np
from scipy import signal


def find_time_index(time_vector: np.ndarray, target_time: float, method: str) -> int:
    """
    Find the index corresponding to a target time within specified bounds.

    Parameters
    ----------
    time_vector: Array of time values
    target_time: Time to find index for
    method: Method to find index, either the first index to s ('first') or ('last')

    Returns
    -------
        idx: The index closest to target_time
    """
    # To remove NaNs in the time_vector
    valid_mask = ~np.isnan(time_vector)

    if method == "first":
        if np.all(time_vector[valid_mask] >= target_time):
            idx = 0
        else:
            idx = np.where(time_vector < target_time)[0][-1]
    elif method == "last":
        if np.all(time_vector[valid_mask] <= target_time):
            idx = len(time_vector) - 1
        else:
            idx = np.where(time_vector > target_time)[0][0]
    else:
        raise ValueError(f"The method should be either 'first' or 'last', got {method}.")
    return idx


def centered_finite_difference(time_vector: np.ndarray, data: np.ndarray) -> np.ndarray:
    """
    Compute the centered finite difference of the data with respect to the time vector.

    Parameters
    ----------
    time_vector: A numpy array of shape (n_frames,) containing the time vector.
    data: A numpy array of shape (n_components, n_frames) containing the data to differentiate.
    """
    velocity = np.zeros(data.shape)
    for i_component in range(data.shape[0]):
        velocity[i_component, 0] = (data[i_component, 1] - data[i_component, 0]) / (time_vector[1] - time_vector[0])
        velocity[i_component, -1] = (data[i_component, -1] - data[i_component, -2]) / (
            time_vector[-1] - time_vector[-2]
        )
        velocity[i_component, 1:-1] = (data[i_component, 2:] - data[i_component, :-2]) / (
            time_vector[2:] - time_vector[:-2]
        )
    return velocity


def filter_data(data: np.ndarray, cutoff_freq: float = 0.2, order: int = 8, padlen: int = 150) -> np.ndarray:
    """
    Apply a Butterworth filter to the data.

    Parameters
    ----------
    data: A numpy array of shape (n_components, n_frames) containing the data to filter.
    cutoff_freq: The cutoff frequency for the filter.
    order: The order of the Butterworth filter.
    padlen: The number of elements by which to extend the data at both ends of axis before applying the filter.
    """
    b, a = signal.butter(order, cutoff_freq)
    filtered_data = np.zeros_like(data)
    for i_component in range(data.shape[0]):
        filtered_data[i_component, :] = signal.filtfilt(b, a, data[i_component, :], padlen=padlen)
    return filtered_data


def interpolate_to_specified_timestamps(
    current_time_vector: np.ndarray[float],
    desired_time_vector: np.ndarray[float],
    angles_to_interpolate: np.ndarray[float]
) -> np.ndarray[float]:
    """
    This function gets the angles_to_interpolate acquired at the current_time_vector instants and interpolates it to
    the desired_time_vector instants.

    Parameters
    ----------
    current_time_vector: The time vector of the data to interpolate (n_frames_current,)
    desired_time_vector: The desired time vector to interpolate the data to (n_frames_desired,)
    angles_to_interpolate: The data to interpolate (3, n_frames_current)
    Returns
    -------
    The modified numpy array of head angles aligned with the eye data timestamps (3, n_frames)
    """
    # Check shapes
    if len(angles_to_interpolate.shape) != 2 or angles_to_interpolate.shape[0] != 3:
        raise NotImplementedError("This function was designed for head angles of shape (3, n_frames). ")

    # Check if there is duplicated frames in the imu data
    frame_diffs = np.linalg.norm(angles_to_interpolate[:, 1:] - angles_to_interpolate[:, :-1], axis=0)
    if not np.all(frame_diffs > 1e-10):
        raise RuntimeError(
            "There were repeated frames in the imu data, which never happened with this eye-tracker. Please notify the developer."
        )

    # Interpolate the head angles to the eye timestamps
    interpolated_data = np.zeros((3, desired_time_vector.shape[0]))
    for i_time, time in enumerate(desired_time_vector):
        if time < current_time_vector[0] or time > current_time_vector[-1]:
            interpolated_data[:, i_time] = np.nan
        else:
            if time in current_time_vector:
                idx = np.where(current_time_vector == time)[0][0]
                interpolated_data[:, i_time] = angles_to_interpolate[:, idx]
            else:
                idx_before = np.where(current_time_vector < time)[0][-1]
                idx_after = np.where(current_time_vector > time)[0][0]
                t_before = current_time_vector[idx_before]
                t_after = current_time_vector[idx_after]
                angles_before = angles_to_interpolate[:, idx_before]
                angles_after = angles_to_interpolate[:, idx_after]
                interpolated_data[:, i_time] = angles_before + (time - t_before) * (
                    (angles_after - angles_before) / (t_after - t_before)
                )
    return interpolated_data