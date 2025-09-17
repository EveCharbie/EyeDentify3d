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
