import numpy as np
import biorbd
from scipy import signal


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
