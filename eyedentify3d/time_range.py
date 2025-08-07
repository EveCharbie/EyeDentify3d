import numpy as np


class TimeRange:
    """A class to represent a time range for data processing in trials."""

    def __init__(self, min_time: float = 0, max_time: float = float("inf")) -> None:
        """
        Parameters
        ----------
        min_time: The time at which to start considering the data in the trial.
        max_time: The time at which to stop considering the data in the trial.
        """
        self.min_time = min_time
        self.max_time = max_time

    def get_indices(self, time_vector: np.ndarray):
        """
        Get the indices of the time vector that fall within the specified time range.

        Parameters
        ----------
        time_vector: A numpy array of time values.

        Returns
        -------
        A numpy array of indices where the time values are within the specified range.
        """
        return np.where(np.logical_and(time_vector >= self.min_time, time_vector <= self.max_time))[0]
