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
        if min_time > max_time:
            raise ValueError("The min_time must be less than or equal to the max_time.")

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
        if np.any(time_vector[-1:] - time_vector[:-1] < 0):
            raise ValueError("The time vector must be strictly increasing.")

        if np.all(time_vector < self.min_time) or np.all(time_vector > self.max_time):
            # If all values are outside the range, return an empty array
            return np.array([], dtype=int)

        # This approach is less clean but is robust no NaNs in the time_vector
        beginning_idx = np.where(time_vector >= self.min_time)[0]
        end_idx = np.where(time_vector > self.max_time)[0]

        if len(beginning_idx) == 0:
            beginning_idx = 0
        else:
            beginning_idx = beginning_idx[0]

        if len(end_idx) == 0:
            end_idx = len(time_vector)
        else:
            end_idx = end_idx[0]

        if beginning_idx >= end_idx:
            return np.array([], dtype=int)
        else:
            return np.arange(beginning_idx, end_idx)
