import numpy as np


def split_sequences(indices: np.ndarray) -> list[np.ndarray]:
    """
    Split an array of indices into an array of sequences of consecutive indices.
    :param indices:
    :return:
    """
    sequence = np.array_split(
        np.array(indices),
        np.flatnonzero(np.diff(np.array(indices)) > 1) + 1,
    )
    return sequence
