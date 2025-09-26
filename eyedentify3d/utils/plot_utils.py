import matplotlib.colors as mcolors
import numpy as np


def format_color_to_rgb(color_name: str) -> np.ndarray[int]:
    """
    Converts a CSS4 or a TABLEAU_COLORS color name into an RGB color in range [0, 1] format.

    Parameters
    ----------
    color_name : The name of the color (e.g., "red", "blue", "green")
    """
    if color_name in mcolors.TABLEAU_COLORS:
        hex_color = mcolors.TABLEAU_COLORS[color_name]
    elif color_name in mcolors.CSS4_COLORS:
        hex_color = mcolors.CSS4_COLORS[color_name]
    else:
        raise NotImplementedError(
            "Your color name does not belong to matplotlib.colors.CSS4_COLORS or matplotlib.colors.TABLEAU_COLORS."
        )

    rgb_normalized = mcolors.to_rgb(hex_color)
    return np.array(rgb_normalized)[np.newaxis, :]
