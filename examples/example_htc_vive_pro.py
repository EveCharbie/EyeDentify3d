"""
In this example, we will load the data from the HTC Vive Pro and extract gaze behavior from the data.
"""

import os

from eyedentify3d import HtcViveProData, TimeRange, ErrorType


def main():

    # Cut the data after the end of the trial (black screen)
    black_screen_time = 7.180  # seconds
    time_range = TimeRange(min_time=0, max_time=black_screen_time)

    # Load the data from the HTC Vive Pro
    data_file_path = "data/HTC_Vive_Pro/TESTNA01_2D_Fist3.csv"
    data = HtcViveProData(data_file_path, error_type=ErrorType.FILE, time_range=time_range)

    # For this example, we will remove all files generated, but in a real case, they should be kept
    if os.path.exists("bad_data_files.txt"):
        os.remove("bad_data_files.txt")


if __name__ == "__main__":
    main()
