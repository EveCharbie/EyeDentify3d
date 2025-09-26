# Eye-tracker data extraction examples

# HTC Vive Pro
```python3
from eyedentify3d import HtcViveProData, TimeRange, ErrorType

# Set the time range that you want to analyze in the trial
time_range = TimeRange(min_time, max_time)

# Load the data from the HTC Vive Pro
data_file_path = "data.csv"
data_object = HtcViveProData(data_file_path, error_type=ErrorType.PRINT, time_range=time_range)
```

## Pupil Invisible
```python3
from eyedentify3d import HtcViveProData, TimeRange, ErrorType

# Set the time range that you want to analyze in the trial
time_range = TimeRange(min_time, max_time)

# Load the data from the Pupil Invisible
data_folder_path = "data"
data_object = PupilInvisibleData(data_folder_path, error_type=ErrorType.PRINT, time_range=time_range)
```

