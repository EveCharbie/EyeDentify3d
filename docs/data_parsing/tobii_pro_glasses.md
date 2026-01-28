# Tobii Pro Glasses 3

The data from the [Tobii Pro Glasses 3](https://www.tobii.com/products/eye-trackers/wearables/tobii-pro-glasses-3) (Tobii Technology, Sweden) eye-tracker is stored in multiple files.
From them, we are interested in the following:
- `gazedata.gz` containing the eye orientation.
- `imudata.gz` containing the head motion data.

## Gaze data
The `gazedata.gz` file contains the following fields:
- `timestamp`: The time vector in seconds.
- `data`:
  - `gaze3d`: The gaze orientation represented as a unit vector in 3D space.
  - `eyeleft`/`eyeright`:
    - `pupildiameter`: The eyes' pupil diameter in millimeters measured by the eye-tracker. Please note that this value is only used to identify invalid timestamps, where the pupil diameter is NaN.


## IMU data
The `imudata.gz` file contains the following fields:
- `timestamp`: The time vector in seconds.
- `data`:
  - `accelerometer`: The head linear acceleration along the x, y, and z axes in meters per seconds squared measured by the accelerometer.
  - `gyroscope`: The head angular velocity along the x, y, and z axes in degrees per second measured by the gyroscope.
  - `magnetometer`: The magnetic field along the x, y, and z axes in microteslas measured by the magnetometer.

Note: This implementation is for Tobii Pro Glasses with firmware version above 1.29.

## How to build a data object
```python3
data_object = TobiiProGlassesData(data_file_path, error_type, time_range)
```

## References
- [User Manual](https://go.tobii.com/tobii-pro-glasses-3-user-manual)
- [IMU coordinate system](https://www.yixinkeyan.com/uploadfile/202303/627a1fdb755d6a0.pdf) p.40
- [Gaze coordinate system](https://www.yixinkeyan.com/uploadfile/202303/627a1fdb755d6a0.pdf) p.40