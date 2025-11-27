# Tobii Pro Glasses 3

The data from the [Tobii Pro Glasses 3](https://www.tobii.com/products/eye-trackers/wearables/tobii-pro-glasses-3) (Tobii Technology, Sweden).
# eye-tracker is stored in a `.csv` file containing the following columns:
# - `time(100ns)`: The time vector in 100 nanoseconds.
# - `eye_valid_L` and `eye_valid_R`: As estimate of the validity of the eye-tracking data. The data can either be valid (`31`) or not (`0`). Please note that this eye-tracker does not provide any indication on the confidence of the gaze orientation estimates, so we consider that as long as the value is `31`, the eye-tracker data is valid.
# - `openness_L` and `openness_R`: The eyes' openness ranging between closed (`0`) and opened (`1`).
# - `gaze_direct_L.x`, `gaze_direct_L.y`, and `gaze_direct_L.z`: The eyes' orientation expressed as a unit vector. 
# - `helmet_rot_x`, `helmet_rot_y`, and `helmet_rot_z`: The head's orientation expressed as Euler angles in degrees.

## How to build a data object
```python3
data_object = TobiiProGlassesData(data_file_path, error_type, time_range)
```

## References
- [User Manual](https://go.tobii.com/tobii-pro-glasses-3-user-manual)
- [IMU coordinate system](https://www.yixinkeyan.com/uploadfile/202303/627a1fdb755d6a0.pdf) p.40
- [Gaze coordinate system](https://www.yixinkeyan.com/uploadfile/202303/627a1fdb755d6a0.pdf) p.40