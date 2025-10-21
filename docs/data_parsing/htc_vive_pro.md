# HTC Vive Pro Eye (Tobii) eye-tracker

The data from the [HTC Vive Pro Eye](https://www.vive.com/fr/product/vive-pro2/overview/) (HTC Coprporation, Taiwan) 
eye-tracker is stored in a `.csv` file containing the following columns:
- `time(100ns)`: The time vector in 100 nanoseconds.
- `eye_valid_L` and `eye_valid_R`: As estimate of the validity of the eye-tracking data. The data can either be valid (`31`) or not (`0`). Please note that this eye-tracker does not provide any indication on the confidence of the gaze orientation estimates, so we consider that as long as the value is `31`, the eye-tracker data is valid.
- `openness_L` and `openness_R`: The eyes' openness ranging between closed (`0`) and opened (`1`).
- `gaze_direct_L.x`, `gaze_direct_L.y`, and `gaze_direct_L.z`: The eyes' orientation expressed as a unit vector. 
- `helmet_rot_x`, `helmet_rot_y`, and `helmet_rot_z`: The head's orientation expressed as Euler angles in degrees.

## How to build a data object
```python3
data_object = HtcViveProData(data_file_path, error_type, time_range)
```

## References
- [User Guide](https://developer.vive.com/documents/719/VIVE_Pro_HMD_User_Guide.pdf)
- [Head and Eye coordinate system](https://www.researchgate.net/figure/Left-Coordinate-system-of-HTC-Vive-Pro-Eye-and-right-a-diagram-showing-gaze-origin_fig4_373699457)