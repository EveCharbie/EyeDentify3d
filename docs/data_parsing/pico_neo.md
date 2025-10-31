# Pico Neo Pro Eye (Tobii) eye-tracker

The data from the [Pico Neo Pro Eye](https://www.picoxr.com/fr/products/neo3-pro-eye) (ByteDance, China) 
eye-tracker is stored in a `.csv` file containing the following columns:
- `Timeline`: The time vector in seconds.
- `Left Eye Pose Status` and `Right Eye Pose Status`: As estimate of the validity of the eye-tracking data. The data can either be valid (`52`) or not (`0`). Please note that this eye-tracker does not provide any indication on the confidence of the gaze orientation estimates, so we consider that as long as the value is `52`, the eye-tracker data is valid.
- `Left Eye Gaze Openness` and `Right Eye Gaze Openness`: The eyes' openness. Please note that for this eye-tracker, the value is only closed (`0`) or opened (`1`).
- `Combine Eye Gaze Vector. x`, `Combine Eye Gaze Vector. y`, and `Combine Eye Gaze Vector. z`: The eyes' orientation expressed as a unit vector. 
- `Head Rotation. x`, `Head Rotation. y`, and `Head Rotation. z`: The head's orientation expressed as Euler angles in degrees.

## How to build a data object
```python3
data_object = PicoNeoData(data_file_path, error_type, time_range)
```

## References
- [User Guide](https://www.picoxr.com/cn/neo3/pdf/PicoNeo3UserGuide.pdf)
- [Developer Guide](https://developer.picoxr.com/reference/unreal/client-api/PXR_HandTracking/#GetPerEyePose)