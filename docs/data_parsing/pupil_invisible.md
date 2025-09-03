# Pupil Invisible eye-tracker

The data from the [Pupil Invisible](https://pupil-labs.com/products/invisible) eye-tracker is stored in multiple files.
From them, we are interested in the following:
- `gaze.csv` containing the eye orientation
- `imu.csv` containing the head motion data
- `blinks.csv` containing the blinks events

## Gaze data
The `gaze.csv` file contains the following columns:
- `timestamp [ns]`: The time vector in nanoseconds
- `worn`: Whether the glasses were worn (1) or not (0). Please note that this eye-tracker does not provide any indication on the confidence of the gaze orientation estimates, so we consider that as long as the eye-tracker is "worn" the eye-tracker data is valid.
- # TODO: VERIFY THIS WITH ANIMATION !!!!!
- `azimuth [deg]`: The eyes' horizontal angle in degrees. The zero defines looking straight ahead, positive values define looking to the right, and negative values define looking to the left.
- `elevation [deg]`: The eyes' vertical angle in degrees. The zero defines looking straight ahead, positive values define looking up, and negative values define looking down.

## IMU data
The `imu.csv` file contains the following columns:
- `timestamp [ns]`: The time vector in nanoseconds
- `gyro x [deg/s]`: The head angular velocity along the extension/flexion axis (pitch) in degrees per second measured by the gyroscope.
- `gyro y [deg/s]`: The head angular velocity along the left/right rotation axis (yaw) in degrees per second measured by the gyroscope.
- `gyro z [deg/s]`: The head angular velocity along the lateral extension/flexion axis (roll) in degrees per second measured by the gyroscope.
- `acceleration x [g]`: The head linear acceleration along the transversal axis (right/left) in g measured by the accelerometer.
- `acceleration y [g]`: The head linear acceleration along the sagittal axis (back/front) in g measured by the accelerometer.
- `acceleration z [g]`: The head linear acceleration along the vertical axis (down/up) in g measured by the accelerometer.
- `roll [deg]`: The lateral extension/flexion (roll) angle in degrees. Positive values define right lateral flexion (right ear getting closer to the right shoulder) and negative values define left lateral flexion (left ear getting closer to the left shoulder).
- `pitch [deg]`: The head extension/flexion (pitch) angle in degrees. Positive values define neck flexion (looking down) and negative values define neck extension (looking up).
- `yaw [deg]`: The head left/right rotation (yaw) angle in degrees. Positive values define rightward rotations (looking right) and negative values define leftward rotations (looking left).
Please note that if no tags are used during the recording the eye-tracker only provide two of the three Euler angles representing the head rotation (roll, pitch). Two reasons motivate this precaution. As there is no magnetometer in the eye-tracker, i) the zero from the yaw angle cannot be defined, and ii) the yaw angle is prone to exponential drift.
Therefore, we recommend using [tags](https://docs.pupil-labs.com/core/software/pupil-capture/#preparing-your-environment) in your experimental setup if possible.
Otherwise, the yaw angle will be approximated by `EyeDentify3D`, since the head orientation is only used to compare timestamps that are close in time, so the drifting effect should stay in a reasonable range.

## Blink data
The `blinks.csv` file contains the following columns:
- `start timestamp [ns]`: The blink start timestamp in nanoseconds.
- `end timestamp [ns]`: The blink end timestamp in nanoseconds.

## How to build a data object
```python3
data_object = PupilInvisibleData(data_folder_path, error_type, time_range)
```

## References
- [Pupil Capture User Manual](https://docs.pupil-labs.com/core/software/pupil-capture/#pupil-capture)
- [IMU coordinate system](https://docs.pupil-labs.com/invisible/assets/pi-imu-diagram.DoPp4CcW.jpg)
- [Gaze coordinate system](https://framerusercontent.com/images/OXOwlMKDg5fYJd2Vv5kQGvBXJw.jpg)