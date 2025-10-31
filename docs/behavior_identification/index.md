# Gaze behavior definition and identification

In the following subsections, you will find an explanation on how the different gaze behaviors are extracted using `EyeDentify3D`. 

![animation.gif](../figures/animation.fig)
Animation 1 - Animation obtained using `gaze_behavior_identifier.animate()`.

![all_gaze_behaviors.png](../figures/all_gaze_behaviors.png)
Figure 1 - Plot obtained using `gaze_behavior_identifier.plot(save_name)`.


```{tableofcontents}
```

For a quick overview, here is a list of the algorithm parameters that you can modify.

| Parameter name                    | Description                                                                                                           | Default value | Related behavior |
|-----------------------------------|-----------------------------------------------------------------------------------------------------------------------|---------------|------------------|
| `eye_openness_threshold`          | Threshold to determine if the eye is closed (0) or open (1).                                                          | 0.5           | Blink            |
| `velocity_factor`                 | The number of times the eye velocity norm should exceed the sliding median to detect a saccade.                       | 5.0           | Saccade          |
| `velocity_window_size`            | Duration (in s) of the sliding window to compute the median eye velocity norm.                                        | 0.52          | Saccade          |
| `min_acceleration_threshold`      | Minimum eye acceleration (in °/s²) to consider a saccade.                                                             | 4000.0        | Saccade          |
| `nb_acceleration_frames`          | Number of frames where the eye acceleration norm should exceed the threshold to consider a saccade.                   | 2             | Saccade          |
| `min_velocity_threshold`          | Minimum gaze velocity norm (in °/s) to consider a visual scanning.                                                    | 100.0         | Visual scanning  |
| `minimal_duration`                | Minimum duration (in s) where the gaze velocity must exceed the threshold to consider a visual scanning.              | 0.04          | Visual scanning  |
| `inter_saccade_minimal_duration`  | Minimum gap duration (in s) between already identified events to consider an inter-saccade sequence.                  | 0.04          | Inter-saccade    |
| `window_duration`                 | Duration (in s) of the windows into which the inter-saccade sequences should be divided.                              | 0.022         | Inter-saccade    |
| `window_overlap`                  | Duration (in s) of the overlap between the windows into which the inter-saccade sequences were divided.               | 0.006         | Inter-saccade    |
| `eta_p`                           | Maximal p-value to consider a sequence as incoherent.                                                                 | 0.001         | Inter-saccade    |
| `eta_d`                           | Dispersion threshold to consider a sequence as a fixation ($>\eta_D$) or smooth pursuit ($<\eta_D$).                  | 0.45          | Inter-saccade    |
| `eta_cd`                          | Direction consistency threshold to consider a sequence as fixation ($<\eta_{CD}$) or smooth pursuit ($>\eta_{CD}$).   | 0.5           | Inter-saccade    |
| `eta_pd`                          | Positional displacement threshold to consider a sequence as fixation ($<\eta_{PD}$) or smooth pursuit ($>\eta_{PD}$). | 0.2           | Inter-saccade    |
| `phi`                             | Maximum angular difference (in °) to consider two sequences to have a displacement in the same direction.             | 45            | Inter-saccade    |
| `main_movement_axis`              | Axis on which the gaze is expected to move the most (0: x-axis, 1: y-axis, 2: z-axis).                                | 0             | Inter-saccade    |
| `eta_max_fixation`                | Maximum spatial range (in °) to consider a sequence as fixation.                                                      | 1.9           | Fixation         |
| `fixation_minimal_duration`       | Minimum duration (in s) to consider a fixation.                                                                       | 0.04          | Fixation         |
| `eta_min_smooth_pursuit`          | Minimum spatial range (in °) to consider a sequence as smooth pursuit.                                                | 1.7           | Smooth pursuit   |
| `smooth_pursuit_minimal_duration` | Minimum duration (in s) to consider a smooth pursuit.                                                                 | 0.04          | Smooth pursuit   |