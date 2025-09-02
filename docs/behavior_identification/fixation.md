# Fixation detection

```{note}
Fixation definition: We define a fixation as a sequence aiming to stabilize the gaze endpoint in the world reference frame.
```

How to detect fixations: 
```python3 
gaze_behavior_identifier.detect_fixation_and_smooth_pursuit_sequences(
        inter_saccade_minimal_duration,
        fixation_minimal_duration,
        smooth_pursuit_minimal_duration,
        window_duration,
        window_overlap,
        eta_p,
        eta_d,
        eta_cd,
        eta_pd,
        eta_max_fixation,
        eta_min_smooth_pursuit,
        phi,
    )
```

Implementation:
Fixations are identified if the following {cite}`Larsson:2015` criteria are met.
    1) `Dispersion` > $\eta_d$ and `Directional consistency` < $\eta_{CD} and `Positional displacement` < $\eta_{PD} and `Spatial range` < $\eta_{maxFix}
    2) `Positional displacement` > $\eta_{PD} and `Spatial range` < $\eta_{minSP}
    3) `Positional displacement` < $\eta_{PD} and `Spatial range` < $\eta_{minFix}

Parameters:
- `window_duration (float)`: The length of the windows. The default is `0.022` s.
- `window_overlap (float)`: The duration by which the windows must overlap at their extremities. The default is `0.006` s.
- `minimal_duration (float)`: The minimal duration for which the gaze behavior must be consistent or inconsistent to consider the inter-saccadic interval. The default is `0.04` s.
- `min_velocity_threshold (float)`: The minimal gaze velocity to consider a visual scanning. The default is `100`°/s. # TODO: CITE
- `eta_p (float)`: The maximal p-value to consider a sequence as incoherent. The default is `0.001`.
- `eta_d (float)`: The minimal dispersion. The default is `0.45`.
- `eta_cd (float)`: The maximal direction consistency. The default is `0.5`.
- `eta_pd (float)`: The maximal positional displacement. The default is `0.2`.
- `eta_max_fixation (float)`: The maximal fixation spatial range. The default is `1.9`°.
- `phi (float)`: The maximal angular difference to consider two sequences to have a displacement in the same direction. The default is `45`°.

![fixation_detection.png](../figures/fixation_detection.png)
Figure 1 - Plot obtained using `gaze_behavior_identifier.fixation.plot(save_name)`.


Available fixation metrics: 
- `gaze_behavior_identifier.fixation.nb_events`: The number of fixations in the trial.
- `gaze_behavior_identifier.fixation.duration`: The duration of each fixation in the trial.
- `gaze_behavior_identifier.fixation.mean_duration`: The mean duration of the fixations in the trial.
- `gaze_behavior_identifier.fixation.max_duration`: The duration of the longest fixation in the trial.
- `gaze_behavior_identifier.fixation.total_duration`: The total time spent doing fixations in the trial.
- `gaze_behavior_identifier.fixation.ratio`: The proportion ot time of the trial spent doing fixations.
- `gaze_behavior_identifier.fixation.search_rate`: The ratio of the number of fixations divided by the mean duration of the fixations.
