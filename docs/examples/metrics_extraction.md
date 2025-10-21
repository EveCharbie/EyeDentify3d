# Metrics extraction examples

## How to save all metrics
If you are interested in saving all available metrics, you can use the `get_results` method:
```python3
import pandas as pd
full_results = gaze_behavior_identifier.get_results(participant_id, trial_id)
full_results.to_csv("results_file_name.csv")
```

## How to extract custom metrics
Otherwise, you can extract the metrics you are interested in individually. Here are some examples of available metrics:
```python3
# Invalid
total_invalid_duration = gaze_behavior_identifier.blink.total_duration()
invalid_ratio = gaze_behavior_identifier.blink.ratio()

# Blinks
nb_blinks = gaze_behavior_identifier.blink.nb_events()
blinks_duration = gaze_behavior_identifier.blink.duration()
mean_blink_duration = gaze_behavior_identifier.blink.mean_duration()
max_blink_duration = gaze_behavior_identifier.blink.max_duration()
total_blink_duration = gaze_behavior_identifier.blink.total_duration()
blink_ratio = gaze_behavior_identifier.blink.ratio()

# Saccades
nb_saccades = gaze_behavior_identifier.saccade.nb_events()
saccades_duration = gaze_behavior_identifier.saccade.duration()
mean_saccade_duration = gaze_behavior_identifier.saccade.mean_duration()
max_saccade_duration = gaze_behavior_identifier.saccade.max_duration()
total_saccade_duration = gaze_behavior_identifier.saccade.total_duration()
saccades_amplitude = gaze_behavior_identifier.saccade.saccade_amplitudes()

# Visual scanning
nb_visual_scannings = gaze_behavior_identifier.visual_scanning.nb_events()
visual_scannings_duration = gaze_behavior_identifier.visual_scanning.duration()
mean_visual_scanning_duration = gaze_behavior_identifier.visual_scanning.mean_duration()
max_visual_scanning_duration = gaze_behavior_identifier.visual_scanning.max_duration()
total_visual_scanning_duration = gaze_behavior_identifier.visual_scanning.total_duration()
visual_scanning_ratio = gaze_behavior_identifier.visual_scanning.ratio()

# Inter-saccadic sequences
inter_saccadic_sequences = gaze_behavior_identifier.inter_saccadic_sequences.sequences()
coherent_sequences = gaze_behavior_identifier.inter_saccadic_sequences.coherent_sequences()
incoherent_sequences = gaze_behavior_identifier.inter_saccadic_sequences.incoherent_sequences()

# Fixations
nb_fixations = gaze_behavior_identifier.fixation.nb_events()
fixations_duration = gaze_behavior_identifier.fixation.duration()
mean_fixation_duration = gaze_behavior_identifier.fixation.mean_duration()
max_fixation_duration = gaze_behavior_identifier.fixation.max_duration()
total_fixation_duration = gaze_behavior_identifier.fixation.total_duration()
fixation_ratio = gaze_behavior_identifier.fixation.ratio()
search_rate = gaze_behavior_identifier.fixation.search_rate()

# Smooth Pursuit
nb_smooth_pursuits = gaze_behavior_identifier.smooth_pursuit.nb_events()
smooth_pursuits_duration = gaze_behavior_identifier.smooth_pursuit.duration()
mean_smooth_pursuit_duration = gaze_behavior_identifier.smooth_pursuit.mean_duration()
max_smooth_pursuit_duration = gaze_behavior_identifier.smooth_pursuit.max_duration()
total_smooth_pursuit_duration = gaze_behavior_identifier.smooth_pursuit.total_duration()
smooth_pursuit_ratio = gaze_behavior_identifier.smooth_pursuit.ratio()
smooth_pursuit_trajectories = gaze_behavior_identifier.smooth_pursuit.smooth_pursuit_trajectories()

# Unidentified
unidentified_frames = gaze_behavior_identifier.unidentified_indices
unidentified_ratio = gaze_behavior_identifier.unidentified_ratio()
```