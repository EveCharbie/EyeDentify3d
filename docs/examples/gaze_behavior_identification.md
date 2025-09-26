# Gaze behavior identification examples

## How identify gaze behaviors
Here is the recommended order in which to identify the behaviors.
```python3
from eyedentify3d import GazeBehaviorIdentifier

# Create a GazeBehaviorIdentifier object
gaze_behavior_identifier = GazeBehaviorIdentifier(data_object)

# Detect gaze behaviors (must be performed in the desired order)
gaze_behavior_identifier.detect_blink_sequences()
gaze_behavior_identifier.detect_invalid_sequences()
gaze_behavior_identifier.detect_saccade_sequences()
gaze_behavior_identifier.detect_visual_scanning_sequences()
gaze_behavior_identifier.detect_fixation_and_smooth_pursuit_sequences()
gaze_behavior_identifier.finalize()  # This is mandatory

# Plot the results if desired
gaze_behavior_identifier.blink.plot(save_name="blink_detection.png")
gaze_behavior_identifier.invalid.plot(save_name="invalid_detection.png")
gaze_behavior_identifier.saccade.plot(save_name="saccade_detection.png")
gaze_behavior_identifier.visual_scanning.plot(save_name="visual_scanning_detection.png")
gaze_behavior_identifier.inter_saccadic_sequences.plot(save_name="inter_saccade_detection.png")
gaze_behavior_identifier.fixation.plot(save_name="fixation_detection.png")
gaze_behavior_identifier.smooth_pursuit.plot(save_name="fixation_detection.png")
gaze_behavior_identifier.plot(save_name="all_gaze_behaviors.png")

# Animate the results if desired
gaze_behavior_identifier.animate()
```
