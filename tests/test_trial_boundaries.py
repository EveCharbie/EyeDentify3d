"""
Tests making sure that a trial can begin and end with any gaze behavior.
The frames at the very beginning and at the very end of a trial are edgy since there is no frame before/after them to
compare with, so every behavior is tested at both extremities of the trial (on the full trial and on trials obtained
after splitting).
"""

import numpy as np
import numpy.testing as npt
import pytest

from eyedentify3d import ReducedData, GazeBehaviorIdentifier
from eyedentify3d.identification.saccade import SaccadeEvent
from eyedentify3d.identification.smooth_pursuit import SmoothPursuitEvent

DT = 0.008  # 125 Hz


def direction_from_angles(yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """
    Get the unit direction vectors (3, n_frames) corresponding to the yaw and pitch angles provided in degrees.
    """
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    return np.vstack(
        [
            np.sin(yaw_rad) * np.cos(pitch_rad),
            np.sin(pitch_rad),
            np.cos(yaw_rad) * np.cos(pitch_rad),
        ]
    )


def build_data_object(segments: list[tuple[str, int]], dt: float = DT, seed: int = 42) -> ReducedData:
    """
    Create a synthetic data object composed of the segments provided.

    Parameters
    ----------
    segments: A list of (behavior, number of frames) defining the trial. The behavior must be one of "fixation",
        "smooth_pursuit", "visual_scanning", "saccade", "blink", or "invalid".
    dt: The time step between two frames.
    seed: The seed of the random number generator used to add a small noise on the angles (a perfectly still gaze
        cannot be classified since the variability decomposition needs some variability).
    """
    random_number_generator = np.random.default_rng(seed)
    nb_frames = sum(segment[1] for segment in segments)
    time_vector = np.arange(nb_frames) * dt

    # The eyes and the head are described through their yaw and pitch angles (in degrees) so that the velocities are
    # easy to impose
    eye_yaw = np.zeros(nb_frames)
    head_yaw = np.zeros(nb_frames)
    eye_openness = np.ones(nb_frames)
    data_invalidity = np.zeros(nb_frames, dtype=bool)

    first_frame = 0
    for behavior, length in segments:
        this_segment = slice(first_frame, first_frame + length)
        if behavior == "fixation":
            # The gaze does not move
            pass
        elif behavior == "smooth_pursuit":
            # The gaze moves slowly (20 deg/s), which is bellow the visual scanning threshold
            head_yaw[this_segment] = np.linspace(0, 20 * length * dt, length)
        elif behavior == "visual_scanning":
            # The gaze moves fast (200 deg/s), but the eyes do not move, so it is not a saccade
            head_yaw[this_segment] = np.linspace(0, 200 * length * dt, length)
        elif behavior == "saccade":
            # The eyes move very fast
            eye_yaw[this_segment] = np.linspace(0, 20, length)
        elif behavior == "blink":
            eye_openness[this_segment] = 0.1
        elif behavior == "invalid":
            data_invalidity[this_segment] = True
        else:
            raise ValueError(f"The behavior {behavior} is not a behavior that can be generated.")

        # Keep the angles continuous from one segment to the next one
        eye_yaw[first_frame + length :] += eye_yaw[first_frame + length - 1]
        head_yaw[first_frame + length :] += head_yaw[first_frame + length - 1]
        first_frame += length

    # Add a small noise so that there is always some variability in the gaze direction
    eye_yaw += random_number_generator.normal(0, 0.01, nb_frames)
    eye_pitch = random_number_generator.normal(0, 0.01, nb_frames)
    head_yaw += random_number_generator.normal(0, 0.01, nb_frames)
    head_pitch = random_number_generator.normal(0, 0.01, nb_frames)

    return ReducedData(
        original_dt=dt,
        original_time_vector=time_vector,
        original_right_eye_openness=eye_openness,
        original_left_eye_openness=eye_openness,
        original_eye_direction=direction_from_angles(eye_yaw, eye_pitch),
        original_head_angles=np.vstack([np.radians(head_pitch), np.radians(head_yaw), np.zeros(nb_frames)]),
        original_gaze_direction=direction_from_angles(eye_yaw + head_yaw, eye_pitch + head_pitch),
        original_head_angular_velocity=np.zeros((3, nb_frames)),
        original_head_velocity_norm=np.zeros(nb_frames),
        original_data_invalidity=data_invalidity,
    )


def identify_behaviors(data_object: ReducedData) -> GazeBehaviorIdentifier:
    """
    Run the whole identification pipeline on the data object provided.
    """
    gaze_behavior_identifier = GazeBehaviorIdentifier(data_object)
    gaze_behavior_identifier.detect_blink_sequences()
    gaze_behavior_identifier.detect_invalid_sequences()
    gaze_behavior_identifier.detect_saccade_sequences()
    gaze_behavior_identifier.detect_visual_scanning_sequences()
    gaze_behavior_identifier.detect_fixation_and_smooth_pursuit_sequences()
    gaze_behavior_identifier.finalize()
    return gaze_behavior_identifier


# Trials beginning and ending with each behavior. The behavior in the middle is only there to separate the two events.
BOUNDARY_SCENARIOS = {
    "blink": [("blink", 20), ("fixation", 60), ("blink", 20)],
    "invalid": [("invalid", 20), ("fixation", 60), ("invalid", 20)],
    "saccade": [("saccade", 6), ("fixation", 60), ("saccade", 6)],
    "visual_scanning": [("visual_scanning", 30), ("fixation", 60), ("visual_scanning", 30)],
    "fixation": [("fixation", 60), ("saccade", 6), ("fixation", 60)],
    "smooth_pursuit": [("smooth_pursuit", 60), ("saccade", 6), ("smooth_pursuit", 60)],
}

# The fixations and smooth pursuits are identified based on the gaze displacement between the current and the next
# frame, so the very last frame of the trial cannot be part of an inter-saccadic sequence.
LAST_FRAME_TOLERANCE = {
    "blink": 0,
    "invalid": 0,
    "saccade": 0,
    "visual_scanning": 0,
    "fixation": 1,
    "smooth_pursuit": 1,
}


@pytest.mark.parametrize("behavior", list(BOUNDARY_SCENARIOS.keys()))
def test_trial_beginning_and_ending_with_each_behavior(behavior):
    """Test that a trial can begin and end with any behavior."""
    data_object = build_data_object(BOUNDARY_SCENARIOS[behavior])
    nb_frames = data_object.time_vector.shape[0]

    gaze_behavior_identifier = identify_behaviors(data_object)

    sequences = getattr(gaze_behavior_identifier, behavior).sequences
    assert len(sequences) == 2

    # The first event begins on the first frame of the trial
    assert sequences[0][0] == 0

    # The last event ends on the last frame of the trial
    assert sequences[-1][-1] >= nb_frames - 1 - LAST_FRAME_TOLERANCE[behavior]

    # The metrics are computed without error on these events (finalize would have raised otherwise)
    assert gaze_behavior_identifier.is_finalized


@pytest.mark.parametrize("behavior", list(BOUNDARY_SCENARIOS.keys()))
def test_split_trial_beginning_and_ending_with_each_behavior(behavior):
    """
    Test that the trials obtained after a split can begin and end with any behavior.
    The split is performed right after the first event so that the first trial ends with this behavior, and the second
    trial begins with the behavior that follows.
    """
    data_object = build_data_object(BOUNDARY_SCENARIOS[behavior])
    gaze_behavior_identifier = identify_behaviors(data_object)

    # Split between the last frame of the first event and the frame that follows (so that no event is cut in two)
    first_sequence = getattr(gaze_behavior_identifier, behavior).sequences[0]
    split_timing = data_object.time_vector[first_sequence[-1]] + DT / 2

    split_gaze_behavior_identifiers = gaze_behavior_identifier.split([split_timing])

    assert len(split_gaze_behavior_identifiers) == 2
    for split_identifier in split_gaze_behavior_identifiers:
        assert split_identifier.is_finalized

    # The first trial ends with the behavior tested
    first_trial = split_gaze_behavior_identifiers[0]
    nb_frames_first_trial = first_trial.data_object.time_vector.shape[0]
    first_trial_sequences = getattr(first_trial, behavior).sequences
    assert len(first_trial_sequences) == 1
    assert first_trial_sequences[0][0] == 0
    assert first_trial_sequences[0][-1] == nb_frames_first_trial - 1

    # The second trial still ends with the behavior tested since the trial ends with it
    second_trial = split_gaze_behavior_identifiers[1]
    nb_frames_second_trial = second_trial.data_object.time_vector.shape[0]
    second_trial_sequences = getattr(second_trial, behavior).sequences
    assert len(second_trial_sequences) == 1
    assert second_trial_sequences[0][-1] >= nb_frames_second_trial - 1 - LAST_FRAME_TOLERANCE[behavior]


def test_smooth_pursuit_trajectory_on_the_last_frame_of_the_trial():
    """
    Test that the smooth pursuit trajectory can be measured when the smooth pursuit reaches the last frame of the
    trial (there is no frame after the last frame to measure the duration of this frame, so the duration of the
    previous frame is used instead).
    """
    data_object = build_data_object([("fixation", 40), ("saccade", 6), ("smooth_pursuit", 60)])
    nb_frames = data_object.time_vector.shape[0]

    smooth_pursuit = SmoothPursuitEvent(
        data_object,
        identified_indices=np.zeros(nb_frames, dtype=bool),
        smooth_pursuit_indices=np.arange(nb_frames - 20, nb_frames),
        minimal_duration=0.04,
    )
    smooth_pursuit.initialize()

    # The smooth pursuit sequence reaches the last frame of the trial
    assert len(smooth_pursuit.sequences) == 1
    assert smooth_pursuit.sequences[0][-1] == nb_frames - 1

    smooth_pursuit.measure_smooth_pursuit_trajectory()

    # The time step is constant, so each frame of the sequence (including the last one) lasts DT
    expected_trajectory = float(np.sum(np.abs(data_object.gaze_angular_velocity[smooth_pursuit.sequences[0]]) * DT))
    npt.assert_almost_equal(smooth_pursuit.smooth_pursuit_trajectories[0], expected_trajectory)


def test_saccade_on_the_first_frame_of_the_trial():
    """
    Test that a saccade beginning on the first frame of the trial is detected (the acceleration of the frame before the
    saccade is used to confirm the saccade, but there is no frame before the first frame of the trial).
    """
    data_object = build_data_object([("saccade", 6), ("fixation", 60)])

    saccade = SaccadeEvent(data_object, identified_indices=np.zeros(data_object.time_vector.shape[0], dtype=bool))
    saccade.initialize()

    assert len(saccade.sequences) == 1
    assert saccade.sequences[0][0] == 0


def test_trial_without_any_saccade():
    """Test that a trial in which no saccade candidate is detected does not raise."""
    data_object = build_data_object([("fixation", 60), ("visual_scanning", 30), ("fixation", 60)])

    saccade = SaccadeEvent(data_object, identified_indices=np.zeros(data_object.time_vector.shape[0], dtype=bool))
    saccade.initialize()

    assert saccade.sequences == []
    npt.assert_array_equal(saccade.frame_indices, np.array([], dtype=int))
