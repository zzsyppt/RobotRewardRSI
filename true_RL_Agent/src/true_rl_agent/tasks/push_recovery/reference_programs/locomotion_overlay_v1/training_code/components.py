"""Reference: retain the robot's locomotion reward with the fixed Push overlay.

The overlay and robot learner are frozen by the task and training binding.
Candidates may add components under the existing Push capability profile.
"""

from true_rl_agent.candidates.components import TrainingComponents


def build_training_components():
    return TrainingComponents(name="push_locomotion_overlay_reference_v1")
