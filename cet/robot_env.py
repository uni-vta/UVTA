"""The robot interface the rollout expects, and why it is not implemented here.

The policy itself is hardware-agnostic: ``uvta/real_env/real_policy.py`` takes
an observation window and returns an unnormalized action chunk. Everything
below that -- transport, message schema, IK -- belongs to whatever arm you are
driving, and the implementation used in the paper is tied to a specific robot
and its vendor SDK, so it is not part of this repository.

To run on your own hardware, implement ``RobotEnv`` and pass it to the rollout
in place of the missing vendor class. The surface is small: the rollout only
ever calls ``get_latest_observation`` and ``send_action``.

Two things the implementation has to get right, because the policy silently
depends on them:

* **Tactile scale and baseline.** The policy consumes ``fsr_region``: each
  finger's 20 taxels pooled into 4 region means, by
  ``deform_to_human/fsr_region.py``. Apply that same pooling to the live
  stream, and subtract the resting baseline the same way training did -- first
  frame for a robot-mounted sensor, clamped at zero. Feeding raw taxels, or
  skipping the baseline, puts the tactile input far outside the training
  distribution and the policy quietly stops using it.

* **Observation timing.** The policy is trained on synchronized frames. The
  camera, joint encoders and tactile sensor have different latencies, so an
  implementation that returns "whatever arrived last" from each will drift
  apart under load. ``cet/observation_timeline.py`` is the buffer used here to
  pick a consistent timestamp across streams.
"""
from __future__ import annotations

import abc
from typing import Any, Dict, Optional


class RobotEnv(abc.ABC):
    """Transport + hardware abstraction for the rollout loop."""

    @abc.abstractmethod
    def get_latest_observation(self) -> Optional[Dict[str, Any]]:
        """Return the most recent synchronized observation, or None if not ready.

        Expected keys (single arm; prefix with ``left_`` / ``right_`` for a
        bimanual setup):

        ``camera_0``       (H, W, 3) uint8 RGB from the wrist camera. Resized
                           to ``dataset.camera_resize_shape`` by the policy, so
                           the native resolution only has to be consistent.
        ``pose``           (6,) float wrist pose as xyz + rotvec, in the same
                           frame the training data used. The policy predicts a
                           RELATIVE pose against this, so a constant frame
                           offset cancels out -- an inconsistent one does not.
        ``proprioception`` (22,) float hand joint angles. Only required when
                           the config sets ``proprio_mode`` to something other
                           than ``none``.
        ``fsr``            (100,) float raw taxels, or ``fsr_region`` (20,) if
                           you pool them yourself. See the module docstring.
        """

    @abc.abstractmethod
    def send_action(self, action: Any, immediate: bool = False) -> None:
        """Command the arm and hand.

        ``action`` is one decoded chunk: absolute wrist poses plus hand joint
        angles, already converted out of the policy's relative rot6d
        representation by the rollout.

        ``immediate`` distinguishes the initial move-to-start (True, execute
        now) from streaming during the loop (False, append to the trajectory
        the interpolator is consuming). Collapsing the two makes the first
        command a step input, which on most arms trips a velocity limit.
        """


class ArmKinematics(abc.ABC):
    """FK / IK for the arm, as the rollout uses it.

    The paper's arm is 7-DoF with a 5-DoF body; ``LightweightIK`` in the
    rollout wraps a vendor solver behind this. Any solver works as long as it
    agrees with the URDF the demonstrations were recorded against -- a
    mismatched kinematic chain shows up as a constant pose offset that looks
    exactly like a badly trained policy.
    """

    ARM_DOF = 7
    BODY_DOF = 5

    @abc.abstractmethod
    def forward(self, joint_angles):
        """Joint angles -> end-effector pose (4x4 homogeneous)."""

    @abc.abstractmethod
    def inverse(self, target_pose, seed_joint_angles):
        """Target pose (4x4) + seed -> joint angles, or None if unreachable."""
