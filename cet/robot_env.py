"""Hardware interfaces for UVTA rollout integration.

Implement robot transport and arm kinematics for your platform; vendor SDKs
are not included. Adapt the topic-based observation and command mappings
in the rollout script to your backend.

Synchronize camera, joint, and tactile timestamps. Use the same fingertip
region pooling as training; RealPolicy applies the configured tactile
baseline correction, so do not subtract it a second time in the backend.
"""
from __future__ import annotations

import abc
from typing import Any, Dict, Optional


class RobotEnv(abc.ABC):
    """Transport + hardware abstraction for the rollout loop."""

    @abc.abstractmethod
    def get_latest_observation(self) -> Optional[Dict[str, Any]]:
        """Return the most recent synchronized observation, or None if not ready.

        The rollout's get_latest_robot_obs() expects topic-keyed wrist RGB,
        arm/hand joint angles, body state, tactile data, and timestamps.
        Adapt that function and build_tactile_from_obs() to your backend.
        These transport keys are distinct from the dataset's array names.
        """

    @abc.abstractmethod
    def send_action(self, action: Any, immediate: bool = False) -> None:
        """Command the arm and hand.

        The rollout supplies topic-keyed joint commands after pose decoding,
        IK, and trajectory limiting; see build_robot_action_buffer().
        Preserve the distinction between immediate initialization commands
        and queued streaming commands (immediate=False).
        """


class ArmKinematics(abc.ABC):
    """FK/IK interface for hardware integration.

    Adapt LightweightIK in the rollout to your solver. Match the robot URDF,
    joint ordering, and wrist coordinate frame used to collect demonstrations.
    """

    ARM_DOF = 7
    BODY_DOF = 5

    @abc.abstractmethod
    def forward(self, joint_angles):
        """Joint angles -> end-effector pose (4x4 homogeneous)."""

    @abc.abstractmethod
    def inverse(self, target_pose, seed_joint_angles):
        """Target pose (4x4) + seed -> joint angles, or None if unreachable."""
