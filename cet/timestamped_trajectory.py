"""Shared timestamped trajectory smoothing for North rollout scripts."""
from __future__ import annotations

import time
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from relative_policy_rollout_v2 import (
    MotorTrajectoryInterpolator,
    PoseTrajectoryInterpolator,
)


class JointKinematicLimiter:
    """Per-joint velocity/acceleration limiter on a monotonic clock.

    The effective integration step is capped at ``nominal_dt``. A long model
    inference stall therefore cannot turn elapsed wall time into permission for
    one large position-setpoint jump when the command loop resumes.
    """

    def __init__(
        self,
        nominal_dt: float,
        max_velocity: float,
        max_acceleration: float,
        max_tracking_error: Optional[float] = None,
    ) -> None:
        if min(nominal_dt, max_velocity, max_acceleration) <= 0:
            raise ValueError("Joint limiter parameters must be positive.")
        if max_tracking_error is not None and max_tracking_error <= 0:
            raise ValueError("max_tracking_error must be positive when set.")
        self.nominal_dt = float(nominal_dt)
        self.max_velocity = float(max_velocity)
        self.max_acceleration = float(max_acceleration)
        self.max_tracking_error = (
            None if max_tracking_error is None else float(max_tracking_error)
        )
        self.previous_position: Optional[np.ndarray] = None
        self.previous_velocity: Optional[np.ndarray] = None
        self.previous_time: Optional[float] = None
        self.last_tracking_reset = False
        self.last_tracking_clamped = False

    def reset(
        self,
        position: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> None:
        now = time.monotonic() if timestamp is None else float(timestamp)
        self.previous_position = np.asarray(position, dtype=np.float32).copy()
        self.previous_velocity = np.zeros_like(self.previous_position)
        self.previous_time = now
        self.last_tracking_reset = False
        self.last_tracking_clamped = False

    def hold(
        self,
        measured: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> np.ndarray:
        """Hold the last command rather than jump to a delayed measurement."""
        now = time.monotonic() if timestamp is None else float(timestamp)
        if self.previous_position is None:
            self.reset(measured, timestamp=now)
        assert self.previous_position is not None
        assert self.previous_velocity is not None
        self.last_tracking_reset = False
        self.last_tracking_clamped = False
        if self.max_tracking_error is not None:
            tracking_error = float(np.linalg.norm(
                self.previous_position - np.asarray(measured, np.float32)
            ))
            if tracking_error > self.max_tracking_error:
                self.previous_position = np.asarray(
                    measured, dtype=np.float32
                ).copy()
                self.last_tracking_reset = True
        self.previous_velocity.fill(0.0)
        self.previous_time = now
        return self.previous_position.copy()

    def limit(
        self,
        target: np.ndarray,
        measured: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> np.ndarray:
        now = time.monotonic() if timestamp is None else float(timestamp)
        target = np.asarray(target, dtype=np.float32)
        measured = np.asarray(measured, dtype=np.float32)
        if target.shape != measured.shape:
            raise ValueError(
                f"target/measured shape mismatch: {target.shape} != {measured.shape}"
            )
        if self.previous_position is None:
            self.reset(measured, timestamp=now - self.nominal_dt)
        assert self.previous_position is not None
        assert self.previous_velocity is not None
        assert self.previous_time is not None

        self.last_tracking_reset = False
        self.last_tracking_clamped = False
        if self.max_tracking_error is not None:
            tracking_error = float(np.linalg.norm(
                self.previous_position - measured
            ))
            if tracking_error > self.max_tracking_error:
                # The actuator has not followed the internally integrated
                # command. Re-anchor the limiter to reality so repeated IK /
                # inference cycles cannot build an arbitrarily large command
                # backlog while the measured arm is stationary.
                self.previous_position = measured.copy()
                self.previous_velocity = np.zeros_like(measured)
                self.previous_time = now - self.nominal_dt
                self.last_tracking_reset = True

        elapsed = max(now - self.previous_time, 1e-4)
        # Position commands are held while inference blocks: the actuator did
        # not traverse a missing ramp during that gap. Limit the next emitted
        # setpoint to one normal control increment.
        dt = min(elapsed, self.nominal_dt)
        desired_velocity = (target - self.previous_position) / dt
        desired_velocity = np.clip(
            desired_velocity,
            -self.max_velocity,
            self.max_velocity,
        )
        max_delta_velocity = self.max_acceleration * dt
        velocity = self.previous_velocity + np.clip(
            desired_velocity - self.previous_velocity,
            -max_delta_velocity,
            max_delta_velocity,
        )
        velocity = np.clip(
            velocity,
            -self.max_velocity,
            self.max_velocity,
        )
        limited = self.previous_position + velocity * dt
        if self.max_tracking_error is not None:
            tracking_delta = limited - measured
            tracking_error = float(np.linalg.norm(tracking_delta))
            if tracking_error > self.max_tracking_error:
                limited = measured + tracking_delta * (
                    self.max_tracking_error / max(tracking_error, 1e-12)
                )
                # Keep the limiter's velocity state consistent with the
                # command that will actually be emitted.
                velocity = (limited - self.previous_position) / dt
                self.last_tracking_clamped = True
        self.previous_position = limited.astype(np.float32)
        self.previous_velocity = velocity.astype(np.float32)
        self.previous_time = now
        return self.previous_position.copy()


def schedule_absolute_trajectory_chunk(
    pose_interp: PoseTrajectoryInterpolator,
    hand_interp: MotorTrajectoryInterpolator,
    pose_chunk: np.ndarray,
    hand_chunk: np.ndarray,
    waypoint_indices: Sequence[int],
    timeline_origin_mono: float,
    waypoint_dt: float,
    arm_latency_s: float,
    hand_latency_s: float,
    safety_margin_s: float,
    max_pos_speed: float,
    max_rot_speed: float,
    max_hand_speed: float,
    now_mono: Optional[float] = None,
) -> Tuple[
    PoseTrajectoryInterpolator,
    MotorTrajectoryInterpolator,
    int,
    int,
    int,
    int,
]:
    """Schedule absolute wrist/hand chunks on their North observation time.

    Stale waypoints are discarded. The first point of each replacement
    interpolator is sampled from the old interpolator at now, providing C0
    continuity across replans. If every predicted point is stale, the final
    target is retained as one future, speed-limited fallback waypoint.
    """
    pose_chunk = np.asarray(pose_chunk, dtype=np.float64)
    hand_chunk = np.asarray(hand_chunk, dtype=np.float64)
    indices = np.asarray(waypoint_indices, dtype=np.int64).reshape(-1)
    if len(indices) == 0:
        raise ValueError("waypoint_indices must not be empty")
    if pose_chunk.shape != (len(indices), 6):
        raise ValueError(
            f"pose_chunk must be ({len(indices)}, 6), got {pose_chunk.shape}"
        )
    if hand_chunk.ndim != 2 or hand_chunk.shape[0] != len(indices):
        raise ValueError(
            "hand_chunk must be a 2-D array with one row per waypoint"
        )
    if np.any(indices[1:] <= indices[:-1]):
        raise ValueError("waypoint_indices must be strictly increasing")
    if min(
        waypoint_dt,
        max_pos_speed,
        max_rot_speed,
        max_hand_speed,
    ) <= 0:
        raise ValueError("Trajectory cadence and speed limits must be positive.")
    if min(arm_latency_s, hand_latency_s, safety_margin_s) < 0:
        raise ValueError("Latency and safety margin must be non-negative.")
    now = time.monotonic() if now_mono is None else float(now_mono)
    times = float(timeline_origin_mono) + indices * float(waypoint_dt)
    arm_offset = int(
        np.searchsorted(
            times,
            now + arm_latency_s + safety_margin_s,
            side="left",
        )
    )
    hand_offset = int(
        np.searchsorted(
            times,
            now + hand_latency_s + safety_margin_s,
            side="left",
        )
    )
    arm_first_index = (
        int(indices[arm_offset]) if arm_offset < len(indices)
        else int(indices[-1] + 1)
    )
    hand_first_index = (
        int(indices[hand_offset]) if hand_offset < len(indices)
        else int(indices[-1] + 1)
    )

    arm_offsets = list(range(arm_offset, len(indices)))
    arm_fallback = not arm_offsets
    if arm_fallback:
        arm_offsets = [len(indices) - 1]
    pose_times = [now]
    pose_values = [np.asarray(pose_interp(now), dtype=np.float64)]
    arm_scheduled = 0
    for offset in arm_offsets:
        deadline = float(times[offset])
        if arm_fallback:
            deadline = max(
                deadline,
                now + arm_latency_s + safety_margin_s + waypoint_dt,
            )
        if deadline <= pose_times[-1]:
            continue
        dt_step = deadline - pose_times[-1]
        previous = pose_values[-1]
        desired = pose_chunk[offset]

        delta_xyz = desired[:3] - previous[:3]
        distance = float(np.linalg.norm(delta_xyz))
        distance_limit = max_pos_speed * dt_step
        if distance > distance_limit:
            delta_xyz *= distance_limit / max(distance, 1e-12)

        previous_rotation = R.from_rotvec(previous[3:])
        desired_rotation = R.from_rotvec(desired[3:])
        delta_rotvec = (
            previous_rotation.inv() * desired_rotation
        ).as_rotvec()
        angle = float(np.linalg.norm(delta_rotvec))
        angle_limit = max_rot_speed * dt_step
        if angle > angle_limit:
            delta_rotvec *= angle_limit / max(angle, 1e-12)

        reachable = np.empty(6, dtype=np.float64)
        reachable[:3] = previous[:3] + delta_xyz
        reachable[3:] = (
            previous_rotation * R.from_rotvec(delta_rotvec)
        ).as_rotvec()
        pose_times.append(deadline)
        pose_values.append(reachable)
        arm_scheduled += 1
    if arm_scheduled:
        pose_interp = PoseTrajectoryInterpolator(
            np.asarray(pose_times),
            np.stack(pose_values, axis=0),
        )

    hand_offsets = list(range(hand_offset, len(indices)))
    hand_fallback = not hand_offsets
    if hand_fallback:
        hand_offsets = [len(indices) - 1]
    hand_times = [now]
    hand_values = [np.asarray(hand_interp(now), dtype=np.float64)]
    hand_scheduled = 0
    for offset in hand_offsets:
        deadline = float(times[offset])
        if hand_fallback:
            deadline = max(
                deadline,
                now + hand_latency_s + safety_margin_s + waypoint_dt,
            )
        if deadline <= hand_times[-1]:
            continue
        dt_step = deadline - hand_times[-1]
        previous = hand_values[-1]
        delta = hand_chunk[offset] - previous
        distance = float(np.linalg.norm(delta))
        distance_limit = max_hand_speed * dt_step
        if distance > distance_limit:
            delta *= distance_limit / max(distance, 1e-12)
        hand_times.append(deadline)
        hand_values.append(previous + delta)
        hand_scheduled += 1
    if hand_scheduled:
        hand_interp = MotorTrajectoryInterpolator(
            np.asarray(hand_times),
            np.stack(hand_values, axis=0),
        )

    return (
        pose_interp,
        hand_interp,
        arm_scheduled,
        hand_scheduled,
        arm_first_index,
        hand_first_index,
    )
