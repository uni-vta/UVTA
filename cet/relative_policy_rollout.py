"""Asynchronous robot rollout for the UVTA relative-pose policy.

Observations follow uvta/diffusion_policy/dataloader/uvta_dataset.py.
Each wrist target is pose[t]^-1 @ pose_action[t+k]; execute the first
predicted target with --anchor_offset 0. Use offset 1 only for legacy
checkpoints whose first target is the identity.

The worker predicts full action chunks while the control loop interpolates
valid waypoints, compensates for latency, solves IK, and applies joint
velocity/acceleration limits. Observation timestamps prevent duplicate
history entries. Proprioceptive inputs follow the saved policy config.
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
import queue
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# ``robot_env`` / ``arm_kinematics`` are only available on the robot's
# deploy machine.  We still want this module to be importable from
# off-robot scripts (smoke tests, IK benchmarks) so users can exercise
# ``FingertipToJointIK`` / ``LightweightIK`` standalone.  Defer the
# hard failure to ``main()``.
try:
    from robot_env import RobotEnv  # noqa: F401
    import arm_kinematics as nk  # noqa: F401
    _ROBOT_ENV_AVAILABLE = True
    _ROBOT_ENV_IMPORT_ERR: Optional[BaseException] = None
except ImportError as _e:  # noqa: BLE001
    RobotEnv = None  # type: ignore[assignment]
    nk = None  # type: ignore[assignment]
    _ROBOT_ENV_AVAILABLE = False
    _ROBOT_ENV_IMPORT_ERR = _e

try:
    from uvta.real_env.real_policy import RealPolicy  # noqa: F401
    _REAL_POLICY_AVAILABLE = True
    _REAL_POLICY_IMPORT_ERR: Optional[BaseException] = None
except ImportError as _e:  # noqa: BLE001
    RealPolicy = None  # type: ignore[assignment]
    _REAL_POLICY_AVAILABLE = False
    _REAL_POLICY_IMPORT_ERR = _e

from observation_timeline import ObservationTimestampBuffer  # noqa: E402
from timestamped_trajectory import (  # noqa: E402
    JointKinematicLimiter,
    MotorTrajectoryInterpolator,
    PoseTrajectoryInterpolator,
    schedule_absolute_trajectory_chunk,
)


# =========================================================
# 基础几何工具
# =========================================================
def vec6dof_to_homogeneous_matrix(translation, rotation_vector):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.from_rotvec(np.asarray(rotation_vector, dtype=np.float32)).as_matrix()
    T[:3, 3] = np.asarray(translation, dtype=np.float32)
    return T


def homogeneous_matrix_to_6dof(T):
    T = np.asarray(T, dtype=np.float32)
    t = T[:3, 3]
    r = R.from_matrix(T[:3, :3]).as_rotvec()
    return np.concatenate([t, r], axis=0).astype(np.float32)


def invert_transformation(T):
    T = np.asarray(T, dtype=np.float32)
    T_inv = np.eye(4, dtype=np.float32)
    Rm = T[:3, :3]
    t = T[:3, 3]
    T_inv[:3, :3] = Rm.T
    T_inv[:3, 3] = -Rm.T @ t
    return T_inv


def parse_t_et_arg(t_et_arg: Optional[str]) -> np.ndarray:
    """Parse ``--t_et`` into a 4x4 homogeneous matrix.

    Accepts either 6 numbers ``x,y,z,rx,ry,rz`` or 16 numbers (row-major
    4x4).  Defaults to the identity.

    NOTE on semantics: Method B trains the relative-pose action target as
    ``T_rel = T_t^{-1} @ T_{t+k}`` *directly in the current wrist frame*,
    so the deploy-time decode is simply ``T_BN = T_BE @ T_rel`` and
    ``T_ET`` does not appear.  Passing a non-identity ``T_ET`` here
    re-introduces the legacy ``T_BE @ T_ET @ T_rel @ T_ET^{-1}`` decode
    (kept for backwards compatibility with old checkpoints trained with
    a wrist offset); using it with a method-B checkpoint will produce a
    silent systematic offset.  We warn at startup if it is not identity.
    """
    if t_et_arg is None or t_et_arg.strip() == "":
        return np.eye(4, dtype=np.float32)

    vals = [float(x) for x in t_et_arg.split(",")]
    if len(vals) == 6:
        return vec6dof_to_homogeneous_matrix(vals[:3], vals[3:])
    if len(vals) == 16:
        return np.asarray(vals, dtype=np.float32).reshape(4, 4)
    raise ValueError("--t_et only accepts 6 or 16 numbers")


# =========================================================
# 图像处理
# =========================================================
def ensure_rgb_hwc(img: np.ndarray) -> np.ndarray:
    """Coerce arbitrary single-frame image input to HWC + RGB uint8."""
    img = np.asarray(img)
    if img.ndim != 3:
        raise ValueError(f"Unexpected image ndim: {img.ndim}, shape={img.shape}")

    if img.shape[0] == 3 and img.shape[-1] != 3:
        img = img.transpose(1, 2, 0)

    if img.shape[-1] != 3:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    # robot vision topics (.../rgb) already deliver RGB, matching the training
    # data. Do NOT convert here: an extra BGR2RGB would swap channels and feed
    # the policy BGR (verified via debug/tmp/policy_input_*.png).
    return img.astype(np.uint8)


# =========================================================
# 观测读取
# =========================================================
def get_latest_robot_obs(robot_env: RobotEnv) -> Dict[str, Any]:
    latest_obs = robot_env.get_latest_observation()
    while latest_obs is None:
        time.sleep(0.01)
        latest_obs = robot_env.get_latest_observation()

    def safe_array(key, dtype=np.float32):
        if key not in latest_obs:
            return None
        return np.array(latest_obs[key], dtype=dtype)

    left_wrist_img = safe_array('/observe/vision/left_wrist/fisheye/rgb', dtype=np.uint8)
    right_wrist_img = safe_array('/observe/vision/right_wrist/fisheye/rgb', dtype=np.uint8)
    left_eye_img = safe_array('/observe/vision/head/stereo/lefteye/rgb', dtype=np.uint8)
    right_eye_img = safe_array('/observe/vision/head/stereo/righteye/rgb', dtype=np.uint8)

    left_arm = safe_array('/state/left_arm/joint_angle')
    left_hand = safe_array('/state/left_hand/joint_angle')
    right_arm = safe_array('/state/right_arm/joint_angle')
    right_hand = safe_array('/state/right_hand/joint_angle')

    motor_angle = safe_array('/state/motor/joint_angle')
    if motor_angle is None:
        body_dof = np.zeros(5, dtype=np.float32)
        neck_dof = np.zeros(2, dtype=np.float32)
    else:
        body_dof = motor_angle[:5].copy()
        neck_dof = motor_angle[5:7].copy()

    return {
        # robot's producer-side timestamp plus the local callback timestamps.
        # Callback timestamps remain identical when a fast control loop polls
        # the same observation more than once, making duplicate detection
        # reliable even if the producer timestamp uses an unknown unit.
        "timestamp": latest_obs.get("timestamp"),
        "receive_wall_time_s": latest_obs.get(
            "_robot_receive_wall_time_s", time.time()
        ),
        "receive_monotonic_s": latest_obs.get(
            "_robot_receive_monotonic_s", time.monotonic()
        ),
        "left_eye_img": left_eye_img,
        "right_eye_img": right_eye_img,
        "left_wrist_img": left_wrist_img,
        "right_wrist_img": right_wrist_img,
        "left_arm": left_arm,
        "left_hand": left_hand,
        "right_arm": right_arm,
        "right_hand": right_hand,
        "body_dof": body_dof,
        "neck_dof": neck_dof,
        "raw": latest_obs,
    }


# =========================================================
# Tactile: build the per-frame tactile vector from the raw robot obs so it
# matches the training-data layout (scripts/flip_book_to_handumi_zarr.py /
# scripts/add_deform_fsr_teleop.py).  Finger order MUST match the zarr build.
# =========================================================
# Order used when concatenating per-finger tactile in the training zarr.
_TACTILE_FINGERS = ["thumb", "index", "middle", "ring", "little"]
# Taxels per finger for the fsr (deform) stream (-> 5 * 20 = 100-D).
_TAXELS_PER_FINGER = 20


def _force6d_to_resultant(force6d: np.ndarray) -> float:
    """``force6d`` = [fx, fy, fz, tx, ty, tz] -> resultant force magnitude
    ``sqrt(fx^2+fy^2+fz^2)`` (L2 norm of the first 3 = force axes).

    Matches the training ``force`` field (per-finger resultant of the 3-axis
    force; see scripts/flip_book_to_handumi_zarr.py ``vec_norm``).
    """
    f = np.asarray(force6d, dtype=np.float32).reshape(-1)[:3]
    return float(np.linalg.norm(f))


# Lazily-constructed deform -> 100-D taxel converter (the SAME class the
# datasets were built with: deform_to_human/deform_to_taxel.py).  Built once
# PER HAND (each loads its per-finger region-label maps) and reused every
# frame.  Bimanual rollout needs both a 'right' and a 'left' converter.
_DEFORM_CONV: Dict[str, Any] = {}


def _get_deform_converter(hand: str = "right"):
    """Return a cached ``DeformToTaxel`` for ``hand`` ('right'/'left',
    reduce='sum'), matching the training-data build.  Imported by file path so
    it works regardless of whether ``deform_to_human`` is on ``sys.path`` / a
    package.  One converter is cached per hand."""
    global _DEFORM_CONV
    if hand not in _DEFORM_CONV:
        import importlib.util

        repo_root = Path(__file__).resolve().parent.parent
        mod_path = repo_root / "deform_to_human" / "deform_to_taxel.py"
        if not mod_path.exists():
            raise FileNotFoundError(
                f"deform->taxel converter not found: {mod_path}"
            )
        spec = importlib.util.spec_from_file_location("deform_to_taxel", mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # reduce='sum' == the teleop/skeleton fsr build recipe.
        _DEFORM_CONV[hand] = mod.DeformToTaxel(hand=hand, reduce="sum")
    return _DEFORM_CONV[hand]


# Lazily-loaded 100-D fsr -> 20-D fsr_region reducer (the SAME single source of
# truth the datasets were pooled with: deform_to_human/fsr_region.py).  Imported
# by file path so it works regardless of sys.path.
_REGION_REDUCER = None


def _get_region_reducer():
    """Return ``reduce_fsr_to_region(fsr, hand)`` from
    ``deform_to_human/fsr_region.py`` (pools each finger's 20 fsr taxels into 4
    region MEANs, matching the ``fsr_region`` training field)."""
    global _REGION_REDUCER
    if _REGION_REDUCER is None:
        import importlib.util

        repo_root = Path(__file__).resolve().parent.parent
        mod_path = repo_root / "deform_to_human" / "fsr_region.py"
        if not mod_path.exists():
            raise FileNotFoundError(f"fsr_region reducer not found: {mod_path}")
        spec = importlib.util.spec_from_file_location("fsr_region", mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _REGION_REDUCER = mod.reduce_fsr_to_region
    return _REGION_REDUCER


def build_tactile_from_obs(
    obs: Dict[str, Any], tactile_key: str, hand: str = "right"
) -> np.ndarray:
    """Assemble the per-frame tactile vector matching the training layout.

    - ``tactile_key == 'force'`` -> per-finger resultant force from
      ``/observe/tactile/{hand}_{finger}/force6d`` (first 3 dims) -> ``(5,)``.
    - ``tactile_key in ('fsr', 'fsr_region')`` -> five ``.../deform`` maps run
      through the dataset's ``DeformToTaxel`` (per-finger region-sum) -> the
      100-D fsr; for ``'fsr_region'`` the 100-D is then pooled to 20-D via the
      SAME reducer used to build the ``fsr_region`` training field.

    Finger order is ``_TACTILE_FINGERS``.  Returns raw (un-normalized) values;
    the policy applies binarize/normalize downstream.
    """
    raw = obs.get("raw", obs)
    if tactile_key == "force":
        parts = []
        for finger in _TACTILE_FINGERS:
            key = f"/observe/tactile/{hand}_{finger}/force6d"
            if key not in raw:
                raise KeyError(f"tactile force key missing from obs: {key}")
            parts.append(np.array([_force6d_to_resultant(raw[key])],
                                  dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)  # (5,)

    # fsr / fsr_region: gather the 5 finger deform maps (finger order) and
    # convert with the exact dataset recipe -> 100-D fsr.
    maps = []
    for finger in _TACTILE_FINGERS:
        key = f"/observe/tactile/{hand}_{finger}/deform"
        if key not in raw:
            raise KeyError(f"tactile deform key missing from obs: {key}")
        maps.append(np.asarray(raw[key]))
    fsr = _get_deform_converter(hand).frame(maps)  # (100,) float32
    if fsr.shape[0] != len(_TACTILE_FINGERS) * _TAXELS_PER_FINGER:
        raise ValueError(
            f"deform->taxel produced {fsr.shape[0]} values, expected "
            f"{len(_TACTILE_FINGERS) * _TAXELS_PER_FINGER}"
        )
    if tactile_key == "fsr_region":
        # Pool each finger's 20 taxels -> 4 region MEANs (5x4 = 20-D), matching
        # the ``fsr_region`` field the policy was trained on.
        return _get_region_reducer()(fsr, hand=hand).astype(np.float32)  # (20,)
    return fsr.astype(np.float32)


class _TactileLog:
    """Durable per-step tactile logger for debugging force RANGE (OOD).

    For each control step it records the raw tactile vector, its min-max
    normalization using the policy's DEPLOY stats (``2*(x-min)/(max-min)-1``,
    in-distribution is ``[-1, 1]``), and a per-dim out-of-range flag.  This is
    exactly what tells us whether a channel exceeds its training量程 before the
    hand even contacts anything.

    Writes ``<run_dir>/tactile.jsonl`` line-buffered (one flushed line per step,
    so it survives a hard stop / SIGKILL) and a ``tactile_summary.txt`` on a
    clean exit.
    """

    def __init__(self, run_dir, stat_min, stat_max, tactile_key, labels=None):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.key = tactile_key
        self.mn = None if stat_min is None else np.asarray(stat_min, np.float32).ravel()
        self.mx = None if stat_max is None else np.asarray(stat_max, np.float32).ravel()
        self.labels = labels
        self._fh = open(os.path.join(run_dir, "tactile.jsonl"), "w", buffering=1)
        self._n = 0
        self._ood_over = None    # per-dim count of force > train max
        self._ood_under = None   # per-dim count of force < train min
        self._obs_min = None
        self._obs_max = None
        self._last_warn = -999

    def _label(self, i):
        if self.labels is not None and i < len(self.labels):
            return self.labels[i]
        return str(i)

    def log(self, step, force):
        f = np.asarray(force, np.float32).ravel()
        d = f.shape[0]
        if self._obs_min is None:
            self._obs_min = f.copy()
            self._obs_max = f.copy()
            self._ood_over = np.zeros(d, np.int64)
            self._ood_under = np.zeros(d, np.int64)
        self._obs_min = np.minimum(self._obs_min, f)
        self._obs_max = np.maximum(self._obs_max, f)

        norm = None
        ood = np.zeros(d, bool)
        if self.mn is not None and self.mx is not None and self.mx.shape == f.shape:
            # Channels whose training range collapsed are never normalized by
            # ``normalize_data``; the policy pins them at the training constant.
            # Dividing by a 1e-8 floor here would report an astronomical value
            # for a millivolt of drift and send debugging down a blind alley,
            # so report them as the pinned in-distribution value instead.
            span = self.mx - self.mn
            frozen = span <= 1e-8
            rng = np.where(frozen, 1.0, span)
            norm = np.where(frozen, -1.0, 2.0 * (f - self.mn) / rng - 1.0)
            # Tolerance 0.05 (=5% beyond the [-1,1] training range) so the
            # marginal norm≈-1.01 at exactly-zero force (train min is a hair
            # above 0) does NOT spam as OOD; only meaningful excursions flag.
            over = norm > 1.05
            under = norm < -1.05
            # A frozen channel that starts moving is genuinely out of
            # distribution -- training only ever showed the constant -- but the
            # normalized view can never say so, so compare the raw reading.
            over |= frozen & (f > self.mx + 1e-3)
            under |= frozen & (f < self.mn - 1e-3)
            ood = over | under
            self._ood_over += over.astype(np.int64)
            self._ood_under += under.astype(np.int64)

        rec = {"step": int(step), "force": [round(float(x), 5) for x in f]}
        if norm is not None:
            rec["norm"] = [round(float(x), 4) for x in norm]
            rec["ood"] = [int(x) for x in ood]
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()
        self._n += 1

        if ood.any() and (step - self._last_warn >= 5):
            self._last_warn = step
            dims = np.where(ood)[0].tolist()
            tmax = self.mx[dims].tolist() if self.mx is not None else None
            print(
                f"[tactile][step {step}] OOD dims="
                f"{[self._label(i) for i in dims]}  "
                f"force={np.round(f[dims], 3).tolist()}  "
                f"train_max={np.round(tmax, 3).tolist() if tmax is not None else '?'}"
            )

    def dump(self):
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass
        lines = [f"tactile_key={self.key}  steps={self._n}"]
        if self._obs_max is not None:
            for i in range(len(self._obs_max)):
                tmn = f"{self.mn[i]:.4f}" if self.mn is not None else "?"
                tmx = f"{self.mx[i]:.4f}" if self.mx is not None else "?"
                over = int(self._ood_over[i]) if self._ood_over is not None else 0
                under = int(self._ood_under[i]) if self._ood_under is not None else 0
                lines.append(
                    f"  [{self._label(i)}] observed[min={self._obs_min[i]:.4f} "
                    f"max={self._obs_max[i]:.4f}]  train[min={tmn} max={tmx}]  "
                    f"OOD steps: over_max={over} under_min={under} / {self._n}"
                )
        with open(os.path.join(self.run_dir, "tactile_summary.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print("[tactile][summary]\n" + "\n".join(lines))


# =========================================================
# 单臂 IK / FK
# =========================================================
class LightweightIK:
    """Thin wrapper around the arm IK backend for single-arm FK / IK."""

    # Mirror nk.ARM_DOF / nk.BODY_DOF.  Hard-coded so the class body does
    # not crash when ``arm_kinematics`` is not importable (smoke tests,
    # off-robot scripts); the actual values are also re-validated against
    # ``nk`` at runtime if the module is available.
    ARM_DOF = 7
    BODY_DOF = 5

    def __init__(self, urdf_path: str):
        self._kin = nk.ArmKinematics()
        if not self._kin.init(urdf_path):
            raise RuntimeError(f"arm kinematics init failed: {urdf_path}")

    def fk(self, arm_angles: np.ndarray, body_angles: np.ndarray,
           side: str = "right") -> np.ndarray:
        arm_angles = np.asarray(arm_angles, dtype=np.float32).ravel()
        body_angles = np.asarray(body_angles, dtype=np.float32).ravel()
        assert arm_angles.shape == (self.ARM_DOF,), \
            f"arm_angles should have {self.ARM_DOF} elements"
        assert body_angles.shape == (self.BODY_DOF,), \
            f"body_angles should have {self.BODY_DOF} elements"

        angles_12 = np.concatenate([arm_angles, body_angles]).astype(np.float32)
        if side == "right":
            flat = self._kin.calc_right_arm_pos_in_base(angles_12)
        elif side == "left":
            flat = self._kin.calc_left_arm_pos_in_base(angles_12)
        else:
            raise ValueError("side must be 'left' or 'right'")
        return np.array(flat, dtype=np.float64).reshape(4, 4, order="F")

    def ik(self, target_pose: np.ndarray, body_angles: np.ndarray,
           init_arm_angles: np.ndarray, side: str = "right"):
        target_pose = np.asarray(target_pose, dtype=np.float32)
        if target_pose.shape == (4, 4):
            target_flat = target_pose.flatten(order="F")
        elif target_pose.size == 16:
            target_flat = target_pose.ravel()
        else:
            raise ValueError("target_pose must be (4,4) or (16,)")

        body_angles = np.asarray(body_angles, dtype=np.float32).ravel()
        init_arm_angles = np.asarray(init_arm_angles, dtype=np.float32).ravel()
        assert body_angles.shape == (self.BODY_DOF,)
        assert init_arm_angles.shape == (self.ARM_DOF,)

        success, angles = self._kin.calc_single_arm_ik(
            target_flat, body_angles, init_arm_angles, side)
        return bool(success), np.array(angles, dtype=np.float64)

    def arm_base_pose(self, body_angles: np.ndarray,
                      side: str = "right") -> np.ndarray:
        body_angles = np.asarray(body_angles, dtype=np.float32).ravel()
        assert body_angles.shape == (self.BODY_DOF,)
        flat = self._kin.calc_arm_base_pos_in_base(body_angles, side)
        return np.array(flat, dtype=np.float64).reshape(4, 4, order="F")


# =========================================================
# Fingertip FK (hand joints -> 5 fingertip poses in wrist frame)
# =========================================================
def _resolve_glove_dir() -> Path:
    """Locate the ``glove_retargeting`` Python sources at runtime.

    Resolution order (first existing path wins):
      1. ``${HUMAN_POLICY_GLOVE_DIR}``  -- explicit env-var override
      2. ``<repo>/third_party/glove_retargeting``  -- vendored snapshot
         (recommended; created by ``third_party/glove_retargeting/``
         PROVENANCE.txt; this is what fresh checkouts get out of the box)
         -- legacy dev-machine fallback (kept so existing dev setups keep
         working unchanged)

    The returned path is also added to ``sys.path`` once so subsequent
    ``import glove_fk_solver`` etc. work without per-class hacks.
    """
    candidates: list[Path] = []
    env_override = os.environ.get("HUMAN_POLICY_GLOVE_DIR", "").strip()
    if env_override:
        candidates.append(Path(env_override))
    repo_root = Path(__file__).resolve().parents[1]
    candidates.append(repo_root / "third_party" / "glove_retargeting")

    for p in candidates:
        # require at least the optimizer file -- so a half-vendored or
        # accidentally empty dir does not silently win the race.
        if p.is_dir() and (p / "hand_retargeting_optimizer.py").is_file():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return p

    # No usable copy found; return the vendored path so error messages
    # below point users at the right place to fix it.
    return repo_root / "third_party" / "glove_retargeting"


_GLOVE_DIR: Path = _resolve_glove_dir()
_FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]


class FingertipFK:
    """Wrap ``GloveFKSolver`` so we can call it per-step at control rate.

    Same code path as ``scripts/add_fingertip_xyz_wrist.py`` so the
    fingertip representation seen at deploy time matches what the trainer
    wrote to ``fingertip_pose_wrist`` in zarr.
    """

    def __init__(self, hand_type: str = "right") -> None:
        # ``_resolve_glove_dir`` already handled sys.path; just import.
        try:
            from glove_fk_solver import GloveFKSolver  # noqa: E402
        except ImportError as e:
            raise ImportError(
                f"FingertipFK could not import GloveFKSolver from "
                f"{_GLOVE_DIR}.  Either vendor `glove_retargeting` into "
                f"`third_party/` (see PROVENANCE.txt) or set "
                f"`HUMAN_POLICY_GLOVE_DIR` to a usable copy.  "
                f"Original error: {e}"
            ) from e
        self._solver = GloveFKSolver(hand_type=hand_type)

    def __call__(self, hand_joint_angles: np.ndarray) -> np.ndarray:
        """``(22,)`` joint angles -> ``(5, 6)`` ``[xyz, rotvec]`` in wrist frame."""
        q = np.asarray(hand_joint_angles, dtype=np.float32).reshape(-1)
        if q.size != 22:
            raise ValueError(f"hand_joint_angles must be 22-D, got {q.shape}")
        ja = self._solver.parse_joint_angles(q)
        tips = self._solver.hand_fk(ja, return_all_joints=False)
        out = np.zeros((5, 6), dtype=np.float32)
        for i, f in enumerate(_FINGER_ORDER):
            p = np.asarray(tips[f], dtype=np.float64)
            pos = p[:3]
            quat_xyzw = np.array([p[4], p[5], p[6], p[3]])
            rotvec = R.from_quat(quat_xyzw).as_rotvec()
            out[i, :3] = pos
            out[i, 3:] = rotvec
        return out


# =========================================================
# Fingertip IK (5 fingertip poses in wrist frame -> 22 hand joints)
# =========================================================
# Tip indices inside the 25-keypoint layout that ``HandRetargetingOptimizer``
# consumes (see KEYPOINT_MAPPING in ``ego/hand_retargeting_node.py``).
_RAW_TIP_IDX = [4, 9, 14, 19, 24]


class FingertipToJointIK:
    """Sync wrapper around ``HandRetargetingOptimizer`` (ego mode).

    Inputs a ``(5, 6)`` fingertip pose ``[xyz, rotvec]`` in the wrist
    (HandBase) frame -- exactly what the policy emits when
    ``hand_action_mode='fingertip'`` -- and returns 22-D HA4 hand joint
    angles ready to send to ``/action/right_hand/joint_angle``.

    Underlying optimizer:
        - ``HandRetargetingOptimizer(..., mode='ego')`` skips the
          ``SurjectionMapper`` (which is teleop-only).
        - The CasADi NLP is built lazily on the first call; warm-start
          after that.  Benchmark (see ``scripts/bench_hand_retargeting.py``)
          shows ~4 ms / 245 Hz on our env, more than enough
          headroom for the 5-30 Hz control rate.

    Important caveat: the optimizer's ``finger_ori_loss`` term needs each
    finger's root position (``raw_finger_roots = [1, 6, 11, 16, 21]``),
    which the policy does not predict.  We leave those slots at zero and
    rely on the *tip-only* terms (``tip_pos_loss``, ``tip_ori_loss``,
    ``pinch_loss``, ``pinch_ori_loss``) -- 4 out of 5 cost terms are
    fully informed, and the benchmark confirms convergence is unchanged.
    """

    def __init__(self, hand_type: str = "right") -> None:
        try:
            from hand_kinematic_casadi import (  # noqa: E402
                HandKinematicCasadi,
                get_ha4_l_config,
                get_ha4_r_config,
            )
            from hand_retargeting_optimizer import (  # noqa: E402
                HandRetargetingOptimizer,
            )
        except ImportError as e:
            raise ImportError(
                f"FingertipToJointIK could not import HandKinematicCasadi / "
                f"HandRetargetingOptimizer from {_GLOVE_DIR}.  Either "
                f"vendor `glove_retargeting` into `third_party/` (see "
                f"PROVENANCE.txt) or set `HUMAN_POLICY_GLOVE_DIR` to a "
                f"usable copy.  Also confirm 'casadi' / 'rich' / "
                f"'transforms3d' are installed.  Original error: {e}"
            ) from e

        cfg = get_ha4_r_config() if hand_type == "right" else get_ha4_l_config()
        hand_model = HandKinematicCasadi(
            urdf_path=cfg["urdf_path"],
            base_link=cfg["base_link"],
            joint_names=cfg["joint_names"],
            keypoint_links=cfg["keypoint_links"],
            keypoint_offsets=cfg["keypoint_offsets"],
        )
        self._opt = HandRetargetingOptimizer(
            hand_model, hand_type=hand_type, mode="ego"
        )
        self._n_dof = int(hand_model.get_n_dof())
        self._last_q = np.zeros(self._n_dof, dtype=np.float64)
        self._last_kp = np.zeros((25, 7), dtype=np.float64)
        # The first ``optimize_with_casadi`` call compiles the symbolic
        # NLP (~150 ms one-shot); warm it up here so the rollout loop sees
        # only steady-state cost.
        warm_kp = np.zeros((25, 7), dtype=np.float64)
        warm_kp[0] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        try:
            self._opt.optimize_with_casadi(
                initial_q=self._last_q,
                raw_keypoints=warm_kp,
                keypoints_last=self._last_kp,
                raw_joint_angles=np.zeros(self._n_dof, dtype=np.float64),
            )
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] FingertipToJointIK warm-up failed ({e}); "
                  "the first real call will compile the NLP (~150 ms).")

    @staticmethod
    def _tip6_to_keypoint7(tip6: np.ndarray) -> np.ndarray:
        """``[x, y, z, rx, ry, rz]`` -> ``[x, y, z, qw, qx, qy, qz]``.

        The optimizer's raw_keypoints layout is scalar-first quaternion
        (see ``finger_fk`` in ``glove_fk_solver.py``: "[x, y, z, qw, qx,
        qy, qz]").  scipy's ``Rotation.as_quat`` returns scalar-last, so
        we reorder explicitly.
        """
        kp = np.zeros(7, dtype=np.float64)
        kp[:3] = tip6[:3]
        quat_xyzw = R.from_rotvec(np.asarray(tip6[3:], dtype=np.float64)).as_quat()
        kp[3] = quat_xyzw[3]  # qw
        kp[4:7] = quat_xyzw[:3]  # qx, qy, qz
        return kp

    def __call__(self, fingertip_5x6: np.ndarray) -> np.ndarray:
        """Run a single CasADi optimisation step.

        Parameters
        ----------
        fingertip_5x6 : (5, 6) array  [thumb, index, middle, ring, pinky]
            Each row is ``[x, y, z, rx, ry, rz]`` in the wrist (HandBase)
            frame, matching the layout written by
            ``add_fingertip_xyz_wrist.py``.

        Returns
        -------
        q : (22,) float32 HA4 joint angles (filtered/clipped by IPOPT
            bounds).
        """
        ft = np.asarray(fingertip_5x6, dtype=np.float64).reshape(5, 6)
        kp = np.zeros((25, 7), dtype=np.float64)
        # Wrist anchor: identity pose at the origin so the
        # ``tip_pos_loss`` (which uses ``tip - wrist``) is well defined.
        kp[0] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        for slot, tip6 in zip(_RAW_TIP_IDX, ft):
            kp[slot] = self._tip6_to_keypoint7(tip6)

        q = self._opt.optimize_with_casadi(
            initial_q=self._last_q,
            raw_keypoints=kp,
            keypoints_last=self._last_kp,
            raw_joint_angles=np.zeros(self._n_dof, dtype=np.float64),
        )
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        self._last_q = q.copy()
        self._last_kp = kp
        return q.astype(np.float32)


# =========================================================
# Relative policy -> absolute target pose
# =========================================================
def recover_absolute_target_pose_from_relative(
    current_pose6d: np.ndarray,
    relative_pose6d: np.ndarray,
    T_ET: np.ndarray,
) -> np.ndarray:
    """``T_BN = T_BE @ T_ET @ T_rel @ inv(T_ET)``.

    With ``T_ET = I`` (the method-B default) this is just
    ``T_BN = T_BE @ T_rel``.  See ``parse_t_et_arg`` for semantics.
    """
    T_BE = vec6dof_to_homogeneous_matrix(current_pose6d[:3], current_pose6d[3:])
    T_rel = vec6dof_to_homogeneous_matrix(relative_pose6d[:3], relative_pose6d[3:])
    T_BN = T_BE @ T_ET @ T_rel @ invert_transformation(T_ET)
    return homogeneous_matrix_to_6dof(T_BN)


def get_current_wrist_pose6d(
    obs: Dict[str, Any],
    ik_solver: Optional[LightweightIK],
    side: str = "right",
) -> np.ndarray:
    """FK the measured joints of the ``side`` ('left'/'right') arm -> current
    wrist pose6d (xyz + rotvec) in the robot base frame."""
    if ik_solver is None:
        raise RuntimeError(
            f"Cannot derive {side} wrist pose: no IK solver provided.  "
            "Pass --urdf_path so LightweightIK can be constructed."
        )
    T_wrist = ik_solver.fk(
        arm_angles=obs[f"{side}_arm"][::-1],  # sensor: [AJ1..AJ7] -> FK needs [AJ7..AJ1]
        body_angles=obs["body_dof"][::-1],    # sensor: [LBJ1..LBJ5] -> FK needs [LBJ5..LBJ1]
        side=side,
    )
    return homogeneous_matrix_to_6dof(T_wrist).astype(np.float32)


def get_current_right_wrist_pose6d(
    obs: Dict[str, Any], ik_solver: Optional[LightweightIK]
) -> np.ndarray:
    """Single-arm compatibility wrapper (right arm)."""
    return get_current_wrist_pose6d(obs, ik_solver, side="right")


def reset_to_episode_start(
    robot_env: RobotEnv,
    ik_solver: LightweightIK,
    zarr_root: str,
    episode: str,
    control_hz: float,
    hold_steps: int = 20,
    ik_pos_tol: float = 0.02,
    ik_rot_tol: float = 0.1,
    mode: str = "right",
) -> None:
    """Reset the robot to the first frame of a recorded episode.

    Reads ``pose[0]`` (xyz + rotvec, robot base frame) and
    ``proprioception[0]`` (22 right-hand joint angles) from
    ``<zarr_root>/<episode>``, solves right-arm IK for that EEF pose, and
    streams the {arm, hand} target for ``hold_steps`` control cycles so the
    robot has time to settle before the rollout starts.
    """
    import zarr

    g = zarr.open_group(os.path.join(zarr_root, episode), mode="r")
    pose0 = np.asarray(g["pose"])[0].astype(np.float64)            # (6,)
    hand0 = np.asarray(g["proprioception"])[0].astype(np.float32)  # (22,)
    print(
        f"[reset2zero] target from {episode}: "
        f"pose0={np.array2string(pose0, precision=3)}"
    )

    obs = get_latest_robot_obs(robot_env)
    target4x4 = vec6dof_to_homogeneous_matrix(pose0[:3], pose0[3:])
    ik_success, ik_angles = ik_solver.ik(
        target_pose=target4x4,
        body_angles=obs["body_dof"][::-1],
        init_arm_angles=obs["right_arm"][::-1],
        side="right",
    )

    ik_pos_err = ik_rot_err = float("inf")
    if ik_success:
        T_reached = ik_solver.fk(
            arm_angles=ik_angles, body_angles=obs["body_dof"][::-1], side="right",
        )
        ik_pos_err = float(np.linalg.norm(T_reached[:3, 3] - target4x4[:3, 3]))
        ik_rot_err = float(np.linalg.norm(T_reached[:3, :3] - target4x4[:3, :3]))
    if not (ik_success and ik_pos_err <= ik_pos_tol and ik_rot_err <= ik_rot_tol):
        raise RuntimeError(
            f"[reset2zero] IK for the episode start pose failed "
            f"(success={ik_success}, pos_err={ik_pos_err * 1000:.1f}mm, "
            f"rot_err={ik_rot_err:.3f}); refusing to send an unreliable reset "
            "target.  Check that the episode pose is reachable from the "
            "current configuration."
        )

    target_right_arm = ik_angles[::-1].astype(np.float32)
    print(
        f"[reset2zero] IK ok (pos_err={ik_pos_err * 1000:.2f}mm "
        f"rot_err={ik_rot_err:.4f}); holding for {hold_steps} steps to settle."
    )

    dt = 1.0 / control_hz
    for _ in range(max(1, hold_steps)):
        obs = get_latest_robot_obs(robot_env)
        action_buffer = build_robot_action_buffer(
            obs=obs,
            target_right_arm=target_right_arm,
            target_right_hand=hand0,
            mode=mode,
        )
        robot_env.send_action(action_buffer, immediate=True)
        time.sleep(dt)
    print("[reset2zero] reset complete; starting rollout.")


# =========================================================
# hand action 处理
# =========================================================
def convert_hand_action_joint(
    hand_action: np.ndarray,
    obs: Dict[str, Any],
    anchor_right_hand: Optional[np.ndarray],
    relative_hand_action: bool,
    clip_hand: bool = False,
    hand_lower: Optional[np.ndarray] = None,
    hand_upper: Optional[np.ndarray] = None,
    measured_hand: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Joint-angle hand action -> 22-D target hand joint angles.

    ``relative_hand_action`` is the deprecated joint-delta encoding kept for
    legacy checkpoints.  IMPORTANT: in training (``uvta_dataset.__getitem__``)
    the relative joint target is ``rel[k] = abs[k] - abs[anchor]`` where the
    anchor is the FIRST frame of the predicted chunk (frame t).  So every
    frame in a chunk is a delta from the SAME fixed anchor -- NOT a step-to-step
    increment.  The deploy code therefore must reconstruct the absolute target
    as ``anchor_hand + rel[k]`` with ``anchor_hand`` latched ONCE per chunk
    (mirroring how ``anchor_right_wrist_pose6d`` anchors the relative eef pose).

    The previous implementation accumulated (``virtual += hand_action`` each
    step), which double-counted the anchor offset and made the fingers drift.

    ``anchor_right_hand`` is the latched per-chunk anchor: pass ``None`` on the
    first call of a new chunk to latch it from the current observation, and
    reuse the returned value for the remaining frames of that chunk.  Method B
    (absolute joints) ignores the anchor entirely.
    """
    if relative_hand_action:
        if anchor_right_hand is None:
            # New chunk: latch the measured hand pose at predict time as the
            # fixed anchor that every rel[k] in this chunk is relative to.
            # ``measured_hand`` lets the bimanual caller latch the correct
            # arm's hand (defaults to the right hand for single-arm callers).
            base_hand = measured_hand if measured_hand is not None else obs["right_hand"]
            anchor_right_hand = np.asarray(base_hand, dtype=np.float32).copy()
        target_right_hand = anchor_right_hand + hand_action
    else:
        target_right_hand = hand_action.astype(np.float32)

    if clip_hand:
        if hand_lower is None or hand_upper is None:
            raise ValueError("clip_hand=True requires --hand_lower and --hand_upper")
        target_right_hand = np.clip(target_right_hand, hand_lower, hand_upper)

    return target_right_hand.astype(np.float32), anchor_right_hand


def convert_hand_action_fingertip(
    hand_action_5x6: np.ndarray,
    ik: FingertipToJointIK,
    clip_hand: bool = False,
    hand_lower: Optional[np.ndarray] = None,
    hand_upper: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fingertip-pose hand action -> 22-D HA4 joint angles.

    Network output for the hand part (when ``hand_action_mode='fingertip'``)
    is 30-D = 5 fingertip xyz+rotvec poses in the right wrist frame.  We
    feed those to the CasADi optimizer to recover joint angles.

    ``clip_hand`` is applied to the *post-IK* joint angles, mirroring the
    joint-mode path.  In normal use it is unnecessary because IPOPT
    already clamps to the URDF joint limits, but we keep it as a safety
    net so users can dial in a tighter envelope.
    """
    if hand_action_5x6.size != 30:
        raise ValueError(
            f"fingertip hand action must be 30-D (5x6), got "
            f"shape {hand_action_5x6.shape}"
        )
    target_right_hand = ik(hand_action_5x6.reshape(5, 6))

    if clip_hand:
        if hand_lower is None or hand_upper is None:
            raise ValueError("clip_hand=True requires --hand_lower and --hand_upper")
        target_right_hand = np.clip(target_right_hand, hand_lower, hand_upper)

    return target_right_hand.astype(np.float32)


# =========================================================
# assemble the robot action buffer
# =========================================================
def build_robot_action_buffer(
    obs: Dict[str, Any],
    target_right_arm: np.ndarray,
    target_right_hand: np.ndarray,
    mode: str = "right",
) -> Dict[str, np.ndarray]:
    if mode != "right":
        raise NotImplementedError("Currently only mode='right' is wired up.")

    return {
        "/action/left_arm/joint_angle": obs["left_arm"].astype(np.float32),
        "/action/left_hand/joint_angle": obs["left_hand"].astype(np.float32),
        "/action/right_arm/joint_angle": target_right_arm.astype(np.float32),
        "/action/right_hand/joint_angle": target_right_hand.astype(np.float32),
        "/action/motor/joint_angle": np.concatenate(
            [obs["body_dof"], obs["neck_dof"]], axis=0
        ).astype(np.float32),
    }


def choose_visual_obs(obs: Dict[str, Any], camera_source: str) -> np.ndarray:
    """Pick a single ``(H, W, 3)`` RGB frame from the multi-camera observation."""
    if camera_source == "right_wrist":
        img = obs["right_wrist_img"]
    elif camera_source == "left_wrist":
        img = obs["left_wrist_img"]
    elif camera_source == "right_eye":
        img = obs["right_eye_img"]
    elif camera_source == "left_eye":
        img = obs["left_eye_img"]
    else:
        raise ValueError(f"Unknown camera_source: {camera_source}")

    if img is None:
        raise RuntimeError(f"{camera_source} image is empty; check observation key.")
    return ensure_rgb_hwc(img)


# =========================================================
# Bimanual (dual-arm) shared helpers
# =========================================================
# chips_teleop camera layout: camera_0 = RIGHT wrist, camera_2 = LEFT wrist,
# camera_1 = ego (unused).  Maps a training ``load_camera_ids`` entry to the
# live-obs key produced by ``get_latest_robot_obs``.
_CAMERA_ID_TO_OBS_KEY = {
    0: "right_wrist_img",
    2: "left_wrist_img",
    1: "left_eye_img",   # ego / head (unused in the bimanual wrist configs)
}
_OBS_KEY_TO_CAMERA_SOURCE = {
    "right_wrist_img": "right_wrist",
    "left_wrist_img": "left_wrist",
    "right_eye_img": "right_eye",
    "left_eye_img": "left_eye",
}


def infer_policy_camera_source(policy) -> str:
    """Infer the primary live camera stream from a policy's training config."""
    camera_ids = [int(c) for c in getattr(policy, "camera_ids", [])]
    if not camera_ids:
        raise ValueError(
            "policy has no camera_ids; cannot infer the deployment camera stream"
        )
    obs_key = _CAMERA_ID_TO_OBS_KEY.get(camera_ids[0])
    source = _OBS_KEY_TO_CAMERA_SOURCE.get(obs_key)
    if source is None:
        raise ValueError(
            f"camera id {camera_ids[0]} has no live camera-source mapping"
        )
    return source


def gather_visual_obs(obs: Dict[str, Any], camera_ids, fallback_source: str):
    """Return the visual observation matching the policy's ``camera_ids``.

    * single camera  -> a single ``(H, W, 3)`` RGB frame (legacy behaviour;
      its camera ID selects the matching live stream; ``fallback_source`` is
      used only when that ID has no known mapping).
    * multiple cameras -> a **list** of ``(H, W, 3)`` frames in ``camera_ids``
      order, ready to hand to ``RealPolicy.push_observation`` (which stacks
      them along the obs-horizon axis in the SAME order the trainer used).
      For chips_teleop that is ``[camera_0 = right wrist, camera_2 = left]``.
    """
    ids = [int(c) for c in camera_ids]
    if not ids:
        return choose_visual_obs(obs, fallback_source)
    frames = []
    for cid in ids:
        key = _CAMERA_ID_TO_OBS_KEY.get(cid)
        if key is None:
            if len(ids) == 1:
                return choose_visual_obs(obs, fallback_source)
            raise ValueError(
                f"camera id {cid} has no live-obs mapping; extend "
                "_CAMERA_ID_TO_OBS_KEY for your robot."
            )
        img = obs.get(key)
        if img is None:
            raise RuntimeError(f"{key} (camera_{cid}) image is empty.")
        frames.append(ensure_rgb_hwc(img))
    return frames[0] if len(frames) == 1 else frames


def _side_of_prefix(prefix: str) -> str:
    """Map an arm prefix ('left_' / 'right_' / '') to the IK side."""
    p = str(prefix).lower()
    if p.startswith("left"):
        return "left"
    if p.startswith("right"):
        return "right"
    return "right"  # single-arm default


def infer_policy_mode(policy) -> str:
    """Infer single-arm side or bimanual mode from ``dataset.arms``."""
    prefixes = list(getattr(policy, "arm_prefixes", [""]))
    if len(prefixes) > 1:
        return "bimanual"
    return _side_of_prefix(prefixes[0] if prefixes else "")


def build_robot_action_buffer_dual(
    obs: Dict[str, Any],
    per_arm: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Assemble a robot action buffer that commands one OR both arms.

    ``per_arm`` maps side ('left'/'right') -> (arm_angles(7), hand_angles(22)).
    Any arm NOT in ``per_arm`` is passed through from the current obs (held in
    place); ``motor`` (body+neck) is always passed through.
    """
    def arm_cmd(side):
        if side in per_arm:
            return np.asarray(per_arm[side][0], dtype=np.float32)
        return obs[f"{side}_arm"].astype(np.float32)

    def hand_cmd(side):
        if side in per_arm:
            return np.asarray(per_arm[side][1], dtype=np.float32)
        return obs[f"{side}_hand"].astype(np.float32)

    return {
        "/action/left_arm/joint_angle": arm_cmd("left"),
        "/action/left_hand/joint_angle": hand_cmd("left"),
        "/action/right_arm/joint_angle": arm_cmd("right"),
        "/action/right_hand/joint_angle": hand_cmd("right"),
        "/action/motor/joint_angle": np.concatenate(
            [obs["body_dof"], obs["neck_dof"]], axis=0
        ).astype(np.float32),
    }


def _build_arm_proprio_inputs(
    obs: Dict[str, Any],
    proprio_mode: str,
    ik_solver: LightweightIK,
    side: str,
) -> Dict[str, Optional[np.ndarray]]:
    """Per-arm analogue of ``_build_per_step_proprio_inputs`` (one ``side``).

    Fingertip proprio is intentionally unsupported for bimanual (it would need
    a per-hand ``FingertipFK``); raise a clear error instead of guessing.
    """
    out = {"joint_proprio": None, "wrist_pose6": None, "fingertip_pose_wrist": None}
    if proprio_mode in ("joint", "both"):
        out["joint_proprio"] = obs[f"{side}_hand"].astype(np.float32)
    if proprio_mode in ("ee_rel", "both"):
        out["wrist_pose6"] = get_current_wrist_pose6d(obs, ik_solver, side=side)
    if proprio_mode in ("fingertip", "fingertip_with_ee"):
        raise NotImplementedError(
            f"proprio_mode={proprio_mode!r} (fingertip) is not wired for the "
            "bimanual rollout -- it needs a per-hand FingertipFK.  Retrain with "
            "proprio_mode in {none, joint, ee_rel, both} or extend "
            "_build_arm_proprio_inputs."
        )
    return out


def push_bimanual_observation(
    policy,
    obs: Dict[str, Any],
    ik_solver: LightweightIK,
    fallback_camera_source: str,
    uses_tactile: bool,
    tactile_key: str,
) -> np.ndarray:
    """Gather multi-camera + per-arm proprio/tactile from the live obs and push
    them to a bimanual ``RealPolicy``.  Returns the primary (first-camera) RGB
    frame for debug/display.

    Camera frames are passed as a list in ``policy.camera_ids`` order and
    per-arm proprio/tactile as lists in ``policy.arm_prefixes`` order -- exactly
    what ``RealPolicy.push_observation`` expects for the bimanual case.
    """
    frames = gather_visual_obs(obs, policy.camera_ids, fallback_camera_source)
    frame_list = frames if isinstance(frames, list) else [frames]

    proprio_mode = policy.proprio_mode
    need_joint = proprio_mode in ("joint", "both")
    need_pose = proprio_mode in ("ee_rel", "both")
    joint_list, pose_list, fsr_list = [], [], []
    for p in policy.arm_prefixes:
        side = _side_of_prefix(p)
        pin = _build_arm_proprio_inputs(obs, proprio_mode, ik_solver, side)
        if need_joint:
            joint_list.append(pin["joint_proprio"])
        if need_pose:
            pose_list.append(pin["wrist_pose6"])
        if uses_tactile:
            fsr_list.append(build_tactile_from_obs(obs, tactile_key, hand=side))

    policy.push_observation(
        visual_obs=frame_list,
        joint_proprio=joint_list if need_joint else None,
        wrist_pose6=pose_list if need_pose else None,
        fingertip_pose_wrist=None,
        fsr=fsr_list if uses_tactile else None,
    )
    return frame_list[0]


def decode_bimanual_action_step(act, arm_prefixes, hand_action_mode):
    """Split ONE decoded bimanual action frame into per-arm parts.

    ``act`` = the per-arm legacy blocks concatenated in ``arm_prefixes`` order
    (as produced by ``RealPolicy._decode_action_to_legacy``).  Each block is
    ``[xyz(3), rotvec(3), hand(...)]`` for eef+hand modes, or the whole block
    is the hand vector for the no-eef hand modes (joint_only/fingertip_only).

    Returns a list of dicts (arm order): ``{side, prefix, relative_pose6d(6),
    hand_action}``.
    """
    num_arms = len(arm_prefixes)
    act = np.asarray(act, dtype=np.float32).reshape(-1)
    if num_arms <= 0 or act.shape[0] % num_arms != 0:
        raise ValueError(
            f"decoded action dim {act.shape[0]} not divisible by "
            f"num_arms={num_arms}"
        )
    per = act.shape[0] // num_arms
    no_eef = hand_action_mode in ("joint_only", "fingertip_only")
    parts = []
    for a, p in enumerate(arm_prefixes):
        blk = act[a * per:(a + 1) * per]
        if no_eef:
            rel = np.zeros(6, dtype=np.float32)
            hand = blk
        else:
            rel = blk[:6]
            hand = blk[6:]
        parts.append(
            {
                "side": _side_of_prefix(p),
                "prefix": p,
                "relative_pose6d": rel.astype(np.float32),
                "hand_action": hand.astype(np.float32),
            }
        )
    return parts


def solve_arm_ik(
    ik_solver: LightweightIK,
    obs: Dict[str, Any],
    target_wrist_pose6d: np.ndarray,
    side: str,
    ik_pos_tol: float,
    ik_rot_tol: float,
    init_arm_sensor_order: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, bool, float, float]:
    """IK a target wrist pose6d for ``side`` with FK-residual validation.

    Mirrors the single-arm accept/reject logic (``calc_single_arm_ik``'s
    success flag alone is unreliable).  Returns ``(target_arm_angles(7, sensor
    order), ik_ok, pos_err_m, rot_err)``; on reject the current arm angles are
    returned so the caller holds position.  ``init_arm_sensor_order`` lets a
    timestamped rollout warm-start from the preceding smoothed command, which
    avoids switching IK branches when multiple solutions are nearby.
    """
    target4x4 = vec6dof_to_homogeneous_matrix(
        translation=target_wrist_pose6d[:3],
        rotation_vector=target_wrist_pose6d[3:],
    )
    ik_seed = (
        np.asarray(init_arm_sensor_order, dtype=np.float32)
        if init_arm_sensor_order is not None
        else np.asarray(obs[f"{side}_arm"], dtype=np.float32)
    )
    ik_success, ik_angles = ik_solver.ik(
        target_pose=target4x4,
        body_angles=obs["body_dof"][::-1],
        init_arm_angles=ik_seed[::-1],
        side=side,
    )
    pos_err = rot_err = float("inf")
    if ik_success:
        T_reached = ik_solver.fk(
            arm_angles=ik_angles, body_angles=obs["body_dof"][::-1], side=side,
        )
        pos_err = float(np.linalg.norm(T_reached[:3, 3] - target4x4[:3, 3]))
        rot_err = float(np.linalg.norm(T_reached[:3, :3] - target4x4[:3, :3]))
    ik_ok = ik_success and pos_err <= ik_pos_tol and rot_err <= ik_rot_tol
    if ik_ok:
        target_arm = ik_angles[::-1].astype(np.float32)
    else:
        target_arm = obs[f"{side}_arm"].astype(np.float32)
    return target_arm, ik_ok, pos_err, rot_err


# =========================================================
# Debug helpers
# =========================================================
def rebuild_debug_video(imgs_dir: str, video_path: str, fps: float = 5.0):
    """Stitch all PNG frames in ``imgs_dir`` into a single mp4.

    Robust against interruption: the ``VideoWriter`` is always released in a
    ``finally`` so the MP4 ``moov`` atom (the trailing index) gets written --
    without it the file has "moov atom not found" and no player can open it.
    The final file is only swapped into place after a successful release, and
    the temp file is cleaned up on any failure so we never leave a dangling
    ``*.tmp.mp4``.
    """
    img_files = sorted(
        [f for f in os.listdir(imgs_dir) if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]),
    )
    if not img_files:
        return
    first = cv2.imread(os.path.join(imgs_dir, img_files[0]))
    if first is None:
        print(f"[DEBUG] video rebuild skipped: cannot read {img_files[0]}")
        return
    h, w = first.shape[:2]
    tmp_path = video_path + ".tmp.mp4"

    # Prefer H.264 (avc1) for broad player compatibility; fall back to mp4v.
    writer = None
    for codec in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
        if writer.isOpened():
            break
        writer.release()
        writer = None
    if writer is None or not writer.isOpened():
        print(f"[DEBUG] video rebuild failed: could not open writer for {tmp_path}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return

    try:
        for fname in img_files:
            frame = cv2.imread(os.path.join(imgs_dir, fname))
            if frame is None:
                continue
            if frame.shape[0] != h or frame.shape[1] != w:
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
    finally:
        # ALWAYS release so the moov atom is flushed even on error/interrupt.
        writer.release()

    if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
        os.replace(tmp_path, video_path)
        print(f"[DEBUG] Video updated: {video_path} ({len(img_files)} frames)")
    else:
        print(f"[DEBUG] video rebuild produced no data: {tmp_path}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _can_imshow() -> bool:
    """Heuristic to decide if cv2.imshow will actually have a display."""
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return True
    if sys.platform == "darwin":
        return True
    return False


class LiveDebugViz:
    """Real-time matplotlib dashboard for debugging a rollout.

    Three panels, refreshed every control step:

    * **Wrist image** -- exactly the (RGB) frame the policy is consuming, so
      you can correlate motion with what the camera sees.
    * **Action chunk 3D** -- the *whole* current chunk decoded to absolute
      wrist XYZ (anchor + every ``act[:6]`` relative pose) is drawn as a 3D
      trajectory (solid), overlaid with the pose actually reachable after
      solving arm IK and FK-ing it back (dashed), with thin grey segments
      linking each target to its IK-reached point so the spatial error is
      visible.  The anchor and currently-executing frame are marked, and the
      title reports the per-chunk mean/max IK position error (mm).  On top of
      the predicted chunk it also overlays the *real-time* executed wrist path
      (red, FK of the angles actually sent each step) and the measured wrist
      path (purple), so predicted vs realised motion can be compared live.
    * **Tracking history** -- rolling per-step curves of the commanded target
      wrist position, the IK-reached position (FK of the IK solution) and the
      live measured wrist position, one line per axis, so you can see if IK is
      faithfully following the policy targets or silently diverging.
    * **Tactile / force** -- rolling per-step curves of the tactile vector
      actually fed to the policy.  For ``tactile_key='force'`` that is the
      per-finger resultant force (one labelled line per finger); for the wider
      ``fsr``/``fsr_region`` streams it degenerates to sum/max over channels.
      Lets you see whether contact is really registering during a rollout.
    """

    def __init__(self, history: int = 200):
        import matplotlib

        # Use whatever interactive backend is configured; fall back silently.
        import matplotlib.pyplot as plt  # noqa: F401

        self._plt = plt
        self._ok = True
        self.history = int(history)
        try:
            plt.ion()
            # mpl_toolkits is needed for the 3D middle panel.
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            self.fig = plt.figure(figsize=(21, 5))
            # 1: wrist image (2D), 2: action chunk (3D), 3: IK tracking (2D),
            # 4: tactile/force input (2D)
            self.ax_img = self.fig.add_subplot(1, 4, 1)
            self.ax_chunk = self.fig.add_subplot(1, 4, 2, projection="3d")
            self.ax_track = self.fig.add_subplot(1, 4, 3)
            self.ax_force = self.fig.add_subplot(1, 4, 4)
            self.fig.canvas.manager.set_window_title(
                "rollout live debug (image | action chunk 3D | IK tracking | force)"
            )
        except Exception as e:  # noqa: BLE001
            print(f"[LIVE] failed to create matplotlib window: {e}")
            self._ok = False
            return

        self._img_artist = None

        # rolling tracking history (deques keep memory bounded)
        from collections import deque

        self._steps = deque(maxlen=self.history)
        self._target_xyz = deque(maxlen=self.history)   # commanded target
        self._reached_xyz = deque(maxlen=self.history)  # FK(ik solution)
        self._live_xyz = deque(maxlen=self.history)     # measured wrist
        self._force = deque(maxlen=self.history)        # tactile fed to policy

    @property
    def ok(self) -> bool:
        return self._ok

    def _decode_chunk_xyz(
        self, output, anchor_pose6d, anchor_offset, t_et
    ) -> np.ndarray:
        """Absolute wrist XYZ for every frame of the current chunk."""
        pts = []
        for k in range(anchor_offset, len(output)):
            rel = output[k][:6].astype(np.float32)
            abs6 = recover_absolute_target_pose_from_relative(
                current_pose6d=anchor_pose6d,
                relative_pose6d=rel,
                T_ET=t_et,
            )
            pts.append(abs6[:3])
        if not pts:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(pts, dtype=np.float32)

    def _decode_chunk_target_and_ik(
        self, output, anchor_pose6d, anchor_offset, t_et, ik_solver, obs
    ):
        """Per-frame chunk targets vs the pose actually reached after IK->FK.

        For every frame of the chunk we recover the absolute target wrist
        pose, solve arm IK for it and FK the solution back to a wrist pose.
        Returns ``(target_xyz, ik_xyz, pos_err_mm)`` so the caller can overlay
        the commanded and reachable paths and read off the IK position error.
        """
        target_pts = []
        ik_pts = []
        errs_mm = []
        # warm-start IK from the live arm angles, then chain frame to frame so
        # successive solutions stay continuous (mirrors the real rollout).
        init_arm = np.asarray(obs["right_arm"], dtype=np.float64)[::-1]
        body = np.asarray(obs["body_dof"], dtype=np.float64)[::-1]
        for k in range(anchor_offset, len(output)):
            rel = output[k][:6].astype(np.float32)
            tgt6 = recover_absolute_target_pose_from_relative(
                current_pose6d=anchor_pose6d,
                relative_pose6d=rel,
                T_ET=t_et,
            )
            target_pts.append(tgt6[:3])
            tgt4x4 = vec6dof_to_homogeneous_matrix(
                translation=tgt6[:3], rotation_vector=tgt6[3:]
            )
            try:
                ok, ang = ik_solver.ik(
                    target_pose=tgt4x4,
                    body_angles=body,
                    init_arm_angles=init_arm,
                    side="right",
                )
                if ok:
                    T_reached = ik_solver.fk(
                        arm_angles=ang, body_angles=body, side="right"
                    )
                    ik_xyz = T_reached[:3, 3].astype(np.float32)
                    init_arm = np.asarray(ang, dtype=np.float64)  # chain
                else:
                    ik_xyz = np.full(3, np.nan, dtype=np.float32)
            except Exception:  # noqa: BLE001
                ik_xyz = np.full(3, np.nan, dtype=np.float32)
            ik_pts.append(ik_xyz)
            errs_mm.append(
                float(np.linalg.norm(ik_xyz - tgt6[:3]) * 1000.0)
            )
        if not target_pts:
            empty = np.zeros((0, 3), dtype=np.float32)
            return empty, empty, np.zeros((0,), dtype=np.float32)
        return (
            np.asarray(target_pts, dtype=np.float32),
            np.asarray(ik_pts, dtype=np.float32),
            np.asarray(errs_mm, dtype=np.float32),
        )

    def _style_3d(self, pts_a, pts_b):
        """Label axes and force an equal aspect ratio on the 3D chunk panel.

        Equal aspect keeps the trajectory's true spatial shape (otherwise mpl
        stretches each axis independently and a small IK error can look huge or
        a real curve can look flat).
        """
        ax = self.ax_chunk
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        stacks = [p for p in (pts_a, pts_b) if p is not None and len(p) > 0]
        if not stacks:
            return
        allp = np.concatenate(stacks, axis=0)
        finite = allp[np.isfinite(allp).all(axis=1)]
        if len(finite) == 0:
            return
        mins = finite.min(axis=0)
        maxs = finite.max(axis=0)
        center = (mins + maxs) / 2.0
        # half-range with a small floor so a near-static chunk is not zoomed in
        # to numerical noise.
        half = max(float((maxs - mins).max()) / 2.0, 0.02)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        try:
            ax.set_box_aspect((1, 1, 1))  # mpl >= 3.3
        except Exception:  # noqa: BLE001
            pass

    def _draw_realtime_hist(self, reached_hist, live_hist):
        """Overlay the real-time executed wrist path onto the 3D chunk panel.

        ``reached_hist`` is the rolling history of FK(sent arm angles) -- i.e.
        where the wrist actually went after IK each control step -- and
        ``live_hist`` the measured wrist pose.  Drawing them here lets you
        compare, in the same 3D view and in real time, the freshly predicted
        chunk against the path the arm has truly executed so far.  Returns the
        concatenated points so the caller can include them in axis autoscaling.
        """
        extras = []
        if reached_hist is not None and len(reached_hist) > 0:
            rh = reached_hist[np.isfinite(reached_hist).all(axis=1)]
            if len(rh) > 0:
                self.ax_chunk.plot(
                    rh[:, 0], rh[:, 1], rh[:, 2],
                    "-", color="tab:red", lw=1.4, alpha=0.9,
                    label="exec IK->FK (live)",
                )
                # highlight the most recent executed point
                self.ax_chunk.scatter(
                    [rh[-1, 0]], [rh[-1, 1]], [rh[-1, 2]],
                    s=40, color="tab:red", marker="*",
                )
                extras.append(rh)
        if live_hist is not None and len(live_hist) > 0:
            lh = live_hist[np.isfinite(live_hist).all(axis=1)]
            if len(lh) > 0:
                self.ax_chunk.plot(
                    lh[:, 0], lh[:, 1], lh[:, 2],
                    ":", color="tab:purple", lw=1.0, alpha=0.7,
                    label="measured (live)",
                )
                extras.append(lh)
        if not extras:
            return None
        return np.concatenate(extras, axis=0)

    def _draw_force(self, steps, tactile_key) -> None:
        """Panel 4: rolling curves of the tactile vector fed to the policy.

        ``force`` is 5-D (one resultant per finger) so every channel gets its
        own labelled line; the wider ``fsr``/``fsr_region`` streams are reduced
        to sum/max over channels because 20-100 lines are unreadable.
        """
        ax = self.ax_force
        ax.cla()
        dim = max((len(f) for f in self._force), default=0)
        if dim == 0:
            ax.set_title("tactile: policy has no tactile input")
            ax.axis("off")
            return
        # A width change mid-run would make the history ragged; keep the rows
        # matching the current width so np.asarray() cannot raise.
        mask = np.array([len(f) == dim for f in self._force], dtype=bool)
        force = np.asarray(
            [f for f, keep in zip(self._force, mask) if keep], dtype=np.float32
        )
        xs = np.asarray(steps)[mask]
        if len(force) == 0:
            return
        cur = force[-1]
        ax.set_title(
            f"tactile '{tactile_key}'  now: max={cur.max():.2f} "
            f"sum={cur.sum():.2f}"
        )
        if tactile_key == "force" and dim == len(_TACTILE_FINGERS):
            for i, name in enumerate(_TACTILE_FINGERS):
                ax.plot(xs, force[:, i], lw=1.3, label=name)
            ax.set_ylabel("per-finger resultant force")
            ax.legend(loc="upper left", fontsize=7, ncol=2)
        else:
            ax.plot(xs, force.sum(1), color="tab:blue", lw=1.4, label="sum")
            ax.plot(xs, force.max(1), color="tab:red", lw=1.1, alpha=0.8,
                    label="max")
            ax.set_ylabel(f"{tactile_key} ({dim}-D)")
            ax.legend(loc="upper left", fontsize=7)
        ax.set_xlabel("control step")
        ax.grid(True, alpha=0.3)

    def update(
        self,
        *,
        wrist_rgb,
        output,
        anchor_pose6d,
        anchor_offset,
        act_index,
        t_et,
        step,
        target_xyz,
        reached_xyz,
        live_xyz,
        ik_ok,
        ik_solver=None,
        obs=None,
        fsr=None,
        tactile_key="force",
    ) -> bool:
        """Redraw all panels.  Returns False if the window was closed."""
        if not self._ok:
            return False
        plt = self._plt
        try:
            # --- panel 1: wrist image -------------------------------------
            if self._img_artist is None:
                self.ax_img.set_title("wrist img (policy input, RGB)")
                self.ax_img.axis("off")
                self._img_artist = self.ax_img.imshow(wrist_rgb)
            else:
                self._img_artist.set_data(wrist_rgb)

            # Append this step to the rolling tracking history first, so both
            # the 3D chunk panel and the 2D tracking panel below can show the
            # latest real-time IK->FK / live wrist points.
            self._steps.append(int(step))
            self._target_xyz.append(np.asarray(target_xyz, dtype=np.float32))
            self._reached_xyz.append(np.asarray(reached_xyz, dtype=np.float32))
            self._live_xyz.append(np.asarray(live_xyz, dtype=np.float32))
            self._force.append(
                np.zeros(0, np.float32) if fsr is None
                else np.asarray(fsr, dtype=np.float32).ravel()
            )

            # --- panel 2: action chunk EEF path in 3D ---------------------
            # Draw the whole chunk's wrist target trajectory in 3D space, and
            # (when an IK solver is available) overlay the path actually
            # reachable after IK->FK, with thin grey segments connecting each
            # target to its IK-reached point so the spatial error is visible.
            # On top of that we overlay the *real-time* executed wrist path
            # (FK of the angles actually sent each control step) so you can
            # compare predicted vs realised motion live.
            self.ax_chunk.cla()
            cur = act_index - anchor_offset - 1
            # real-time executed (IK->FK) and measured wrist history in 3D
            reached_hist = np.asarray(self._reached_xyz, dtype=np.float32)
            live_hist = np.asarray(self._live_xyz, dtype=np.float32)
            if ik_solver is not None and obs is not None:
                tgt_xyz, ik_xyz, err_mm = self._decode_chunk_target_and_ik(
                    output, anchor_pose6d, anchor_offset, t_et, ik_solver, obs
                )
                if len(err_mm) > 0 and np.isfinite(err_mm).any():
                    finite = err_mm[np.isfinite(err_mm)]
                    err_str = (
                        f"IK err(mm) mean={finite.mean():.1f} "
                        f"max={finite.max():.1f}"
                    )
                else:
                    err_str = "IK err: n/a"
                self.ax_chunk.set_title(
                    f"action chunk 3D  (exec {act_index}/{len(output)})\n"
                    f"{err_str}"
                )
                if len(tgt_xyz) > 0:
                    # target trajectory (solid blue, dots per frame)
                    self.ax_chunk.plot(
                        tgt_xyz[:, 0], tgt_xyz[:, 1], tgt_xyz[:, 2],
                        "-o", ms=3, color="tab:blue", label="target",
                    )
                    # IK->FK trajectory (dashed orange, x per frame)
                    finite_mask = np.isfinite(ik_xyz).all(axis=1)
                    if finite_mask.any():
                        self.ax_chunk.plot(
                            ik_xyz[finite_mask, 0],
                            ik_xyz[finite_mask, 1],
                            ik_xyz[finite_mask, 2],
                            "--x", ms=4, color="tab:orange", label="IK->FK",
                        )
                    # error segments target<->IK for each frame
                    for k in range(len(tgt_xyz)):
                        if finite_mask[k]:
                            self.ax_chunk.plot(
                                [tgt_xyz[k, 0], ik_xyz[k, 0]],
                                [tgt_xyz[k, 1], ik_xyz[k, 1]],
                                [tgt_xyz[k, 2], ik_xyz[k, 2]],
                                "-", color="grey", lw=0.8, alpha=0.6,
                            )
                    # mark the anchor (chunk start) and the current frame
                    self.ax_chunk.scatter(
                        [tgt_xyz[0, 0]], [tgt_xyz[0, 1]], [tgt_xyz[0, 2]],
                        s=60, marker="^", color="green", label="anchor",
                    )
                    if 0 <= cur < len(tgt_xyz):
                        self.ax_chunk.scatter(
                            [tgt_xyz[cur, 0]], [tgt_xyz[cur, 1]],
                            [tgt_xyz[cur, 2]], s=110, facecolors="none",
                            edgecolors="red", linewidths=2,
                            label="current",
                        )
                    # real-time executed wrist path (FK of sent angles)
                    extra = self._draw_realtime_hist(reached_hist, live_hist)
                    self._style_3d(
                        np.concatenate([tgt_xyz, ik_xyz[finite_mask]], axis=0),
                        extra,
                    )
                    self.ax_chunk.legend(loc="upper left", fontsize=7)
            else:
                # No IK solver: just the target trajectory in 3D.
                chunk_xyz = self._decode_chunk_xyz(
                    output, anchor_pose6d, anchor_offset, t_et
                )
                self.ax_chunk.set_title(
                    f"action chunk 3D  (exec {act_index}/{len(output)})"
                )
                if len(chunk_xyz) > 0:
                    self.ax_chunk.plot(
                        chunk_xyz[:, 0], chunk_xyz[:, 1], chunk_xyz[:, 2],
                        "-o", ms=3, color="tab:blue", label="target",
                    )
                    self.ax_chunk.scatter(
                        [chunk_xyz[0, 0]], [chunk_xyz[0, 1]], [chunk_xyz[0, 2]],
                        s=60, marker="^", color="green", label="anchor",
                    )
                    if 0 <= cur < len(chunk_xyz):
                        self.ax_chunk.scatter(
                            [chunk_xyz[cur, 0]], [chunk_xyz[cur, 1]],
                            [chunk_xyz[cur, 2]], s=110, facecolors="none",
                            edgecolors="red", linewidths=2, label="current",
                        )
                    extra = self._draw_realtime_hist(reached_hist, live_hist)
                    self._style_3d(chunk_xyz, extra)
                    self.ax_chunk.legend(loc="upper left", fontsize=7)

            # --- panel 3: rolling tracking history ------------------------
            # (history already appended above so the 3D panel stays in sync)
            self.ax_track.cla()
            self.ax_track.set_title(
                f"IK tracking  (last step accepted={ik_ok})"
            )
            steps = np.asarray(self._steps)
            tgt = np.asarray(self._target_xyz)
            rch = np.asarray(self._reached_xyz)
            liv = np.asarray(self._live_xyz)
            colors = ["tab:blue", "tab:orange", "tab:green"]
            for ax_i, name in enumerate("xyz"):
                c = colors[ax_i]
                self.ax_track.plot(
                    steps, tgt[:, ax_i], "-", color=c, lw=1.6,
                    label=f"target {name}",
                )
                self.ax_track.plot(
                    steps, rch[:, ax_i], "--", color=c, lw=1.2,
                    label=f"IK {name}",
                )
                self.ax_track.plot(
                    steps, liv[:, ax_i], ":", color=c, lw=1.0, alpha=0.7,
                    label=f"live {name}",
                )
            self.ax_track.set_xlabel("control step")
            self.ax_track.set_ylabel("wrist xyz (m)")
            self.ax_track.legend(loc="upper left", fontsize=7, ncol=3)
            self.ax_track.grid(True, alpha=0.3)

            # --- panel 4: tactile / force actually fed to the policy -------
            self._draw_force(steps, tactile_key)

            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

            # window closed?
            if not plt.fignum_exists(self.fig.number):
                self._ok = False
                return False
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[LIVE] update failed, disabling live viz: {e}")
            self._ok = False
            return False

    def close(self):
        if not self._ok:
            return
        try:
            self._plt.close(self.fig)
        except Exception:  # noqa: BLE001
            pass


def _can_import_finger_ik() -> bool:
    """Probe whether ``FingertipToJointIK``'s deps are importable.

    We probe (without actually constructing the optimizer) so that a
    missing dep on a joint-mode checkpoint does not crash startup.
    sys.path has already been adjusted in ``_resolve_glove_dir``.
    """
    try:
        import casadi  # noqa: F401
        import hand_kinematic_casadi  # noqa: F401
        import hand_retargeting_optimizer  # noqa: F401
        return True
    except ImportError:
        return False


# =========================================================
# 主循环
# =========================================================
def _build_per_step_proprio_inputs(
    obs: Dict[str, Any],
    proprio_mode: str,
    ik_solver: LightweightIK,
    fingertip_fk: Optional[FingertipFK],
    legacy_joint_proprio: Optional[np.ndarray],
) -> Dict[str, Optional[np.ndarray]]:
    """Build the per-frame proprio kwargs that ``RealPolicy.push_observation``
    expects, based on ``policy.proprio_mode``.

    Returns a dict with keys ``joint_proprio`` / ``wrist_pose6`` /
    ``fingertip_pose_wrist``; any field not required by the mode is left
    ``None``.

    ``legacy_joint_proprio`` is honoured only when the policy explicitly
    needs joint angles; it lets users override which 22-D vector to feed
    if their observation keys differ.
    """
    out = {
        "joint_proprio": None,
        "wrist_pose6": None,
        "fingertip_pose_wrist": None,
    }
    if proprio_mode in ("joint", "both"):
        out["joint_proprio"] = (
            legacy_joint_proprio
            if legacy_joint_proprio is not None
            else obs["right_hand"].astype(np.float32)
        )
    if proprio_mode in ("ee_rel", "both", "fingertip_with_ee"):
        out["wrist_pose6"] = get_current_right_wrist_pose6d(obs, ik_solver)
    if proprio_mode in ("fingertip", "fingertip_with_ee"):
        if fingertip_fk is None:
            raise RuntimeError(
                f"proprio_mode={proprio_mode} needs fingertip_pose_wrist; "
                "FingertipFK is not initialised (it requires GloveFKSolver "
                "under third_party/)."
            )
        out["fingertip_pose_wrist"] = fingertip_fk(obs["right_hand"])
    # 'none' mode -> all three stay None and the policy uses vision-only condition.
    return out


def _decode_single_absolute_chunk(
    output: np.ndarray,
    waypoint_indices: np.ndarray,
    obs: Dict[str, Any],
    anchor_wrist_pose6d: np.ndarray,
    t_et: np.ndarray,
    hand_action_mode: str,
    relative_hand_action: bool,
    no_eef_policy: bool,
    eef_only_policy: bool,
    fingertip_ik: Optional[FingertipToJointIK],
    clip_hand: bool,
    hand_lower: Optional[np.ndarray],
    hand_upper: Optional[np.ndarray],
    held_wrist_pose6d: Optional[np.ndarray],
    held_hand: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Decode selected network frames into absolute wrist and hand waypoints."""
    output = np.asarray(output)
    indices = np.asarray(waypoint_indices, dtype=np.int64)
    if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= len(output)):
        raise ValueError("waypoint_indices must select at least one output frame")

    if no_eef_policy:
        if held_wrist_pose6d is None:
            held_wrist_pose6d = np.asarray(
                anchor_wrist_pose6d, dtype=np.float32
            ).copy()
        pose_chunk = np.repeat(
            held_wrist_pose6d[None], len(indices), axis=0
        )
    else:
        pose_chunk = np.stack(
            [
                recover_absolute_target_pose_from_relative(
                    current_pose6d=anchor_wrist_pose6d,
                    relative_pose6d=np.asarray(output[k, :6], dtype=np.float32),
                    T_ET=t_et,
                )
                for k in indices
            ],
            axis=0,
        )

    if eef_only_policy:
        if held_hand is None:
            held_hand = np.asarray(obs["right_hand"], dtype=np.float32).copy()
        hand_chunk = np.repeat(held_hand[None], len(indices), axis=0)
    elif hand_action_mode in ("fingertip", "fingertip_only"):
        if fingertip_ik is None:
            raise RuntimeError("Fingertip action chunk requires fingertip IK.")
        hand_chunk = np.stack(
            [
                convert_hand_action_fingertip(
                    hand_action_5x6=(
                        output[k] if no_eef_policy else output[k, 6:]
                    ).reshape(5, 6),
                    ik=fingertip_ik,
                    clip_hand=clip_hand,
                    hand_lower=hand_lower,
                    hand_upper=hand_upper,
                )
                for k in indices
            ],
            axis=0,
        )
    else:
        hand_anchor = (
            np.asarray(obs["right_hand"], dtype=np.float32).copy()
            if relative_hand_action else None
        )
        hand_rows = []
        for k in indices:
            hand_action = output[k] if no_eef_policy else output[k, 6:]
            hand_target, _ = convert_hand_action_joint(
                hand_action=np.asarray(hand_action, dtype=np.float32),
                obs=obs,
                anchor_right_hand=hand_anchor,
                relative_hand_action=relative_hand_action,
                clip_hand=clip_hand,
                hand_lower=hand_lower,
                hand_upper=hand_upper,
            )
            hand_rows.append(hand_target)
        hand_chunk = np.stack(hand_rows, axis=0)

    return (
        np.asarray(pose_chunk, dtype=np.float32),
        np.asarray(hand_chunk, dtype=np.float32),
        held_wrist_pose6d,
        held_hand,
    )


def _relative_replan_request_deadline(
    *,
    timeline_origin_mono: float,
    result_applied_mono: float,
    first_valid_index: int,
    last_index: int,
    exec_steps: int,
    waypoint_dt: float,
    control_dt: float,
    inference_lead_s: float,
) -> float:
    """Start the next async inference early enough to overlap execution.

    ``first_valid_index`` is the later of the arm/hand first usable indices.
    We execute at most ``exec_steps`` coherent waypoint phases from there, but
    request the next plan roughly one measured inference duration before that
    execution boundary.  A fully stale result is retried on the next control
    tick instead of commanding its last (old) row.
    """
    if int(first_valid_index) > int(last_index):
        return float(result_applied_mono) + float(control_dt)
    final_executed_index = min(
        int(last_index), int(first_valid_index) + int(exec_steps) - 1
    )
    execution_boundary = (
        float(timeline_origin_mono)
        + final_executed_index * float(waypoint_dt)
    )
    request_deadline = execution_boundary - max(float(inference_lead_s), 0.0)
    return max(
        request_deadline,
        float(result_applied_mono) + float(control_dt),
    )


def _blend_relative_startup_targets(
    *,
    hold_wrist6: np.ndarray,
    target_wrist6: np.ndarray,
    hold_hand: np.ndarray,
    target_hand: np.ndarray,
    elapsed_s: float,
    ramp_s: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Blend the first executable plan from the captured startup state."""
    if ramp_s <= 0.0:
        alpha = 1.0
    else:
        u = float(np.clip(elapsed_s / ramp_s, 0.0, 1.0))
        alpha = u * u * (3.0 - 2.0 * u)

    hold_wrist6 = np.asarray(hold_wrist6, dtype=np.float64)
    target_wrist6 = np.asarray(target_wrist6, dtype=np.float64)
    wrist = hold_wrist6.copy()
    wrist[:3] = (
        hold_wrist6[:3]
        + alpha * (target_wrist6[:3] - hold_wrist6[:3])
    )
    hold_rot = R.from_rotvec(hold_wrist6[3:])
    target_rot = R.from_rotvec(target_wrist6[3:])
    relative_rot = (hold_rot.inv() * target_rot).as_rotvec()
    wrist[3:] = (hold_rot * R.from_rotvec(alpha * relative_rot)).as_rotvec()
    hand = np.asarray(hold_hand, dtype=np.float64) + alpha * (
        np.asarray(target_hand, dtype=np.float64)
        - np.asarray(hold_hand, dtype=np.float64)
    )
    return wrist.astype(np.float32), hand.astype(np.float32), alpha


@dataclass(frozen=True)
class _AsyncRelativeObservationRequest:
    """Immutable observation/history item passed to the inference worker."""

    visual_obs: np.ndarray
    joint_proprio: Optional[np.ndarray]
    wrist_pose6: Optional[np.ndarray]
    fingertip_pose_wrist: Optional[np.ndarray]
    fsr: Optional[np.ndarray]
    request_id: Optional[int] = None
    source_observation_time: Optional[float] = None
    source_anchor_wrist6: Optional[np.ndarray] = None
    source_right_hand: Optional[np.ndarray] = None
    used_source_timestamp: bool = False


@dataclass(frozen=True)
class _AsyncRelativePredictionResult:
    """One prediction tied to the exact observation that requested it."""

    request_id: int
    source_observation_time: float
    source_anchor_wrist6: np.ndarray
    source_right_hand: np.ndarray
    used_source_timestamp: bool
    inference_started: float
    inference_finished: float
    output: Optional[np.ndarray] = None
    error: Optional[BaseException] = None
    error_traceback: Optional[str] = None


class _AsyncRelativePlanner:
    """Own policy history and CUDA inference outside the control thread.

    Both ``push_observation`` and ``predict_action`` run on this one worker.
    This is important: updating RealPolicy's deques in the main thread while
    the worker is constructing a strided observation window would otherwise
    create a history race and occasionally mix timestamps inside one input.
    """

    _STOP = object()

    def __init__(self, *, policy, prediction_frames: int, queue_capacity: int):
        self.policy = policy
        self.prediction_frames = max(1, int(prediction_frames))
        self._requests: "queue.Queue[Any]" = queue.Queue(
            maxsize=max(1, int(queue_capacity))
        )
        self._results: "queue.Queue[_AsyncRelativePredictionResult]" = (
            queue.Queue()
        )
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="Relative-Policy-Inference",
            daemon=True,
        )
        self._thread.start()

    def submit(self, request: _AsyncRelativeObservationRequest) -> bool:
        """Submit without blocking; a requested plan is never silently lost."""
        if self._closed:
            raise RuntimeError("cannot submit to a closed async planner")
        try:
            self._requests.put_nowait(request)
            return True
        except queue.Full:
            if request.request_id is not None:
                raise RuntimeError(
                    "async inference observation queue is full; increase "
                    "--observation_buffer_size or investigate a stalled model"
                )
            return False

    def poll_result(self) -> Optional[_AsyncRelativePredictionResult]:
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def close(self, timeout_s: float = 1.0) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._requests.put_nowait(self._STOP)
        except queue.Full:
            return
        self._thread.join(timeout=max(0.0, float(timeout_s)))

    def _run(self) -> None:
        while True:
            request = self._requests.get()
            if request is self._STOP:
                return
            assert isinstance(request, _AsyncRelativeObservationRequest)
            inference_started = time.monotonic()
            try:
                self.policy.push_observation(
                    visual_obs=request.visual_obs,
                    joint_proprio=request.joint_proprio,
                    wrist_pose6=request.wrist_pose6,
                    fingertip_pose_wrist=request.fingertip_pose_wrist,
                    fsr=request.fsr,
                )
                if request.request_id is None:
                    continue

                assert request.source_observation_time is not None
                assert request.source_anchor_wrist6 is not None
                assert request.source_right_hand is not None
                inference_started = time.monotonic()
                with torch.no_grad():
                    output = self.policy.predict_action(
                        proprioception=None,
                        fsr=None,
                        visual_obs=None,
                        num_frames=self.prediction_frames,
                    )
                result = _AsyncRelativePredictionResult(
                    request_id=int(request.request_id),
                    source_observation_time=float(
                        request.source_observation_time
                    ),
                    source_anchor_wrist6=np.asarray(
                        request.source_anchor_wrist6, np.float32
                    ).copy(),
                    source_right_hand=np.asarray(
                        request.source_right_hand, np.float32
                    ).copy(),
                    used_source_timestamp=bool(request.used_source_timestamp),
                    inference_started=inference_started,
                    inference_finished=time.monotonic(),
                    output=np.asarray(output, np.float32),
                )
            except BaseException as exc:  # noqa: BLE001
                result = _AsyncRelativePredictionResult(
                    request_id=(
                        -1
                        if request.request_id is None
                        else int(request.request_id)
                    ),
                    source_observation_time=float(
                        request.source_observation_time or time.monotonic()
                    ),
                    source_anchor_wrist6=(
                        np.zeros(6, np.float32)
                        if request.source_anchor_wrist6 is None
                        else np.asarray(
                            request.source_anchor_wrist6, np.float32
                        ).copy()
                    ),
                    source_right_hand=(
                        np.zeros(22, np.float32)
                        if request.source_right_hand is None
                        else np.asarray(
                            request.source_right_hand, np.float32
                        ).copy()
                    ),
                    used_source_timestamp=bool(request.used_source_timestamp),
                    inference_started=inference_started,
                    inference_finished=time.monotonic(),
                    error=exc,
                    error_traceback=traceback.format_exc(),
                )
            self._results.put(result)


def _decode_bimanual_absolute_chunks(
    output: np.ndarray,
    waypoint_indices: np.ndarray,
    obs: Dict[str, Any],
    arm_prefixes: list,
    anchor_wrist: Dict[str, np.ndarray],
    t_et: np.ndarray,
    hand_action_mode: str,
    relative_hand_action: bool,
    no_eef_policy: bool,
    eef_only_policy: bool,
    clip_hand: bool,
    hand_lower: Optional[np.ndarray],
    hand_upper: Optional[np.ndarray],
    held_wrist: Dict[str, Optional[np.ndarray]],
    held_hand: Dict[str, Optional[np.ndarray]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Decode a bimanual network chunk into per-arm absolute trajectories."""
    output = np.asarray(output)
    indices = np.asarray(waypoint_indices, dtype=np.int64)
    if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= len(output)):
        raise ValueError("waypoint_indices must select at least one output frame")

    pose_rows: Dict[str, list] = {p: [] for p in arm_prefixes}
    hand_rows: Dict[str, list] = {p: [] for p in arm_prefixes}
    hand_anchors: Dict[str, Optional[np.ndarray]] = {
        p: (
            np.asarray(
                obs[f"{_side_of_prefix(p)}_hand"],
                dtype=np.float32,
            ).copy()
            if relative_hand_action else None
        )
        for p in arm_prefixes
    }

    for k in indices:
        parts = decode_bimanual_action_step(
            output[k],
            arm_prefixes,
            hand_action_mode,
        )
        for part in parts:
            prefix = part["prefix"]
            side = part["side"]
            if no_eef_policy:
                if held_wrist[prefix] is None:
                    held_wrist[prefix] = anchor_wrist[prefix].copy()
                wrist_target = held_wrist[prefix]
            else:
                wrist_target = recover_absolute_target_pose_from_relative(
                    current_pose6d=anchor_wrist[prefix],
                    relative_pose6d=part["relative_pose6d"],
                    T_ET=t_et,
                )
            pose_rows[prefix].append(wrist_target)

            hand_action = part["hand_action"]
            if eef_only_policy or hand_action.size == 0:
                if held_hand[prefix] is None:
                    held_hand[prefix] = np.asarray(
                        obs[f"{side}_hand"], dtype=np.float32
                    ).copy()
                hand_target = held_hand[prefix]
            else:
                hand_target, _ = convert_hand_action_joint(
                    hand_action=hand_action,
                    obs=obs,
                    anchor_right_hand=hand_anchors[prefix],
                    relative_hand_action=relative_hand_action,
                    clip_hand=clip_hand,
                    hand_lower=hand_lower,
                    hand_upper=hand_upper,
                    measured_hand=obs[f"{side}_hand"],
                )
            hand_rows[prefix].append(hand_target)

    pose_chunks = {
        p: np.asarray(pose_rows[p], dtype=np.float32)
        for p in arm_prefixes
    }
    hand_chunks = {
        p: np.asarray(hand_rows[p], dtype=np.float32)
        for p in arm_prefixes
    }
    return pose_chunks, hand_chunks


def run_bimanual_rollout(
    args,
    policy,
    robot_env,
    ik_solver: LightweightIK,
    proprio_mode: str,
    hand_action_mode: str,
    action_horizon: int,
    down_sample_steps: int,
) -> None:
    """Dual-arm variant of the single-model rollout loop.

    !!! UNTESTED ON HARDWARE !!!  ``RealPolicy`` inference (per-arm stats,
    camera stacking, per-arm tactile, per-arm action decode) is verified
    offline by ``scripts/smoke_bimanual_rollout.py``, but the closed-loop
    dual-arm COMMAND path below has NOT been validated on the robot.  Start with
    ``--debug``, a low ``--control_hz`` and a ready e-stop.

    Per arm (``policy.arm_prefixes`` order): latch a wrist anchor at predict
    time, decode the arm's relative eef pose -> absolute -> IK(side) -> arm
    joints; decode the arm's hand action -> joint command; command BOTH arms
    each step via ``build_robot_action_buffer_dual``.
    """
    if hand_action_mode in ("fingertip", "fingertip_only"):
        raise NotImplementedError(
            "bimanual rollout supports hand_action_mode in {joint, none, "
            "joint_only}; fingertip modes need a per-hand FingertipToJointIK.  "
            "Retrain the hand as joint / joint_only or extend "
            "run_bimanual_rollout with per-hand fingertip IK."
        )

    arm_prefixes = list(policy.arm_prefixes)
    T_ET = parse_t_et_arg(args.t_et)
    if not np.allclose(T_ET, np.eye(4), atol=1e-6):
        print("[WARN] --t_et != I; only meaningful for legacy checkpoints.")
    control_dt = 1.0 / float(args.control_hz)
    waypoint_dt = 1.0 / float(args.waypoint_hz)
    arm_latency_s = float(args.arm_latency_ms) / 1000.0
    hand_latency_s = float(args.hand_latency_ms) / 1000.0
    safety_margin_s = float(args.safety_margin_ms) / 1000.0
    ANCHOR_OFFSET = int(getattr(args, "anchor_offset", 0))
    chunk_exec_steps = max(1, action_horizon - ANCHOR_OFFSET)
    if args.exec_horizon is not None:
        chunk_exec_steps = max(1, min(int(args.exec_horizon), chunk_exec_steps))

    hand_lower = (
        np.array(args.hand_lower, dtype=np.float32)
        if args.hand_lower is not None else None
    )
    hand_upper = (
        np.array(args.hand_upper, dtype=np.float32)
        if args.hand_upper is not None else None
    )

    uses_tactile = bool(getattr(policy, "enable_fsr", False))
    tactile_key = str(getattr(policy, "tactile_key", "fsr"))
    relative_hand_action = bool(
        getattr(policy.model_cfg.dataset, "relative_hand_action", False)
    )
    no_eef_policy = hand_action_mode in ("joint_only", "fingertip_only")
    eef_only_policy = hand_action_mode == "none"

    policy.reset_history()

    # per-arm rolling state (keyed by arm prefix)
    anchor_wrist: Dict[str, Optional[np.ndarray]] = {p: None for p in arm_prefixes}
    held_wrist: Dict[str, Optional[np.ndarray]] = {p: None for p in arm_prefixes}
    held_hand: Dict[str, Optional[np.ndarray]] = {p: None for p in arm_prefixes}

    output: Optional[np.ndarray] = None
    act_index = ANCHOR_OFFSET
    obs_timeline = ObservationTimestampBuffer(
        capacity=args.observation_buffer_size,
        max_clock_skew_s=args.observation_timestamp_max_skew_s,
        max_transport_age_s=args.observation_timestamp_max_age_ms / 1000.0,
    )
    pose_interps: Dict[str, Optional[PoseTrajectoryInterpolator]] = {
        p: None for p in arm_prefixes
    }
    hand_interps: Dict[str, Optional[MotorTrajectoryInterpolator]] = {
        p: None for p in arm_prefixes
    }
    arm_limiters = {
        p: JointKinematicLimiter(
            nominal_dt=control_dt,
            max_velocity=args.arm_joint_max_velocity,
            max_acceleration=args.arm_joint_max_acceleration,
        )
        for p in arm_prefixes
    }
    hand_limiters = {
        p: JointKinematicLimiter(
            nominal_dt=control_dt,
            max_velocity=args.hand_joint_max_velocity,
            max_acceleration=args.hand_joint_max_acceleration,
        )
        for p in arm_prefixes
    }
    next_tick = time.monotonic()
    next_replan = next_tick
    plan_origin_mono: Optional[float] = None
    missed_ticks_total = 0
    primary_frame: Optional[np.ndarray] = None

    debug_act_dir = None
    run_dir = args.debug_dir
    if args.debug:
        run_dir = os.path.join(
            args.debug_dir, "run_bimanual_" + time.strftime("%Y%m%d_%H%M%S")
        )
        debug_act_dir = os.path.join(run_dir, "act")
        os.makedirs(debug_act_dir, exist_ok=True)
        print(f"[DEBUG] saving bimanual actions to {run_dir}")

    print("======================================")
    print("Start BIMANUAL rollout   [UNTESTED ON HARDWARE]")
    print(f"  model_path        : {args.model_path} (ckpt={args.ckpt})")
    print(f"  arms              : {arm_prefixes}")
    print(f"  cameras           : {policy.camera_ids}")
    print(f"  proprio_mode      : {proprio_mode}")
    print(f"  hand_action_mode  : {hand_action_mode}")
    print(f"  action_horizon    : {action_horizon}  (executable = {chunk_exec_steps})")
    print(f"  control/waypoint  : {args.control_hz:g} / {args.waypoint_hz:g} Hz")
    print(
        "  observation time  : robot producer timestamp when valid; "
        "otherwise callback monotonic receive time"
    )
    print(
        f"  actuator latency  : arm={args.arm_latency_ms:.1f}ms "
        f"hand={args.hand_latency_ms:.1f}ms "
        f"safety={args.safety_margin_ms:.1f}ms"
    )
    print(f"  tactile           : {uses_tactile} (key={tactile_key})")
    print(f"  relative_hand_act : {relative_hand_action}")
    print("======================================")

    if args.reset2zero:
        print(
            "[WARN] --reset2zero uses the single-(right)-arm reset routine; "
            "skipping it for the bimanual rollout (reset both arms manually "
            "to a safe start pose before running)."
        )

    enable_show = args.show and _can_imshow()
    debug_step = 0
    try:
        while True:
            obs = get_latest_robot_obs(robot_env)
            obs_stamp = obs_timeline.append(obs)

            if obs_stamp is not None:
                primary_frame = push_bimanual_observation(
                    policy=policy,
                    obs=obs,
                    ik_solver=ik_solver,
                    fallback_camera_source=args.camera_source,
                    uses_tactile=uses_tactile,
                    tactile_key=tactile_key,
                )

            now = time.monotonic()
            need_predict = (
                (
                    any(pose_interps[p] is None for p in arm_prefixes)
                    or now >= next_replan
                )
                and obs_stamp is not None
            )
            if need_predict:
                inference_started = time.monotonic()
                with torch.no_grad():
                    output = policy.predict_action(
                        proprioception=None, fsr=None, visual_obs=None,
                    )
                inference_finished = time.monotonic()
                assert obs_stamp is not None
                waypoint_indices = np.arange(
                    ANCHOR_OFFSET, len(output), dtype=np.int64
                )
                for p in arm_prefixes:
                    anchor_wrist[p] = get_current_wrist_pose6d(
                        obs, ik_solver, side=_side_of_prefix(p)
                    )
                pose_chunks, hand_chunks = _decode_bimanual_absolute_chunks(
                    output=output,
                    waypoint_indices=waypoint_indices,
                    obs=obs,
                    arm_prefixes=arm_prefixes,
                    anchor_wrist=anchor_wrist,
                    t_et=T_ET,
                    hand_action_mode=hand_action_mode,
                    relative_hand_action=relative_hand_action,
                    no_eef_policy=no_eef_policy,
                    eef_only_policy=eef_only_policy,
                    clip_hand=args.clip_hand,
                    hand_lower=hand_lower,
                    hand_upper=hand_upper,
                    held_wrist=held_wrist,
                    held_hand=held_hand,
                )
                schedule_summary = []
                for p in arm_prefixes:
                    side = _side_of_prefix(p)
                    if pose_interps[p] is None:
                        pose_interps[p] = PoseTrajectoryInterpolator(
                            np.array([inference_finished]),
                            np.asarray(anchor_wrist[p])[None],
                        )
                        hand_interps[p] = MotorTrajectoryInterpolator(
                            np.array([inference_finished]),
                            np.asarray(
                                obs[f"{side}_hand"], dtype=np.float32
                            )[None],
                        )
                    assert pose_interps[p] is not None
                    assert hand_interps[p] is not None
                    (
                        pose_interps[p],
                        hand_interps[p],
                        arm_n,
                        hand_n,
                        arm_first,
                        hand_first,
                    ) = schedule_absolute_trajectory_chunk(
                        pose_interp=pose_interps[p],
                        hand_interp=hand_interps[p],
                        pose_chunk=pose_chunks[p],
                        hand_chunk=hand_chunks[p],
                        waypoint_indices=waypoint_indices,
                        timeline_origin_mono=obs_stamp.control_time_s,
                        waypoint_dt=waypoint_dt,
                        arm_latency_s=arm_latency_s,
                        hand_latency_s=hand_latency_s,
                        safety_margin_s=safety_margin_s,
                        max_pos_speed=args.max_pos_speed,
                        max_rot_speed=args.max_rot_speed,
                        max_hand_speed=args.max_hand_speed,
                    )
                    schedule_summary.append(
                        f"{side}:A{arm_n}@{arm_first}/H{hand_n}@{hand_first}"
                    )
                plan_origin_mono = obs_stamp.control_time_s
                next_replan = plan_origin_mono + (
                    ANCHOR_OFFSET + chunk_exec_steps
                ) * waypoint_dt
                act_index = ANCHOR_OFFSET
                timestamp_source = (
                    "robot" if obs_stamp.used_source_timestamp else "receive"
                )
                print(
                    f"New timestamped bimanual chunk: {output.shape} "
                    f"infer={(inference_finished-inference_started)*1000.0:.1f}ms "
                    f"obs_age={(inference_finished-plan_origin_mono)*1000.0:.1f}ms "
                    f"time={timestamp_source} {' '.join(schedule_summary)}"
                )

            assert output is not None
            assert plan_origin_mono is not None
            sample_time = time.monotonic()
            act_index = int(
                np.clip(
                    np.floor(
                        (sample_time + arm_latency_s - plan_origin_mono)
                        / waypoint_dt
                    ),
                    ANCHOR_OFFSET,
                    len(output) - 1,
                )
            )
            if args.debug:
                np.save(
                    os.path.join(
                        debug_act_dir,
                        f"act_{debug_step:06d}.npy",
                    ),
                    output[act_index],
                )

            per_arm_cmd: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            for p in arm_prefixes:
                side = _side_of_prefix(p)
                assert pose_interps[p] is not None
                assert hand_interps[p] is not None
                target_wrist6d = np.asarray(
                    pose_interps[p](sample_time + arm_latency_s),
                    dtype=np.float32,
                )
                measured_arm = np.asarray(
                    obs[f"{side}_arm"], dtype=np.float32
                )
                target_arm_raw, ik_ok, pos_err, rot_err = solve_arm_ik(
                    ik_solver,
                    obs,
                    target_wrist6d,
                    side,
                    args.ik_pos_tol,
                    args.ik_rot_tol,
                    init_arm_sensor_order=arm_limiters[p].previous_position,
                )
                if ik_ok:
                    target_arm = arm_limiters[p].limit(
                        target_arm_raw,
                        measured=measured_arm,
                        timestamp=sample_time,
                    )
                else:
                    target_arm = arm_limiters[p].hold(
                        measured_arm,
                        timestamp=sample_time,
                    )
                target_hand = hand_limiters[p].limit(
                    np.asarray(
                        hand_interps[p](sample_time + hand_latency_s),
                        dtype=np.float32,
                    ),
                    measured=np.asarray(
                        obs[f"{side}_hand"], dtype=np.float32
                    ),
                    timestamp=sample_time,
                )
                print(
                    f"[IK:{side:>5s}] step={debug_step} accepted={ik_ok} "
                    f"pos_err={pos_err * 1000:.2f}mm rot_err={rot_err:.4f}"
                )
                per_arm_cmd[side] = (target_arm, target_hand)

            # 7) send BOTH arms
            action_buffer = build_robot_action_buffer_dual(obs, per_arm_cmd)
            robot_env.send_action(action_buffer, immediate=False)
            debug_step += 1

            # 8) optional display (first camera)
            if enable_show:
                assert primary_frame is not None
                show_img = cv2.cvtColor(primary_frame, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    show_img, f"step={debug_step} act_index={act_index}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
                cv2.imshow("robot_bimanual_rollout", show_img)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    print("Quit by keyboard.")
                    break

            tick_finished = time.monotonic()
            next_tick += control_dt
            if next_tick <= tick_finished:
                missed = int(
                    (tick_finished - next_tick) // control_dt
                ) + 1
                next_tick += missed * control_dt
                missed_ticks_total += missed
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        print(
            f"[timing] bimanual control ticks skipped={missed_ticks_total}"
        )
        if enable_show:
            cv2.destroyAllWindows()


def main(args):
    # -------- hard-required deploy deps --------
    if not _ROBOT_ENV_AVAILABLE:
        raise RuntimeError(
            "robot_env / arm_kinematics not importable -- this script "
            "can only run on the robot's deploy machine.  "
            f"(original error: {_ROBOT_ENV_IMPORT_ERR})"
        )
    if not _REAL_POLICY_AVAILABLE:
        raise RuntimeError(
            "uvta.real_env.real_policy not importable -- check "
            f"PYTHONPATH points to the UVTA repo root.  "
            f"(original error: {_REAL_POLICY_IMPORT_ERR})"
        )

    # -------- policy --------
    policy = RealPolicy(
        model_path=args.model_path, ckpt=args.ckpt, use_ema=args.use_ema
    )
    # This script drives ONE model and sends its output straight to the robot, so
    # the checkpoint must predict the recorded COMMAND (predict_action=True).  A
    # state-only policy predicts where the arm will be, not what to command; it
    # needs a stage-2 inverse map to turn that into a command, which is what
    # two_stage_policy_rollout.py does.  Refuse rather than execute a state as
    # though it were a command.
    if "action" not in policy.motor_blocks:
        raise SystemExit(
            f"{args.model_path} predicts {policy.motor_blocks} -- no action "
            "block, so it cannot be executed directly.  Run it as stage 1 of "
            "cet/two_stage_policy_rollout.py instead (that script pairs it with "
            "a stage-2 model that maps the predicted state to a command)."
        )
    if len(policy.motor_blocks) > 1:
        print(
            f"[rollout] policy predicts {policy.motor_blocks}; executing the "
            f"{policy.executed_block!r} block (the rest is an auxiliary "
            "training signal and is dropped here)."
        )
    if getattr(policy, "aux_targets", None):
        print(
            f"[rollout] auxiliary targets {policy.aux_targets} come from "
            "regression heads (aux_as_head=True), so they never entered the "
            "diffusion trajectory; nothing to drop here."
        )
    proprio_mode = policy.proprio_mode
    hand_action_mode = policy.hand_action_mode
    action_horizon = int(policy.action_horizon)
    down_sample_steps = int(policy.down_sample_steps)
    # Deployment topology comes from the checkpoint's dataset config.  CLI
    # values remain optional emergency overrides for backwards compatibility.
    if args.mode is None:
        args.mode = infer_policy_mode(policy)
    if args.camera_source is None:
        args.camera_source = infer_policy_camera_source(policy)
    print(
        f"[config] inferred mode={args.mode!r}, "
        f"camera_source={args.camera_source!r} from model"
    )
    if min(args.control_hz, args.waypoint_hz) <= 0:
        raise ValueError("control_hz and waypoint_hz must be positive.")
    if args.control_hz > 60.0:
        raise ValueError(
            "control_hz must not exceed robot's ~60-Hz action publisher "
            "when timestamp smoothing uses non-immediate FIFO commands."
        )
    if args.observation_buffer_size <= 0:
        raise ValueError("observation_buffer_size must be positive.")
    if min(
        args.observation_timestamp_max_skew_s,
        args.observation_timestamp_max_age_ms,
        args.arm_latency_ms,
        args.hand_latency_ms,
        args.safety_margin_ms,
        args.startup_ramp_s,
    ) < 0:
        raise ValueError(
            "Timestamp tolerances, latencies, and startup_ramp_s must be "
            "non-negative."
        )
    if min(
        args.max_pos_speed,
        args.max_rot_speed,
        args.max_hand_speed,
        args.arm_joint_max_velocity,
        args.arm_joint_max_acceleration,
        args.hand_joint_max_velocity,
        args.hand_joint_max_acceleration,
    ) <= 0:
        raise ValueError("Trajectory speed and acceleration limits must be positive.")

    # Hand action mode dispatch (Bug 6).
    #
    # For ``hand_action_mode='fingertip'`` we drive
    # ``HandRetargetingOptimizer`` (mode='ego', from
    # ``third_party/glove_retargeting/``) at control rate.
    # The optimizer compiles its CasADi NLP on the first call, so it is
    # constructed up-front (with a zero-input warm-up call) to avoid a
    # 150 ms spike on the first control step.  Benchmark shows ~4 ms /
    # 245 Hz per call afterwards (see ``scripts/bench_hand_retargeting.py``).
    if hand_action_mode in ("fingertip", "fingertip_only"):
        if not _can_import_finger_ik():
            raise NotImplementedError(
                f"Policy was trained with hand_action_mode={hand_action_mode!r}, "
                f"but the CasADi-based fingertip IK could not be imported from "
                f"{_GLOVE_DIR} -- check that 'casadi', 'transforms3d', "
                "'rich' are installed and that the path exists.  Either "
                "fix the env or retrain with hand_action_mode='joint'."
            )

    # -------- RobotEnv --------
    # The tactile keys (``/observe/tactile/<finger>/force6d`` etc.) are only
    # populated in the live obs when ``enable_tactile=True``.  If the trained
    # policy consumes tactile, force it on regardless of the CLI flag so
    # ``build_tactile_from_obs`` does not KeyError below.
    policy_uses_tactile = bool(getattr(policy, "enable_fsr", False))
    enable_tactile = bool(args.enable_tactile) or policy_uses_tactile
    if policy_uses_tactile and not args.enable_tactile:
        print(
            "[tactile] policy requires tactile -> auto-enabling RobotEnv "
            "tactile stream (override the --enable_tactile flag)."
        )
    robot_env = RobotEnv(
        enable_tactile=enable_tactile,
        action_output=[
            "/action/left_arm/joint_angle",
            "/action/left_hand/joint_angle",
            "/action/right_arm/joint_angle",
            "/action/right_hand/joint_angle",
            "/action/motor/joint_angle",
        ],
    )

    # -------- IK/FK --------
    ik_solver = LightweightIK(args.urdf_path)

    # -------- bimanual dispatch --------
    # A bimanual policy (arms=['left_','right_']) decodes a concatenated
    # [left|right] action and expects per-arm tactile + both wrist cameras.
    # Run the dual-arm loop and return; the single-arm code below is untouched.
    if getattr(policy, "num_arms", 1) > 1:
        return run_bimanual_rollout(
            args=args,
            policy=policy,
            robot_env=robot_env,
            ik_solver=ik_solver,
            proprio_mode=proprio_mode,
            hand_action_mode=hand_action_mode,
            action_horizon=action_horizon,
            down_sample_steps=down_sample_steps,
        )

    fingertip_fk: Optional[FingertipFK] = None
    if proprio_mode in ("fingertip", "fingertip_with_ee"):
        fingertip_fk = FingertipFK(hand_type="right")

    # Build the fingertip IK *only* when needed; constructing it compiles
    # the CasADi NLP (~150 ms warm-up) so we avoid the cost in joint mode.
    fingertip_ik: Optional[FingertipToJointIK] = None
    if hand_action_mode in ("fingertip", "fingertip_only"):
        print("[init] Compiling fingertip->joint CasADi IK "
              "(one-shot ~150 ms warm-up) ...")
        t0 = time.time()
        fingertip_ik = FingertipToJointIK(hand_type="right")
        print(f"[init] FingertipToJointIK ready ({(time.time() - t0) * 1000:.1f} ms)")

    # -------- debug dirs --------
    # Each run gets its OWN timestamped sub-directory under --debug_dir so
    # consecutive experiments (e.g. grabbing the ball at different positions)
    # never overwrite each other.  This lets you compare the saved actions
    # across runs to tell whether the policy memorised a trajectory or
    # generalised.
    debug_imgs_dir = debug_act_dir = debug_ik_dir = debug_pose_dir = None
    run_dir = args.debug_dir
    if args.debug:
        run_name = "run_" + time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.debug_dir, run_name)
        debug_imgs_dir = os.path.join(run_dir, "imgs")
        debug_act_dir = os.path.join(run_dir, "act")
        debug_ik_dir = os.path.join(run_dir, "ik")
        debug_pose_dir = os.path.join(run_dir, "pose")
        for d in (debug_imgs_dir, debug_act_dir, debug_ik_dir, debug_pose_dir):
            os.makedirs(d, exist_ok=True)
        print(f"[DEBUG] Saving debug data to {run_dir} (new timestamped run)")

    # -------- params --------
    T_ET = parse_t_et_arg(args.t_et)
    if not np.allclose(T_ET, np.eye(4), atol=1e-6):
        print(
            "[WARN] --t_et != I.  Method-B training emits action targets "
            "directly in the current wrist frame (T_BN = T_BE @ T_rel), so a "
            "non-identity T_ET will introduce a systematic offset at deploy "
            "time.  Only enable this for legacy (pre-method-B) checkpoints."
        )

    control_dt = 1.0 / float(args.control_hz)
    waypoint_dt = 1.0 / float(args.waypoint_hz)
    arm_latency_s = float(args.arm_latency_ms) / 1000.0
    hand_latency_s = float(args.hand_latency_ms) / 1000.0
    safety_margin_s = float(args.safety_margin_ms) / 1000.0
    mode = args.mode
    if mode != "right":
        raise NotImplementedError("Currently only --mode right is supported.")

    hand_lower = (
        np.array(args.hand_lower, dtype=np.float32) if args.hand_lower is not None else None
    )
    hand_upper = (
        np.array(args.hand_upper, dtype=np.float32) if args.hand_upper is not None else None
    )

    # Reset the policy's rolling history so we start the deque empty.
    policy.reset_history()
    # The worker requests the whole diffusion prediction, not merely the
    # trained action_horizon prefix.  The extra suffix keeps the old plan alive
    # while the next inference overlaps it.
    prediction_frames = int(getattr(policy, "pred_horizon", action_horizon))
    prediction_frames = max(action_horizon, prediction_frames)
    planner = _AsyncRelativePlanner(
        policy=policy,
        prediction_frames=prediction_frames,
        queue_capacity=args.observation_buffer_size,
    )

    output: Optional[np.ndarray] = None
    # Which chunk frame to start EXECUTING from.  Current convention: the eef
    # action comes from ``pose_action`` anchored on the current state
    # ``pose[t]``, so ``action[0] = pose[t]^-1 @ pose_action[t]`` is the REAL
    # first commanded step and must be executed -> ANCHOR_OFFSET = 0 (default).
    # Legacy checkpoints that reused ``pose`` as the action had
    # ``action[0] = identity`` (a no-op) and skipped it with --anchor_offset 1.
    ANCHOR_OFFSET = int(getattr(args, "anchor_offset", 0))
    act_index = ANCHOR_OFFSET
    obs_timeline = ObservationTimestampBuffer(
        capacity=args.observation_buffer_size,
        max_clock_skew_s=args.observation_timestamp_max_skew_s,
        max_transport_age_s=args.observation_timestamp_max_age_ms / 1000.0,
    )
    pose_interp: Optional[PoseTrajectoryInterpolator] = None
    hand_interp: Optional[MotorTrajectoryInterpolator] = None
    next_tick = time.monotonic()
    next_replan = next_tick
    plan_origin_mono: Optional[float] = None
    waypoint_indices: Optional[np.ndarray] = None
    missed_ticks_total = 0
    dropped_worker_observations = 0
    inference_pending = False
    next_request_id = 0
    chunk_id = -1
    inference_lead_ema_s = 0.0
    startup_warmup_completed = not bool(args.startup_inference_warmup)
    startup_hold_right_arm: Optional[np.ndarray] = None
    startup_hold_right_hand: Optional[np.ndarray] = None
    startup_hold_wrist6: Optional[np.ndarray] = None
    startup_motion_start_mono: Optional[float] = None
    arm_limiter = JointKinematicLimiter(
        nominal_dt=control_dt,
        max_velocity=args.arm_joint_max_velocity,
        max_acceleration=args.arm_joint_max_acceleration,
    )
    hand_limiter = JointKinematicLimiter(
        nominal_dt=control_dt,
        max_velocity=args.hand_joint_max_velocity,
        max_acceleration=args.hand_joint_max_acceleration,
    )
    # EEF-only policies (e.g. the v30 config with hand_action_mode='none')
    # predict a 9-D action = relative eef pose only and do NOT control the
    # hand.  In that case we latch the hand joints once at the start of the
    # rollout and keep commanding the same pose every step so the fingers
    # stay unchanged.  ``None`` until the first step latches it.
    held_right_hand: Optional[np.ndarray] = None
    eef_only_policy = (hand_action_mode == "none")
    # ``joint_only`` is the mirror image of eef-only: the policy predicts the
    # 22-D joint vector ONLY (no relative eef pose), so the action has no
    # ``act[:6]`` wrist component.  We hold the wrist at the pose latched at
    # the start of the rollout and command the predicted joints every step.
    joint_only_policy = (hand_action_mode == "joint_only")
    # ``fingertip_only`` is the fingertip analogue of joint_only: the policy
    # predicts ONLY the 45-D fingertip block (decoded to 30-D xyz+rotvec by
    # the policy server) and does NOT control the wrist.  We hold the wrist at
    # the latched start pose and feed the fingertip action through the
    # fingertip IK to recover joint angles.
    fingertip_only_policy = (hand_action_mode == "fingertip_only")
    # Policies that do not control the wrist (no eef block in the action).
    no_eef_policy = joint_only_policy or fingertip_only_policy
    held_right_wrist_pose6d: Optional[np.ndarray] = None
    # The EEF relative pose ``act[:6]`` is, by training-time construction
    # (uvta_dataset.__getitem__: ``T_anchor_inv @ T_k`` with anchor = the
    # chunk's first frame t), expressed relative to the wrist pose at the
    # moment the chunk was predicted -- NOT relative to the live pose at each
    # executed step.  So we latch the current wrist pose once per replan and
    # reuse it as the fixed anchor for every frame of that chunk.  Using the
    # live pose per-step (as before) made every chunk restart its relative
    # offset from ~0 before the arm could track it, so the arm only crept by
    # a few mm and looked frozen while the (absolute) hand joints still moved.
    anchor_right_wrist_pose6d: Optional[np.ndarray] = None

    # Per-step raw tactile vector (assembled from the robot obs below when the
    # trained policy uses tactile).  ``None`` when the model has no tactile.
    fsr: Optional[np.ndarray] = None
    policy_tactile_key = str(getattr(policy, "tactile_key", "fsr"))
    if policy_uses_tactile:
        _tac_desc = {
            "force": "force6d->resultant(5)",
            "fsr": "deform->taxel(100)",
            "fsr_region": "deform->taxel(100)->region_mean(20)",
        }.get(policy_tactile_key, f"deform->{policy_tactile_key}")
        print(
            f"[tactile] policy uses tactile: source='{policy_tactile_key}' "
            f"({_tac_desc}), fingers={_TACTILE_FINGERS}"
        )

    # Durable per-step tactile range logger (debug the '接触前就 OOD' question):
    # records raw force, its min-max normalization using the deploy stats, and a
    # per-dim out-of-range flag so we can see which channel exceeds its training
    # 量程.  Active only with --debug + a tactile policy.
    tactile_log: Optional[_TactileLog] = None
    if args.debug and policy_uses_tactile:
        _tst = None
        try:
            _tst = policy.stats.get(policy_tactile_key)
        except Exception:  # noqa: BLE001
            _tst = None
        tactile_log = _TactileLog(
            run_dir,
            stat_min=_tst["min"] if _tst is not None else None,
            stat_max=_tst["max"] if _tst is not None else None,
            tactile_key=policy_tactile_key,
            labels=_TACTILE_FINGERS if policy_tactile_key == "force" else None,
        )
        if _tst is None:
            print(f"[tactile][WARN] no '{policy_tactile_key}' stats found; "
                  "logging raw force only (no OOD flag).")
        print(f"[tactile] range-debug logging -> "
              f"{os.path.join(run_dir, 'tactile.jsonl')}")
    debug_step = 0
    visual_obs_rgb: Optional[np.ndarray] = None
    # Per-step IK tracking error log (FK(ik_solution) vs the requested target
    # wrist pose).  Saved to ``debug/ik_tracking_error.{npy,csv}`` at exit so
    # we can tell offline whether the arm IK is faithfully tracking the
    # policy's EEF targets or silently diverging.
    ik_err_log: list = []
    # how many executable frames live in each chunk (action_horizon minus the
    # anchor offset; with the pose_action convention ANCHOR_OFFSET=0 so all
    # predicted frames are executable).
    chunk_exec_steps = max(1, action_horizon - ANCHOR_OFFSET)
    if args.exec_horizon is not None:
        # Optional override; clamp to what the network actually predicts so
        # we never index past the end of the chunk.
        chunk_exec_steps = max(1, min(int(args.exec_horizon), chunk_exec_steps))
    enable_show = args.show and _can_imshow()
    if args.show and not enable_show:
        print(
            "[WARN] --show requested but no DISPLAY env var found; "
            "skipping cv2.imshow().  Use --debug to save frames instead."
        )

    # Live debug dashboard: when --debug is on (and a display exists, unless
    # --no_live_viz was passed) pop up a matplotlib window that streams the
    # wrist image, the decoded action chunk and the IK tracking history so the
    # user can watch in real time whether the action chunk + IK are correct and
    # tracking the wrist image.
    live_viz: Optional[LiveDebugViz] = None
    want_live = args.debug and not args.no_live_viz
    if want_live and not _can_imshow():
        print(
            "[WARN] --debug live viz requested but no DISPLAY found; "
            "skipping live dashboard (data is still saved to --debug_dir)."
        )
        want_live = False
    if want_live:
        live_viz = LiveDebugViz(history=args.live_viz_history)
        if not live_viz.ok:
            live_viz = None
        else:
            print(
                "[LIVE] real-time debug dashboard open "
                "(wrist img | action chunk | IK tracking). "
                "Close the window or press q in the cv2 window to stop."
            )

    legacy_joint_proprio: Optional[np.ndarray] = None
    print("======================================")
    print("Start Rollout with relative policy")
    print(f"  model_path        : {args.model_path}")
    print(f"  ckpt              : {args.ckpt}")
    print(f"  use_ema           : {args.use_ema if args.use_ema is not None else '(follow config)'}")
    print(f"  control_hz        : {args.control_hz}")
    print(f"  waypoint_hz       : {args.waypoint_hz}")
    print(
        "  observation time  : robot producer timestamp when valid; "
        "otherwise callback monotonic receive time"
    )
    print(
        f"  actuator latency  : arm={args.arm_latency_ms:.1f}ms "
        f"hand={args.hand_latency_ms:.1f}ms "
        f"safety={args.safety_margin_ms:.1f}ms"
    )
    print(f"  camera_source     : {args.camera_source}")
    print(f"  proprio_mode      : {proprio_mode}")
    print(f"  hand_action_mode  : {hand_action_mode}")
    print(
        f"  horizons          : action={action_horizon}, "
        f"prediction={prediction_frames}, executable={chunk_exec_steps}"
    )
    print(f"  down_sample_steps : {down_sample_steps}  (obs stride; "
          f"buffer warm-up needs >= {(policy.obs_horizon - 1) * down_sample_steps + 1} frames)")
    if proprio_mode in ("fingertip", "fingertip_with_ee"):
        print("  fingertip FK      : enabled (right hand, GloveFKSolver)")
    if hand_action_mode in ("fingertip", "fingertip_only"):
        print("  fingertip IK      : enabled (CasADi HandRetargetingOptimizer, ego mode)")
    if eef_only_policy:
        print("  hand control      : EEF-only policy (hand_action_mode='none'); "
              "holding hand joints constant")
    if joint_only_policy:
        print("  wrist control     : joint-only policy (hand_action_mode="
              "'joint_only'); holding wrist pose constant, commanding joints")
    if fingertip_only_policy:
        print("  wrist control     : fingertip-only policy (hand_action_mode="
              "'fingertip_only'); holding wrist pose constant, commanding "
              "fingertips via IK")
    if proprio_mode in ("fingertip", "fingertip_with_ee") or \
            hand_action_mode in ("fingertip", "fingertip_only"):
        print(f"  glove_retargeting : {_GLOVE_DIR}")
    if args.use_right_proprio:
        print("  legacy proprio    : --use_right_proprio (override joint_proprio)")
    print(
        "  relative_hand_act :",
        bool(getattr(policy.model_cfg.dataset, "relative_hand_action", False)),
    )
    print(
        "  inference/control : asynchronous worker / continuous control; "
        "next request is advanced by measured inference time"
    )
    print(
        "  startup safety    : "
        + (
            "discard one warmup result, hard-hold joints, "
            f"then ramp over {args.startup_ramp_s:.2f}s"
            if args.startup_inference_warmup
            else f"hard-hold until first plan, then ramp over "
                 f"{args.startup_ramp_s:.2f}s"
        )
    )
    print("======================================")

    # Optionally reset the robot to a recorded episode's first frame before
    # starting the rollout, so inference always begins from a known pose.
    if args.reset2zero:
        reset_to_episode_start(
            robot_env=robot_env,
            ik_solver=ik_solver,
            zarr_root=args.reset2zero_zarr_root,
            episode=args.reset2zero_episode,
            control_hz=args.control_hz,
            hold_steps=args.reset2zero_hold_steps,
            ik_pos_tol=args.ik_pos_tol,
            ik_rot_tol=args.ik_rot_tol,
            mode=mode,
        )

    try:
        while True:
            obs = get_latest_robot_obs(robot_env)
            obs_stamp = obs_timeline.append(obs)

            # The current frame is retained for debug/display every tick, but
            # only a fresh coherent robot bundle is allowed into policy history.
            visual_obs_rgb = choose_visual_obs(obs, args.camera_source)
            if policy_uses_tactile:
                fsr = build_tactile_from_obs(obs, policy_tactile_key)
                if tactile_log is not None:
                    tactile_log.log(debug_step, fsr)

            proprio_inputs: Optional[Dict[str, Optional[np.ndarray]]] = None
            if obs_stamp is not None:
                if args.use_right_proprio:
                    legacy_joint_proprio = np.asarray(
                        obs["right_hand"], dtype=np.float32
                    )
                proprio_inputs = _build_per_step_proprio_inputs(
                    obs=obs,
                    proprio_mode=proprio_mode,
                    ik_solver=ik_solver,
                    fingertip_fk=fingertip_fk,
                    legacy_joint_proprio=legacy_joint_proprio,
                )
                if pose_interp is None:
                    # Async startup needs a valid command while CUDA is busy.
                    # Capture and hard-hold the measured joints rather than
                    # solving FK->IK for an equivalent wrist pose (a 7-DoF IK
                    # can otherwise jump to a different null-space branch).
                    hold_time = time.monotonic()
                    startup_hold_right_arm = np.asarray(
                        obs["right_arm"], np.float32
                    ).copy()
                    startup_hold_right_hand = np.asarray(
                        obs["right_hand"], np.float32
                    ).copy()
                    startup_hold_wrist6 = get_current_right_wrist_pose6d(
                        obs, ik_solver
                    )
                    anchor_right_wrist_pose6d = startup_hold_wrist6.copy()
                    arm_limiter.reset(
                        startup_hold_right_arm, timestamp=hold_time
                    )
                    hand_limiter.reset(
                        startup_hold_right_hand, timestamp=hold_time
                    )
                    pose_interp = PoseTrajectoryInterpolator(
                        np.array([hold_time]), startup_hold_wrist6[None]
                    )
                    hand_interp = MotorTrajectoryInterpolator(
                        np.array([hold_time]), startup_hold_right_hand[None]
                    )
                    plan_origin_mono = obs_stamp.control_time_s

            # Polling never blocks.  While a plan is pending, the main thread
            # continues sampling and sending the previous interpolation.
            now = time.monotonic()
            need_predict = (
                (chunk_id < 0 or now >= next_replan)
                and not inference_pending
                and obs_stamp is not None
            )
            prediction_result = planner.poll_result()
            if prediction_result is not None and not startup_warmup_completed:
                inference_pending = False
                if prediction_result.error is not None:
                    raise RuntimeError(
                        "asynchronous startup warmup inference failed:\n"
                        + str(prediction_result.error_traceback)
                    ) from prediction_result.error
                if prediction_result.request_id < 0:
                    raise RuntimeError("async observation worker failed")
                warmup_s = (
                    prediction_result.inference_finished
                    - prediction_result.inference_started
                )
                inference_lead_ema_s = warmup_s
                startup_warmup_completed = True
                print(
                    "[relative][startup] discarded warmup result: "
                    f"infer={warmup_s * 1000.0:.1f}ms; "
                    "arm/hand remain at the captured startup joints."
                )
                prediction_result = None

            if prediction_result is not None:
                result_applied_mono = time.monotonic()
                inference_pending = False
                if prediction_result.error is not None:
                    raise RuntimeError(
                        "asynchronous relative-policy inference failed:\n"
                        + str(prediction_result.error_traceback)
                    ) from prediction_result.error
                if prediction_result.request_id < 0:
                    raise RuntimeError("async observation worker failed")
                assert prediction_result.output is not None
                candidate_output = np.asarray(
                    prediction_result.output, np.float32
                )
                if candidate_output.ndim != 2 or len(candidate_output) == 0:
                    raise RuntimeError(
                        "relative policy must return a non-empty (H,D) chunk, "
                        f"got {candidate_output.shape}"
                    )
                candidate_waypoint_indices = np.arange(
                    ANCHOR_OFFSET, len(candidate_output), dtype=np.int64
                )
                if len(candidate_waypoint_indices) == 0:
                    raise RuntimeError(
                        f"anchor_offset={ANCHOR_OFFSET} leaves no action in "
                        f"a {len(candidate_output)}-frame prediction"
                    )
                source_obs = {
                    "right_hand": prediction_result.source_right_hand.copy()
                }
                (
                    pose_chunk,
                    hand_chunk,
                    candidate_held_wrist,
                    candidate_held_hand,
                ) = _decode_single_absolute_chunk(
                    output=candidate_output,
                    waypoint_indices=candidate_waypoint_indices,
                    obs=source_obs,
                    anchor_wrist_pose6d=(
                        prediction_result.source_anchor_wrist6
                    ),
                    t_et=T_ET,
                    hand_action_mode=hand_action_mode,
                    relative_hand_action=bool(
                        getattr(
                            policy.model_cfg.dataset,
                            "relative_hand_action",
                            False,
                        )
                    ),
                    no_eef_policy=no_eef_policy,
                    eef_only_policy=eef_only_policy,
                    fingertip_ik=fingertip_ik,
                    clip_hand=args.clip_hand,
                    hand_lower=hand_lower,
                    hand_upper=hand_upper,
                    held_wrist_pose6d=held_right_wrist_pose6d,
                    held_hand=held_right_hand,
                )
                assert pose_interp is not None
                assert hand_interp is not None
                (
                    candidate_pose_interp,
                    candidate_hand_interp,
                    arm_scheduled,
                    hand_scheduled,
                    arm_first_index,
                    hand_first_index,
                ) = schedule_absolute_trajectory_chunk(
                    pose_interp=pose_interp,
                    hand_interp=hand_interp,
                    pose_chunk=pose_chunk,
                    hand_chunk=hand_chunk,
                    waypoint_indices=candidate_waypoint_indices,
                    timeline_origin_mono=(
                        prediction_result.source_observation_time
                    ),
                    waypoint_dt=waypoint_dt,
                    arm_latency_s=arm_latency_s,
                    hand_latency_s=hand_latency_s,
                    safety_margin_s=safety_margin_s,
                    max_pos_speed=args.max_pos_speed,
                    max_rot_speed=args.max_rot_speed,
                    max_hand_speed=args.max_hand_speed,
                )
                first_valid_index = max(
                    arm_first_index, hand_first_index
                )
                last_index = int(candidate_waypoint_indices[-1])
                plan_exhausted = (
                    first_valid_index > last_index
                    or arm_scheduled <= 0
                    or hand_scheduled <= 0
                )
                inference_s = (
                    prediction_result.inference_finished
                    - prediction_result.inference_started
                )
                if inference_lead_ema_s <= 0.0:
                    inference_lead_ema_s = inference_s
                else:
                    inference_lead_ema_s = (
                        0.8 * inference_lead_ema_s + 0.2 * inference_s
                    )
                apply_age_s = max(
                    result_applied_mono
                    - prediction_result.source_observation_time,
                    0.0,
                )
                result_wait_s = max(
                    result_applied_mono
                    - prediction_result.inference_finished,
                    0.0,
                )
                if plan_exhausted:
                    # The scheduler's generic last-row fallback is useful for
                    # many callers but unsafe at startup: it would execute an
                    # old relative target. Preserve the running hold/old plan.
                    next_replan = result_applied_mono + control_dt
                else:
                    pose_interp = candidate_pose_interp
                    hand_interp = candidate_hand_interp
                    output = candidate_output
                    waypoint_indices = candidate_waypoint_indices
                    held_right_wrist_pose6d = candidate_held_wrist
                    held_right_hand = candidate_held_hand
                    anchor_right_wrist_pose6d = (
                        prediction_result.source_anchor_wrist6.copy()
                    )
                    plan_origin_mono = (
                        prediction_result.source_observation_time
                    )
                    next_replan = _relative_replan_request_deadline(
                        timeline_origin_mono=plan_origin_mono,
                        result_applied_mono=result_applied_mono,
                        first_valid_index=first_valid_index,
                        last_index=last_index,
                        exec_steps=chunk_exec_steps,
                        waypoint_dt=waypoint_dt,
                        control_dt=control_dt,
                        inference_lead_s=inference_lead_ema_s,
                    )
                    act_index = first_valid_index
                    chunk_id += 1
                    if startup_motion_start_mono is None:
                        startup_motion_start_mono = result_applied_mono
                        print(
                            "[relative][startup] first executable plan "
                            f"accepted; ramping for {args.startup_ramp_s:.2f}s."
                        )
                timestamp_source = (
                    "robot"
                    if prediction_result.used_source_timestamp
                    else "receive"
                )
                plan_label = "discarded-stale" if plan_exhausted else str(chunk_id)
                print(
                    f"[relative plan {plan_label}] shape={candidate_output.shape} "
                    f"infer={inference_s * 1000.0:.1f}ms "
                    f"result_wait={result_wait_s * 1000.0:.1f}ms "
                    f"apply_age={apply_age_s * 1000.0:.1f}ms "
                    f"time={timestamp_source} "
                    f"drop_arm={max(arm_first_index-ANCHOR_OFFSET, 0)}/"
                    f"{len(candidate_waypoint_indices)} "
                    f"drop_hand={max(hand_first_index-ANCHOR_OFFSET, 0)}/"
                    f"{len(candidate_waypoint_indices)} "
                    f"valid_arm={arm_scheduled}@{arm_first_index} "
                    f"valid_hand={hand_scheduled}@{hand_first_index} "
                    f"next_in={max(next_replan-result_applied_mono, 0.0)*1000.0:.1f}ms"
                )

            # All RealPolicy history access lives in the worker. A fresh frame
            # without a plan id only advances history; a frame with an id also
            # snapshots the exact timestamp and anchors used by that plan.
            if obs_stamp is not None:
                assert proprio_inputs is not None
                request_id = next_request_id if need_predict else None
                request_anchor = (
                    get_current_right_wrist_pose6d(obs, ik_solver)
                    if need_predict else None
                )
                submitted = planner.submit(
                    _AsyncRelativeObservationRequest(
                        visual_obs=np.asarray(visual_obs_rgb).copy(),
                        joint_proprio=(
                            None
                            if proprio_inputs["joint_proprio"] is None
                            else np.asarray(
                                proprio_inputs["joint_proprio"], np.float32
                            ).copy()
                        ),
                        wrist_pose6=(
                            None
                            if proprio_inputs["wrist_pose6"] is None
                            else np.asarray(
                                proprio_inputs["wrist_pose6"], np.float32
                            ).copy()
                        ),
                        fingertip_pose_wrist=(
                            None
                            if proprio_inputs["fingertip_pose_wrist"] is None
                            else np.asarray(
                                proprio_inputs["fingertip_pose_wrist"],
                                np.float32,
                            ).copy()
                        ),
                        fsr=(
                            None
                            if fsr is None
                            else np.asarray(fsr, np.float32).copy()
                        ),
                        request_id=request_id,
                        source_observation_time=(
                            obs_stamp.control_time_s
                            if need_predict else None
                        ),
                        source_anchor_wrist6=request_anchor,
                        source_right_hand=(
                            np.asarray(obs["right_hand"], np.float32).copy()
                            if need_predict else None
                        ),
                        used_source_timestamp=obs_stamp.used_source_timestamp,
                    )
                )
                if need_predict:
                    inference_pending = True
                    next_request_id += 1
                elif not submitted:
                    dropped_worker_observations += 1

            # debug: save the frame seen by the policy + rebuild video
            if args.debug:
                img_bgr = cv2.cvtColor(visual_obs_rgb, cv2.COLOR_RGB2BGR)
                img_path = os.path.join(debug_imgs_dir, f"img_{debug_step:06d}.png")
                cv2.imwrite(img_path, img_bgr)
                # Rebuilding the mp4 from scratch every step gets expensive
                # quickly; throttle to every ``--debug_video_every`` steps.
                if args.debug_video_every > 0 and (
                    (debug_step + 1) % args.debug_video_every == 0
                ):
                    video_path = os.path.join(run_dir, "rollout_video.mp4")
                    rebuild_debug_video(debug_imgs_dir, video_path, fps=args.control_hz)

            assert pose_interp is not None and hand_interp is not None
            assert plan_origin_mono is not None
            assert startup_hold_right_arm is not None
            assert startup_hold_right_hand is not None
            assert startup_hold_wrist6 is not None
            sample_time = time.monotonic()
            if output is None:
                act_index = ANCHOR_OFFSET
                act = np.zeros(6, dtype=np.float32)
                relative_pose6d = np.zeros(6, dtype=np.float32)
            else:
                act_index = int(
                    np.clip(
                        np.floor(
                            (sample_time + arm_latency_s - plan_origin_mono)
                            / waypoint_dt
                        ),
                        ANCHOR_OFFSET,
                        len(output) - 1,
                    )
                )
                act = output[act_index]
                relative_pose6d = (
                    np.zeros(6, dtype=np.float32)
                    if no_eef_policy else act[:6].astype(np.float32)
                )
            if args.debug:
                np.save(os.path.join(debug_act_dir, f"act_{debug_step:06d}.npy"), act)

            # Sample the effect trajectory ahead by each actuator's latency.
            # Wrist interpolation is linear in xyz plus SO(3) Slerp; the hand
            # is linear in joint space and then passes a per-joint limiter.
            current_right_wrist_pose6d = get_current_right_wrist_pose6d(
                obs, ik_solver
            )
            target_right_wrist_pose6d = np.asarray(
                pose_interp(sample_time + arm_latency_s),
                dtype=np.float32,
            )
            target_right_hand = np.asarray(
                hand_interp(sample_time + hand_latency_s),
                dtype=np.float32,
            )
            startup_hard_hold = startup_motion_start_mono is None
            if startup_hard_hold:
                target_right_wrist_pose6d = startup_hold_wrist6.copy()
                target_right_hand = startup_hold_right_hand.copy()
            else:
                (
                    target_right_wrist_pose6d,
                    target_right_hand,
                    _startup_alpha,
                ) = _blend_relative_startup_targets(
                    hold_wrist6=startup_hold_wrist6,
                    target_wrist6=target_right_wrist_pose6d,
                    hold_hand=startup_hold_right_hand,
                    target_hand=target_right_hand,
                    elapsed_s=sample_time - startup_motion_start_mono,
                    ramp_s=args.startup_ramp_s,
                )
            measured_right_hand = np.asarray(
                obs["right_hand"], dtype=np.float32
            )
            if startup_hard_hold:
                target_right_hand = hand_limiter.hold(
                    measured_right_hand, timestamp=sample_time
                )
            else:
                target_right_hand = hand_limiter.limit(
                    target_right_hand,
                    measured=measured_right_hand,
                    timestamp=sample_time,
                )
            if args.debug:
                np.save(
                    os.path.join(debug_pose_dir, f"pose_{debug_step:06d}.npy"),
                    current_right_wrist_pose6d,
                )

            # 6) arm IK
            measured_right_arm = np.asarray(
                obs["right_arm"], dtype=np.float32
            )
            ik_success = True
            ik_angles = measured_right_arm[::-1].copy()
            target_right_wrist_pose4x4 = vec6dof_to_homogeneous_matrix(
                translation=target_right_wrist_pose6d[:3],
                rotation_vector=target_right_wrist_pose6d[3:],
            )
            # ``calc_single_arm_ik`` returns success=True even for targets it
            # cannot actually reach (it only flags False for grossly
            # out-of-range, metre-scale targets).  For cm-scale targets in the
            # workspace's degenerate/boundary regions it happily returns a
            # solution whose FK is hundreds of mm off, which drives the arm in
            # the wrong direction.  So we do NOT trust ``ik_success`` alone:
            # we FK the returned solution and reject it when the pose error
            # exceeds the tolerances.  This mirrors ``verify_ik.py``,
            # which also scores IK quality via FK residual rather than the
            # raw success flag.
            ik_pos_err = 0.0 if startup_hard_hold else float("inf")
            ik_rot_err = 0.0 if startup_hard_hold else float("inf")
            if not startup_hard_hold:
                ik_seed_arm = (
                    arm_limiter.previous_position
                    if arm_limiter.previous_position is not None
                    else measured_right_arm
                )
                ik_success, ik_angles = ik_solver.ik(
                    target_pose=target_right_wrist_pose4x4,
                    body_angles=obs["body_dof"][::-1],
                    init_arm_angles=ik_seed_arm[::-1],
                    side="right",
                )
            if not startup_hard_hold and ik_success:
                T_reached = ik_solver.fk(
                    arm_angles=ik_angles,
                    body_angles=obs["body_dof"][::-1],
                    side="right",
                )
                ik_pos_err = float(
                    np.linalg.norm(
                        T_reached[:3, 3] - target_right_wrist_pose4x4[:3, 3]
                    )
                )
                ik_rot_err = float(
                    np.linalg.norm(
                        T_reached[:3, :3] - target_right_wrist_pose4x4[:3, :3]
                    )
                )
            ik_ok = startup_hard_hold or (
                ik_success
                and ik_pos_err <= args.ik_pos_tol
                and ik_rot_err <= args.ik_rot_tol
            )
            if startup_hard_hold:
                target_right_arm = arm_limiter.hold(
                    measured_right_arm, timestamp=sample_time
                )
            elif ik_ok:
                target_right_arm = arm_limiter.limit(
                    ik_angles[::-1].astype(np.float32),
                    measured=measured_right_arm,
                    timestamp=sample_time,
                )
            else:
                print(
                    f"[WARN] IK rejected (success={ik_success}, "
                    f"pos_err={ik_pos_err * 1000:.1f}mm, rot_err={ik_rot_err:.3f}); "
                    "holding previous smoothed arm command."
                )
                target_right_arm = arm_limiter.hold(
                    measured_right_arm,
                    timestamp=sample_time,
                )

            # Always report + log the IK tracking error (FK residual against
            # the requested target), regardless of accept/reject.
            print(
                f"[IK] step={debug_step} accepted={ik_ok} "
                f"pos_err={ik_pos_err * 1000:.2f}mm rot_err={ik_rot_err:.4f}"
            )
            ik_err_log.append(
                (int(debug_step), int(bool(ik_ok)), ik_pos_err, ik_rot_err)
            )

            if args.debug:
                np.save(os.path.join(debug_ik_dir, f"ik_{debug_step:06d}.npy"), ik_angles)
            debug_step += 1

            # Hand joints were decoded for the full chunk and are sampled
            # continuously above before their final kinematic limiter.

            # 8) send
            action_buffer = build_robot_action_buffer(
                obs=obs,
                target_right_arm=target_right_arm,
                target_right_hand=target_right_hand,
                mode=mode,
            )
            robot_env.send_action(action_buffer, immediate=False)

            # 9) optional live display
            if enable_show:
                show_img = cv2.cvtColor(visual_obs_rgb, cv2.COLOR_RGB2BGR)
                txt1 = f"act_index={act_index}"
                txt2 = f"rel_pose={np.array2string(relative_pose6d, precision=3)}"
                cv2.putText(show_img, txt1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(show_img, txt2[:80], (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                cv2.imshow("robot_relative_policy_rollout", show_img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Quit by keyboard.")
                    break

            # 9b) live matplotlib dashboard (image + action chunk + tracking)
            if (
                live_viz is not None
                and output is not None
                and anchor_right_wrist_pose6d is not None
            ):
                # Where the arm will actually end up = FK of the angles we just
                # commanded (accepted IK solution, or the held current angles
                # when IK was rejected).
                try:
                    T_sent = ik_solver.fk(
                        arm_angles=target_right_arm[::-1],
                        body_angles=obs["body_dof"][::-1],
                        side="right",
                    )
                    reached_xyz = T_sent[:3, 3].astype(np.float32)
                except Exception:  # noqa: BLE001
                    reached_xyz = target_right_wrist_pose6d[:3]
                alive = live_viz.update(
                    wrist_rgb=visual_obs_rgb,
                    output=output,
                    anchor_pose6d=anchor_right_wrist_pose6d,
                    anchor_offset=ANCHOR_OFFSET,
                    act_index=act_index,
                    t_et=T_ET,
                    step=debug_step,
                    target_xyz=target_right_wrist_pose6d[:3],
                    reached_xyz=reached_xyz,
                    live_xyz=current_right_wrist_pose6d[:3],
                    ik_ok=ik_ok,
                    ik_solver=ik_solver,
                    obs=obs,
                    fsr=fsr,
                    tactile_key=policy_tactile_key,
                )
                if not alive:
                    print("Live debug window closed; stopping rollout.")
                    break

            # Absolute monotonic deadlines prevent catch-up command bursts.
            tick_finished = time.monotonic()
            next_tick += control_dt
            if next_tick <= tick_finished:
                missed = int(
                    (tick_finished - next_tick) // control_dt
                ) + 1
                next_tick += missed * control_dt
                missed_ticks_total += missed
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

    finally:
        planner.close()
        print(
            f"[timing] single-arm control ticks skipped={missed_ticks_total}; "
            f"worker observations dropped={dropped_worker_observations}"
        )
        if enable_show:
            cv2.destroyAllWindows()
        if live_viz is not None:
            live_viz.close()
        if tactile_log is not None:
            tactile_log.dump()

        # Save the IK tracking-error log (independent of --debug so it is
        # always available after a run).  When --debug is on this lands inside
        # the per-run timestamped dir alongside the other data.
        if ik_err_log:
            os.makedirs(run_dir, exist_ok=True)
            err_arr = np.array(
                [(s, ok, p, r) for (s, ok, p, r) in ik_err_log],
                dtype=np.float64,
            )  # columns: step, accepted, pos_err_m, rot_err
            npy_path = os.path.join(run_dir, "ik_tracking_error.npy")
            csv_path = os.path.join(run_dir, "ik_tracking_error.csv")
            np.save(npy_path, err_arr)
            header = "step,accepted,pos_err_m,rot_err"
            np.savetxt(csv_path, err_arr, delimiter=",", header=header,
                       comments="", fmt=["%d", "%d", "%.6f", "%.6f"])
            pos_mm = err_arr[:, 2] * 1000.0
            rot = err_arr[:, 3]
            n_rej = int((err_arr[:, 1] == 0).sum())
            print(
                "[IK] tracking-error summary over "
                f"{len(err_arr)} steps: "
                f"pos_err(mm) mean={pos_mm.mean():.2f} median={np.median(pos_mm):.2f} "
                f"max={pos_mm.max():.2f} | "
                f"rot_err mean={rot.mean():.4f} max={rot.max():.4f} | "
                f"rejected={n_rej}/{len(err_arr)}"
            )
            print(f"[IK] saved error log to {npy_path} and {csv_path}")

        if args.debug:
            # Final video flush so users always get a complete file even if
            # ``debug_video_every`` did not land on the last step.
            try:
                video_path = os.path.join(run_dir, "rollout_video.mp4")
                rebuild_debug_video(debug_imgs_dir, video_path, fps=args.control_hz)
            except Exception as e:  # noqa: BLE001
                print(f"[DEBUG] final video flush failed: {e}")
            print(
                f"[DEBUG] Saved {debug_step} steps: imgs={debug_imgs_dir}, "
                f"act={debug_act_dir}, ik={debug_ik_dir}, pose={debug_pose_dir}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Rollout with the UVTA relative-control policy")

    # policy
    parser.add_argument("--model_path", type=str, required=True, help="UVTA training output dir")
    parser.add_argument("--ckpt", type=int, required=True, help="checkpoint id")
    parser.add_argument(
        "--use_ema",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use the EMA checkpoint (checkpoints/ema_epoch_<ckpt>.ckpt) "
            "instead of the raw checkpoint (checkpoints/epoch_<ckpt>.ckpt).  "
            "Pass --use_ema to force EMA, --no-use_ema to force the raw "
            "weights; leave unset to follow the training config's use_ema."
        ),
    )

    # control
    parser.add_argument(
        "--control_hz",
        type=float,
        default=50.0,
        help="Rate for sampling and sending the interpolated command trajectory.",
    )
    parser.add_argument(
        "--waypoint_hz",
        type=float,
        default=29.64,
        help="Dataset/action-chunk waypoint rate; keep equal to recording rate.",
    )
    parser.add_argument(
        "--observation_buffer_size",
        type=int,
        default=256,
        help="Unique timestamped robot observations retained for diagnostics.",
    )
    parser.add_argument(
        "--observation_timestamp_max_skew_s",
        type=float,
        default=5.0,
        help="Maximum producer/local wall-clock skew before receive-time fallback.",
    )
    parser.add_argument(
        "--observation_timestamp_max_age_ms",
        type=float,
        default=500.0,
        help="Maximum credible producer-to-receive observation age.",
    )
    parser.add_argument("--arm_latency_ms", type=float, default=100.0)
    parser.add_argument("--hand_latency_ms", type=float, default=10.0)
    parser.add_argument(
        "--safety_margin_ms",
        type=float,
        default=1000.0 / 29.64,
        help="Future-time margin required before accepting a predicted waypoint.",
    )
    parser.add_argument(
        "--startup_inference_warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Discard the first (usually CUDA-warmup) async prediction while "
            "hard-holding the captured startup arm and hand joints."
        ),
    )
    parser.add_argument(
        "--startup_ramp_s",
        type=float,
        default=0.5,
        help=(
            "Seconds used to blend from the captured startup wrist/hand "
            "targets into the first executable plan (0 disables the blend)."
        ),
    )
    parser.add_argument("--max_pos_speed", type=float, default=0.25)
    parser.add_argument("--max_rot_speed", type=float, default=0.16)
    parser.add_argument("--max_hand_speed", type=float, default=2.0)
    parser.add_argument("--arm_joint_max_velocity", type=float, default=1.0)
    parser.add_argument("--arm_joint_max_acceleration", type=float, default=4.0)
    parser.add_argument("--hand_joint_max_velocity", type=float, default=2.0)
    parser.add_argument("--hand_joint_max_acceleration", type=float, default=8.0)
    parser.add_argument(
        "--exec_horizon",
        type=int,
        default=None,
        help=(
            "Optional override for how many frames of a chunk to execute "
            "before re-planning.  Defaults to action_horizon - anchor_offset, "
            "clamped to the predicted chunk length.  Leave unset to follow the "
            "trained action_horizon."
        ),
    )
    parser.add_argument(
        "--anchor_offset",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "Index of the first chunk frame to EXECUTE.  With the current "
            "training convention the eef action comes from `pose_action` "
            "anchored on the current state pose[t], so action[0] = "
            "pose[t]^-1 @ pose_action[t] is the real first commanded step and "
            "MUST be executed -> use 0 (default).  Set 1 ONLY for LEGACY "
            "checkpoints trained with `pose` reused as the action, where "
            "action[0] is the anchor identity (a no-op) and should be skipped."
        ),
    )
    parser.add_argument(
        "--mode", type=str, default=None, choices=["right"],
        help="deployment arm side (default: infer from the model's dataset.arms)",
    )

    # camera
    parser.add_argument(
        "--camera_source",
        type=str,
        default=None,
        choices=["right_wrist", "left_wrist", "right_eye", "left_eye"],
        help=(
            "camera stream to feed to the policy "
            "(default: infer from dataset.load_camera_ids)"
        ),
    )

    # urdf
    parser.add_argument(
        "--urdf_path",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "allset_kinematics/urdf/robot_description/urdf/robot_model.urdf",
        ),
        help="arm URDF path",
    )

    # IK acceptance tolerances.  ``calc_single_arm_ik``'s success flag is
    # unreliable (returns True for unreachable cm-scale targets), so we
    # validate every solution by FK residual and reject it if it exceeds
    # these bounds, holding the current arm pose instead of sending a bad
    # solution that would drive the arm in the wrong direction.
    parser.add_argument(
        "--ik_pos_tol",
        type=float,
        default=0.02,
        help="max FK position error (m) for an IK solution to be accepted",
    )
    parser.add_argument(
        "--ik_rot_tol",
        type=float,
        default=0.1,
        help=(
            "max FK rotation error (Frobenius norm of the 3x3 rotation "
            "matrix difference) for an IK solution to be accepted"
        ),
    )

    # T_ET
    parser.add_argument(
        "--t_et",
        type=str,
        default=None,
        help=(
            'Legacy fixed wrist offset.  Either "x,y,z,rx,ry,rz" or 16 '
            "comma-separated numbers (row-major 4x4).  For method-B "
            "checkpoints leave this unset (identity) -- a non-identity "
            "value triggers the legacy decode and will warn at startup."
        ),
    )

    # hand command shaping
    parser.add_argument("--clip_hand", action="store_true")
    parser.add_argument("--hand_lower", nargs="+", type=float, default=None)
    parser.add_argument("--hand_upper", nargs="+", type=float, default=None)

    # legacy proprio override
    parser.add_argument(
        "--use_right_proprio",
        action="store_true",
        help=(
            "Override the joint_proprio fed to the policy with the raw "
            "right_hand observation.  Only meaningful for proprio_mode "
            "values that actually consume joint angles (joint, both); "
            "ignored otherwise."
        ),
    )

    # reset-to-zero (move the robot to a recorded episode's first frame
    # before starting inference)
    parser.add_argument(
        "--reset2zero",
        action="store_true",
        help=(
            "Before the rollout, read the first frame's pose + "
            "proprioception from --reset2zero_zarr_root/--reset2zero_episode, "
            "solve arm IK for that pose and stream it so the robot resets to "
            "that start configuration, then begin inference."
        ),
    )
    parser.add_argument(
        "--reset2zero_zarr_root",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data/20260514_teleop",
        ),
        help="zarr root holding the reset reference episode",
    )
    parser.add_argument(
        "--reset2zero_episode",
        type=str,
        default="episode_1",
        help="episode name whose first frame is used as the reset target",
    )
    parser.add_argument(
        "--reset2zero_hold_steps",
        type=int,
        default=20,
        help="how many control cycles to stream the reset target to settle",
    )

    # env
    parser.add_argument("--enable_tactile", action="store_true")
    parser.add_argument("--show", action="store_true")

    # debug
    parser.add_argument("--debug", action="store_true", help="save per-step images, actions, IK and pose plus a video")
    parser.add_argument(
        "--no_live_viz",
        action="store_true",
        help=(
            "Disable the real-time matplotlib debug dashboard that --debug "
            "opens by default (wrist image + action chunk EEF trajectory + "
            "IK tracking history).  Data is still saved to --debug_dir."
        ),
    )
    parser.add_argument(
        "--live_viz_history",
        type=int,
        default=200,
        help="number of recent control steps to keep in the IK tracking plot",
    )
    parser.add_argument(
        "--debug_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "debug",
        ),
        help="debug data root dir",
    )
    parser.add_argument(
        "--debug_video_every",
        type=int,
        default=0,
        help=(
            "Rebuild the debug rollout video every N control steps.  Each "
            "rebuild re-encodes ALL PNGs in the debug dir from scratch, so a "
            "non-zero value makes the loop progressively slower (O(N^2)) and "
            "starves the observation thread.  Default 0 = only rebuild once "
            "at exit (recommended).  Set >0 only for short debugging runs."
        ),
    )

    args = parser.parse_args()
    main(args)
