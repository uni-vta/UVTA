"""Robot rollout v2: receding-horizon + trajectory-interpolation smoothing.

Same Method-B relative-pose policy as ``relative_policy_rollout.py``, but the
action-chunk execution path follows DexUMI ``eval_xhand.py`` rather than ACT
temporal ensembling (that lives in ``relative_policy_rollout_v1.py``):

    obs -> predict_action (full chunk)
        -> relative pose -> absolute pose
        -> discard past actions by latency
        -> schedule_waypoint -> Pose/Motor trajectory interpolation
        -> sample interpolators each control step -> arm IK -> the robot
        -> after exec_horizon * dt, replan (no cross-chunk weighted mix)

Aligned with the *Method B* sample layout used by the trainer (see
``DexUMI/dexumi/diffusion_policy/dataloader/uvta_dataset.py`` and the
``train_diffusion_policy_v18+`` configs):

    sample length = (obs_horizon - 1) * down_sample_steps + pred_horizon
    obs window    = sample[0 :: d, ..., (H-1)*d]      (anchor = t @ (H-1)*d)
    action target = sample[(H-1)*d : (H-1)*d + pred_horizon]
                    -> action[k] = T_state(t)^{-1} @ T_action(t+k)

The eef action comes from the ``pose_action`` stream anchored on STATE
``pose[t]``, so ``action[0]`` is the real first commanded step
(``--anchor_offset 0``).  Legacy identity-anchor checkpoints use
``--anchor_offset 1``.

Proprio handling is automatic from ``policy.proprio_mode``; the legacy
``--use_right_proprio`` flag still works for back-compat.
"""

from __future__ import annotations

import numbers
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union

import cv2
import numpy as np
import torch
import scipy.interpolate as si
import scipy.spatial.transform as st
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
# (loads the per-finger region-label maps) and reused every frame.
_DEFORM_CONV = None


def _get_deform_converter():
    """Return a cached ``DeformToTaxel`` (right hand, reduce='sum'), matching
    the training-data build.  Imported by file path so it works regardless of
    whether ``deform_to_human`` is on ``sys.path`` / a package."""
    global _DEFORM_CONV
    if _DEFORM_CONV is None:
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
        # reduce='sum' + hand='right' == the teleop/skeleton fsr build recipe.
        _DEFORM_CONV = mod.DeformToTaxel(hand="right", reduce="sum")
    return _DEFORM_CONV


def build_tactile_from_obs(
    obs: Dict[str, Any], tactile_key: str, hand: str = "right"
) -> np.ndarray:
    """Assemble the per-frame tactile vector matching the training layout.

    - ``tactile_key == 'force'`` -> per-finger resultant force from
      ``/observe/tactile/{hand}_{finger}/force6d`` (first 3 dims) -> ``(5,)``.
    - otherwise (``'fsr'``/deform) -> five ``.../deform`` maps run through the
      dataset's ``DeformToTaxel`` (per-finger region-sum) -> ``(100,)``.

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

    # fsr / deform: gather the 5 finger deform maps (finger order) and convert
    # with the exact dataset recipe.
    maps = []
    for finger in _TACTILE_FINGERS:
        key = f"/observe/tactile/{hand}_{finger}/deform"
        if key not in raw:
            raise KeyError(f"tactile deform key missing from obs: {key}")
        maps.append(np.asarray(raw[key]))
    fsr = _get_deform_converter().frame(maps)  # (100,) float32
    if fsr.shape[0] != len(_TACTILE_FINGERS) * _TAXELS_PER_FINGER:
        raise ValueError(
            f"deform->taxel produced {fsr.shape[0]} values, expected "
            f"{len(_TACTILE_FINGERS) * _TAXELS_PER_FINGER}"
        )
    return fsr.astype(np.float32)


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


def get_current_right_wrist_pose6d(
    obs: Dict[str, Any], ik_solver: Optional[LightweightIK]
) -> np.ndarray:
    if ik_solver is None:
        raise RuntimeError(
            "Cannot derive right wrist pose: no IK solver provided.  "
            "Pass --urdf_path so LightweightIK can be constructed."
        )
    T_wrist = ik_solver.fk(
        arm_angles=obs["right_arm"][::-1],   # sensor: [AJ1..AJ7] -> FK needs [AJ7..AJ1]
        body_angles=obs["body_dof"][::-1],   # sensor: [LBJ1..LBJ5] -> FK needs [LBJ5..LBJ1]
        side="right",
    )
    return homogeneous_matrix_to_6dof(T_wrist).astype(np.float32)


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
            anchor_right_hand = obs["right_hand"].copy()
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


# chips_teleop camera layout: camera_0 = RIGHT wrist, camera_2 = LEFT wrist,
# camera_1 = ego (unused).  Maps a training ``load_camera_ids`` entry to the
# live-obs key produced by ``get_latest_robot_obs``.
_CAMERA_ID_TO_OBS_KEY = {
    0: "right_wrist_img",
    2: "left_wrist_img",
    1: "left_eye_img",   # ego / head (unused in the bimanual wrist configs)
}


def gather_visual_obs(obs: Dict[str, Any], camera_ids, fallback_source: str):
    """Return the visual observation matching the policy's ``camera_ids``.

    * single camera  -> a single ``(H, W, 3)`` RGB frame (legacy behaviour;
      ``fallback_source`` picks the stream when ``camera_ids`` is unknown).
    * multiple cameras -> a **list** of ``(H, W, 3)`` frames in ``camera_ids``
      order, ready to hand straight to ``RealPolicy.push_observation`` /
      ``predict_action`` (which stacks them along the obs-horizon axis in the
      SAME order the trainer used).  For chips_teleop that is
      ``[camera_0 = right wrist, camera_2 = left wrist]``.
    """
    ids = [int(c) for c in camera_ids]
    if len(ids) <= 1:
        # Single-camera policy: honour the explicit --camera_source.
        return choose_visual_obs(obs, fallback_source)
    frames = []
    for cid in ids:
        key = _CAMERA_ID_TO_OBS_KEY.get(cid)
        if key is None:
            raise ValueError(
                f"camera id {cid} has no live-obs mapping; extend "
                "_CAMERA_ID_TO_OBS_KEY for your robot."
            )
        img = obs.get(key)
        if img is None:
            raise RuntimeError(f"{key} (camera_{cid}) image is empty.")
        frames.append(ensure_rgb_hwc(img))
    return frames


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

            self.fig = plt.figure(figsize=(17, 5))
            # 1: wrist image (2D), 2: action chunk (3D), 3: IK tracking (2D)
            self.ax_img = self.fig.add_subplot(1, 3, 1)
            self.ax_chunk = self.fig.add_subplot(1, 3, 2, projection="3d")
            self.ax_track = self.fig.add_subplot(1, 3, 3)
            self.fig.canvas.manager.set_window_title(
                "rollout live debug (image | action chunk 3D | IK tracking)"
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


# =========================================================
# V2 DexUMI-style receding-horizon + trajectory interpolation
# =========================================================
# Fixed deploy defaults (not CLI flags), mirroring v1's reproducible-variant
# style.  Tunable after measuring the real robot.  These match DexUMI
# ``eval_xhand`` / ``ur5.py`` defaults where applicable.
_V2_ROBOT_ACTION_LATENCY = 0.1
_V2_HAND_ACTION_LATENCY = 0.3
_V2_MAX_POS_SPEED = 0.25  # m/s
_V2_MAX_ROT_SPEED = 0.16  # rad/s
_V2_MAX_HAND_SPEED = 2.0  # rad/s (joint-space L2)


def _rotation_distance(a: st.Rotation, b: st.Rotation) -> float:
    return (b * a.inv()).magnitude()


def _pose_distance(start_pose, end_pose):
    start_pose = np.asarray(start_pose)
    end_pose = np.asarray(end_pose)
    pos_dist = float(np.linalg.norm(end_pose[:3] - start_pose[:3]))
    rot_dist = _rotation_distance(
        st.Rotation.from_rotvec(start_pose[3:]),
        st.Rotation.from_rotvec(end_pose[3:]),
    )
    return pos_dist, rot_dist


class PoseTrajectoryInterpolator:
    """Linear position + Slerp orientation interpolator (DexUMI copy).

    Vendored from ``dexumi/real_env/common/pose_trajectory_interpolator.py``
    so this file stays self-contained and omits the upstream debug print.
    """

    def __init__(self, times: np.ndarray, poses: np.ndarray):
        assert len(times) >= 1
        assert len(poses) == len(times)
        times = np.asarray(times)
        poses = np.asarray(poses)
        if len(times) == 1:
            self.single_step = True
            self._times = times
            self._poses = poses
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])
            self.pos_interp = si.interp1d(
                times, poses[:, :3], axis=0, assume_sorted=True
            )
            self.rot_interp = st.Slerp(times, st.Rotation.from_rotvec(poses[:, 3:]))

    @property
    def times(self) -> np.ndarray:
        return self._times if self.single_step else self.pos_interp.x

    @property
    def poses(self) -> np.ndarray:
        if self.single_step:
            return self._poses
        poses = np.zeros((len(self.times), 6))
        poses[:, :3] = self.pos_interp.y
        poses[:, 3:] = self.rot_interp(self.times).as_rotvec()
        return poses

    def trim(self, start_t: float, end_t: float) -> "PoseTrajectoryInterpolator":
        assert start_t <= end_t
        times = self.times
        keep_times = times[(start_t < times) & (times < end_t)]
        all_times = np.unique(np.concatenate([[start_t], keep_times, [end_t]]))
        return PoseTrajectoryInterpolator(times=all_times, poses=self(all_times))

    def schedule_waypoint(
        self,
        pose,
        time,
        max_pos_speed=np.inf,
        max_rot_speed=np.inf,
        curr_time=None,
        last_waypoint_time=None,
    ) -> "PoseTrajectoryInterpolator":
        assert max_pos_speed > 0 and max_rot_speed > 0
        if last_waypoint_time is not None:
            assert curr_time is not None

        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
                return self
            start_time = max(curr_time, start_time)
            if last_waypoint_time is not None:
                end_time = (
                    curr_time
                    if time <= last_waypoint_time
                    else max(last_waypoint_time, curr_time)
                )
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)
        assert start_time <= end_time <= time

        trimmed_interp = self.trim(start_time, end_time)
        duration = time - end_time
        end_pose = trimmed_interp(end_time)
        pos_dist, rot_dist = _pose_distance(pose, end_pose)
        duration = max(
            duration,
            max(pos_dist / max_pos_speed, rot_dist / max_rot_speed),
        )
        last_waypoint_time = end_time + duration
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.poses, [pose], axis=0)
        return PoseTrajectoryInterpolator(times, poses)

    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        is_single = isinstance(t, numbers.Number)
        if is_single:
            t = np.array([t])
        if self.single_step:
            pose = np.tile(self._poses[0], (len(t), 1))
        else:
            t = np.clip(t, self.times[0], self.times[-1])
            pose = np.zeros((len(t), 6))
            pose[:, :3] = self.pos_interp(t)
            pose[:, 3:] = self.rot_interp(t).as_rotvec()
        return pose[0] if is_single else pose


class MotorTrajectoryInterpolator:
    """Linear joint-space interpolator (DexUMI copy, no debug print)."""

    def __init__(self, times: np.ndarray, values: np.ndarray):
        assert len(times) >= 1 and len(values) == len(times)
        times = np.asarray(times)
        values = np.asarray(values)
        if len(times) == 1:
            self.single_step = True
            self._times = times
            self._values = values
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])
            self.interp = si.interp1d(times, values, axis=0, assume_sorted=True)

    @property
    def times(self) -> np.ndarray:
        return self._times if self.single_step else self.interp.x

    @property
    def values(self) -> np.ndarray:
        return self._values if self.single_step else self.interp.y

    def trim(self, start_t: float, end_t: float) -> "MotorTrajectoryInterpolator":
        assert start_t <= end_t
        times = self.times
        keep_times = times[(start_t < times) & (times < end_t)]
        all_times = np.unique(np.concatenate([[start_t], keep_times, [end_t]]))
        return MotorTrajectoryInterpolator(times=all_times, values=self(all_times))

    def schedule_waypoint(
        self,
        value,
        time,
        max_speed=np.inf,
        curr_time=None,
        last_waypoint_time=None,
    ) -> "MotorTrajectoryInterpolator":
        assert max_speed > 0
        if last_waypoint_time is not None:
            assert curr_time is not None

        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
                return self
            start_time = max(curr_time, start_time)
            if last_waypoint_time is not None:
                end_time = (
                    curr_time
                    if time <= last_waypoint_time
                    else max(last_waypoint_time, curr_time)
                )
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)
        assert start_time <= end_time <= time

        trimmed_interp = self.trim(start_time, end_time)
        duration = time - end_time
        end_value = trimmed_interp(end_time)
        duration = max(
            duration,
            float(np.linalg.norm(value - end_value)) / max_speed,
        )
        last_waypoint_time = end_time + duration
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        values = np.append(trimmed_interp.values, [value], axis=0)
        return MotorTrajectoryInterpolator(times, values)

    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        is_single = isinstance(t, numbers.Number)
        if is_single:
            t = np.array([t])
        if self.single_step:
            values = np.tile(self._values[0], (len(t), 1))
        else:
            t = np.clip(t, self.times[0], self.times[-1])
            values = self.interp(t)
        return values[0] if is_single else values


def _decode_chunk_absolute_targets(
    output: np.ndarray,
    anchor_offset: int,
    anchor_wrist_pose6d: np.ndarray,
    obs: Dict[str, Any],
    t_et: np.ndarray,
    hand_action_mode: str,
    relative_hand_action: bool,
    no_eef_policy: bool,
    eef_only_policy: bool,
    held_right_wrist_pose6d: Optional[np.ndarray],
    held_right_hand: Optional[np.ndarray],
    fingertip_ik: Optional["FingertipToJointIK"],
    clip_hand: bool,
    hand_lower: Optional[np.ndarray],
    hand_upper: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Decode one predicted chunk into absolute EEF poses and hand joints.

    Returns
    -------
    poses : (T, 6) absolute wrist xyz+rotvec (base frame)
    hands : (T, 22) absolute hand joint targets
    held_right_wrist_pose6d, held_right_hand : possibly latched hold poses
    """
    output = np.asarray(output, dtype=np.float32)
    n = len(output)
    poses = np.zeros((n, 6), dtype=np.float32)
    hands = np.zeros((n, 22), dtype=np.float32)
    hand_anchor: Optional[np.ndarray] = None

    for k in range(anchor_offset, n):
        act = output[k]
        if no_eef_policy:
            hand_action = act.astype(np.float32)
            assert held_right_wrist_pose6d is not None
            poses[k] = held_right_wrist_pose6d
        else:
            relative_pose6d = act[:6].astype(np.float32)
            hand_action = act[6:].astype(np.float32)
            poses[k] = recover_absolute_target_pose_from_relative(
                current_pose6d=anchor_wrist_pose6d,
                relative_pose6d=relative_pose6d,
                T_ET=t_et,
            )

        if eef_only_policy or hand_action.size == 0:
            if held_right_hand is None:
                held_right_hand = obs["right_hand"].astype(np.float32)
            hands[k] = held_right_hand
        elif hand_action_mode in ("fingertip", "fingertip_only"):
            assert fingertip_ik is not None
            hands[k] = convert_hand_action_fingertip(
                hand_action_5x6=hand_action.reshape(5, 6),
                ik=fingertip_ik,
                clip_hand=clip_hand,
                hand_lower=hand_lower,
                hand_upper=hand_upper,
            )
        else:
            hands[k], hand_anchor = convert_hand_action_joint(
                hand_action=hand_action,
                obs=obs,
                anchor_right_hand=hand_anchor,
                relative_hand_action=relative_hand_action,
                clip_hand=clip_hand,
                hand_lower=hand_lower,
                hand_upper=hand_upper,
            )

    return poses, hands, held_right_wrist_pose6d, held_right_hand


def _schedule_chunk_into_interpolators(
    pose_interp: PoseTrajectoryInterpolator,
    hand_interp: MotorTrajectoryInterpolator,
    poses: np.ndarray,
    hands: np.ndarray,
    anchor_offset: int,
    dt: float,
    t_actual_inference_mono: float,
    robot_action_latency: float,
    hand_action_latency: float,
    max_pos_speed: float,
    max_rot_speed: float,
    max_hand_speed: float,
) -> Tuple[PoseTrajectoryInterpolator, MotorTrajectoryInterpolator, int, int]:
    """Schedule still-valid chunk waypoints (DexUMI eval_xhand discard rule)."""
    n_action = len(poses)
    t_exec = time.monotonic()
    robot_times_mono = t_actual_inference_mono + np.arange(n_action) * dt
    hand_times_mono = t_actual_inference_mono + np.arange(n_action) * dt
    valid_robot = robot_times_mono >= (t_exec + robot_action_latency + dt)
    valid_hand = hand_times_mono >= (t_exec + hand_action_latency + dt)
    # Convert to wall clock for schedule_waypoint (matches eval_xhand).
    robot_times = robot_times_mono - time.monotonic() + time.time()
    hand_times = hand_times_mono - time.monotonic() + time.time()

    robot_scheduled = 0
    last_robot_wp = None
    for k in np.where(valid_robot)[0]:
        if k < anchor_offset:
            continue
        curr_time = time.time()
        pose_interp = pose_interp.schedule_waypoint(
            pose=poses[k],
            time=float(robot_times[k]),
            max_pos_speed=max_pos_speed,
            max_rot_speed=max_rot_speed,
            curr_time=curr_time,
            last_waypoint_time=last_robot_wp,
        )
        last_robot_wp = float(robot_times[k])
        robot_scheduled += 1

    hand_scheduled = 0
    last_hand_wp = None
    for k in np.where(valid_hand)[0]:
        if k < anchor_offset:
            continue
        curr_time = time.time()
        hand_interp = hand_interp.schedule_waypoint(
            value=hands[k],
            time=float(hand_times[k]),
            max_speed=max_hand_speed,
            curr_time=curr_time,
            last_waypoint_time=last_hand_wp,
        )
        last_hand_wp = float(hand_times[k])
        hand_scheduled += 1

    return pose_interp, hand_interp, robot_scheduled, hand_scheduled


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
    proprio_mode = policy.proprio_mode
    hand_action_mode = policy.hand_action_mode
    action_horizon = int(policy.action_horizon)
    down_sample_steps = int(policy.down_sample_steps)

    # -------- bimanual guard --------
    #
    # ``RealPolicy`` itself is fully bimanual (per-arm stats, multi-camera
    # stacking via ``gather_visual_obs``, per-arm FSR, per-arm action decode --
    # verified offline by ``scripts/smoke_bimanual_rollout.py``).  This *control
    # loop*, however, still assembles a single (right-arm) tactile/proprio
    # observation and commands a single arm through one anchor / interpolator /
    # IK side.  Running a bimanual checkpoint here would silently mis-map the
    # 2*N-D action onto one arm, so refuse up front and say exactly what is
    # missing.
    if getattr(policy, "num_arms", 1) > 1:
        raise NotImplementedError(
            "Loaded a BIMANUAL policy "
            f"(arms={policy.arm_prefixes}, cameras={policy.camera_ids}), but "
            "relative_policy_rollout_v2.py's control loop is single-arm.\n"
            "Ready for bimanual: RealPolicy inference + camera stacking "
            "(use gather_visual_obs(obs, policy.camera_ids, args.camera_source)).\n"
            "Still single-arm and needs wiring before a dual-arm deploy:\n"
            "  1. push_observation: per-arm FSR list [left_fsr, right_fsr] "
            "(and per-arm proprio if proprio_mode!='none');\n"
            "  2. per-arm anchor pose + relative->absolute decode "
            "(RealPolicy returns [left|right] concatenated action blocks);\n"
            "  3. per-arm IK (LightweightIK side='left'/'right') + "
            "per-arm receding-horizon interpolators;\n"
            "  4. send both /action/left_arm & /action/right_arm (RobotEnv "
            "already exposes both).\n"
            "Validate offline first: scripts/smoke_bimanual_rollout.py "
            "--zarr data/chips_teleop --episode <ep>."
        )

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
    robot_env = RobotEnv(
        enable_tactile=args.enable_tactile,
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

    dt = 1.0 / args.control_hz
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

    output: Optional[np.ndarray] = None
    # Which chunk frame to start EXECUTING from.  Current convention: the eef
    # action comes from ``pose_action`` anchored on the current state
    # ``pose[t]``, so ``action[0] = pose[t]^-1 @ pose_action[t]`` is the REAL
    # first commanded step and must be executed -> ANCHOR_OFFSET = 0 (default).
    # Legacy checkpoints that reused ``pose`` as the action had
    # ``action[0] = identity`` (a no-op) and skipped it with --anchor_offset 1.
    ANCHOR_OFFSET = int(getattr(args, "anchor_offset", 0))
    act_index = ANCHOR_OFFSET
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
    policy_uses_tactile = bool(getattr(policy, "enable_fsr", False))
    policy_tactile_key = str(getattr(policy, "tactile_key", "fsr"))
    if policy_uses_tactile:
        print(
            f"[tactile] policy uses tactile: source='{policy_tactile_key}' "
            f"({'force6d->resultant' if policy_tactile_key == 'force' else 'deform->taxel20'}), "
            f"fingers={_TACTILE_FINGERS}"
        )
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
    relative_hand_action = bool(
        getattr(policy.model_cfg.dataset, "relative_hand_action", False)
    )
    # Steps executed since the last predict+schedule (DexUMI receding horizon).
    steps_since_replan = chunk_exec_steps  # force first-loop replan
    pose_interp: Optional[PoseTrajectoryInterpolator] = None
    hand_interp: Optional[MotorTrajectoryInterpolator] = None
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
    print("Start Rollout with relative policy (v2 DexUMI smoothing)")
    print(f"  model_path        : {args.model_path}")
    print(f"  ckpt              : {args.ckpt}")
    print(f"  use_ema           : {args.use_ema if args.use_ema is not None else '(follow config)'}")
    print(f"  control_hz        : {args.control_hz}")
    print(f"  camera_source     : {args.camera_source}")
    print(f"  proprio_mode      : {proprio_mode}")
    print(f"  hand_action_mode  : {hand_action_mode}")
    print(f"  action_horizon    : {action_horizon}  (executable = {chunk_exec_steps})")
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
        relative_hand_action,
    )
    print(
        "  smoothing_v2      : receding-horizon + Pose/Motor trajectory "
        "interpolation (DexUMI eval_xhand style; NO temporal ensemble)"
    )
    print(
        "  v2 latencies      : "
        f"robot={_V2_ROBOT_ACTION_LATENCY:.3f}s, "
        f"hand={_V2_HAND_ACTION_LATENCY:.3f}s"
    )
    print(
        "  v2 speed limits   : "
        f"pos={_V2_MAX_POS_SPEED:.3f} m/s, "
        f"rot={_V2_MAX_ROT_SPEED:.3f} rad/s, "
        f"hand={_V2_MAX_HAND_SPEED:.3f} rad/s"
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

    # Seed trajectory interpolators with the live state so the first
    # schedule_waypoint has a valid trim origin (DexUMI starts the same way).
    _seed_obs = get_latest_robot_obs(robot_env)
    _seed_pose = get_current_right_wrist_pose6d(_seed_obs, ik_solver)
    _seed_hand = _seed_obs["right_hand"].astype(np.float32)
    _seed_t = time.time()
    pose_interp = PoseTrajectoryInterpolator(
        times=np.array([_seed_t], dtype=np.float64),
        poses=_seed_pose.reshape(1, 6).astype(np.float64),
    )
    hand_interp = MotorTrajectoryInterpolator(
        times=np.array([_seed_t], dtype=np.float64),
        values=_seed_hand.reshape(1, -1).astype(np.float64),
    )

    try:
        while True:
            loop_t0 = time.time()

            # 1) latest observation
            for i in range(100):
                obs = get_latest_robot_obs(robot_env)

            # 2) pick the camera frame and push to the rolling buffer
            visual_obs_rgb = choose_visual_obs(obs, args.camera_source)
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

            # Assemble the raw per-frame tactile vector from the live obs when
            # the trained model uses tactile (matches the training zarr layout).
            if policy_uses_tactile:
                fsr = build_tactile_from_obs(obs, policy_tactile_key)

            # We push *every* step (even when reusing the previous chunk) so
            # the deque keeps marching forward and ``_obs_window_indices``
            # sees the correct strided history when we do re-predict.
            policy.push_observation(
                visual_obs=visual_obs_rgb,
                joint_proprio=proprio_inputs["joint_proprio"],
                wrist_pose6=proprio_inputs["wrist_pose6"],
                fingertip_pose_wrist=proprio_inputs["fingertip_pose_wrist"],
                fsr=fsr,
            )

            # 3) DexUMI receding-horizon: replan every ``chunk_exec_steps``,
            # decode the full chunk to absolute targets, discard past
            # waypoints by latency, and schedule the rest into interpolators.
            # There is NO overlapping-chunk temporal ensemble (that is v1).
            need_predict = (
                output is None
                or steps_since_replan >= chunk_exec_steps
            )
            if need_predict:
                t_actual_inference = time.monotonic()
                anchor_right_wrist_pose6d = get_current_right_wrist_pose6d(
                    obs, ik_solver
                )
                if no_eef_policy and held_right_wrist_pose6d is None:
                    held_right_wrist_pose6d = anchor_right_wrist_pose6d.copy()
                with torch.no_grad():
                    # ``visual_obs=None`` / ``fsr=None`` make predict_action use
                    # the rolling buffers we have been populating each step (so
                    # the strided obs window is built correctly for the tactile
                    # stream too, not just a single frame).
                    output = policy.predict_action(
                        proprioception=None,
                        fsr=None,
                        visual_obs=None,
                    )
                chunk_poses, chunk_hands, held_right_wrist_pose6d, held_right_hand = (
                    _decode_chunk_absolute_targets(
                        output=output,
                        anchor_offset=ANCHOR_OFFSET,
                        anchor_wrist_pose6d=anchor_right_wrist_pose6d,
                        obs=obs,
                        t_et=T_ET,
                        hand_action_mode=hand_action_mode,
                        relative_hand_action=relative_hand_action,
                        no_eef_policy=no_eef_policy,
                        eef_only_policy=eef_only_policy,
                        held_right_wrist_pose6d=held_right_wrist_pose6d,
                        held_right_hand=held_right_hand,
                        fingertip_ik=fingertip_ik,
                        clip_hand=args.clip_hand,
                        hand_lower=hand_lower,
                        hand_upper=hand_upper,
                    )
                )
                assert pose_interp is not None and hand_interp is not None
                pose_interp, hand_interp, n_robot_wp, n_hand_wp = (
                    _schedule_chunk_into_interpolators(
                        pose_interp=pose_interp,
                        hand_interp=hand_interp,
                        poses=chunk_poses,
                        hands=chunk_hands,
                        anchor_offset=ANCHOR_OFFSET,
                        dt=dt,
                        t_actual_inference_mono=t_actual_inference,
                        robot_action_latency=_V2_ROBOT_ACTION_LATENCY,
                        hand_action_latency=_V2_HAND_ACTION_LATENCY,
                        max_pos_speed=_V2_MAX_POS_SPEED,
                        max_rot_speed=_V2_MAX_ROT_SPEED,
                        max_hand_speed=_V2_MAX_HAND_SPEED,
                    )
                )
                steps_since_replan = 0
                act_index = ANCHOR_OFFSET
                print(
                    f"New chunk predicted: shape={output.shape}  "
                    f"executable_steps={chunk_exec_steps}  "
                    f"scheduled robot={n_robot_wp} hand={n_hand_wp}"
                )

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

            # 4) Sample the scheduled Pose/Motor trajectories at wall-clock
            # time (DexUMI's servo loop does the same via the interpolator).
            assert pose_interp is not None and hand_interp is not None
            assert output is not None
            t_now = time.time()
            target_right_wrist_pose6d = np.asarray(
                pose_interp(t_now), dtype=np.float32
            )
            target_right_hand = np.asarray(hand_interp(t_now), dtype=np.float32)
            act_index = min(
                ANCHOR_OFFSET + steps_since_replan,
                len(output) - 1,
            )
            # Approximate relative pose for debug overlay only.
            if no_eef_policy:
                relative_pose6d = np.zeros(6, dtype=np.float32)
            else:
                relative_pose6d = output[act_index][:6].astype(np.float32)
            act = output[act_index]
            steps_since_replan += 1

            if args.debug:
                np.save(os.path.join(debug_act_dir, f"act_{debug_step:06d}.npy"), act)

            # 5) live wrist pose (debug / viz); absolute target already sampled
            current_right_wrist_pose6d = get_current_right_wrist_pose6d(obs, ik_solver)
            if anchor_right_wrist_pose6d is None:
                anchor_right_wrist_pose6d = current_right_wrist_pose6d
            if args.debug:
                np.save(
                    os.path.join(debug_pose_dir, f"pose_{debug_step:06d}.npy"),
                    current_right_wrist_pose6d,
                )

            # 6) arm IK
            target_right_wrist_pose4x4 = vec6dof_to_homogeneous_matrix(
                translation=target_right_wrist_pose6d[:3],
                rotation_vector=target_right_wrist_pose6d[3:],
            )
            ik_success, ik_angles = ik_solver.ik(
                target_pose=target_right_wrist_pose4x4,
                body_angles=obs["body_dof"][::-1],
                init_arm_angles=obs["right_arm"][::-1],
                side="right",
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
            ik_pos_err = float("inf")
            ik_rot_err = float("inf")
            if ik_success:
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
            ik_ok = (
                ik_success
                and ik_pos_err <= args.ik_pos_tol
                and ik_rot_err <= args.ik_rot_tol
            )
            if not ik_ok:
                print(
                    f"[WARN] IK rejected (success={ik_success}, "
                    f"pos_err={ik_pos_err * 1000:.1f}mm, rot_err={ik_rot_err:.3f}); "
                    "holding current arm angles."
                )
                target_right_arm = obs["right_arm"].astype(np.float32)
            else:
                target_right_arm = ik_angles[::-1].astype(np.float32)

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

            # 7) hand target already sampled from MotorTrajectoryInterpolator
            # (fingertip IK / relative-joint decode happened at schedule time).

            # 8) send
            action_buffer = build_robot_action_buffer(
                obs=obs,
                target_right_arm=target_right_arm,
                target_right_hand=target_right_hand,
                mode=mode,
            )
            robot_env.send_action(action_buffer, immediate=True)

            # 9) optional live display
            if enable_show:
                show_img = cv2.cvtColor(visual_obs_rgb, cv2.COLOR_RGB2BGR)
                txt1 = f"act_index={act_index} replan={steps_since_replan}/{chunk_exec_steps}"
                txt2 = f"rel_pose={np.array2string(relative_pose6d, precision=3)}"
                cv2.putText(show_img, txt1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(show_img, txt2[:80], (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                cv2.imshow("robot_relative_policy_rollout_v2", show_img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Quit by keyboard.")
                    break

            # 9b) live matplotlib dashboard (image + action chunk + tracking)
            if live_viz is not None:
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
                )
                if not alive:
                    print("Live debug window closed; stopping rollout.")
                    break

            # 10) sleep to maintain control_hz (required for timed waypoints)
            elapsed = time.time() - loop_t0
            time.sleep(max(0.0, dt - elapsed))

    finally:
        if enable_show:
            cv2.destroyAllWindows()
        if live_viz is not None:
            live_viz.close()

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
    parser = argparse.ArgumentParser(
        "Robot rollout v2 with DexUMI receding-horizon + trajectory interpolation"
    )

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
    parser.add_argument("--control_hz", type=int, default=5, help="control frequency (Hz)")
    parser.add_argument(
        "--exec_horizon",
        type=int,
        default=None,
        help=(
            "How many control steps to execute from a scheduled chunk before "
            "re-planning (DexUMI receding horizon).  Defaults to "
            "action_horizon - anchor_offset.  Smaller values replan more often."
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
    parser.add_argument("--mode", type=str, default="right", choices=["right"])

    # camera
    parser.add_argument(
        "--camera_source",
        type=str,
        default="right_wrist",
        choices=["right_wrist", "left_wrist", "right_eye", "left_eye"],
        help="which camera stream to feed to the policy",
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
