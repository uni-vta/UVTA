"""Observation windows and action targets for the UVTA diffusion policy.

Observations are sampled at [t-(H-1)*d, ..., t], where H is obs_horizon
and d is down_sample_steps. Indices before episode start repeat the first
frame. proprio_mode selects joint, relative wrist, or fingertip inputs.

Stored poses use xyz + axis-angle rotation vectors (6-D). Network poses use
xyz + the first two rows of the rotation matrix (9-D continuous rotation;
Zhou et al., 2019). Position and hand targets are range-normalized; rotation
entries use identity normalization. Output blocks are selected by
predict_action, predict_state, and predict_tactile.
"""

from __future__ import annotations

import numpy as np
from uvta.common.utility.matrix import (
    homogeneous_matrix_to_6dof,
    invert_transformation,
    vec6dof_to_homogeneous_matrix,
)
from tqdm import tqdm

from .diffusion_bc_dataset import (
    DiffusionBCDataset,
    get_data_stats,
    normalize_data,
    process_image,
    sample_sequence,
)
from .replay_buffer import UVTAReplayBuffer

# ---------------------------------------------------------------------------
# Pose helpers.
#
# - zarr / disk storage:    6-D xyz + axis-angle rotvec
# - dataset internal math:  4x4 SE(3) matrices
# - network-facing tensors: 9-D xyz + rot6d  (Zhou et al. 2019)
# ---------------------------------------------------------------------------


def _pose6_to_mat(pose6: np.ndarray) -> np.ndarray:
    """``(6,)`` xyz + rotvec -> ``(4, 4)`` homogeneous transform."""
    return vec6dof_to_homogeneous_matrix(pose6[:3], pose6[3:6])


def _mat_to_pose6(mat: np.ndarray) -> np.ndarray:
    """``(4, 4)`` -> ``(6,)`` xyz + rotvec (float32)."""
    return homogeneous_matrix_to_6dof(mat).astype(np.float32)


# ------------- rot6d helpers (UMI-style, mat <-> 6D rotation) -------------
# rot6d is defined as the first two rows of the 3x3 rotation matrix,
# flattened to 6.  Decoding via Gram-Schmidt makes this a continuous
# SO(3) -> R^6 embedding (no axis-angle wraparound / no quaternion double
# cover), which is ideal as a diffusion target.  See Zhou et al. 2019
# "On the Continuity of Rotation Representations in Neural Networks".


def _mat_to_rot6d(mat3: np.ndarray) -> np.ndarray:
    """``(..., 3, 3)`` rotation matrix -> ``(..., 6)`` rot6d.

    rot6d = flatten of the first two rows of ``mat3``.  Matches UMI's
    ``mat_to_rot6d`` (``umi/umi/common/pose_util.py``).
    """
    batch = mat3.shape[:-2]
    return np.asarray(mat3[..., :2, :], dtype=np.float32).reshape(batch + (6,))


def _rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    """``(..., 6)`` rot6d -> ``(..., 3, 3)`` rotation matrix via Gram-Schmidt.

    Mirrors UMI's ``rot6d_to_mat`` and is the inverse used at deploy time
    (``RealPolicy``) to turn network output back into a usable rotation.
    """
    d6 = np.asarray(d6, dtype=np.float64)  # higher precision for normalize/cross
    a1, a2 = d6[..., :3], d6[..., 3:]

    def _normalize(v):
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.clip(n, 1e-12, None)

    b1 = _normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = _normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2).astype(np.float32)


def _mat_to_pose9(mat: np.ndarray) -> np.ndarray:
    """``(..., 4, 4)`` -> ``(..., 9)`` ``[x, y, z, rot6d]`` (network format)."""
    xyz = np.asarray(mat[..., :3, 3], dtype=np.float32)
    rot6d = _mat_to_rot6d(mat[..., :3, :3])
    return np.concatenate([xyz, rot6d], axis=-1).astype(np.float32)


def _pose9_to_mat(pose9: np.ndarray) -> np.ndarray:
    """``(..., 9)`` -> ``(..., 4, 4)`` homogeneous transform (deploy-side)."""
    pose9 = np.asarray(pose9)
    xyz = pose9[..., :3]
    rotmat = _rot6d_to_mat(pose9[..., 3:9])
    out = np.zeros(pose9.shape[:-1] + (4, 4), dtype=np.float32)
    out[..., :3, :3] = rotmat
    out[..., :3, 3] = xyz
    out[..., 3, 3] = 1.0
    return out


def _pose6_to_pose9(pose6: np.ndarray) -> np.ndarray:
    """``(..., 6)`` xyz + rotvec (zarr storage) -> ``(..., 9)`` xyz + rot6d."""
    mats = np.stack([_pose6_to_mat(p) for p in pose6.reshape(-1, 6)], axis=0)
    mats = mats.reshape(pose6.shape[:-1] + (4, 4))
    return _mat_to_pose9(mats)


def _stride_indices(obs_horizon: int, down_sample_steps: int) -> np.ndarray:
    """Indices, relative to the anchor ``t``, of the obs window.

    Returns oldest-first, anchor-last: ``[-(H-1)d, ..., -d, 0]`` so that
    ``window[-1]`` is the "current" frame ``t``.
    """
    H = int(obs_horizon)
    d = int(down_sample_steps)
    assert H >= 1 and d >= 1, f"obs_horizon={H}, down_sample_steps={d}"
    return np.arange(H, dtype=np.int64)[::-1] * (-d)  # e.g. H=3, d=2 -> [-4, -2, 0]


VALID_PROPRIO_MODES = (
    "none",                 # NO proprio stream; condition = vision (+fsr) only
    "joint",                # 22-D hand joint angles only (legacy)
    "ee_rel",               #  9-D wrist pose relative to current frame
                            #     (xyz + rot6d, current frame is identity)
    "both",                 # joint + ee_rel  (22 + 9 = 31)
    "fingertip",            # 5 fingertips x 9-D (xyz + rot6d) in wrist frame
                            #     -> 45-D, NO joint angles
    "fingertip_with_ee",    # fingertip (45) + ee_rel (9) = 54
)

# Per-frame feature dims for the lo-D proprio streams (rot6d network I/O).
JOINT_DIM = 22
EE_REL_DIM = 9         # xyz (3) + rot6d (6)
FINGERTIP_DIM = 5 * 9  # 45 = 5 fingertips × 9-D pose


ROUTABLE_TARGETS = ("state", "tactile")


def resolve_output_routing(
    *,
    predict_action: bool,
    predict_state: bool,
    predict_tactile: bool,
    aux_as_head,
):
    """Split the predicted targets between the diffusion vector and regression heads.

    ONE definition, shared by the dataset, the trainer and ``RealPolicy``, so a
    saved config cannot be read as two different architectures at train and
    deploy time.

    ``aux_as_head`` names which targets LEAVE the diffusion trajectory:
        False / None / []   nothing leaves; every target rides in the trajectory
        True                every AUXILIARY target leaves, i.e. everything
                            except the block deploy executes
        ["tactile"]         exactly the named ones

    The executed block is never routable.  It is the quantity that reaches the
    robot -- or, for a two-stage stage 1, the one stage 2 is conditioned on -- so
    it has to stay a SAMPLE from the trajectory.  A regression head gives the
    conditional MEAN, which mode-averages across genuinely multimodal futures and
    is therefore not something you can execute or invert.

    So the two useful settings fall straight out of that rule:
        predict a+s+t, aux_as_head=True   -> diffusion [action], heads [state, tactile]
        predict   s+t, aux_as_head=True   -> diffusion [state],  heads [tactile]
    the second being the two-stage stage 1: the state stays sampled because
    stage 2 consumes it, while the sparse tactile leaves the trajectory instead
    of injecting its noise into the state at every reverse step.

    Returns ``(motor_blocks, executed_block, head_targets)`` in a fixed order.
    """
    predicted = {
        "action": bool(predict_action),
        "state": bool(predict_state),
        "tactile": bool(predict_tactile),
    }
    if not any(predicted.values()):
        raise ValueError(
            "predict_action / predict_state / predict_tactile are all False: "
            "there would be no target to regress."
        )
    executed = "action" if predicted["action"] else "state"
    if not predicted[executed]:
        raise ValueError(
            "a tactile-only target has nothing for deploy to execute: enable "
            "predict_action (for direct rollout) or predict_state (for a "
            "two-stage stage 1)."
        )

    auxiliary = [t for t in ROUTABLE_TARGETS if predicted[t] and t != executed]
    if aux_as_head is True:
        heads = list(auxiliary)
    elif not aux_as_head:
        heads = []
    else:
        requested = {str(x) for x in aux_as_head}
        unknown = requested - set(("action",) + ROUTABLE_TARGETS)
        if unknown:
            raise ValueError(
                f"aux_as_head names unknown target(s) {sorted(unknown)}; "
                f"valid names are {list(ROUTABLE_TARGETS)}."
            )
        if executed in requested:
            raise ValueError(
                f"aux_as_head cannot contain {executed!r}: it is the block "
                "deploy executes, so it must stay a sample from the diffusion "
                "trajectory rather than a regression head's conditional mean."
            )
        not_predicted = {t for t in requested if not predicted.get(t, False)}
        if not_predicted:
            raise ValueError(
                f"aux_as_head names {sorted(not_predicted)}, which "
                f"predict_{'/predict_'.join(sorted(not_predicted))} turns off; "
                "a target cannot be routed to a head without being predicted."
            )
        heads = [t for t in ROUTABLE_TARGETS if t in requested]

    motor_blocks = [
        b for b in ("action", "state") if predicted[b] and b not in heads
    ]
    if not motor_blocks:
        raise ValueError(
            "routing left the diffusion target empty; at least the executed "
            f"block ({executed!r}) must stay in the trajectory."
        )
    return motor_blocks, executed, heads


def proprio_dim_for_mode(mode: str) -> int:
    """Return the per-frame proprioception feature dim for a given mode."""
    if mode == "none":
        return 0
    if mode == "joint":
        return JOINT_DIM
    if mode == "ee_rel":
        return EE_REL_DIM
    if mode == "both":
        return JOINT_DIM + EE_REL_DIM
    if mode == "fingertip":
        return FINGERTIP_DIM
    if mode == "fingertip_with_ee":
        return FINGERTIP_DIM + EE_REL_DIM
    raise ValueError(
        f"invalid proprio_mode={mode!r}; choose from {VALID_PROPRIO_MODES}"
    )


def _mode_needs_joint(mode: str) -> bool:
    return mode in ("joint", "both")


def _mode_needs_ee_rel(mode: str) -> bool:
    return mode in ("ee_rel", "both", "fingertip_with_ee")


def _mode_needs_fingertip(mode: str) -> bool:
    return mode in ("fingertip", "fingertip_with_ee")


VALID_HAND_ACTION_MODES = (
    "joint", "fingertip", "none", "joint_only", "fingertip_only",
)


def _mode_has_eef(mode: str) -> bool:
    """Whether the action target includes the leading 9-D relative EEF pose.

    All modes prepend the 9-D wrist pose EXCEPT ``joint_only`` and
    ``fingertip_only``, whose action is just the hand block (joint angles or
    fingertip pose respectively); the wrist is not controlled by the policy.
    """
    return mode not in ("joint_only", "fingertip_only")


def hand_action_dim_for_mode(mode: str) -> int:
    """Return the hand-action dim for a given mode.

    ``joint`` -> 22 joint angles, ``fingertip`` -> 45 fingertip pose,
    ``none`` -> 0 (the action carries only the 9-D relative EEF pose),
    ``joint_only`` -> 22 joint angles (and NO eef block; see
    ``_mode_has_eef``).
    """
    if mode == "none":
        return 0
    if mode in ("joint", "joint_only"):
        return JOINT_DIM
    if mode in ("fingertip", "fingertip_only"):
        return FINGERTIP_DIM
    raise ValueError(
        f"invalid hand_action_mode={mode!r}; choose from {VALID_HAND_ACTION_MODES}"
    )


def _default_hand_action_mode_for(proprio_mode: str) -> str:
    """The user can leave ``hand_action_mode`` unset and we'll mirror the
    proprio choice: fingertip-style proprio -> fingertip-style action,
    joint-style proprio (or ee_rel only) -> joint-angle action.
    """
    return "fingertip" if _mode_needs_fingertip(proprio_mode) else "joint"


class UVTADataset(DiffusionBCDataset):
    """Single-arm + multi-finger dataset with configurable obs window.

    Parameters
    ----------
    proprio_mode : {"joint", "ee_rel", "both", "fingertip", "fingertip_with_ee"}
        See module docstring.
    hand_action_mode : {"joint", "fingertip", "none"} or None
        Controls what the diffusion action target represents for the hand:
            - ``"joint"``     : 22-D hand_action (joint angles) [legacy]
            - ``"fingertip"`` : 45-D = 5 fingertips x 9-D (xyz + rot6d) in
              wrist frame.
            - ``"none"``      : NO hand component; the action is only the
              9-D relative EEF pose (xyz + rot6d).  Use this to train a
              pure end-effector policy with no joint / fingertip targets.
        If ``None``, mirrors ``proprio_mode``: fingertip-style proprio -> 
        fingertip-style action, otherwise joint-angle action.  This is the
        recommended default and matches the "exclusive" semantic of the
        joint-vs-fingertip switch.
    down_sample_steps : int
        Temporal stride for the obs window.  ``obs[k] = traj[t - k*d]`` for
        ``k = 0, ..., obs_horizon - 1`` (i.e. ``obs[0] == traj[t]`` and the
        time index decreases for earlier frames).  Default 1 keeps the old
        contiguous behaviour.
    """

    def __init__(
        self,
        data_dirs,
        max_episode=None,
        load_camera_ids=[0],
        camera_resize_shape=None,
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=16,
        unnormal_list=[],
        seed=0,
        optional_transforms=None,
        replay_buffer_cls=UVTAReplayBuffer,
        relative_hand_action=False,
        proprio_mode: str = "joint",
        hand_action_mode: str | None = None,
        down_sample_steps: int = 1,
        norm_clip_percentile=None,
        norm_method: str = "minmax",
        norm_clip_sigma=None,
        norm_per_embodiment: bool = False,
        stats_primary_embodiment: str = "robot",
        share_joint_stats: bool = False,
        action_from_next_state: bool | None = None,
        predict_future_tactile: bool | None = None,
        predict_action: bool | None = None,
        predict_state: bool | None = None,
        predict_tactile: bool | None = None,
        # False / True / a list of target names -- see ``resolve_output_routing``.
        aux_as_head=False,
        arms: list[str] | None = None,
        **replay_buffer_kwargs,
    ):
        # --- bimanual arms ----------------------------------------------
        # ``arms``: list of per-arm field prefixes.  Single-arm data uses bare
        # field names, so the default ``[""]`` reproduces the legacy single-arm
        # behaviour exactly.  Dual-arm data (``chips_teleop``) stores per-arm
        # streams under ``left_*`` / ``right_*``, so pass
        # ``arms=["left_", "right_"]``.  The action / proprioception / tactile
        # tensors are then built PER ARM and concatenated (each arm's EEF pose is
        # relativized to its OWN current wrist frame, matching the single-arm
        # convention).  Cameras are shared across arms.
        if arms is None:
            arms = [""]
        self.arm_prefixes = [str(p) for p in arms]
        if len(self.arm_prefixes) == 0:
            raise ValueError("arms must be a non-empty list (default [''])")
        self.num_arms = len(self.arm_prefixes)
        self.is_bimanual = self.num_arms > 1
        # Forward the prefixes to the replay buffer so it loads the per-arm
        # streams under the matching prefixed keys.
        replay_buffer_kwargs["arm_prefixes"] = self.arm_prefixes

        self.relative_hand_action = relative_hand_action
        # --- output composition: three INDEPENDENT prediction blocks ----
        # The action target is the concatenation, per arm, of whichever of these
        # is enabled, with the tactile blocks grouped at the very end:
        #   ACTION  [eef_rel(9) | joint(22)]  from pose_action / hand_action
        #                                     -- the recorded teleop COMMAND.
        #   STATE   [eef_rel(9) | joint(22)]  from pose[t+1] / proprioception[t+1]
        #                                     -- the state the command LED TO.
        #   TACTILE [F]                       from fsr[t+1..], all arms at the end.
        # On robot data the first two genuinely differ (measured: wrist 1.7-6.4 mm
        # mean, joints 4-38 mrad mean); on human captures pose_action IS pose[t+1],
        # so the two blocks are duplicates there and their losses move together.
        #
        # Legacy knobs express the older either/or form and still work:
        #   action_from_next_state=False -> action only
        #   action_from_next_state=True  -> state only (it OVERWROTE the command)
        #   predict_future_tactile       -> the tactile block
        # Mixing legacy and new knobs is rejected rather than silently ranked.
        # With neither given, all three default to True.
        _legacy = (action_from_next_state is not None) or (
            predict_future_tactile is not None
        )
        _explicit = any(
            x is not None for x in (predict_action, predict_state, predict_tactile)
        )
        if _legacy and _explicit:
            raise ValueError(
                "specify EITHER the legacy {action_from_next_state, "
                "predict_future_tactile} or the new {predict_action, "
                "predict_state, predict_tactile}, not both: they describe the "
                "same output composition and there is no sane precedence."
            )
        if _explicit:
            self.predict_action = True if predict_action is None else bool(predict_action)
            self.predict_state = True if predict_state is None else bool(predict_state)
            self.predict_tactile = (
                True if predict_tactile is None else bool(predict_tactile)
            )
        elif _legacy:
            _afns = bool(action_from_next_state)
            self.predict_action = not _afns
            self.predict_state = _afns
            self.predict_tactile = bool(predict_future_tactile)
        else:
            self.predict_action = self.predict_state = self.predict_tactile = True
        # ``aux_as_head``: which targets leave the diffusion vector and come back
        # as separate keys for a model that regresses them from the conditioning
        # with their own heads.  See ``resolve_output_routing`` for the rules.
        #
        # Why this exists: a shared trajectory is denoised as ONE vector, so the
        # residual uncertainty on every dimension re-enters every other dimension
        # at each of the reverse steps.  Measured on book_teleop over 256 anchors
        # (identical decode path, all models executing their own executed block),
        # sampled wrist error: 2.6 mm predicting action alone, 6.0 mm with the
        # tactile block glued on, 7.6 mm with both auxiliary blocks.  The training
        # loss does not show this at all -- it scores one denoising step, while
        # deployment runs sixteen coupled ones -- so the fix has to be structural:
        # the auxiliary targets have to leave the trajectory, not just be
        # down-weighted in it.
        self.aux_as_head = aux_as_head
        (
            self.motor_blocks,
            self.executed_block,
            self.aux_targets,
        ) = resolve_output_routing(
            predict_action=self.predict_action,
            predict_state=self.predict_state,
            predict_tactile=self.predict_tactile,
            aux_as_head=aux_as_head,
        )
        # Per-target routing predicates, used everywhere below instead of the raw
        # flag: with only state + tactile predicted, ``aux_as_head=True`` routes
        # the TACTILE out while the state stays in the trajectory, so a single
        # boolean cannot answer "is this target on a head?".
        self.state_as_head = "state" in self.aux_targets
        self.tactile_as_head = "tactile" in self.aux_targets
        # ``predict_future_tactile``: append the FUTURE tactile window to the
        # diffusion ACTION target so the policy also predicts where the tactile
        # signal is going.  Kept as an alias of ``predict_tactile`` because the
        # deploy side and several scripts read it by this name.
        self.predict_future_tactile = self.predict_tactile
        self.tactile_action_dim = 0
        # ``action_from_next_state`` (LEGACY): the OVERWRITING form of a
        # state-only target -- the replay buffer replaces pose_action /
        # hand_action with the shifted state, so the command is unrecoverable.
        # Still honoured for reproducing older runs, and still what the deploy
        # side keys off, so it stays True exactly when the target is state-only.
        # ``predict_state`` alone uses the non-destructive ``pose_next`` /
        # ``joint_next`` streams instead and can coexist with the command block.
        self.action_from_next_state = self.predict_state and not self.predict_action

        # --- normalization config safety net ----------------------------
        # mean-std (z-score) normalization is UNBOUNDED.  If the diffusion
        # noise scheduler uses ``clip_sample=True``, as in the default
        # training config, an unbounded action target gets clamped to [-1, 1]
        # at sample time, silently destroying any action beyond ~1 sigma.
        # The fix is ``norm_clip_sigma`` (clamp to ±sigma and rescale onto
        # [-1, 1]).  It is easy to forget to set it when switching to
        # ``meanstd``, so we auto-supply a safe default here and warn loudly
        # rather than letting the bug through.
        _DEFAULT_CLIP_SIGMA = 3.0
        if norm_method not in ("minmax", "meanstd"):
            raise ValueError(
                f"invalid norm_method={norm_method!r}; "
                "choose 'minmax' or 'meanstd'"
            )
        if norm_method == "meanstd" and not norm_clip_sigma:
            norm_clip_sigma = _DEFAULT_CLIP_SIGMA
            print(
                "[UVTADataset] WARNING: norm_method='meanstd' but "
                "norm_clip_sigma was not set.  mean-std output is UNBOUNDED "
                "and would be silently clamped by clip_sample=True, "
                f"corrupting large actions.  Auto-setting norm_clip_sigma="
                f"{_DEFAULT_CLIP_SIGMA} (±{_DEFAULT_CLIP_SIGMA} sigma -> "
                "[-1, 1]).  Set it explicitly in yaml to silence this, or "
                "set clip_sample=False if you truly want raw z-score."
            )

        # --- proprio_mode + hand_action_mode bookkeeping ----------------
        if proprio_mode not in VALID_PROPRIO_MODES:
            raise ValueError(
                f"invalid proprio_mode={proprio_mode!r}; "
                f"choose from {VALID_PROPRIO_MODES}"
            )
        self.proprio_mode = proprio_mode
        self.needs_joint_proprio = _mode_needs_joint(proprio_mode)
        self.needs_ee_rel_proprio = _mode_needs_ee_rel(proprio_mode)
        self.needs_fingertip_proprio = _mode_needs_fingertip(proprio_mode)

        if hand_action_mode is None:
            hand_action_mode = _default_hand_action_mode_for(proprio_mode)
        if hand_action_mode not in VALID_HAND_ACTION_MODES:
            raise ValueError(
                f"invalid hand_action_mode={hand_action_mode!r}; "
                f"choose from {VALID_HAND_ACTION_MODES}"
            )
        self.hand_action_mode = hand_action_mode
        self.needs_fingertip_action = hand_action_mode in (
            "fingertip", "fingertip_only",
        )
        # ``none`` -> the action is the 9-D relative EEF pose only; no joint
        # nor fingertip hand component is concatenated.
        self.needs_hand_action = hand_action_mode != "none"
        # ``joint_only`` -> the action is JUST the 22-D joint block, with no
        # leading 9-D relative EEF pose (the wrist is not policy-controlled).
        # Every other mode prepends the eef pose.
        self.action_has_eef = _mode_has_eef(hand_action_mode)

        # The action-side wrist pose is ALWAYS taken from the REAL action stream
        # ``pose_action`` (the commanded next-state pose, shift +1 from the
        # observed ``pose`` STATE) whenever the action carries an eef pose block
        # -- modes 'none' (eef-only), 'joint', 'fingertip'.  For 'joint_only' /
        # 'fingertip_only' the wrist is not policy-controlled so there is no eef
        # action.  The relativization ANCHOR always stays the current STATE
        # (``pose[t]``) so obs (ee_rel) and action share the same reference
        # frame and deploy (which anchors on the live observed wrist)
        # reconstructs the true target.  ``action_uses_pose_action`` is
        # finalized after the buffer is built (it also requires the
        # ``pose_action`` stream to actually be present in the data).
        self._action_has_eef_pose = self.hand_action_mode not in (
            "joint_only", "fingertip_only",
        )

        # --- state-block compatibility ----------------------------------
        # The next state is only defined for the absolute-joint / eef-only
        # modes: fingertip actions come from the separate ``fingertip_action``
        # stream (nothing to shift) and the relative-joint path subtracts
        # hand_action[0], which is meaningless against a state.
        if self.predict_state:
            if self.hand_action_mode not in ("none", "joint", "joint_only"):
                raise ValueError(
                    "predict_state=True only supports hand_action_mode in "
                    f"{{'none','joint','joint_only'}}; got "
                    f"{self.hand_action_mode!r}.  Fingertip actions come from "
                    "the fingertip_action stream and are not shifted."
                )
            if self.relative_hand_action:
                raise ValueError(
                    "predict_state=True is incompatible with "
                    "relative_hand_action=True (the state joints are the "
                    "absolute proprioception[t+1])."
                )
            # The buffer needs the proprioception stream to build the joint
            # part, even when proprio_mode does not use joints as an obs.
            replay_buffer_kwargs["skip_proprioception"] = False
            if self.action_from_next_state:
                # State-only: keep the legacy OVERWRITING synthesis so runs from
                # before the split reproduce byte-for-byte.
                replay_buffer_kwargs["action_from_next_state"] = True
                print(
                    "[UVTADataset] state-only target (legacy "
                    "action_from_next_state) -> the OBSERVED next state "
                    "(pose[t+1] wrist"
                    + (", proprioception[t+1] joints"
                       if self.hand_action_mode in ("joint", "joint_only") else "")
                    + ") is written over pose_action/hand_action in the buffer."
                )
            else:
                # Command AND state: the state needs its own streams so the
                # recorded command survives alongside it.
                replay_buffer_kwargs["synthesize_next_state"] = True

        # --- tactile-block compatibility --------------------------------
        # The future tactile is appended to the action target, so the tactile
        # stream must actually be loaded (enable_fsr).  The tactile blocks are
        # appended AFTER every motor block, for every hand_action_mode.
        if self.predict_tactile and not bool(
            replay_buffer_kwargs.get("enable_fsr", False)
        ):
            raise ValueError(
                "predict_tactile=True requires enable_fsr=True (the future "
                "tactile window is part of the action target)."
            )
        print(
            "[UVTADataset] output blocks: "
            f"action={self.predict_action}  state={self.predict_state}  "
            f"tactile={self.predict_tactile}  -> deploy executes the "
            f"{self.executed_block!r} block"
        )
        print(
            f"[UVTADataset] routing: diffusion={self.motor_blocks}"
            + (f" + tactile" if self.predict_tactile and not self.tactile_as_head
               else "")
            + f"  heads={self.aux_targets or 'none'}"
        )

        if self.hand_action_mode == "none":
            print(
                "[UVTADataset] hand_action_mode='none' -> action is the "
                "9-D relative EEF pose only (no joint / fingertip target)."
            )
        if self.hand_action_mode == "joint_only":
            print(
                "[UVTADataset] hand_action_mode='joint_only' -> action is the "
                "22-D joint-angle block only (no EEF pose target)."
            )
        if self.hand_action_mode == "fingertip_only":
            print(
                "[UVTADataset] hand_action_mode='fingertip_only' -> action is "
                "the 45-D fingertip block only (5 x 9-D, no EEF pose target)."
            )

        # Quick exclusivity sanity: joint-style proprio + fingertip-style
        # action (or vice-versa) is allowed but odd; warn loudly so the user
        # noticed if they mismatched the two modes.
        if self.needs_fingertip_proprio and self.hand_action_mode == "joint":
            print(
                "[UVTADataset] WARNING: proprio_mode includes fingertip but "
                "hand_action_mode='joint'.  This is allowed but unusual; "
                "double-check that this is what you want."
            )
        if self.needs_joint_proprio and self.hand_action_mode == "fingertip":
            print(
                "[UVTADataset] WARNING: proprio_mode includes joint but "
                "hand_action_mode='fingertip'.  This is allowed but unusual; "
                "double-check that this is what you want."
            )

        # The "ee_rel" stream is computed on-the-fly from ``pose`` and is
        # NOT stored in ``buffer.memory_buffer``.  It is also kept raw
        # (no normalization), since the relative pose is naturally near
        # zero around the current frame.  See UMI which keeps rot6d of the
        # relative pose unnormalized.

        # Fingertip proprio (observation) -> need the *state* stream
        # ``fingertip_pose_wrist`` (T, 5, 6)  = FK(proprioception[t]).
        # Fingertip action (target)         -> need the *next-state* stream
        # ``fingertip_action`` (T, 5, 6)  = FK(proprioception[t+1]) with
        # last frame repeating, matching the hand_action shift-+1 rule.
        # These two arrays are independent so we toggle them independently.
        # The user does not have to set the load flags manually.
        if self.needs_fingertip_proprio:
            replay_buffer_kwargs["load_fingertip"] = True
        if self.needs_fingertip_action:
            replay_buffer_kwargs["load_fingertip_action"] = True

        # Auto-set ``skip_proprioception`` based on proprio_mode so the
        # user does not have to remember to flip it in yaml when switching
        # away from joint-style proprio.  Rules:
        #   * proprio_mode needs joint  -> skip_proprioception=False (load joint)
        #   * proprio_mode does NOT need joint -> skip_proprioception=True
        #     (so the 22-D buffer is not allocated and the zarr's
        #     ``proprioception`` field is not even required to exist).
        # If the user explicitly set ``skip_proprioception`` in yaml, we
        # honor their choice but still validate consistency below.
        user_set_skip = "skip_proprioception" in replay_buffer_kwargs
        if not user_set_skip:
            replay_buffer_kwargs["skip_proprioception"] = (
                not self.needs_joint_proprio
            )
            if replay_buffer_kwargs["skip_proprioception"]:
                print(
                    f"[UVTADataset] proprio_mode={proprio_mode!r} does not "
                    "need joint angles -> auto-setting skip_proprioception=True"
                )

        # Consistency check: joint-style proprio + skip_proprioception=True
        # would silently drop the input the dataset needs, so we hard-fail.
        skip_proprio = bool(replay_buffer_kwargs["skip_proprioception"])
        if self.needs_joint_proprio and skip_proprio:
            raise ValueError(
                f"proprio_mode={proprio_mode!r} requires the joint-angle "
                "proprioception stream, but skip_proprioception=True is set "
                "(either explicitly in yaml or computed).  Set "
                "skip_proprioception=False (or change proprio_mode to "
                "'ee_rel' / 'fingertip')."
            )

        # --- unnormal_list management -----------------------------------
        # ``hand_action`` is kept un-normalized when ``relative_hand_action``
        # is True (matches the legacy behaviour).  We also force the
        # raw 3-D ``fingertip_pose_wrist`` array (if loaded into buffer) to be
        # unnormalized — the parent's normalize_data doesn't handle 3-D
        # tensors anyway.
        # All of the per-arm skip-list bookkeeping below is applied ONCE PER ARM
        # so bimanual runs manage ``left_*`` / ``right_*`` keys (single-arm keeps
        # the bare ``pose`` / ``hand_action`` / ... names via prefix "").
        for p in self.arm_prefixes:
            hand_key = f"{p}hand_action"
            if not self.needs_hand_action:
                # EEF-only action: the hand_action stream is never folded into
                # the action target, so it must not be normalized as part of the
                # action.  Drop it from unnormal_list management entirely.
                if hand_key in unnormal_list:
                    unnormal_list.remove(hand_key)
                    print(f"Removing {hand_key} from unnormal_list (eef-only action)")
            elif self.relative_hand_action:
                if hand_key not in unnormal_list:
                    unnormal_list.append(hand_key)
                    print(f"Adding {hand_key} to unnormal_list")
                assert hand_key in unnormal_list
            else:
                if hand_key in unnormal_list:
                    unnormal_list.remove(hand_key)
                    print(f"Removing {hand_key} from unnormal_list")
                assert hand_key not in unnormal_list
            # ``pose`` (observed wrist STATE) is relativized on the fly, so it
            # must stay raw in the buffer.  Auto-add per arm (single-arm configs
            # that already list "pose" are a no-op).
            pose_key = f"{p}pose"
            if pose_key not in unnormal_list:
                unnormal_list.append(pose_key)
                print(f"Adding {pose_key} to unnormal_list")
            assert pose_key in unnormal_list
            # fingertip_pose_wrist (state, T,5,6) and fingertip_action (target,
            # T,5,6) are both 3-D tensors that the parent's normalize_data
            # cannot handle out of the box; we skip default normalization and
            # apply our own (xyz range-normalized, rot6d identity) inside
            # ``__getitem__`` / ``get_relative_action_normalization_stats``.
            if self.needs_fingertip_proprio and \
                    f"{p}fingertip_pose_wrist" not in unnormal_list:
                unnormal_list.append(f"{p}fingertip_pose_wrist")
                print(f"Adding {p}fingertip_pose_wrist to unnormal_list")
            if self.needs_fingertip_action and \
                    f"{p}fingertip_action" not in unnormal_list:
                unnormal_list.append(f"{p}fingertip_action")
                print(f"Adding {p}fingertip_action to unnormal_list")
            # ``pose_action`` (real action wrist pose, T,6 = xyz+rotvec) is turned
            # into a relative pose on the fly like ``pose``, so it must NOT be
            # range-normalized as a raw buffer stream.  Adding it to the skip-list
            # is harmless even when the data has no such stream (the
            # normalization loop only iterates existing buffer keys).
            if self._action_has_eef_pose and f"{p}pose_action" not in unnormal_list:
                unnormal_list.append(f"{p}pose_action")
                print(f"Adding {p}pose_action to unnormal_list")
            # ``pose_next`` (the STATE block's wrist source) is relativized the
            # same way, so it stays raw too.  ``joint_next`` is NOT listed: it is
            # an ordinary joint stream and must be normalized on the joint scale
            # (``share_joint_stats`` ties it to hand_action / proprioception).
            if (
                self.predict_state
                and not self.action_from_next_state
                and self._action_has_eef_pose
                and f"{p}pose_next" not in unnormal_list
            ):
                unnormal_list.append(f"{p}pose_next")
                print(f"Adding {p}pose_next to unnormal_list")

        self.down_sample_steps = int(down_sample_steps)
        # Sample layout (UMI-style, "method B"):
        #
        #   [pad_before (H-1)*d frames | obs anchor t | future actions pred_horizon-1 frames]
        #
        # Total length = (H-1)*d + pred_horizon.  ``obs window`` indices into
        # the sample are [0, d, 2d, ..., (H-1)*d]; the anchor t sits exactly
        # at index (H-1)*d; the action target is the contiguous slice
        # [(H-1)*d : (H-1)*d + pred_horizon] of length ``pred_horizon``.
        #
        # Distinct from the legacy layout where ``sequence_length=pred_horizon``
        # and the action target overlapped with the obs region: now ``action``
        # is strictly the future (anchor included as identity), and yaml's
        # ``pred_horizon`` regains its UMI meaning of "future horizon
        # predicted by the network".
        self._sample_length = (
            (int(obs_horizon) - 1) * self.down_sample_steps + int(pred_horizon)
        )
        # We pass ``sample_length`` to the parent via its ``pred_horizon``
        # arg (the parent uses it as ``sequence_length`` for slicing).  And
        # we inflate ``obs_horizon`` so the parent's pad_before reaches all
        # the way back to ``t - (H-1)*d``.
        effective_obs_horizon_for_pad = (
            (int(obs_horizon) - 1) * self.down_sample_steps + 1
        )

        super().__init__(
            data_dirs,
            max_episode,
            load_camera_ids,
            camera_resize_shape,
            # NOTE: pass ``self._sample_length`` here, NOT the user-facing
            # ``pred_horizon``.  The parent uses this as the slicing length;
            # we restore the real ``pred_horizon`` on ``self`` below.
            self._sample_length,
            # Pass an inflated obs_horizon so that pad_before is large enough
            # to cover the strided window's deepest reach.
            obs_horizon=effective_obs_horizon_for_pad,
            action_horizon=action_horizon,
            unnormal_list=unnormal_list,
            seed=seed,
            replay_buffer_cls=replay_buffer_cls,
            norm_clip_percentile=norm_clip_percentile,
            norm_method=norm_method,
            norm_clip_sigma=norm_clip_sigma,
            norm_per_embodiment=norm_per_embodiment,
            stats_primary_embodiment=stats_primary_embodiment,
            share_joint_stats=share_joint_stats,
            **replay_buffer_kwargs,
        )
        # Restore the *true* obs_horizon / pred_horizon on self.  The parent
        # set ``self.pred_horizon = self._sample_length`` (slicing length);
        # we override so downstream code (loss, deploy) sees the user's
        # original "future horizon" value.
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)
        # Cache stride indices.
        self._obs_window_offsets = _stride_indices(
            self.obs_horizon, self.down_sample_steps,
        )
        # ``_anchor_index_in_sample`` is the index inside the
        # ``self._sample_length``-long sample that corresponds to the anchor
        # t.  By construction (pad_before = (H-1)*d) the anchor is exactly
        # at position (H-1)*d.
        self._anchor_index_in_sample = (self.obs_horizon - 1) * self.down_sample_steps

        self.optional_transforms = optional_transforms

        # Per-sample dataset (data_dir) index, used by co-training uniform
        # sampling.  ``self.indices[i][0]`` is the ``buffer_start_idx`` of
        # training sample ``i`` in the concatenated memory buffer; mapping it
        # through ``eps_end`` tells us which episode (and hence which data_dir
        # / embodiment) the sample was drawn from.  Computed lazily on demand.
        # NOTE: set BEFORE ``get_relative_action_normalization_stats`` because
        # per-embodiment action stats need the per-sample embodiment mapping.
        self._sample_data_dir_idx = None
        self._sample_embodiment = None
        # Per-sample INTEGER embodiment id (into ``self.embodiment_names``),
        # attached to each batch item so the trainer can split the loss by
        # embodiment (robot vs human).  Computed lazily; see
        # ``get_sample_embodiment_ids``.
        self._sample_embodiment_id = None
        self.embodiment_names = None

        # Finalize the action wrist-pose source now that the buffer exists:
        # use ``pose_action`` when the action carries an eef pose AND the data
        # actually provides the ``pose_action`` stream; otherwise fall back to
        # ``pose`` (state) so old datasets without ``pose_action`` still work.
        self.action_uses_pose_action = self._action_has_eef_pose and bool(
            getattr(self.buffer, "has_pose_action", False)
        )
        if self.action_uses_pose_action:
            print(
                "[UVTADataset] action wrist pose taken from 'pose_action' "
                "(real action, next-state target); anchored on current 'pose' "
                "state."
            )
        elif self._action_has_eef_pose:
            print(
                "[UVTADataset] WARNING: action carries an eef pose but the "
                "data has no 'pose_action' stream; falling back to 'pose' "
                "(state) as the action target."
            )

        # Finalize the tactile action dim (per-frame tactile feature width) now
        # that the buffer exists.  The tactile stream is stored (and already
        # normalized in place) under ``buffer.fsr_source_key``.
        # Per-frame tactile feature width (per arm).  For the trailing tactile
        # action block we concatenate every arm's future tactile window, so the
        # total trailing width is ``per_arm_tactile_dim * num_arms``.
        self.per_arm_tactile_dim = 0
        if self.predict_future_tactile:
            if not getattr(self.buffer, "enable_fsr", False):
                raise ValueError(
                    "predict_future_tactile=True but the buffer has no tactile "
                    "stream (enable_fsr is off)."
                )
            tk = self.buffer.fsr_source_key
            self.per_arm_tactile_dim = int(
                self.buffer.memory_buffer[f"{self.arm_prefixes[0]}{tk}"].shape[-1]
            )
            if self.tactile_as_head:
                # The tactile future is a HEAD target, so it is not part of the
                # diffusion vector.  ``tactile_action_dim`` stays 0 -- it is
                # specifically "how much tactile is glued to the action" --
                # while ``per_arm_tactile_dim`` still sizes the head's output.
                print(
                    f"[UVTADataset] tactile future ('{tk}', "
                    f"{self.per_arm_tactile_dim}-D x {self.num_arms} arm(s)) is an "
                    "AUXILIARY HEAD target, not part of the diffusion vector."
                )
            else:
                self.tactile_action_dim = self.per_arm_tactile_dim * self.num_arms
                print(
                    f"[UVTADataset] predict_future_tactile=True -> appending the "
                    f"future tactile window ('{tk}', {self.per_arm_tactile_dim}-D per "
                    f"frame x {self.num_arms} arm(s) = {self.tactile_action_dim}-D) to "
                    "the action target; the deploy side slices it off."
                )

        self.get_relative_action_normalization_stats()

    # ------------------------------------------------------------------
    # Co-training uniform sampling
    # ------------------------------------------------------------------

    def get_sample_data_dir_idx(self) -> np.ndarray:
        """``(len(self),)`` int array: the data_dir index of every sample.

        A training sample maps to the episode that contains its
        ``buffer_start_idx``; each episode's originating data_dir index is
        recorded by ``UVTAReplayBuffer`` in ``episode_data_dir_idx``.
        """
        if self._sample_data_dir_idx is not None:
            return self._sample_data_dir_idx

        eps_end = np.asarray(self.buffer.eps_end)
        episode_data_dir_idx = getattr(
            self.buffer, "episode_data_dir_idx", None
        )
        if episode_data_dir_idx is None:
            raise AttributeError(
                "replay buffer does not expose ``episode_data_dir_idx``; "
                "uniform co-training sampling requires UVTAReplayBuffer."
            )
        episode_data_dir_idx = np.asarray(episode_data_dir_idx, dtype=np.int64)

        # ``buffer_start_idx`` of every sample (first column of self.indices).
        buffer_start = np.asarray(self.indices)[:, 0]
        # ``np.searchsorted(eps_end, x, side="right")`` returns the index of the
        # first episode whose (exclusive) end is strictly greater than x, i.e.
        # the episode that owns frame x.  eps_end is the cumulative frame count
        # per episode, so this is exactly the episode id of each sample.
        episode_ids = np.searchsorted(eps_end, buffer_start, side="right")
        episode_ids = np.clip(episode_ids, 0, len(episode_data_dir_idx) - 1)
        self._sample_data_dir_idx = episode_data_dir_idx[episode_ids]
        return self._sample_data_dir_idx

    def get_sample_embodiment(self) -> np.ndarray:
        """``(len(self),)`` object array: canonical embodiment of every sample.

        Maps each sample -> its data_dir -> the data_dir's embodiment
        (``self.buffer.embodiments``).  Used for per-embodiment action-stat
        selection in ``__getitem__`` and by
        ``get_relative_action_normalization_stats``.
        """
        if self._sample_embodiment is not None:
            return self._sample_embodiment
        dir_idx = self.get_sample_data_dir_idx()
        embodiments = np.asarray(self.buffer.embodiments, dtype=object)
        self._sample_embodiment = embodiments[dir_idx]
        return self._sample_embodiment

    def get_sample_embodiment_ids(self) -> np.ndarray:
        """``(len(self),)`` int64 array: integer embodiment id of every sample.

        The id indexes ``self.embodiment_names`` (the sorted set of unique
        embodiment strings), so ``embodiment_names[ids[i]]`` is the canonical
        embodiment of sample ``i``.  This is attached to every batch item as
        ``embodiment_id`` so the trainer can split the diffusion loss by
        embodiment (e.g. robot vs human) for per-source convergence curves.

        Defensive: if the buffer cannot expose per-sample embodiments (e.g. a
        replay buffer without ``episode_data_dir_idx``), we fall back to a
        single ``"all"`` embodiment with id 0 for every sample so training is
        never blocked.
        """
        if self._sample_embodiment_id is not None:
            return self._sample_embodiment_id
        try:
            emb = self.get_sample_embodiment()  # (len,) object array of strings
            names = sorted({str(e) for e in emb.tolist()})
            name_to_id = {n: i for i, n in enumerate(names)}
            ids = np.asarray(
                [name_to_id[str(e)] for e in emb.tolist()], dtype=np.int64
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(
                f"[UVTADataset] per-sample embodiment unavailable ({exc}); "
                "labelling every sample as 'all'."
            )
            names = ["all"]
            ids = np.zeros(len(self), dtype=np.int64)
        self.embodiment_names = names
        self._sample_embodiment_id = ids
        return self._sample_embodiment_id

    def compute_uniform_sample_weights(self) -> np.ndarray:
        """``(len(self),)`` float64 weights giving each data_dir equal mass.

        Every dataset (data_dir) contributes the same total probability per
        epoch regardless of how many samples it has: a sample from a dataset
        holding ``n`` samples gets weight ``1 / (num_datasets * n)``, so the
        summed weight of each dataset is ``1 / num_datasets``.  Datasets with
        the same embodiment are still counted as separate datasets (one weight
        bucket per data_dir); this matches "each listed dataset is sampled with
        equal probability in one epoch".
        """
        sample_dir_idx = self.get_sample_data_dir_idx()
        num_data_dirs = len(self.buffer.data_path)
        counts = np.bincount(sample_dir_idx, minlength=num_data_dirs).astype(
            np.float64
        )
        present = counts > 0
        num_present = int(present.sum())
        if num_present == 0:
            raise ValueError("no samples found while building uniform weights")
        per_sample = np.zeros_like(counts)
        # Each present dataset shares 1/num_present of the mass, split evenly
        # across its samples.  Absent datasets (count 0) contribute nothing.
        per_sample[present] = 1.0 / (num_present * counts[present])
        weights = per_sample[sample_dir_idx]
        return weights

    def build_balanced_coverage_sampler(self, seed: int = 0):
        """Sampler that upsamples every data_dir to the LARGEST one's size, with
        guaranteed coverage.

        Unlike ``build_uniform_sampler`` (weighted, WITH replacement), this draws
        exactly ``N = max_d(n_d)`` indices from EVERY data_dir per epoch, built
        from whole permutations plus a without-replacement remainder.  So per
        epoch:

          * the largest data_dir is a plain shuffle -- every sample exactly once;
          * a smaller one with ``n_d`` samples has every sample at least
            ``floor(N / n_d)`` times, and ``N mod n_d`` of them one extra time.

        With human=1000 / robot=400 that is 1000 draws each (2000 total): every
        human sample once, every robot sample twice, and 200 robot samples a
        third time.  Nothing is ever skipped, which is what the weighted sampler
        cannot promise (it leaves ~49% of the larger set untouched per epoch).

        Cost: the epoch grows to ``N * num_data_dirs`` samples, which is larger
        than ``len(self)`` whenever the datasets are unbalanced.

        DDP: ``__iter__`` is deterministic given (seed, epoch) and the epoch
        counter advances once per iteration, so every rank produces the SAME
        sequence and ``accelerator.prepare`` can shard it safely.
        """
        import torch

        sample_dir_idx = np.asarray(self.get_sample_data_dir_idx())
        num_dirs = len(self.buffer.data_path)
        per_dir = [
            np.where(sample_dir_idx == d)[0] for d in range(num_dirs)
        ]
        per_dir = [ix for ix in per_dir if len(ix)]
        if not per_dir:
            raise ValueError("no samples found while building the sampler")
        target = max(len(ix) for ix in per_dir)

        class _BalancedCoverageSampler(torch.utils.data.Sampler):
            def __init__(self, groups, target, seed):
                self.groups = groups
                self.target = int(target)
                self.seed = int(seed)
                self.epoch = 0
                self._len = self.target * len(self.groups)

            def __len__(self):
                return self._len

            def __iter__(self):
                # Mix seed and epoch through SeedSequence rather than adding
                # them: ``seed + epoch`` would make run(seed=7)'s 2nd epoch
                # identical to run(seed=8)'s 1st, so two "different seeds" runs
                # would just be shifted copies of each other.
                g = np.random.default_rng([self.seed, self.epoch])
                self.epoch += 1
                out = []
                for ix in self.groups:
                    n = len(ix)
                    reps, rem = divmod(self.target, n)
                    parts = [g.permutation(ix) for _ in range(reps)]
                    if rem:
                        parts.append(g.permutation(ix)[:rem])
                    out.append(np.concatenate(parts) if parts else ix[:0])
                allidx = np.concatenate(out)
                g.shuffle(allidx)
                return iter(allidx.tolist())

        return _BalancedCoverageSampler(per_dir, target, seed)

    def describe_balanced_coverage(self) -> str:
        """One-line-per-dataset summary of what the coverage sampler will do."""
        sample_dir_idx = np.asarray(self.get_sample_data_dir_idx())
        num_dirs = len(self.buffer.data_path)
        counts = np.bincount(sample_dir_idx, minlength=num_dirs)
        present = [(d, int(c)) for d, c in enumerate(counts) if c]
        target = max(c for _, c in present)
        lines = [
            f"target draws per data_dir = {target} "
            f"(largest dataset); epoch = {target * len(present)} samples"
        ]
        embs = list(getattr(self.buffer, "embodiments", []))
        for d, c in present:
            reps, rem = divmod(target, c)
            emb = embs[d] if d < len(embs) else "?"
            extra = f", {rem} of them {reps + 1}x" if rem else ""
            lines.append(
                f"  data_dir[{d}] ({emb}): {c} samples -> each {reps}x{extra}"
            )
        return "\n".join(lines)

    def build_uniform_sampler(self, num_samples: int | None = None, seed: int = 0):
        """Return a ``WeightedRandomSampler`` giving each data_dir equal mass.

        ``num_samples`` defaults to ``len(self)`` so one epoch still draws the
        same number of samples as the full concatenated dataset (with
        replacement), but re-weighted so each dataset appears with equal
        probability.  Pass this as the DataLoader ``sampler=`` (and leave
        ``shuffle`` unset); ``accelerator.prepare`` shards it across GPUs.
        """
        import torch

        weights = self.compute_uniform_sample_weights()
        if num_samples is None:
            num_samples = len(self)
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        return torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=int(num_samples),
            replacement=True,
            generator=generator,
        )

    # ------------------------------------------------------------------
    # Action-side normalization (unchanged semantics, modulo the use of
    # the corrected ``_pose6_to_mat`` helper).
    # ------------------------------------------------------------------

    def get_relative_action_normalization_stats(self):
        """Compute normalization stats for the *position* parts of the action.

        UMI convention (see ``umi/.../diffusion_policy/dataset/umi_dataset.py``
        ``get_normalizer``): **xyz is range-normalized, rot6d is identity**.
        We therefore only compute min/max for the position components and
        rely on identity normalization (no-op) for the rot6d half.

        Resulting ``self.stats`` keys (added by this method):
            - ``"relative_pose_xyz"``     : (3,) min/max of the relative
              wrist xyz on the ACTION side (the future ``pose_action`` window
              anchored on the current state ``pose[t]``).
            - ``"relative_pose_xyz_obs"`` : (3,) min/max of the relative wrist
              xyz on the OBSERVATION side (the past ``pose`` obs window anchored
              on its own last frame = time t).  Computed separately from the
              action stat so obs and action each use their OWN range.  Only
              present when ``proprio_mode`` uses ee_rel (``needs_ee_rel_proprio``).
            - ``"fingertip_xyz_action"``  : (15,) min/max of fingertip xyz
              in wrist frame, only present when
              ``hand_action_mode == 'fingertip'``.
            - ``"relative_hand_action"``  : preserved for the legacy
              ``relative_hand_action=True`` joint-action path.
        """
        print("Computing relative normalization stats (xyz only; rot6d=identity)")
        # Per-arm accumulators (single-arm -> one entry under prefix "").  The
        # stats keys are prefixed too (``relative_pose_xyz`` single-arm,
        # ``left_relative_pose_xyz`` / ``right_relative_pose_xyz`` dual-arm), so
        # each arm normalizes its OWN (own-frame) relative pose range.
        all_relative_xyz = {p: [] for p in self.arm_prefixes}
        # The STATE block has its own xyz range (it tracks where the wrist
        # actually went, not what was commanded), so it gets its own stat rather
        # than borrowing the command block's.  Only needed when both blocks
        # coexist; the legacy state-only mode reuses ``relative_pose_xyz``.
        self._needs_state_xyz_stat = (
            self.predict_state
            and not self.action_from_next_state
            and self._action_has_eef_pose
        )
        all_state_rel_xyz = {p: [] for p in self.arm_prefixes}
        all_obs_rel_xyz = {p: [] for p in self.arm_prefixes}
        all_relative_hand_action = {p: [] for p in self.arm_prefixes}
        all_fingertip_xyz = {p: [] for p in self.arm_prefixes}
        anchor = self._anchor_index_in_sample
        # Obs-window indices into a sample (strided past frames up to t); used
        # to build the OBSERVATION-side relative xyz stats independently.
        win_idx = self._window_indices_into_sample()
        # Take only the future slice (anchor included, length pred_horizon),
        # NOT the full sample.  This way the stats describe the actual range
        # of relative-to-t deltas the network has to regress, rather than
        # the larger range that would include "past" obs-region frames.
        future_end = anchor + self.pred_horizon
        for idx in tqdm(range(len(self))):
            (
                buffer_start_idx,
                buffer_end_idx,
                sample_start_idx,
                sample_end_idx,
            ) = self.indices[idx]
            nsample = sample_sequence(
                train_data=self.buffer.memory_buffer,
                sequence_length=self._sample_length,
                buffer_start_idx=buffer_start_idx,
                buffer_end_idx=buffer_end_idx,
                sample_start_idx=sample_start_idx,
                sample_end_idx=sample_end_idx,
            )
            for p in self.arm_prefixes:
                rel_mats = self._relative_action_pose_mats(
                    nsample, anchor, future_end, prefix=p
                )  # (pred_horizon, 4, 4); anchored on this arm's state pose[t]
                rel_xyz = rel_mats[:, :3, 3].astype(np.float32)
                all_relative_xyz[p].append(rel_xyz)
                if self._needs_state_xyz_stat:
                    state_mats = self._relative_action_pose_mats(
                        nsample, anchor, future_end, prefix=p, block="state"
                    )
                    all_state_rel_xyz[p].append(
                        state_mats[:, :3, 3].astype(np.float32)
                    )
                if self.needs_ee_rel_proprio:
                    # Obs-side ee_rel: relativize the PAST obs window to its own
                    # last frame (= current time t), matching __getitem__.
                    ee_rel9 = self._build_ee_rel_obs(nsample[f"{p}pose"][win_idx])
                    all_obs_rel_xyz[p].append(ee_rel9[:, :3].astype(np.float32))
                if self.needs_fingertip_action:
                    ft = nsample[f"{p}fingertip_action"][anchor:future_end]
                    ft_xyz = ft[..., :3].reshape(self.pred_horizon, 5 * 3)
                    all_fingertip_xyz[p].append(ft_xyz.astype(np.float32))
                elif self.needs_hand_action and self.relative_hand_action:
                    hand_action = nsample[f"{p}hand_action"][anchor:future_end]
                    relative_hand_action = np.array(
                        [h - hand_action[0] for h in hand_action],
                        dtype=np.float32,
                    )
                    all_relative_hand_action[p].append(relative_hand_action)

        clip_p = getattr(self, "norm_clip_percentile", None)
        method = getattr(self, "norm_method", "minmax")
        clip_sigma = getattr(self, "norm_clip_sigma", None)
        norm_kw = dict(clip_percentile=clip_p, method=method, clip_sigma=clip_sigma)

        # ------------------------------------------------------------------
        # Add the action-side stats to ``self.stats`` under these keys.  When
        # ``norm_per_embodiment`` is on, ``_add_action_stat`` fills the
        # top-level key from the primary embodiment (robot) and stores the
        # per-embodiment breakdown under ``self.stats['_per_embodiment']``.
        # ------------------------------------------------------------------
        def _add_action_stat(key, all_arr):
            if not self.norm_per_embodiment:
                self.stats[key] = get_data_stats(all_arr, **norm_kw)
                return
            sample_emb = self.get_sample_embodiment()
            embodiments = sorted(set(sample_emb.tolist()))
            per_emb = self.stats.setdefault("_per_embodiment", {})
            primary = self.stats.get("_primary_embodiment",
                                     self.stats_primary_embodiment)
            if primary not in embodiments:
                primary = embodiments[0]
                self.stats["_primary_embodiment"] = primary
            for emb in embodiments:
                mask = sample_emb == emb
                emb_stat = get_data_stats(all_arr[mask], **norm_kw)
                per_emb.setdefault(emb, {})[key] = emb_stat
            self.stats[key] = per_emb[primary][key]

        print(f"[norm] method={method!r} clip_percentile={clip_p} "
              f"clip_sigma={clip_sigma} per_embodiment={self.norm_per_embodiment} "
              f"arms={self.arm_prefixes}")
        for p in self.arm_prefixes:
            rel_key = f"{p}relative_pose_xyz"
            arr = np.array(all_relative_xyz[p])
            print(f"all_relative xyz shape [{p or 'single'}]", arr.shape)
            _add_action_stat(rel_key, arr)
            print(f"relative pose xyz stats (action) [{p or 'single'}]",
                  self.stats[rel_key])
            if self._needs_state_xyz_stat:
                st_key = f"{p}relative_state_xyz"
                st_arr = np.array(all_state_rel_xyz[p])
                _add_action_stat(st_key, st_arr)
                print(f"relative pose xyz stats (state) [{p or 'single'}] "
                      f"min: {self.stats[st_key]['min']} "
                      f"max: {self.stats[st_key]['max']}")
            if self.needs_ee_rel_proprio:
                obs_key = f"{p}relative_pose_xyz_obs"
                _add_action_stat(obs_key, np.array(all_obs_rel_xyz[p]))
                print(f"relative pose xyz stats (obs) [{p or 'single'}]",
                      self.stats[obs_key])
            if self.needs_fingertip_action:
                ft_key = f"{p}fingertip_xyz_action"
                ft_arr = np.array(all_fingertip_xyz[p])
                print(f"fingertip action xyz shape [{p or 'single'}]", ft_arr.shape)
                _add_action_stat(ft_key, ft_arr)
                print(f"fingertip xyz stats [{p or 'single'}] min:",
                      self.stats[ft_key]["min"], "max:", self.stats[ft_key]["max"])
            elif self.needs_hand_action and self.relative_hand_action:
                rha_key = f"{p}relative_hand_action"
                rha_arr = np.array(all_relative_hand_action[p])
                print(f"all_relative hand action shape [{p or 'single'}]",
                      rha_arr.shape)
                _add_action_stat(rha_key, rha_arr)
                print(f"relative hand action stats [{p or 'single'}]",
                      self.stats[rha_key])

    # ------------------------------------------------------------------
    # Obs-side helpers
    # ------------------------------------------------------------------

    def _window_indices_into_sample(self) -> np.ndarray:
        """Return the obs-window indices into a sample of length ``self._sample_length``.

        "Method B" sample layout (1-D time axis, length ``(H-1)*d + pred_horizon``):

            [past obs frames | anchor t  |          future action            ]
            ^t-(H-1)d        ^t        ^t+1                              ^t + pred_horizon - 1
             sample[0]      sample[(H-1)*d]                              sample[-1]

        So the obs window occupies indices ``0, d, 2d, ..., (H-1)*d`` inside
        the sample (selecting the strided observations leading up to t), and
        the anchor is at index ``(H-1)*d == self._anchor_index_in_sample``.
        The action target is the future slice
        ``sample[(H-1)*d : (H-1)*d + pred_horizon]`` of length ``pred_horizon``.
        """
        H = self.obs_horizon
        d = self.down_sample_steps
        return np.arange(H, dtype=np.int64) * d

    def _slice_obs_window(self, arr: np.ndarray) -> np.ndarray:
        """Pick the ``obs_horizon`` frames from a sliced sample array."""
        return arr[self._window_indices_into_sample()]

    def _build_ee_rel_obs(self, pose_window: np.ndarray) -> np.ndarray:
        """``(obs_horizon, 9)`` ee_rel obs in **xyz + rot6d** (network format).

        ``pose_window`` is the obs-window slice of the ``pose`` array
        (xyz + axis-angle rotvec of the wrist in the robot_base frame),
        oldest-first.  We relativize every frame to the **last frame**
        (= the anchor / current time) so that ``ee_rel[-1] == identity``
        (xyz=0, rot6d=[1,0,0, 0,1,0]) and
        ``ee_rel[k] = T_t^{-1} · T_k`` for ``k < H-1``.

        The returned vector is ``[x, y, z, m00, m01, m02, m10, m11, m12]``
        — same network-facing layout used by the action target rotation.
        Identity normalization (no-op) is applied to the rot6d half; the
        xyz half is range-normalized by ``__getitem__`` using
        ``self.stats['relative_pose_xyz_obs']`` (the obs-window range, computed
        separately from the action stat ``relative_pose_xyz``).
        """
        T_world_curr = _pose6_to_mat(pose_window[-1])
        T_curr_world = invert_transformation(T_world_curr)
        H = pose_window.shape[0]
        rel_mats = np.stack(
            [T_curr_world @ _pose6_to_mat(pose_window[k]) for k in range(H)],
            axis=0,
        )  # (H, 4, 4)
        return _mat_to_pose9(rel_mats)  # (H, 9)

    def _resolve_stat_alias_pairs(self):
        """Put ``joint_next`` on ``hand_action``'s joint scale.

        The two blocks are only comparable if an identical joint angle maps to
        an identical normalized value, and the STATE block must not perturb the
        ACTION block's scale -- otherwise toggling ``predict_state`` would move
        the command target too and the ablation would measure two things at
        once.  ``joint_next`` is a shifted copy of ``proprioception``, so it adds
        no range; pooling it would only re-weight the percentile mixture.
        """
        buf = self.buffer.memory_buffer
        pairs = []
        for p in self.arm_prefixes:
            src, alias = f"{p}hand_action", f"{p}joint_next"
            if (
                alias in buf
                and src in buf
                and alias not in self.unnormal_list
                and src not in self.unnormal_list
            ):
                pairs.append((src, alias))
        return pairs

    def _block_source_keys(self, prefix, block):
        """Stream + stat names for one output block of one arm.

        ``block`` is ``"action"`` (the recorded command) or ``"state"`` (the
        observed next state).  Under the legacy overwriting mode there is only
        one block and it lives in the command streams, so the state block reads
        those and keeps the historical stat name -- that is what makes older
        runs and their ``stats.pickle`` reproduce unchanged.
        """
        if block == "action" or self.action_from_next_state:
            return (
                f"{prefix}pose_action" if self.action_uses_pose_action
                else f"{prefix}pose",
                f"{prefix}hand_action",
                f"{prefix}relative_pose_xyz",
            )
        if block != "state":
            raise ValueError(f"unknown output block {block!r}")
        return (
            f"{prefix}pose_next",
            f"{prefix}joint_next",
            f"{prefix}relative_state_xyz",
        )

    def _relative_action_pose_mats(
        self, nsample, anchor, future_end, prefix="", block="action"
    ):
        """``(pred_horizon, 4, 4)`` future wrist action poses relative to t.

        The ANCHOR (reference frame) is ALWAYS the current STATE wrist pose
        ``pose[anchor]`` (= time t).  The CONTENT comes from ``pose_action``
        (the real action = commanded next-state target) when
        ``action_uses_pose_action`` is on, else from ``pose`` (legacy: the
        state trajectory reused as the action).

        Result ``rel[k] = T_state(t)^{-1} · T_src(t+k)`` so that at deploy the
        absolute target is recovered as ``T_observed(t) · rel[k]`` — matching
        ``recover_absolute_target_pose_from_relative`` in the rollout, which
        anchors on the live observed wrist pose.
        """
        T_anchor_inv = invert_transformation(
            _pose6_to_mat(nsample[f"{prefix}pose"][anchor])
        )
        src_key, _, _ = self._block_source_keys(prefix, block)
        src = nsample[src_key][anchor:future_end]  # (pred_horizon, 6)
        n = src.shape[0]
        Ts = np.stack([_pose6_to_mat(src[i]) for i in range(n)], axis=0)
        return np.einsum("ij,njk->nik", T_anchor_inv, Ts)  # (n, 4, 4)

    # ------------------------------------------------------------------
    # Per-embodiment stats lookup
    # ------------------------------------------------------------------

    def _stats_for_sample(self, idx: int, key: str) -> dict:
        """Return the stats dict for ``key`` appropriate to sample ``idx``.

        Under per-embodiment normalization each sample must be normalized with
        its OWN embodiment's stats; otherwise (global normalization) we return
        the single top-level ``self.stats[key]``.
        """
        if not self.norm_per_embodiment:
            return self.stats[key]
        emb = self.get_sample_embodiment()[idx]
        per_emb = self.stats.get("_per_embodiment", {})
        emb_stats = per_emb.get(emb, {})
        # Fall back to the top-level (primary) stats if an embodiment somehow
        # lacks this key (shouldn't happen for present embodiments).
        return emb_stats.get(key, self.stats[key])

    # ------------------------------------------------------------------
    # Per-arm block builders (shared by single- and dual-arm __getitem__)
    # ------------------------------------------------------------------

    def _build_arm_motor_action(
        self, nsample, idx, prefix, anchor, future_end, block="action"
    ):
        """One arm's MOTOR block (eef + hand), WITHOUT the tactile block.

        Returns ``(pred_horizon, per_arm_block_dim)`` float32.  ``block`` selects
        the COMMAND streams (``"action"``) or the observed-next-state streams
        (``"state"``); the two differ only in which streams and which xyz stat
        are read, so the eef/joint layout and the relativization anchor (always
        the current state ``pose[t]``) are identical.  Keyed by ``prefix`` so each
        arm uses its OWN frame and stats.
        """
        pose_key, joint_key, xyz_stat = self._block_source_keys(prefix, block)
        rel_mats = self._relative_action_pose_mats(
            nsample, anchor, future_end, prefix=prefix, block=block
        )
        rel_xyz = rel_mats[:, :3, 3].astype(np.float32)       # (pred_horizon, 3)
        rel_rot6d = _mat_to_rot6d(rel_mats[:, :3, :3])        # (pred_horizon, 6)
        rel_xyz = normalize_data(
            rel_xyz, self._stats_for_sample(idx, xyz_stat)
        )
        rel_pose9 = np.concatenate([rel_xyz, rel_rot6d], axis=1)

        if not self.needs_hand_action:
            # EEF-only action: just the 9-D relative wrist pose.
            return rel_pose9.astype(np.float32)
        if self.hand_action_mode == "joint_only":
            # Just the joint block (no eef prepended).
            hand_action = nsample[joint_key][anchor:future_end]
            if self.relative_hand_action:
                hand_action = np.array(
                    [h - hand_action[0] for h in hand_action], dtype=np.float32,
                )
                hand_action = normalize_data(
                    hand_action,
                    self._stats_for_sample(idx, f"{prefix}relative_hand_action"),
                )
            return hand_action.astype(np.float32)
        if self.needs_fingertip_action:
            # Absolute fingertip pose in wrist frame (5 x (xyz+rot6d) = 45D),
            # from the next-state ``fingertip_action`` stream.
            ft = nsample[f"{prefix}fingertip_action"][anchor:future_end]
            ft_xyz = ft[..., :3].astype(np.float32)              # (pred, 5, 3)
            ft_mats = np.stack(
                [_pose6_to_mat(pp) for pp in ft.reshape(-1, 6)],
                axis=0,
            ).reshape(self.pred_horizon, 5, 4, 4)
            ft_rot6d = _mat_to_rot6d(ft_mats[..., :3, :3])       # (pred, 5, 6)
            ft_xyz_flat = ft_xyz.reshape(self.pred_horizon, 5 * 3)
            ft_xyz_flat = normalize_data(
                ft_xyz_flat,
                self._stats_for_sample(idx, f"{prefix}fingertip_xyz_action"),
            )
            ft_xyz = ft_xyz_flat.reshape(self.pred_horizon, 5, 3)
            ft_pose9 = np.concatenate([ft_xyz, ft_rot6d], axis=-1)
            ft_flat = ft_pose9.reshape(self.pred_horizon, FINGERTIP_DIM)
            if self.action_has_eef:
                return np.concatenate([rel_pose9, ft_flat], axis=1).astype(
                    np.float32
                )
            # fingertip_only: no leading eef pose.
            return ft_flat.astype(np.float32)
        if self.relative_hand_action:
            hand_action = nsample[joint_key][anchor:future_end]
            relative_hand_action = np.array(
                [h - hand_action[0] for h in hand_action], dtype=np.float32,
            )
            relative_hand_action = normalize_data(
                relative_hand_action,
                self._stats_for_sample(idx, f"{prefix}relative_hand_action"),
            )
            return np.concatenate(
                [rel_pose9, relative_hand_action], axis=1,
            ).astype(np.float32)
        hand_action = nsample[joint_key][anchor:future_end]
        return np.concatenate([rel_pose9, hand_action], axis=1).astype(np.float32)

    def _build_arm_tactile_future(self, nsample, prefix, anchor, future_end):
        """One arm's FUTURE tactile action block ``(pred_horizon, F)``.

        Shifted +1 (``fsr[t+1..]``, last frame repeated) so it is temporally
        aligned with the next-state wrist / joint action (see the single-arm
        rationale that this replaces).
        """
        tk = self.buffer.fsr_source_key
        tac = nsample[f"{prefix}{tk}"][anchor:future_end].astype(np.float32)
        return np.concatenate([tac[1:], tac[-1:]], axis=0)

    def _build_arm_proprio(self, nsample, idx, prefix, win_idx):
        """One arm's proprioception obs block ``(obs_horizon, per_arm_prop_dim)``.

        Returns ``None`` when ``proprio_mode`` contributes nothing for this arm.
        """
        prop_parts = []
        if self.needs_joint_proprio:
            prop_parts.append(
                nsample[f"{prefix}proprioception"][win_idx].astype(np.float32)
            )
        if self.needs_ee_rel_proprio:
            # PAST obs window, relativized to its own last frame (= time t).
            ee_rel9 = self._build_ee_rel_obs(nsample[f"{prefix}pose"][win_idx])
            ee_rel_xyz = ee_rel9[:, :3]
            ee_rel_rot6d = ee_rel9[:, 3:]
            ee_rel_xyz = normalize_data(
                ee_rel_xyz,
                self._stats_for_sample(idx, f"{prefix}relative_pose_xyz_obs"),
            )
            prop_parts.append(
                np.concatenate([ee_rel_xyz, ee_rel_rot6d], axis=-1).astype(
                    np.float32
                )
            )
        if self.needs_fingertip_proprio:
            ft_obs = nsample[f"{prefix}fingertip_pose_wrist"][win_idx]  # (H,5,6)
            ft_obs_xyz = ft_obs[..., :3].astype(np.float32)
            ft_obs_mats = np.stack(
                [_pose6_to_mat(pp) for pp in ft_obs.reshape(-1, 6)],
                axis=0,
            ).reshape(self.obs_horizon, 5, 4, 4)
            ft_obs_rot6d = _mat_to_rot6d(ft_obs_mats[..., :3, :3])
            ft_stat_key = f"{prefix}fingertip_xyz_action"
            if ft_stat_key in self.stats:
                ft_obs_xyz_flat = ft_obs_xyz.reshape(self.obs_horizon, 5 * 3)
                ft_obs_xyz_flat = normalize_data(
                    ft_obs_xyz_flat, self._stats_for_sample(idx, ft_stat_key)
                )
                ft_obs_xyz = ft_obs_xyz_flat.reshape(self.obs_horizon, 5, 3)
            ft_pose9 = np.concatenate([ft_obs_xyz, ft_obs_rot6d], axis=-1)
            prop_parts.append(
                ft_pose9.reshape(self.obs_horizon, FINGERTIP_DIM).astype(
                    np.float32
                )
            )
        if not prop_parts:
            return None
        return (
            prop_parts[0]
            if len(prop_parts) == 1
            else np.concatenate(prop_parts, axis=-1)
        )

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
        ) = self.indices[idx]
        nsample = sample_sequence(
            train_data=self.buffer.memory_buffer,
            sequence_length=self._sample_length,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        # ----- Vision (shared across arms): obs-window frames + transforms ---
        win_idx = self._window_indices_into_sample()
        for camera_id in self.buffer.load_camera_ids:
            cam = nsample[f"camera_{camera_id}"][win_idx]   # (obs_horizon, H, W, C)
            nsample[f"camera_{camera_id}"] = process_image(
                cam,
                self.optional_transforms,
                resize_shape=self.buffer.camera_resize_shape,
            )

        anchor = self._anchor_index_in_sample
        future_end = anchor + self.pred_horizon

        # ----- Action target (per-arm MOTOR blocks, then per-arm tactile) ----
        # Bimanual layout (UMI-style per-robot blocks), with each arm's motor
        # part being the enabled output blocks back to back:
        #   [ (arm0: action | state) , (arm1: action | state) , (arm0_tac | arm1_tac) ]
        # The motor blocks come first (one per arm) and the FUTURE tactile blocks
        # are grouped at the very end so the trainer can still strip a single
        # trailing tactile block of width ``tactile_action_dim =
        # per_arm_tactile_dim * num_arms``.  Single-arm (arm_prefixes == [""])
        # with action-only reduces to the legacy single block.
        motor_blocks = [
            self._build_arm_motor_action(
                nsample, idx, p, anchor, future_end, block=b
            )
            for p in self.arm_prefixes
            for b in self.motor_blocks
        ]
        action = (
            motor_blocks[0]
            if len(motor_blocks) == 1
            else np.concatenate(motor_blocks, axis=1)
        )
        if self.predict_tactile and not self.tactile_as_head:
            tac_blocks = [
                self._build_arm_tactile_future(nsample, p, anchor, future_end)
                for p in self.arm_prefixes
            ]
            action = np.concatenate([action, *tac_blocks], axis=1)
        nsample["action"] = action.astype(np.float32)

        # ----- HEAD targets (kept out of the diffusion vector) -----
        # Same per-arm builders as the in-vector blocks, so a head-routed run and
        # an in-vector run regress numerically identical quantities -- only the
        # route differs.  That equality is what makes the two a clean A/B.
        if self.state_as_head:
            blocks = [
                self._build_arm_motor_action(
                    nsample, idx, p, anchor, future_end, block="state"
                )
                for p in self.arm_prefixes
            ]
            nsample["aux_state"] = (
                blocks[0] if len(blocks) == 1
                else np.concatenate(blocks, axis=1)
            ).astype(np.float32)
        if self.tactile_as_head:
            blocks = [
                self._build_arm_tactile_future(nsample, p, anchor, future_end)
                for p in self.arm_prefixes
            ]
            nsample["aux_tactile"] = (
                blocks[0] if len(blocks) == 1
                else np.concatenate(blocks, axis=1)
            ).astype(np.float32)

        # ----- Proprioception (per-arm blocks concatenated) ------------------
        prop_blocks = [
            self._build_arm_proprio(nsample, idx, p, win_idx)
            for p in self.arm_prefixes
        ]
        prop_blocks = [b for b in prop_blocks if b is not None]
        if prop_blocks:
            nsample["proprioception"] = (
                prop_blocks[0]
                if len(prop_blocks) == 1
                else np.concatenate(prop_blocks, axis=-1)
            )
        else:
            nsample.pop("proprioception", None)

        # ----- FSR / tactile OBS (per-arm blocks concatenated) ---------------
        # Sliced to the obs window and merged into ONE ``fsr`` / ``force`` key
        # so the model / trainer consume a single tactile obs tensor.
        tk = self.buffer.fsr_source_key
        if self.buffer.enable_fsr:
            fsr_blocks = [
                nsample[f"{p}{tk}"][win_idx] for p in self.arm_prefixes
            ]
            nsample[tk] = (
                fsr_blocks[0]
                if self.num_arms == 1
                else np.concatenate(fsr_blocks, axis=-1)
            )

        # Strip the leftover full-length raw per-arm arrays that downstream code
        # never consumes (they were folded into action / proprioception / fsr
        # above).  For single-arm (prefix "") this pops the bare ``pose`` /
        # ``hand_action`` / ... exactly as before; the shared ``proprioception``
        # and tactile keys (prefix "") are kept as the obs tensors.
        for p in self.arm_prefixes:
            nsample.pop(f"{p}pose", None)
            nsample.pop(f"{p}pose_action", None)
            nsample.pop(f"{p}fingertip_pose_wrist", None)
            nsample.pop(f"{p}fingertip_action", None)
            nsample.pop(f"{p}hand_action", None)
            nsample.pop(f"{p}pose_next", None)
            nsample.pop(f"{p}joint_next", None)
            if p:
                # Dual-arm: the per-arm proprio / tactile raw keys have already
                # been merged into the shared keys above, so drop them.
                nsample.pop(f"{p}proprioception", None)
                nsample.pop(f"{p}{tk}", None)

        # Per-sample embodiment id (int into ``self.embodiment_names``) so the
        # trainer can break the loss down by embodiment (robot vs human).
        nsample["embodiment_id"] = int(self.get_sample_embodiment_ids()[idx])

        return nsample
