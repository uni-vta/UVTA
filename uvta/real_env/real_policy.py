"""Inference-side wrapper for ``UVTADataset`` policies.

This is the deployment counterpart to ``UVTADataset`` in
``uvta.diffusion_policy.dataloader.uvta_dataset``.  It is responsible
for:

* Reading the trained ``stats.pickle`` and synthesizing the action-side
  ``stats`` dict so the network output can be un-normalized.  The action
  vector is laid out as **xyz (3, range-norm) + rot6d (6, identity) +
  hand action** where the hand action is:
      - ``"joint"``     : 22-D joint angles (range-norm)
      - ``"fingertip"`` : 5 × (xyz_3 range-norm + rot6d_6 identity) = 45D
  ``cfg.dataset.hand_action_mode`` selects the hand action mode (falling
  back to the joint-vs-fingertip toggle inferred from
  ``cfg.dataset.proprio_mode``).

* Building the obs-window tensors (vision + proprioception) from a
  rolling buffer of observations.  Proprioception layout mirrors the
  action layout: xyz (range-norm) + rot6d (identity) + joint angles.

* Slicing the predicted ``pred_horizon`` trajectory to remove the
  observation-padding prefix.  In the trainer we deliberately keep the
  EEF rel and hand action targets relative to the *first* frame in the
  slice (= ``traj[t - (H-1)*d]``); at deploy time we want the actions
  starting from t so the executed step lives at position ``(H-1)*d`` of
  the predicted trajectory.

Rotation convention (network I/O): 6D continuous rotation (Zhou et al.
2019), matches UMI exactly.  See ``uvta_dataset._mat_to_rot6d`` /
``_rot6d_to_mat`` for the conversion.  Identity normalization is
implemented by setting ``stats["min"] == stats["max"] == 0`` for those
dims so ``unnormalize_data`` short-circuits (threshold = 5e-2).
"""

import collections
import os
import time

import cv2
import numpy as np
import torch
from uvta.common.utility.file import read_pickle
from uvta.common.utility.model import load_config, load_diffusion_model
from uvta.constants import INPAINT_RESIZE_RATIO
from uvta.diffusion_policy.dataloader.uvta_dataset import (
    _mat_to_pose9,
    _mat_to_rot6d,
    _pose6_to_mat,
    _rot6d_to_mat,
    resolve_output_routing,
)
from uvta.diffusion_policy.dataloader.diffusion_bc_dataset import (
    normalize_data,
    process_image,
    unnormalize_data,
)
from uvta.diffusion_policy.dataloader.ring_overlay import canonical_embodiment


# Per-frame dims in the **network I/O** (rot6d-based, NOT zarr xyz+rotvec).
JOINT_DIM = 22
EE_REL_DIM = 9          # xyz (3) + rot6d (6)
EE_REL_XYZ_DIM = 3
EE_REL_ROT6D_DIM = 6
FINGERTIP_DIM = 5 * 9   # 45 = 5 fingertips × (xyz + rot6d)
FINGERTIP_XYZ_DIM = 5 * 3  # 15
FINGERTIP_ROT6D_DIM = 5 * 6  # 30


def _canon_embodiments(names) -> set:
    """Canonicalize a config embodiment list, tolerating None / OmegaConf lists.

    Unknown names are dropped rather than raised on: a deploy session must not
    die because an unrelated embodiment alias in the training config is stale.
    """
    if not names:
        return set()
    out = set()
    for n in names:
        try:
            out.add(canonical_embodiment(str(n)))
        except ValueError:
            print(f"[RealPolicy] ignoring unknown embodiment {n!r} in config")
    return out


def _resolve_modes(cfg):
    """Return (proprio_mode, hand_action_mode) for both old and new configs.

    Old v13-style configs don't have ``proprio_mode`` / ``hand_action_mode``
    at all; default them to the legacy joint-angle behaviour.
    """
    dataset_cfg = cfg.dataset
    proprio_mode = getattr(dataset_cfg, "proprio_mode", "joint")
    hand_action_mode = getattr(dataset_cfg, "hand_action_mode", None)
    if hand_action_mode is None:
        hand_action_mode = (
            "fingertip"
            if proprio_mode in ("fingertip", "fingertip_with_ee")
            else "joint"
        )
    return proprio_mode, hand_action_mode


class RealPolicy:
    """Inference wrapper compatible with the new UVTADataset feature set."""

    def __init__(
        self,
        model_path: str,
        ckpt: int,
        use_ema: bool | None = None,
        baseline_correct: bool | None = None,
        baseline_frames: int | None = None,
    ):
        if baseline_frames is not None and int(baseline_frames) <= 0:
            raise ValueError("baseline_frames must be positive when provided")
        model_cfg = load_config(model_path)
        # ``use_ema`` selects ``ema_epoch_<ckpt>.ckpt`` vs ``epoch_<ckpt>.ckpt``.
        # ``None`` (the default) follows the training config; pass an explicit
        # bool to override it at inference time (e.g. force the EMA weights).
        if use_ema is None:
            use_ema = model_cfg.training.use_ema
        print(f"[RealPolicy] use_ema={use_ema} (ckpt={ckpt})")
        model, noise_scheduler = load_diffusion_model(
            model_path, ckpt, use_ema=use_ema
        )
        stats = read_pickle(os.path.join(model_path, "stats.pickle"))

        self.pred_horizon = model_cfg.dataset.pred_horizon
        # ``action_horizon`` decides how many of the pred_horizon frames to
        # execute per rollout step.  Required to be present in v14+ configs.
        self.action_horizon = int(model_cfg.dataset.action_horizon)
        self.action_dim = model_cfg.action_dim
        self.obs_horizon = int(model_cfg.dataset.obs_horizon)
        # ``down_sample_steps`` is new in v14; default to 1 for legacy configs.
        self.down_sample_steps = int(
            getattr(model_cfg.dataset, "down_sample_steps", 1)
        )

        self.proprio_mode, self.hand_action_mode = _resolve_modes(model_cfg)
        print(
            f"[RealPolicy] proprio_mode={self.proprio_mode}  "
            f"hand_action_mode={self.hand_action_mode}  "
            f"obs_horizon={self.obs_horizon}  "
            f"down_sample_steps={self.down_sample_steps}"
        )

        self.model = model.eval()
        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = model_cfg.num_inference_steps
        self.stats = stats
        self.camera_resize_shape = model_cfg.dataset.camera_resize_shape

        # ------------------------------------------------------------------
        # Bimanual: arm prefixes + camera stacking (mirror UVTADataset).
        #
        # * ``dataset.arms`` (e.g. ["left_", "right_"]) selects the per-arm
        #   streams; the action / obs tensors are per-arm blocks concatenated
        #   back-to-back in THIS order.  Single-arm configs (no ``arms``)
        #   default to [""], reproducing the exact legacy single-block layout.
        # * ``dataset.load_camera_ids`` (e.g. [0, 2]) lists the cameras the
        #   trainer stacked along the obs-horizon axis and fed through ONE
        #   shared vision backbone.  Deploy MUST feed them in the SAME order
        #   (chips_teleop: camera_0 = right wrist, camera_2 = left wrist).
        # ------------------------------------------------------------------
        _arms = getattr(model_cfg.dataset, "arms", None)
        self.arm_prefixes = [str(a) for a in _arms] if _arms else [""]
        self.num_arms = len(self.arm_prefixes)
        _cam_ids = getattr(model_cfg.dataset, "load_camera_ids", None)
        if _cam_ids is None:
            _cam_ids = [getattr(model_cfg.dataset, "camera_id", 0)]
        self.camera_ids = [int(c) for c in _cam_ids]
        self.num_cameras = len(self.camera_ids)
        print(
            f"[RealPolicy] arms={self.arm_prefixes} (num_arms={self.num_arms})  "
            f"cameras={self.camera_ids}"
        )

        # ------------------------------------------------------------------
        # Tactile (FSR / force) conditioning.  The training dataset stores the
        # tactile stream under its SOURCE field name (dataset.fsr_source_key,
        # default "fsr") and applies the same binarize / cutoff preprocessing.
        # We mirror that here so deploy-time normalization matches training:
        #   * ``tactile_key``       -> which stats.pickle key to normalize with;
        #   * ``fsr_binarize`` +
        #     ``fsr_binary_cutoff`` -> optional 0/1 thresholding applied BEFORE
        #                              normalization (must match training).
        # ``tactile_key`` also falls back to "fsr" if the configured key is
        # absent from the saved stats (older checkpoints).
        _ds_cfg = model_cfg.dataset
        self.enable_fsr = bool(_ds_cfg.get("enable_fsr", False))
        # ``tactile_key`` is the BASE stream name ("fsr" / "force"); the actual
        # stats key is resolved per-arm via ``_arm_stat_key`` (e.g. "left_fsr").
        self.tactile_key = str(_ds_cfg.get("fsr_source_key", "fsr"))
        if self._arm_stat_key(self.arm_prefixes[0], self.tactile_key) not in self.stats:
            if self._arm_stat_key(self.arm_prefixes[0], "fsr") in self.stats:
                self.tactile_key = "fsr"
        self.fsr_binarize = bool(_ds_cfg.get("fsr_binarize", True))
        _cutoff = _ds_cfg.get("fsr_binary_cutoff", None)
        self.fsr_binary_cutoff = (
            np.array(_cutoff, dtype=np.float32) if _cutoff is not None else None
        )
        # ------------------------------------------------------------------
        # Tactile baseline correction on the LIVE stream, mirroring what
        # ``replay_buffer.py`` applied per episode at TRAIN time.  Deploy is
        # always the ROBOT embodiment, so the live stream needs the correction
        # iff training corrected 'robot':
        #   * robot in ``fsr_baseline_first_frame_embodiments`` -> N = 1
        #   * robot in ``fsr_baseline_embodiments``             -> N =
        #     ``fsr_baseline_frames``   (first-frame wins, as in training)
        # The baseline is the mean of the session's first N RAW frames, frozen
        # once latched, and subtraction is clamped at 0 exactly like training
        # (``np.maximum(x - baseline, 0)``) so "0 = no contact" holds on both
        # sides.  Skipping this on a model trained WITH robot correction leaves
        # the live stream sitting a full resting offset above the training
        # distribution (measured at ~2% of peak on light_teleop2 / bottle_teleop).
        #
        # ``baseline_correct``: None = follow the training config (default),
        # True/False = force on/off. ``baseline_frames`` optionally overrides
        # the training frame count at deployment; 1 means latch and freeze the
        # very first live frame. Forcing settings that differ from training can
        # change the input distribution, so only override deliberately.
        # ------------------------------------------------------------------
        _bl_train_on = bool(_ds_cfg.get("fsr_baseline_correct", False))
        _bl_mean_embs = _canon_embodiments(
            _ds_cfg.get("fsr_baseline_embodiments", None)
        )
        _bl_first_embs = _canon_embodiments(
            _ds_cfg.get("fsr_baseline_first_frame_embodiments", None)
        )
        _robot_first = "robot" in _bl_first_embs
        _robot_mean = "robot" in _bl_mean_embs
        _bl_auto = _bl_train_on and (_robot_first or _robot_mean)
        self.fsr_baseline_correct = (
            _bl_auto if baseline_correct is None else bool(baseline_correct)
        )
        # An explicit deployment override takes precedence, followed by the
        # training first-frame mode and finally the configured mean window.
        self.fsr_baseline_frames = int(
            baseline_frames
            if baseline_frames is not None
            else 1
            if _robot_first
            else _ds_cfg.get("fsr_baseline_frames", 5)
        )
        if self.enable_fsr:
            print(
                f"Tactile enabled: stats key='{self.tactile_key}', "
                f"binarize={self.fsr_binarize}, cutoff={self.fsr_binary_cutoff}"
            )
            if self.fsr_baseline_correct:
                _src_parts = []
                if baseline_correct:
                    _src_parts.append("correction forced ON")
                if baseline_frames is not None:
                    _src_parts.append("frame count overridden at deployment")
                _src = ", ".join(_src_parts) or "from train config"
                print(
                    f"Tactile baseline correction ({_src}): subtract the mean of "
                    f"the first {self.fsr_baseline_frames} live frame(s) per arm, "
                    "clamped at 0."
                )
            else:
                _why = (
                    "forced OFF"
                    if baseline_correct is False
                    else f"train config corrected {sorted(_bl_mean_embs | _bl_first_embs)}"
                    if _bl_train_on
                    else "training did not baseline-correct"
                )
                print(f"Tactile baseline correction: OFF ({_why}).")

        # ------------------------------------------------------------------
        # Future-tactile action prediction (predict_future_tactile).  When the
        # policy was trained to ALSO predict the future tactile trajectory, the
        # action vector carries a trailing tactile block of width
        # ``tactile_action_dim`` (per-frame tactile feature dim).  Deploy only
        # executes the wrist / hand part, so we (a) extend the action stats with
        # a tactile normalization segment (so unnormalize covers the full width)
        # and (b) slice the predicted tactile off before decoding.  The tactile
        # feature dim is read from the saved tactile stats.
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Output-block composition (predict_action / predict_state / tactile).
        # A policy may predict the recorded COMMAND, the resulting next STATE, or
        # both, plus the tactile block.  Per arm the motor part is the enabled
        # blocks back to back: ``[action | state]``.  Deploy executes ONE of
        # them -- the command when present, else the state, which is exactly what
        # the legacy ``action_from_next_state=True`` meant -- so the block that
        # is not executed is sliced off before decoding and the external
        # contract stays the historical ``[eef | joint]`` per arm.
        # ------------------------------------------------------------------
        _pa = _ds_cfg.get("predict_action", None)
        _ps = _ds_cfg.get("predict_state", None)
        _pt = _ds_cfg.get("predict_tactile", None)
        _afns = _ds_cfg.get("action_from_next_state", None)
        _pft = _ds_cfg.get("predict_future_tactile", None)
        if any(x is not None for x in (_pa, _ps, _pt)):
            _predict_action = True if _pa is None else bool(_pa)
            _predict_state = True if _ps is None else bool(_ps)
            _predict_tactile = True if _pt is None else bool(_pt)
        else:
            _afns = bool(_afns)
            _predict_action = not _afns
            _predict_state = _afns
            _predict_tactile = bool(_pft)
        # Does the state block carry its own wrist-xyz range?  Only when the
        # command AND the next state are both predicted -- then the dataset reads
        # the state off ``pose_next`` / ``relative_state_xyz`` instead of
        # overwriting the command streams (``_block_source_keys``).  Independent
        # of whether the state travels in the diffusion vector or on a head.
        self._state_own_xyz = bool(_predict_action and _predict_state)
        # ``aux_as_head`` names the targets that were regressed by separate MLP
        # heads off the conditioning vector instead of riding in the diffusion
        # trajectory.  For the trajectory's LAYOUT a head-routed target is
        # indistinguishable from one that was never predicted, so the routing has
        # to be resolved with the exact same function the dataset used -- hence
        # the shared ``resolve_output_routing`` rather than a second copy of the
        # rules here.
        (
            self.motor_blocks,
            self.executed_block,
            self.aux_targets,
        ) = resolve_output_routing(
            predict_action=_predict_action,
            predict_state=_predict_state,
            predict_tactile=_predict_tactile,
            aux_as_head=_ds_cfg.get("aux_as_head", False),
        )
        self.aux_as_head = bool(self.aux_targets)
        self.executed_block_index = self.motor_blocks.index(self.executed_block)
        # In-vector tactile only.  A head-routed tactile is served from
        # ``predict_world`` and must NOT be sliced off the trajectory.
        self.predict_future_tactile = bool(
            _predict_tactile and "tactile" not in self.aux_targets
        )
        # ``per_arm_tactile_dim`` = one arm's tactile feature width;
        # ``tactile_action_dim`` = the TOTAL trailing tactile block appended to
        # the action (one per arm, grouped at the very end): [... | L_tac | R_tac].
        self.per_arm_tactile_dim = 0
        self.tactile_action_dim = 0
        # A tactile world head needs the same per-arm width to unnormalize its
        # output, so resolve the width whenever tactile is predicted at all and
        # keep ``tactile_action_dim`` (the in-vector block) at 0 for the head.
        if self.predict_future_tactile or "tactile" in self.aux_targets:
            k0 = self._arm_stat_key(self.arm_prefixes[0], self.tactile_key)
            if k0 not in self.stats:
                raise KeyError(
                    "tactile prediction is enabled but tactile stats key "
                    f"'{k0}' is missing from stats.pickle."
                )
            self.per_arm_tactile_dim = int(
                np.asarray(self.stats[k0]["min"]).size
            )
        if self.predict_future_tactile:
            self.tactile_action_dim = self.per_arm_tactile_dim * self.num_arms
            print(
                f"Future-tactile prediction enabled: appending "
                f"{self.per_arm_tactile_dim}-D x {self.num_arms} arm(s) = "
                f"{self.tactile_action_dim}-D tactile block to the action "
                "(sliced off at execution)."
            )

        # ------------------------------------------------------------------
        # Build action stats.
        #
        # Action layout (network I/O):
        #     [eef_xyz(3, range), eef_rot6d(6, identity),
        #      hand( joint 22 range  OR  fingertip 5x(xyz 3 range + rot6d 6 identity) )]
        #
        # Scheme A: which dims are normalized is decided by an explicit
        # ``identity_mask`` (True = skip, i.e. rot6d) carried in the stats
        # dict, NOT by a range threshold.  xyz / joint dims are ALWAYS
        # normalized; only rot6d channels are flagged identity.
        # ------------------------------------------------------------------
        # Normalization method / sigma-clamp are carried per-stream in the
        # saved stats (all streams share the dataset's config); inherit from
        # the FIRST arm's eef-xyz stats (or the hand stats for joint_only,
        # which has no eef block).  Legacy stats without ``method`` -> minmax.
        def _seg(src_stats, n):
            """Return (min,max,mean,std,mask) for a normalized sub-block of
            length ``n`` drawn from ``src_stats`` (mask all-False = normalize).
            mean/std fall back to min/max for legacy stats that lack them."""
            mn = np.asarray(src_stats["min"], dtype=np.float32)
            mx = np.asarray(src_stats["max"], dtype=np.float32)
            mean = np.asarray(src_stats.get("mean", mn), dtype=np.float32)
            std = np.asarray(src_stats.get("std", mx - mn), dtype=np.float32)
            mask = np.zeros(n, dtype=bool)
            return mn[:n], mx[:n], mean[:n], std[:n], mask

        def _identity_seg(n):
            """A skipped (identity) sub-block of length ``n``: all zeros, mask
            all-True so both minmax and meanstd short-circuit it."""
            z = np.zeros(n, dtype=np.float32)
            return z, z, z.copy(), z.copy(), np.ones(n, dtype=bool)

        def _arm_motor_segs(prefix, block="action"):
            """Segments for ONE arm's ONE output block (eef + hand).

            Keyed by this arm's own stats (``{prefix}relative_pose_xyz`` etc.),
            mirroring ``UVTADataset._build_arm_motor_action`` for the matching
            ``block``."""
            def sk(base):
                return self._arm_stat_key(prefix, base)

            eef_xyz = _seg(stats[sk("relative_pose_xyz")], EE_REL_XYZ_DIM)
            eef_rot6d = _identity_seg(EE_REL_ROT6D_DIM)

            if self.hand_action_mode == "none":
                hand_segs = []
            elif self.hand_action_mode in ("fingertip", "fingertip_only"):
                hand_xyz_stats = stats[sk("fingertip_xyz_action")]
                # Per finger: [xyz_3 normalize, rot6d_6 identity], interleaved.
                hand_segs = []
                for f in range(5):
                    sub = {
                        k: np.asarray(hand_xyz_stats[k])[f * 3: f * 3 + 3]
                        for k in ("min", "max", "mean", "std")
                        if k in hand_xyz_stats
                    }
                    hand_segs.append(_seg(sub, 3))
                    hand_segs.append(_identity_seg(6))
            elif model_cfg.dataset.relative_hand_action:
                s = stats[sk("relative_hand_action")]
                hand_segs = [_seg(s, np.asarray(s["min"]).size)]
            else:
                s = stats[sk("hand_action")]
                hand_segs = [_seg(s, np.asarray(s["min"]).size)]
            if block == "state" and self._state_own_xyz:
                # Only when the two blocks COEXIST does the state block have its
                # own wrist range (whether the state rides in the diffusion
                # vector or comes off a world head).  State-only is the legacy
                # overwriting mode:
                # the state lives in the command streams and keeps the historical
                # ``relative_pose_xyz``, exactly as
                # ``UVTADataset._block_source_keys`` resolves it.  Either way
                # the joints reuse hand_action's stats (the dataset aliases
                # joint_next onto them), so ``hand_segs`` above already applies.
                s_xyz = stats.get(sk("relative_state_xyz"))
                if s_xyz is None:
                    raise KeyError(
                        f"{sk('relative_state_xyz')} missing from stats.pickle "
                        "but the config predicts both the action and the state "
                        "block; the checkpoint predates the block split."
                    )
                eef_xyz = _seg(s_xyz, 3)

            if self.hand_action_mode == "joint_only":
                # NO eef block: the arm's block is just the joint angles.
                src = (stats[sk("relative_hand_action")]
                       if model_cfg.dataset.relative_hand_action
                       else stats[sk("hand_action")])
                return [_seg(src, np.asarray(src["min"]).size)]
            if self.hand_action_mode == "fingertip_only":
                # NO eef block: just the 45-D fingertip block.
                return hand_segs
            return [eef_xyz, eef_rot6d] + hand_segs

        # Meta (method / clip_sigma) from arm-0's driving stream.
        p0 = self.arm_prefixes[0]
        if self.hand_action_mode == "joint_only":
            _src_for_meta = (
                stats.get(self._arm_stat_key(p0, "hand_action"))
                or stats.get(self._arm_stat_key(p0, "relative_hand_action"))
            )
        elif self.hand_action_mode == "fingertip_only":
            _src_for_meta = stats.get(self._arm_stat_key(p0, "fingertip_xyz_action"))
        else:
            _src_for_meta = stats.get(self._arm_stat_key(p0, "relative_pose_xyz"))
        _src_for_meta = _src_for_meta or {}
        norm_method = _src_for_meta.get("method", "minmax")
        clip_sigma = _src_for_meta.get("clip_sigma", None)
        print(
            f"Action layout: hand_action_mode={self.hand_action_mode}  "
            f"relative_hand_action={model_cfg.dataset.relative_hand_action}"
        )

        # Bimanual layout (mirror UVTADataset.__getitem__):
        #   [ arm0_motor | arm1_motor | ... | arm0_tac | arm1_tac | ... ]
        # motor blocks (eef+hand) come first, one per arm; the future-tactile
        # blocks are grouped at the very end (one per arm).  Single-arm
        # (arm_prefixes == [""]) reduces to the legacy single block.
        # Each arm contributes one segment run per enabled output block; the
        # blocks share the joint stats (``joint_next`` is a stat alias of
        # ``hand_action``) and differ only in the wrist xyz stat, which is
        # ``relative_state_xyz`` for the state block.
        motor_segs_per_arm = [
            [s for b in self.motor_blocks for s in _arm_motor_segs(p, b)]
            for p in self.arm_prefixes
        ]
        self.per_arm_motor_dim = int(
            sum(np.asarray(s[0]).size for s in motor_segs_per_arm[0])
        )
        # Width of ONE block for one arm = what the rollout ultimately consumes.
        self.per_arm_block_dim = self.per_arm_motor_dim // len(self.motor_blocks)
        segs = [s for arm_segs in motor_segs_per_arm for s in arm_segs]

        # Future-tactile blocks: one per arm, range-normalized with that arm's
        # own tactile stats (``{prefix}fsr``), appended after ALL motor blocks.
        if self.tactile_action_dim > 0:
            for p in self.arm_prefixes:
                tk = self._arm_stat_key(p, self.tactile_key)
                segs.append(_seg(self.stats[tk], self.per_arm_tactile_dim))

        self.stats["action"] = {
            "min": np.concatenate([s[0] for s in segs]),
            "max": np.concatenate([s[1] for s in segs]),
            "mean": np.concatenate([s[2] for s in segs]),
            "std": np.concatenate([s[3] for s in segs]),
            "identity_mask": np.concatenate([s[4] for s in segs]),
            "method": norm_method,
            "clip_sigma": clip_sigma,
        }
        # Stats for the STATE world head.  Built from the same per-arm segment
        # machinery as the in-vector state block, so the two routes decode
        # identically; it just is not part of stats["action"] any more.
        if "state" in self.aux_targets:
            _s_segs = [
                s for p in self.arm_prefixes for s in _arm_motor_segs(p, "state")
            ]
            self.stats["aux_state"] = {
                "min": np.concatenate([s[0] for s in _s_segs]),
                "max": np.concatenate([s[1] for s in _s_segs]),
                "mean": np.concatenate([s[2] for s in _s_segs]),
                "std": np.concatenate([s[3] for s in _s_segs]),
                "identity_mask": np.concatenate([s[4] for s in _s_segs]),
                "method": norm_method,
                "clip_sigma": clip_sigma,
            }
        if self.aux_targets:
            print(
                f"[aux heads] {self.aux_targets} regressed from the conditioning "
                f"vector; the diffusion vector holds {self.motor_blocks}"
                + (" + tactile" if self.tactile_action_dim else "")
                + f", and {self.executed_block!r} is the sampled block deploy uses."
            )
        print(
            f"action stats method={norm_method} clip_sigma={clip_sigma} | "
            f"total={self.stats['action']['min'].size} "
            f"per_arm_motor={self.per_arm_motor_dim} "
            f"per_arm_block={self.per_arm_block_dim} "
            f"blocks={self.motor_blocks} exec={self.executed_block!r} "
            f"tactile={self.tactile_action_dim} | "
            "min head:", self.stats["action"]["min"][:9],
        )
        self.model_cfg = model_cfg

        # Per-frame proprio dim from the saved stats (only for ``joint`` /
        # ``both``; ee_rel & fingertip streams are not in stats since they
        # live in unnormal_list).
        if "proprioception" in stats:
            self.proprio_joint_dim = int(np.asarray(stats["proprioception"]["min"]).size)
        else:
            self.proprio_joint_dim = JOINT_DIM  # safe default

        # ------------------------------------------------------------------
        # Rolling observation buffers.  These let the caller push a single
        # new observation each step via ``push_observation`` and we manage
        # the ``obs_horizon * down_sample_steps`` history internally.
        # ------------------------------------------------------------------
        self._buf_capacity = (self.obs_horizon - 1) * self.down_sample_steps + 1
        # Per-CAMERA visual history (keyed by camera id) and per-ARM proprio /
        # tactile history (keyed by arm prefix).  Single-arm / single-camera
        # collapses to a one-entry dict, so the legacy
        # ``push_observation(frame, joint_proprio=...)`` call still works.
        self._visual_history = {
            cid: collections.deque(maxlen=self._buf_capacity)
            for cid in self.camera_ids
        }
        self._joint_history = {
            p: collections.deque(maxlen=self._buf_capacity)
            for p in self.arm_prefixes
        }
        self._pose_history = {
            p: collections.deque(maxlen=self._buf_capacity)
            for p in self.arm_prefixes
        }
        self._fingertip_history = {
            p: collections.deque(maxlen=self._buf_capacity)
            for p in self.arm_prefixes
        }
        # ``_fsr_history`` deliberately holds the RAW pushed values so debug
        # dumps stay faithful to the sensor; the baseline is subtracted only
        # when a window is built.  The baseline itself is latched separately
        # because the deque's ``maxlen`` would otherwise evict the very first
        # frames it is computed from.
        self._fsr_history = {
            p: collections.deque(maxlen=self._buf_capacity)
            for p in self.arm_prefixes
        }
        self._fsr_baseline = {p: None for p in self.arm_prefixes}
        self._fsr_baseline_acc = {p: [] for p in self.arm_prefixes}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_history(self):
        for d in self._visual_history.values():
            d.clear()
        for hist in (
            self._joint_history,
            self._pose_history,
            self._fingertip_history,
            self._fsr_history,
        ):
            for d in hist.values():
                d.clear()
        # A new episode re-latches the baseline: the resting offset drifts
        # between takes, which is exactly why training re-estimates it per
        # episode instead of once per dataset.
        for p in self.arm_prefixes:
            self._fsr_baseline[p] = None
            self._fsr_baseline_acc[p].clear()

    def get_latest_tactile_physical(self) -> "np.ndarray | None":
        """Return the latest tactile input after deployment preprocessing.

        ``_fsr_history`` stores RAW frames, so the latched baseline is applied
        here to yield physical ``fsr`` / ``fsr_region`` units after the same
        subtraction and non-negative clamp inference uses, but before
        binarization and normalization.  This makes it directly comparable with
        ``predict_future_tactile`` once that output is unnormalized back to
        physical units.
        """
        parts = []
        for p in self.arm_prefixes:
            hist = self._fsr_history[p]
            if not hist:
                return None
            frame = np.asarray(hist[-1], dtype=np.float32).reshape(-1)
            parts.append(self._apply_fsr_baseline(p, frame))
        out = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=-1)
        return np.asarray(out, dtype=np.float32).copy()

    def push_observation(
        self,
        visual_obs,
        joint_proprio=None,
        wrist_pose6=None,
        fingertip_pose_wrist=None,
        fsr=None,
    ):
        """Append a single new observation to the rolling history.

        Single-arm / single-camera policies accept a single array per field
        (legacy behaviour).  Bimanual / multi-camera policies accept per-camera
        / per-arm values as either a **list** (aligned to ``load_camera_ids`` /
        ``arms``) or a **dict** (keyed by camera id / arm prefix):

        Parameters
        ----------
        visual_obs : (H, W, 3) frame, or a list/dict of one (H, W, 3) frame
            per camera in ``self.camera_ids`` order (e.g. [cam0, cam2]).
        joint_proprio : (22,) joint vector per arm (needed for joint proprio).
        wrist_pose6 : (6,) xyz+rotvec wrist pose in base frame, per arm
            (needed for ee_rel proprio).
        fingertip_pose_wrist : (5, 6) fingertip pose in wrist frame, per arm.
        fsr : FSR / tactile vector per arm (needed for tactile conditioning).
        """
        for cid, frame in self._as_camera_map(visual_obs).items():
            frame = np.asarray(frame)
            if frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError(
                    f"camera {cid} frame must be (H, W, 3), got {frame.shape}"
                )
            if cid not in self._visual_history:
                raise KeyError(
                    f"camera {cid} not in load_camera_ids={self.camera_ids}"
                )
            self._visual_history[cid].append(frame.astype(np.uint8, copy=True))
        for p, v in self._as_arm_map(joint_proprio).items():
            self._joint_history[p].append(
                np.asarray(v, dtype=np.float32).reshape(-1)
            )
        for p, v in self._as_arm_map(wrist_pose6).items():
            self._pose_history[p].append(
                np.asarray(v, dtype=np.float32).reshape(6)
            )
        for p, v in self._as_arm_map(fingertip_pose_wrist).items():
            self._fingertip_history[p].append(
                np.asarray(v, dtype=np.float32).reshape(5, 6)
            )
        for p, v in self._as_arm_map(fsr).items():
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            self._fsr_history[p].append(v)
            self._latch_fsr_baseline(p, v)

    # ------------------------------------------------------------------
    # Tactile baseline (live-stream counterpart of the train-time correction)
    # ------------------------------------------------------------------

    def _latch_fsr_baseline(self, prefix: str, frame: np.ndarray):
        """Accumulate the session's first N raw frames, then freeze their mean."""
        if not self.fsr_baseline_correct or self._fsr_baseline[prefix] is not None:
            return
        acc = self._fsr_baseline_acc[prefix]
        acc.append(frame.copy())
        if len(acc) >= self.fsr_baseline_frames:
            self._fsr_baseline[prefix] = np.mean(
                np.stack(acc, axis=0), axis=0
            ).astype(np.float32)
            acc.clear()
            print(
                f"[RealPolicy] tactile baseline latched for arm "
                f"{prefix or 'single'}: mean={self._fsr_baseline[prefix].mean():.1f} "
                f"max={self._fsr_baseline[prefix].max():.1f}"
            )

    def _apply_fsr_baseline(self, prefix: str, fsr: np.ndarray) -> np.ndarray:
        """Subtract this arm's latched baseline, clamped at 0 (train parity).

        Before the baseline is fully latched the partial mean of whatever frames
        have arrived is used, so the first few inference steps of an episode are
        still corrected rather than silently running uncorrected.
        """
        if not self.fsr_baseline_correct:
            return fsr
        bl = self._fsr_baseline[prefix]
        if bl is None:
            acc = self._fsr_baseline_acc[prefix]
            if not acc:
                return fsr
            bl = np.mean(np.stack(acc, axis=0), axis=0).astype(np.float32)
        if bl.shape[-1] != fsr.shape[-1]:
            raise ValueError(
                f"tactile baseline width {bl.shape[-1]} != stream width "
                f"{fsr.shape[-1]} for arm {prefix or 'single'}"
            )
        return np.maximum(fsr - bl, 0.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _arm_stat_key(self, prefix, base):
        """Resolve a stats key for ``base`` under ``prefix``.

        Returns ``f"{prefix}{base}"`` when present in the saved stats (bimanual,
        e.g. ``"left_fsr"``), otherwise the bare ``base`` (single-arm / shared).
        """
        k = f"{prefix}{base}"
        if k in self.stats:
            return k
        return base

    def _as_camera_map(self, visual_obs):
        """Normalize ``visual_obs`` into ``{camera_id: (H, W, 3)}``.

        Accepts a single frame (only when the policy uses ONE camera), a
        list/tuple aligned to ``self.camera_ids``, or a dict keyed by camera id.
        """
        if isinstance(visual_obs, dict):
            return {int(k): np.asarray(v) for k, v in visual_obs.items()}
        if isinstance(visual_obs, (list, tuple)):
            if len(visual_obs) != self.num_cameras:
                raise ValueError(
                    f"expected {self.num_cameras} camera frame(s) "
                    f"(load_camera_ids={self.camera_ids}), got {len(visual_obs)}"
                )
            return {
                cid: np.asarray(f)
                for cid, f in zip(self.camera_ids, visual_obs)
            }
        arr = np.asarray(visual_obs)
        if arr.ndim == 3:
            if self.num_cameras != 1:
                raise ValueError(
                    f"policy uses {self.num_cameras} cameras "
                    f"(ids={self.camera_ids}); pass a list/dict of frames "
                    "in that order, not a single array."
                )
            return {self.camera_ids[0]: arr}
        raise ValueError(
            f"cannot interpret visual_obs of shape {arr.shape} as camera frames"
        )

    def _as_arm_map(self, value):
        """Normalize a per-arm input into ``{prefix: array}`` (``None`` -> {}).

        Accepts a single array (single-arm / arm-0), a list/tuple aligned to
        ``self.arm_prefixes``, or a dict keyed by prefix.
        """
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, (list, tuple)):
            if len(value) != self.num_arms:
                raise ValueError(
                    f"expected {self.num_arms} per-arm entries "
                    f"(arms={self.arm_prefixes}), got {len(value)}"
                )
            return {p: v for p, v in zip(self.arm_prefixes, value)}
        return {self.arm_prefixes[0]: value}

    @staticmethod
    def _visual_input_frame_ndim(visual_obs) -> int:
        """ndim of a single element of ``visual_obs`` (3 = fresh frame, 4 =
        pre-stacked window), regardless of single / list / dict packaging."""
        if isinstance(visual_obs, dict):
            return np.asarray(next(iter(visual_obs.values()))).ndim
        if isinstance(visual_obs, (list, tuple)):
            return np.asarray(visual_obs[0]).ndim
        return np.asarray(visual_obs).ndim

    def _legacy_window_tensor(self, visual_obs) -> torch.Tensor:
        """Build the stacked visual tensor from pre-formed ``(N, H, W, 3)``
        window(s) (one per camera), bypassing the rolling history.

        Each camera window is padded/trimmed to ``obs_horizon`` (pad_before
        semantics), converted to a tensor, then the cameras are concatenated
        along the obs-horizon axis in ``self.camera_ids`` order.
        """
        if isinstance(visual_obs, dict):
            cam_arrays = [np.asarray(visual_obs[cid]) for cid in self.camera_ids]
        elif isinstance(visual_obs, (list, tuple)):
            if len(visual_obs) != self.num_cameras:
                raise ValueError(
                    f"expected {self.num_cameras} camera window(s), "
                    f"got {len(visual_obs)}"
                )
            cam_arrays = [np.asarray(v) for v in visual_obs]
        else:
            arr = np.asarray(visual_obs)
            if self.num_cameras != 1:
                raise ValueError(
                    f"policy uses {self.num_cameras} cameras (ids={self.camera_ids}); "
                    "pass a list/dict of (N,H,W,3) windows in that order."
                )
            cam_arrays = [arr]

        tensors = []
        for arr in cam_arrays:
            N = arr.shape[0]
            if N != self.obs_horizon:
                pad = self.obs_horizon - N
                if pad > 0:
                    arr = np.concatenate([arr[:1]] * pad + [arr], axis=0)
                else:
                    arr = arr[-self.obs_horizon:]
            tensors.append(self._frames_to_tensor(arr))
        return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=1)

    def _normalize_explicit_fsr(self, fsr) -> torch.Tensor:
        """Normalize an explicitly-passed tactile vector into ``(1, 1, F)``.

        For bimanual, ``fsr`` is the concatenated ``[arm0 | arm1 | ...]`` vector;
        it is split per arm and each block gets that arm's own baseline
        subtraction, binarization and ``{prefix}fsr`` normalization, in the same
        order training applied them.

        Callers that only ever pass tactile explicitly (never via
        ``push_observation``) still get a baseline: it is latched off these
        vectors instead.
        """
        fsr_arr = np.asarray(fsr, dtype=np.float32).reshape(1, -1)
        per = fsr_arr.shape[-1] // self.num_arms
        chunks = []
        for i, p in enumerate(self.arm_prefixes):
            seg = fsr_arr[:, i * per: (i + 1) * per]
            if not self._fsr_history[p]:
                self._latch_fsr_baseline(p, seg[0])
            seg = self._apply_fsr_baseline(p, seg)
            if self.fsr_binarize and self.fsr_binary_cutoff is not None:
                seg = np.where(
                    seg >= self.fsr_binary_cutoff, 1.0, 0.0
                ).astype(np.float32)
            chunks.append(
                normalize_data(
                    seg, self.stats[self._arm_stat_key(p, self.tactile_key)]
                )
            )
        fsr_arr = chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=-1)
        return torch.from_numpy(fsr_arr).unsqueeze(0).cuda()

    def _obs_window_indices(self, deque_len: int) -> list:
        """Return the indices into the rolling deque that correspond to the
        obs window ``[t - (H-1)*d, ..., t - d, t]``.

        When the history is shorter than the full window length we pad
        from the left (oldest) by repeating index 0, matching the
        ``pad_before`` semantics of ``ReplayBuffer.sample_sequence``.
        """
        H = self.obs_horizon
        d = self.down_sample_steps
        # deepest needed: index (deque_len - 1) - (H-1)*d, where the
        # deque is ordered oldest-first.  Pad on the left if negative.
        idxs = []
        for k in range(H - 1, -1, -1):
            i = (deque_len - 1) - k * d
            if i < 0:
                i = 0
            idxs.append(i)
        return idxs

    def _frames_to_tensor(self, frames: np.ndarray) -> torch.Tensor:
        """``(N, H, W, 3)`` uint8 RGB -> ``(1, N, C, H', W')`` cuda tensor.

        Resizes by ``INPAINT_RESIZE_RATIO`` then runs the deterministic
        train-time transforms (Resize + CenterCrop are size-safe).
        """
        _, H, W, _ = frames.shape
        frames = np.array(
            [
                cv2.resize(
                    f,
                    (
                        int(W * INPAINT_RESIZE_RATIO),
                        int(H * INPAINT_RESIZE_RATIO),
                    ),
                )
                for f in frames
            ]
        )
        frames_t = process_image(
            frames,
            optional_transforms=["Resize", "CenterCrop"],
            resize_shape=self.camera_resize_shape,
        )
        return frames_t.unsqueeze(0).cuda()  # (1, N, C, H, W)

    def _build_visual_window(self) -> torch.Tensor:
        """``(1, num_cameras * obs_horizon, C, H, W)`` torch tensor on cuda.

        Each camera's obs window is built independently, then the cameras are
        stacked along the obs-horizon axis in ``self.camera_ids`` order --
        mirroring ``train_diffusion_policy._gather_visual_obs`` so the shared
        vision backbone sees the frames in the SAME order as training
        (chips_teleop: [camera_0 = right wrist, camera_2 = left wrist]).
        """
        cams = []
        for cid in self.camera_ids:
            hist = self._visual_history[cid]
            if not hist:
                raise RuntimeError(
                    f"visual history for camera {cid} is empty; call "
                    "push_observation() first or pass frames to predict_action()."
                )
            idxs = self._obs_window_indices(len(hist))
            frames = np.stack([hist[i] for i in idxs], axis=0)  # (H_obs,H,W,3)
            cams.append(self._frames_to_tensor(frames))  # (1, obs_horizon, C,H,W)
        if len(cams) == 1:
            return cams[0]
        return torch.cat(cams, dim=1)  # stack along horizon axis, camera order

    def _build_proprio_window(self) -> "torch.Tensor | None":
        """Construct ``proprioception`` (1, obs_horizon, P) for the model.

        Per-arm blocks are built independently (each with its OWN stats) and
        concatenated along the feature axis in ``self.arm_prefixes`` order,
        mirroring ``UVTADataset.__getitem__``.  ``None`` is returned when the
        proprio_mode contributes nothing (e.g. ``proprio_mode='none'``).
        """
        arm_blocks = []
        for p in self.arm_prefixes:
            blk = self._build_arm_proprio_window(p)
            if blk is not None:
                arm_blocks.append(blk)
        if not arm_blocks:
            return None
        out = (
            arm_blocks[0]
            if len(arm_blocks) == 1
            else np.concatenate(arm_blocks, axis=-1)
        )
        return torch.from_numpy(out).unsqueeze(0).cuda()  # (1, H, P)

    def _build_arm_proprio_window(self, prefix) -> "np.ndarray | None":
        """One arm's proprio block ``(obs_horizon, per_arm_P)`` or ``None``.

        Layout per frame (matches ``UVTADataset._build_arm_proprio``):
            [joint(22, range)? + ee_rel( xyz_3 range + rot6d_6 identity )?
             + fingertip( 5 × (xyz_3 range + rot6d_6 identity) )?]
        keyed by this arm's own stats (``{prefix}proprioception`` etc.).
        """
        def sk(base):
            return self._arm_stat_key(prefix, base)

        parts = []
        if self.proprio_mode in ("joint", "both"):
            hist = self._joint_history[prefix]
            if not hist:
                raise RuntimeError(
                    f"proprio_mode={self.proprio_mode!r} needs joint angles for "
                    f"arm {prefix!r} but none have been pushed."
                )
            idxs = self._obs_window_indices(len(hist))
            joint = np.stack([hist[i] for i in idxs], axis=0)
            joint = normalize_data(joint, self.stats[sk("proprioception")])
            parts.append(joint.astype(np.float32))
        if self.proprio_mode in ("ee_rel", "both", "fingertip_with_ee"):
            hist = self._pose_history[prefix]
            if not hist:
                raise RuntimeError(
                    f"proprio_mode={self.proprio_mode!r} needs the wrist pose "
                    f"for arm {prefix!r} but none have been pushed."
                )
            idxs = self._obs_window_indices(len(hist))
            pose_win = np.stack([hist[i] for i in idxs], axis=0)
            ee_rel9 = self._build_ee_rel(pose_win)  # (H, 9) xyz + rot6d
            ee_xyz = ee_rel9[:, :EE_REL_XYZ_DIM]
            ee_rot6d = ee_rel9[:, EE_REL_XYZ_DIM:]
            # Obs-side ee_rel uses its OWN stat (obs-window range); fall back to
            # the action stat for legacy checkpoints saved before the split.
            _obs_xyz_stats = self.stats.get(
                sk("relative_pose_xyz_obs"), self.stats[sk("relative_pose_xyz")]
            )
            ee_xyz = normalize_data(ee_xyz, _obs_xyz_stats)
            parts.append(
                np.concatenate([ee_xyz, ee_rot6d], axis=-1).astype(np.float32)
            )
        if self.proprio_mode in ("fingertip", "fingertip_with_ee"):
            hist = self._fingertip_history[prefix]
            if not hist:
                raise RuntimeError(
                    f"proprio_mode={self.proprio_mode!r} needs fingertip poses "
                    f"for arm {prefix!r} but none have been pushed."
                )
            idxs = self._obs_window_indices(len(hist))
            ft = np.stack(
                [hist[i] for i in idxs], axis=0
            )  # (H, 5, 6) xyz + rotvec
            H = ft.shape[0]
            ft_xyz = ft[..., :3].astype(np.float32)  # (H, 5, 3)
            ft_mats = np.stack(
                [_pose6_to_mat(p) for p in ft.reshape(-1, 6)], axis=0,
            ).reshape(H, 5, 4, 4)
            ft_rot6d = _mat_to_rot6d(ft_mats[..., :3, :3])  # (H, 5, 6)
            # Range-normalize xyz if the trainer computed those stats.
            ft_stat_key = sk("fingertip_xyz_action")
            if ft_stat_key in self.stats:
                ft_xyz_flat = ft_xyz.reshape(H, FINGERTIP_XYZ_DIM)
                ft_xyz_flat = normalize_data(
                    ft_xyz_flat, self.stats[ft_stat_key]
                )
                ft_xyz = ft_xyz_flat.reshape(H, 5, 3)
            ft_pose9 = np.concatenate([ft_xyz, ft_rot6d], axis=-1)  # (H, 5, 9)
            parts.append(
                ft_pose9.reshape(H, FINGERTIP_DIM).astype(np.float32)
            )
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=-1)

    @staticmethod
    def _build_ee_rel(pose_window: np.ndarray) -> np.ndarray:
        """``(obs_horizon, 9)`` ee_rel: ``[xyz, rot6d]`` of T_t^{-1} @ T_k.

        Last frame is identity (xyz=0, rot6d=[1,0,0, 0,1,0]).  Input
        ``pose_window`` is the zarr storage format (xyz + axis-angle
        rotvec, oldest-first, anchor-last).
        """
        H = pose_window.shape[0]
        Ts = np.stack(
            [_pose6_to_mat(pose_window[k]) for k in range(H)], axis=0,
        )
        T_curr_inv = np.linalg.inv(Ts[-1])
        rel_mats = np.einsum("ij,njk->nik", T_curr_inv, Ts)
        return _mat_to_pose9(rel_mats).astype(np.float32)

    def _build_fsr_window(self) -> "torch.Tensor | None":
        """``(1, obs_horizon, fsr_dim)`` windowed tactile from the rolling
        history, matching ``UVTADataset.__getitem__`` (``nsample[tk][win_idx]``
        over the ALREADY-preprocessed buffer).

        Applies the same train-time preprocessing that ``replay_buffer.py`` did
        at load: optional binarization against ``fsr_binary_cutoff``, then range
        normalization with ``stats[tactile_key]``.  Returns ``None`` when no
        tactile has been pushed.

        The optional baseline subtraction runs FIRST (before binarize and
        normalize), matching the order ``replay_buffer.py`` used at train time.

        Bimanual: each arm's tactile window is normalized with that arm's own
        stats (``{prefix}fsr``) then concatenated along the feature axis in
        ``self.arm_prefixes`` order -> ``(1, obs_horizon, per_arm*num_arms)``.
        """
        parts = []
        for p in self.arm_prefixes:
            hist = self._fsr_history[p]
            if not hist:
                return None
            idxs = self._obs_window_indices(len(hist))
            fsr = np.stack([hist[i] for i in idxs], axis=0)  # (H, per_arm)
            fsr = self._apply_fsr_baseline(p, fsr)
            if self.fsr_binarize and self.fsr_binary_cutoff is not None:
                fsr = np.where(
                    fsr >= self.fsr_binary_cutoff, 1.0, 0.0
                ).astype(np.float32)
            _sk = self._arm_stat_key(p, self.tactile_key)
            fsr = normalize_data(fsr, self.stats[_sk]).astype(np.float32)
            parts.append(fsr)
        out = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=-1)
        return torch.from_numpy(out).unsqueeze(0).cuda()  # (1, H, dim)

    # ------------------------------------------------------------------
    # Inference entry point
    # ------------------------------------------------------------------

    def predict_action(
        self,
        proprioception=None,
        fsr=None,
        visual_obs=None,
        joint_proprio=None,
        wrist_pose6=None,
        fingertip_pose_wrist=None,
        return_tactile=False,
        tactile_normalized=False,
        num_frames=None,
        block=None,
        profile_inference=False,
    ):
        """Run one inference step.

        ``return_tactile`` : when the policy was trained with
        ``predict_future_tactile=True`` (the tactile block is part of the
        action), return ``(action, tactile_pred)`` where ``tactile_pred`` is the
        predicted FUTURE tactile chunk -- the same block that is normally sliced
        off and discarded.  This is what the two-stage rollout feeds into the
        stage-2 MLP.  When the policy has no tactile action block,
        ``tactile_pred`` is ``None``.

        ``tactile_normalized`` : return the tactile block on the NETWORK scale
        ([-1, 1], bounded by ``clip_sample``) instead of un-normalizing it to
        physical units.  Prefer this when handing tactile to another network:
        the un-normalize / re-normalize round trip is lossy, needs both sides to
        share one set of tactile stats, and silently passes through channels
        whose training range collapsed (``_should_normalize_dim`` skips them, so
        a raw prediction lands next to neighbours bounded by 1).

        ``num_frames`` : how many leading frames of the predicted trajectory to
        return.  ``None`` (default) uses ``action_horizon`` -- what the rollout
        executes per replan.  Pass a larger value (up to ``pred_horizon``) when a
        downstream consumer needs more of the future than is executed; the
        two-stage rollout does this because stage 2 measures its residual against
        stage 1's state for EVERY output frame, so it needs one predicted state
        per stage-2 output frame, not per executed frame.

        There are two calling conventions:

        Legacy:  pass ``proprioception``, ``fsr``, ``visual_obs`` as fully
        prepared tensors of shape ``(1, P)`` / ``(1, F)`` / ``(N, H, W, 3)``
        respectively.  This is what ``relative_policy_rollout.py`` currently
        does.  Only works for the legacy joint-style proprio with
        ``obs_horizon == 1`` and ``down_sample_steps == 1``.

        Window mode: pass ``visual_obs=<single frame>`` plus any of
        ``joint_proprio`` / ``wrist_pose6`` / ``fingertip_pose_wrist`` and
        we use the rolling history.  Push observations first via
        ``push_observation`` for full-history mode.

        Returns a ``(action_horizon, action_dim)`` array of *raw*
        (un-normalized) action vectors.  Caller is responsible for
        splitting eef_rel and hand parts.
        """
        inference_profile = {} if profile_inference else None
        profile_total_started = time.perf_counter()
        self.last_inference_profile = None

        def phase_started():
            return time.perf_counter() if inference_profile is not None else None

        def finish_phase(name, started):
            if inference_profile is not None and started is not None:
                inference_profile[name] = (
                    time.perf_counter() - started
                ) * 1000.0

        # --- Visual observation --------------------------------------------
        phase_start = phase_started()
        # visual_obs may be: None (use rolling history); a single (H,W,3) frame
        # or (N,H,W,3) window (single-camera); or a list/dict of those, one per
        # camera in ``self.camera_ids`` order (multi-camera / bimanual).
        _have_hist = any(len(d) for d in self._visual_history.values())
        if visual_obs is None:
            if not _have_hist:
                raise RuntimeError(
                    "predict_action: no visual history; call "
                    "push_observation() first or pass visual_obs."
                )
            visual_t = self._build_visual_window()
        elif self._visual_input_frame_ndim(visual_obs) == 3:
            # Fresh single frame(s): push (with proprio) and use rolling buffer.
            self.push_observation(
                visual_obs=visual_obs,
                joint_proprio=joint_proprio,
                wrist_pose6=wrist_pose6,
                fingertip_pose_wrist=fingertip_pose_wrist,
            )
            visual_t = self._build_visual_window()
        else:
            # Legacy full-window path: (N,H,W,3) per camera, bypass history.
            visual_t = self._legacy_window_tensor(visual_obs)
        finish_phase("visual_prepare_wall_ms", phase_start)

        # --- Proprioception ------------------------------------------------
        phase_start = phase_started()
        _have_proprio_hist = any(
            len(d)
            for hist in (
                self._joint_history,
                self._pose_history,
                self._fingertip_history,
            )
            for d in hist.values()
        )
        if proprioception is not None:
            # Legacy single-arm path: caller already built the proprio vector.
            proprio_t = normalize_data(
                proprioception.reshape(1, -1),
                self.stats[self._arm_stat_key(self.arm_prefixes[0], "proprioception")],
            )
            proprio_t = torch.from_numpy(proprio_t).unsqueeze(0).cuda()
        elif _have_proprio_hist:
            proprio_t = self._build_proprio_window()
        else:
            proprio_t = None
        finish_phase("proprio_prepare_wall_ms", phase_start)

        phase_start = phase_started()
        _have_fsr_hist = any(len(d) for d in self._fsr_history.values())
        if fsr is not None:
            # Explicit tactile passed by the caller.  For bimanual this is the
            # concatenated [arm0 | arm1 | ...] vector; split + normalize per arm.
            fsr = self._normalize_explicit_fsr(fsr)
        elif self.enable_fsr and _have_fsr_hist:
            # Preferred path: build the strided obs window from the rolling
            # history that ``push_observation(fsr=...)`` has been filling, so
            # obs_horizon > 1 is handled correctly (mirrors visual / proprio).
            fsr = self._build_fsr_window()
        finish_phase("tactile_prepare_wall_ms", phase_start)

        # --- Diffusion sample ---------------------------------------------
        phase_start = phase_started()
        trajectory = torch.randn(1, self.pred_horizon, self.action_dim).cuda()
        finish_phase("noise_allocate_wall_ms", phase_start)
        diffusion_profile = {} if inference_profile is not None else None
        diffusion_started = phase_started()
        trajectory = self.model.inference(
            proprioception=proprio_t,
            fsr=fsr,
            visual_obs=visual_t,
            trajectory=trajectory,
            noise_scheduler=self.noise_scheduler,
            num_inference_steps=self.num_inference_steps,
            inference_profile=diffusion_profile,
        )
        if inference_profile is not None:
            inference_profile["diffusion_submit_wall_ms"] = (
                time.perf_counter() - diffusion_started
            ) * 1000.0
        d2h_started = phase_started()
        trajectory = trajectory.detach().to("cpu").numpy()
        finish_phase("diffusion_d2h_sync_wall_ms", d2h_started)
        finish_phase("diffusion_end_to_end_wall_ms", diffusion_started)

        if inference_profile is not None and diffusion_profile is not None:
            def cuda_elapsed_ms(events):
                if not events or events[0] is None or events[1] is None:
                    return 0.0
                return float(events[0].elapsed_time(events[1]))

            step_events = diffusion_profile.pop(
                "_denoise_step_cuda_events", []
            )
            step_ms = [cuda_elapsed_ms(events) for events in step_events]
            inference_profile["diffusion_gpu_ms"] = cuda_elapsed_ms(
                diffusion_profile.pop("_diffusion_total_cuda_events", None)
            )
            inference_profile["vision_gpu_ms"] = cuda_elapsed_ms(
                diffusion_profile.pop("_vision_cuda_events", None)
            )
            inference_profile["denoise_step_gpu_ms"] = step_ms
            inference_profile["denoise_gpu_sum_ms"] = float(sum(step_ms))
            inference_profile["diffusion_gpu_unattributed_ms"] = max(
                inference_profile["diffusion_gpu_ms"]
                - inference_profile["vision_gpu_ms"]
                - inference_profile["denoise_gpu_sum_ms"],
                0.0,
            )

        phase_start = phase_started()
        naction = trajectory[0]
        action_pred = unnormalize_data(naction, stats=self.stats["action"])
        finish_phase("action_unnormalize_wall_ms", phase_start)

        # --- World heads (aux_as_head) -------------------------------------
        # One extra forward pass through the trunk, no diffusion.  Cached so
        # ``predict_state_block`` / the returned tactile can serve them without a
        # second pass.
        self._last_world = {}
        world_started = phase_started()
        if self.aux_as_head:
            world_cuda_start = world_cuda_end = None
            if inference_profile is not None and torch.cuda.is_available():
                world_cuda_start = torch.cuda.Event(enable_timing=True)
                world_cuda_end = torch.cuda.Event(enable_timing=True)
                world_cuda_start.record()
            world_outputs = self.model.predict_world(
                proprioception=proprio_t, fsr=fsr, visual_obs=visual_t
            )
            if world_cuda_end is not None:
                world_cuda_end.record()
            if inference_profile is not None:
                inference_profile["world_submit_wall_ms"] = (
                    time.perf_counter() - world_started
                ) * 1000.0
            self._last_world = {
                k: v.detach().to("cpu").numpy()[0]
                for k, v in world_outputs.items()
            }
            if inference_profile is not None and world_cuda_start is not None:
                inference_profile["world_gpu_ms"] = float(
                    world_cuda_start.elapsed_time(world_cuda_end)
                )
        elif inference_profile is not None:
            inference_profile["world_submit_wall_ms"] = 0.0
            inference_profile["world_gpu_ms"] = 0.0
        finish_phase("world_end_to_end_wall_ms", world_started)

        postprocess_started = phase_started()
        n_keep = self.action_horizon if num_frames is None else int(num_frames)
        if n_keep > self.pred_horizon:
            raise ValueError(
                f"num_frames={n_keep} exceeds pred_horizon={self.pred_horizon}; "
                "the network does not predict that far."
            )

        # Split off the predicted FUTURE-tactile block (auxiliary target).  It is
        # not part of the executed command, but the two-stage rollout wants it
        # (``return_tactile``), so capture it before dropping.  Taken from the
        # NORMALIZED trajectory when the caller asked for the network scale.
        tactile_pred = None
        if self.tactile_action_dim > 0:
            src = naction if tactile_normalized else action_pred
            tactile_pred = src[:n_keep, -self.tactile_action_dim:].astype(
                np.float32
            )
            action_pred = action_pred[:, : -self.tactile_action_dim]
        elif "tactile" in self._last_world:
            # From the world head, which regresses the NORMALIZED target, so the
            # unnormalized branch has to invert it per arm.
            ntac = self._last_world["tactile"][:n_keep]
            tactile_pred = (
                ntac if tactile_normalized else self._unnormalize_tactile(ntac)
            ).astype(np.float32)

        # Keep only the block that is executed.  Everything downstream (the
        # rot6d decode, the rollouts, the IK) expects one ``[eef | joint]`` run
        # per arm, so an auxiliary block must not reach them.  The predicted
        # STATE block is still available via ``predict_state_block``.
        # ``block`` selects WHICH output block to return, defaulting to the one
        # deploy executes.  The two-stage rollout asks for ``"state"``: stage 2 is
        # conditioned on where the arm will BE, not on what was commanded.
        self._last_motor_full = action_pred.copy()
        want_block = self.executed_block if block is None else str(block)
        if want_block == "state" and "state" in self.aux_targets:
            # Head-served block (aux_as_head).  Swap in the head's output here so
            # it goes through the same truncation and rot6d decode as an
            # in-trajectory block, keeping the caller's contract identical:
            # ``block="state"`` returns the decoded legacy layout either way.
            action_pred = self._unnormalize_state_head(self._last_world[want_block])
        elif want_block not in self.motor_blocks:
            raise ValueError(
                f"this policy does not predict the {want_block!r} block "
                f"(it predicts {self.motor_blocks}"
                + (f", heads {self.aux_targets}" if self.aux_targets else "")
                + ").  A policy with predict_action=True is driven by "
                "relative_policy_rollout.py; one without it needs the two-stage "
                "rollout."
            )
        elif len(self.motor_blocks) > 1:
            action_pred = self._extract_block(action_pred, want_block)

        # --- Pick the executable horizon ----------------------------------
        # In the "method B" sample layout, the network predicts a strictly-
        # future trajectory of length ``pred_horizon``.  With the pose_action
        # convention frame 0 = pose[t]^-1 @ pose_action[t] is the real first
        # commanded step (no longer the anchor identity), so the whole
        # ``[:action_horizon]`` slice is executable; the rollout consumes from
        # index ``--anchor_offset`` (0 by default) and re-plans afterwards.
        # ``num_frames`` widens this when a consumer needs more of the future
        # than is executed (see the two-stage rollout).
        action_pred = action_pred[:n_keep, :]

        # --- Decode rot6d -> rotvec for deploy-friendly output -------------
        # Network output:    [eef_xyz(3), eef_rot6d(6), hand( 22 OR 5*(3+6) )]
        # External contract: [eef_xyz(3), eef_rotvec(3), hand( 22 OR 5*(3+3) )]
        # The decode keeps the rollout-side code (cet/relative_policy_rollout.py
        # etc.) unchanged: ``act[:6]`` stays xyz+rotvec, ``act[6:]`` stays
        # joint angles or fingertip xyz+rotvec.
        decoded = self._decode_action_to_legacy(action_pred)
        finish_phase("postprocess_wall_ms", postprocess_started)
        if inference_profile is not None:
            inference_profile["total_wall_ms"] = (
                time.perf_counter() - profile_total_started
            ) * 1000.0
            primary_keys = (
                "visual_prepare_wall_ms",
                "proprio_prepare_wall_ms",
                "tactile_prepare_wall_ms",
                "noise_allocate_wall_ms",
                "diffusion_end_to_end_wall_ms",
                "action_unnormalize_wall_ms",
                "world_end_to_end_wall_ms",
                "postprocess_wall_ms",
            )
            inference_profile["wall_unattributed_ms"] = max(
                inference_profile["total_wall_ms"]
                - sum(float(inference_profile.get(k, 0.0)) for k in primary_keys),
                0.0,
            )
            self.last_inference_profile = inference_profile
        if return_tactile:
            return decoded, tactile_pred
        return decoded

    def _extract_block(self, motor: np.ndarray, block: str) -> np.ndarray:
        """Pull one output block out of a per-arm-concatenated motor tensor.

        ``motor`` is ``(T, num_arms * len(motor_blocks) * per_arm_block_dim)``
        laid out ``[arm0: action | state, arm1: action | state, ...]``.  The
        result keeps the arm-major layout with a single block per arm, i.e.
        exactly what a policy with only that block would have produced.
        """
        if block not in self.motor_blocks:
            raise ValueError(
                f"block {block!r} was not predicted (have {self.motor_blocks})"
            )
        i = self.motor_blocks.index(block)
        nb, per = len(self.motor_blocks), self.per_arm_block_dim
        expect = self.num_arms * nb * per
        if motor.shape[-1] != expect:
            raise ValueError(
                f"motor width {motor.shape[-1]} != num_arms({self.num_arms}) * "
                f"blocks({nb}) * per_arm_block_dim({per}) = {expect}"
            )
        return np.concatenate(
            [
                motor[:, (a * nb + i) * per: (a * nb + i + 1) * per]
                for a in range(self.num_arms)
            ],
            axis=-1,
        )

    def predict_state_block(self) -> np.ndarray | None:
        """The predicted next-STATE block from the last ``predict_action`` call.

        ``None`` when the policy does not predict it.  Returned in the same
        arm-major ``[eef(9) | joint(22)]`` per-arm layout as the executed block,
        un-normalized -- this is what a stage-2 model is conditioned on.
        """
        if "state" in getattr(self, "aux_targets", []):
            # aux_as_head: the head regresses the normalized target, so invert
            # with the same ``action`` stats the in-vector state block used --
            # per arm, since those stats describe one arm's block.
            world = getattr(self, "_last_world", None)
            if not world or "state" not in world:
                raise RuntimeError(
                    "predict_state_block(): call predict_action() first."
                )
            return self._unnormalize_state_head(world["state"])
        if "state" not in self.motor_blocks:
            return None
        motor = getattr(self, "_last_motor_full", None)
        if motor is None:
            raise RuntimeError(
                "predict_state_block(): call predict_action() first."
            )
        if len(self.motor_blocks) == 1:
            return motor.copy()
        return self._extract_block(motor, "state")

    def _unnormalize_state_head(self, nstate: np.ndarray) -> np.ndarray:
        """Invert normalization for the state world head's output.

        Uses ``stats['aux_state']``, which differs from ``stats['action']`` in
        the wrist-xyz range (``relative_state_xyz``), so a straight per-arm slice
        of the action stats would decode the translation on the wrong scale.
        """
        return unnormalize_data(nstate, stats=self.stats["aux_state"])

    def _unnormalize_tactile(self, ntac: np.ndarray) -> np.ndarray:
        """Invert normalization for the tactile world head, per arm."""
        per = self.per_arm_tactile_dim
        return np.concatenate(
            [
                unnormalize_data(
                    ntac[:, a * per: (a + 1) * per],
                    stats=self.stats[
                        self._arm_stat_key(self.arm_prefixes[a], self.tactile_key)
                    ],
                )
                for a in range(self.num_arms)
            ],
            axis=-1,
        )

    def _decode_action_to_legacy(self, action: np.ndarray) -> np.ndarray:
        """Decode the network's rot6d action to the legacy xyz+rotvec layout.

        ``action`` shape: ``(T, motor_total)`` (the future-tactile block, if
        any, has already been stripped).  For bimanual the motor part is the
        per-arm blocks concatenated ([arm0 | arm1 | ...], ``per_arm_motor_dim``
        each); every arm is decoded independently and the per-arm legacy blocks
        are concatenated back-to-back.  So for a joint-hand bimanual policy the
        output is ``[L_xyz(3) L_rotvec(3) L_joint(22) | R_xyz(3) R_rotvec(3)
        R_joint(22)]`` and the caller slices per arm (``self.per_arm_motor_dim``
        network dims -> 28 legacy dims per arm here).
        """
        if self.num_arms == 1:
            return self._decode_arm_block(action)
        # One block per arm by this point (``predict_action`` already dropped the
        # auxiliary block), so slice on the BLOCK width, not the full motor one.
        per = self.per_arm_block_dim
        outs = [
            self._decode_arm_block(action[:, a * per: (a + 1) * per])
            for a in range(self.num_arms)
        ]
        return np.concatenate(outs, axis=-1)

    def _decode_arm_block(self, action_9d: np.ndarray) -> np.ndarray:
        """Decode ONE arm's motor block (rot6d -> legacy xyz+rotvec).

        ``action_9d`` shape: ``(T, per_arm_motor_dim)``.  Returns
        ``(T, per_arm_legacy_dim)``.
        """
        from scipy.spatial.transform import Rotation as R

        from scipy.spatial.transform import Rotation as _R  # noqa: F811

        T = action_9d.shape[0]
        if self.hand_action_mode == "joint_only":
            # No eef block: the network output IS the 22-D joint vector.
            # External contract for joint hand is raw joint angles, so return
            # the action unchanged (caller holds the wrist pose itself).
            return action_9d.astype(np.float32)
        if self.hand_action_mode == "fingertip_only":
            # No eef block: the network output IS the 45-D fingertip vector
            # (T, 5, 9) = xyz(3) + rot6d(6).  Decode to legacy (T, 30) =
            # (T, 5, 6) xyz + rotvec; the caller holds the wrist pose itself.
            hand_part = action_9d.reshape(T, 5, 9)
            ft_xyz = hand_part[..., :3]
            ft_rot6d = hand_part[..., 3:9]
            ft_rotmat = _rot6d_to_mat(ft_rot6d.reshape(-1, 6)).reshape(T, 5, 3, 3)
            ft_rotvec = (
                _R.from_matrix(ft_rotmat.reshape(-1, 3, 3))
                .as_rotvec()
                .reshape(T, 5, 3)
                .astype(np.float32)
            )
            return np.concatenate([ft_xyz, ft_rotvec], axis=-1).reshape(T, 30)
        eef_xyz = action_9d[:, :EE_REL_XYZ_DIM]
        eef_rot6d = action_9d[:, EE_REL_XYZ_DIM:EE_REL_DIM]
        eef_rotmat = _rot6d_to_mat(eef_rot6d)
        eef_rotvec = R.from_matrix(eef_rotmat).as_rotvec().astype(np.float32)
        eef_legacy = np.concatenate([eef_xyz, eef_rotvec], axis=-1)  # (T, 6)

        hand_part = action_9d[:, EE_REL_DIM:]
        if self.hand_action_mode == "none":
            # EEF-only policy: nothing past the 9-D wrist pose.
            return eef_legacy
        if self.hand_action_mode == "fingertip":
            # network: (T, 45) = (T, 5, 9)  ;  legacy: (T, 30) = (T, 5, 6)
            hand_part = hand_part.reshape(T, 5, 9)
            ft_xyz = hand_part[..., :3]
            ft_rot6d = hand_part[..., 3:9]
            ft_rotmat = _rot6d_to_mat(ft_rot6d.reshape(-1, 6)).reshape(T, 5, 3, 3)
            ft_rotvec = (
                R.from_matrix(ft_rotmat.reshape(-1, 3, 3))
                .as_rotvec()
                .reshape(T, 5, 3)
                .astype(np.float32)
            )
            hand_legacy = np.concatenate([ft_xyz, ft_rotvec], axis=-1).reshape(T, 30)
            return np.concatenate([eef_legacy, hand_legacy], axis=-1)
        # joint mode: hand_part is already raw joint angles (22-D).
        return np.concatenate([eef_legacy, hand_part], axis=-1)
