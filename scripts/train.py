import copy
import datetime
import math
import os
import sys
import time
import warnings
from collections import defaultdict

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_P2P_DISABLE"] = "1"
import pickle

import hydra
import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator
from uvta.common.utility.model import load_config
from uvta.diffusion_policy.dataloader.diffusion_bc_dataset import (
    normalize_data,
    unnormalize_data,
)
from diffusers.optimization import get_scheduler
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm.auto import tqdm

# dist.init_process_group(
#     backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=5400)
# )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

# Image transforms allowed in val.  RandomCrop / ColorJitter / GaussianBlur /
# RandomGrayscale are random and would inject noise into the val loss; we map
# them to a deterministic chain (Resize+CenterCrop, both 224x224 at the end).
_VAL_DETERMINISTIC_TRANSFORMS = ["Resize", "CenterCrop"]

# Action-tensor layout produced by ``UVTADataset.__getitem__``:
#   nsample["action"] = concat([relative_vec6dof (B, T, 6), hand_action (B, T, K)], dim=-1)
# Per-component MSE breakdown of the diffusion noise-prediction loss.
#
# Action layout (network-facing, all rotations are rot6d since the 2026-05
# flip):
#
#     [ eef_xyz(3)  eef_rot6d(6) | hand( joint OR fingertip ) ]
#       \------- _EEF_DIM = 9 ----/
#
# Hand part is one of:
#   * ``hand_action_mode='joint'``    : 22-D joint angles.
#   * ``hand_action_mode='fingertip'``: 5 × (xyz_3 + rot6d_6) = 45 D
#                                       (one block per finger).
#
# We split the per-batch noise / noise-prediction tensor along these
# boundaries to produce a fine-grained set of MSE numbers as bookkeeping
# alongside the overall MSE that drives backprop.  None of these affect
# optimization — they are wandb metrics only.
_EEF_DIM = 9        # xyz(3) + rot6d(6) for the wrist
_EEF_XYZ_DIM = 3
_EEF_ROT6D_DIM = 6
_FT_PER_FINGER = 9  # xyz(3) + rot6d(6) per fingertip
_FT_PER_XYZ = 3
_FT_PER_ROT6D = 6
_NUM_FINGERS = 5


def _nanmean(values) -> float:
    """``np.nanmean`` that returns NaN (silently) for all-NaN / empty input.

    In eef-only mode (``hand_action_mode='none'``) the ``hand*`` buckets are
    always NaN, so a plain ``np.nanmean`` emits a noisy
    ``RuntimeWarning: Mean of empty slice`` every epoch.  This wrapper
    suppresses that warning while preserving the NaN result.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(values))


def _mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(a, b)


def _reshape_arms(
    pred: torch.Tensor, tgt: torch.Tensor, num_arms: int = 1
):
    """Reshape the trailing MOTOR dim into ``(..., num_arms, per_arm)``.

    Bimanual actions are laid out ``[arm0(eef|hand), arm1(eef|hand), ...]`` (the
    tactile block, when present, must already be stripped off the tail).
    Reshaping the motor dim to ``(num_arms, per_arm)`` puts every arm's channels
    on the SAME trailing per-arm axis, so the existing eef/hand/xyz/rot6d slices
    (all on the last axis) and their ``mse_loss`` (a mean over ALL elements)
    automatically AGGREGATE the metric across arms.  ``num_arms<=1`` is a no-op,
    so the single-arm path is numerically unchanged.
    """
    if num_arms and num_arms > 1:
        lead = pred.shape[:-1]
        d = pred.shape[-1]
        if d % num_arms != 0:
            raise ValueError(
                f"motor action dim {d} not divisible by num_arms={num_arms}"
            )
        per = d // num_arms
        pred = pred.reshape(*lead, num_arms, per)
        tgt = tgt.reshape(*lead, num_arms, per)
    return pred, tgt


def _gather_visual_obs(batch, camera_ids, device):
    """Stack the requested camera streams along the obs-horizon axis.

    Each ``camera_{id}`` entry is ``(B, obs_horizon, C, H, W)``.  For a bimanual
    setup we feed BOTH wrist cameras (chips_teleop: camera_0 = right wrist,
    camera_2 = left wrist; camera_1 = ego, unused) through the SAME vision
    backbone by concatenating them along the horizon axis ->
    ``(B, num_cameras*obs_horizon, C, H, W)``; the model then flattens every
    (camera, frame) embedding into the global condition.  A single camera
    returns its tensor unchanged, so the single-arm path is untouched.
    """
    obs = []
    for cid in camera_ids:
        key = f"camera_{cid}"
        if key not in batch:
            raise KeyError(
                f"{key} not found in batch (load_camera_ids={list(camera_ids)}); "
                "check dataset.load_camera_ids."
            )
        obs.append(batch[key].to(device))
    if len(obs) == 1:
        return obs[0]
    return torch.cat(obs, dim=1)


def _reconstruct_x0(
    noisy_actions: torch.Tensor,
    noise_pred: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler,
) -> torch.Tensor:
    """Reconstruct the clean action ``x0`` from an epsilon-prediction.

    For the ``prediction_type='epsilon'`` parameterization the forward process
    is ``x_t = sqrt(a_bar_t) * x0 + sqrt(1 - a_bar_t) * eps`` so the model's
    noise prediction ``eps_theta`` inverts to::

        x0_hat = (x_t - sqrt(1 - a_bar_t) * eps_theta) / sqrt(a_bar_t)

    ``a_bar_t`` is the scheduler's ``alphas_cumprod`` gathered per-sample by
    ``timesteps`` and broadcast over the trailing ``(T, action_dim)`` dims.

    This is used ONLY to compute the (eval-only) action-space reconstruction
    MSE.  We deliberately do NOT clip to ``[-1, 1]`` (even though the scheduler
    has ``clip_sample=True`` at inference): an unclipped reconstruction is a
    more faithful measure of raw model error for monitoring.
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(
        device=noisy_actions.device, dtype=noisy_actions.dtype
    )
    alpha_bar = alphas_cumprod[timesteps]
    while alpha_bar.dim() < noisy_actions.dim():
        alpha_bar = alpha_bar.unsqueeze(-1)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
    return (noisy_actions - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar


def _recon_loss_parts(
    noisy_actions: torch.Tensor,
    noise_pred: torch.Tensor,
    timesteps: torch.Tensor,
    actions: torch.Tensor,
    noise_scheduler,
    hand_action_mode: str,
    tactile_action_dim: int = 0,
    num_arms: int = 1,
) -> tuple[dict, dict]:
    """Compute the (eval-only) action-space reconstruction MSE breakdowns.

    Returns ``(recon, recon_clip)`` -- both dicts keyed by ``_LOSS_KEYS``:

    - ``recon``      : MSE between the unclipped ``x0_hat`` and the true
                       action.  This is the faithful raw-model-error metric.
    - ``recon_clip`` : same, but ``x0_hat`` is first clamped to
                       ``[-r, +r]`` with ``r = clip_sample_range`` (default
                       1.0).  This mirrors what the DDIM sampler actually does
                       at inference when ``clip_sample=True`` (the action
                       normalization maps everything into ``[-1, 1]``), so it
                       is the more deploy-representative reconstruction error.
    """
    x0_hat = _reconstruct_x0(noisy_actions, noise_pred, timesteps, noise_scheduler)
    recon = _split_action_mse(
        x0_hat, actions, hand_action_mode, tactile_action_dim, num_arms
    )
    clip_range = float(getattr(noise_scheduler.config, "clip_sample_range", 1.0))
    x0_hat_clip = x0_hat.clamp(-clip_range, clip_range)
    recon_clip = _split_action_mse(
        x0_hat_clip, actions, hand_action_mode, tactile_action_dim, num_arms
    )
    return recon, recon_clip


def _split_action_mse(
    noise_pred: torch.Tensor,
    noise: torch.Tensor,
    hand_action_mode: str = "joint",
    tactile_action_dim: int = 0,
    num_arms: int = 1,
) -> dict:
    """Compute a per-component MSE breakdown of the noise prediction loss.

    ``tactile_action_dim`` : when > 0 the action carries a trailing
    ``tactile_action_dim``-wide FUTURE tactile block (predict_future_tactile).
    Its MSE is reported under ``tactile`` and stripped off before the eef / hand
    split so those numbers stay clean.  ``total`` still covers the WHOLE action
    (eef + hand + tactile), matching the back-prop loss.

    ``num_arms`` : for a bimanual action ``[arm0(eef|hand), arm1(eef|hand), ...]
    | tactile]`` the (tactile-stripped) motor part is reshaped to
    ``(..., num_arms, per_arm)`` so every eef / hand / xyz / rot6d number is the
    aggregate ACROSS arms.  ``num_arms=1`` leaves the single-arm math unchanged.

    Returned dict (every value is a Python float):

    - ``total``       : MSE over the full action tensor (backprop target).
    - ``tactile``     : MSE over the trailing tactile block (NaN if none).
    - ``eef``         : MSE over the 9-D EEF block.
    - ``eef_xyz``     : MSE over EEF xyz (3 dims, aggregate).
    - ``eef_x``       : MSE over the EEF x translation dim.
    - ``eef_y``       : MSE over the EEF y translation dim.
    - ``eef_z``       : MSE over the EEF z translation dim.
    - ``eef_rot6d``   : MSE over EEF rot6d (6 dims).
    - ``hand``        : MSE over the hand block (22 D joint OR 45 D fingertip).
                        Backwards-compatible alias also returned via the
                        legacy ``joint`` key so existing wandb panels keep
                        showing the same number.
    - ``joint``       : alias of ``hand`` (legacy key name).
    - ``hand_joint``  : MSE over 22-D joint angles, only when
                        ``hand_action_mode == 'joint'``; NaN otherwise.
    - ``hand_xyz``    : MSE over the 5 × 3 = 15 xyz dims (only fingertip mode).
    - ``hand_rot6d``  : MSE over the 5 × 6 = 30 rot6d dims (only fingertip mode).

    ``noise_pred`` / ``noise`` have last-dim ``action_dim``; leading shape
    ``(B, T, ...)`` is reduced by ``mse_loss`` over all elements.

    The fingertip xyz / rot6d losses are reported as **mean over all
    fingers** (not per-finger), matching the joint-mode hand_joint
    semantics so they live on the same axis in plots.
    """
    nan = torch.tensor(float("nan"))
    out = {}

    # ``total`` covers the WHOLE action (incl. any tactile block) to match the
    # back-prop MSE.  We then strip the trailing tactile block (if present) so
    # the eef / hand split below only sees the wrist + hand action channels.
    out["total"] = float(_mse(noise_pred, noise).item())
    if tactile_action_dim and tactile_action_dim > 0:
        out["tactile"] = float(
            _mse(
                noise_pred[..., -tactile_action_dim:],
                noise[..., -tactile_action_dim:],
            ).item()
        )
        noise_pred = noise_pred[..., :-tactile_action_dim]
        noise = noise[..., :-tactile_action_dim]
    else:
        out["tactile"] = float("nan")
    # Bimanual: reshape motor to (..., num_arms, per_arm) so the eef / hand /
    # xyz / rot6d slices below (all on the last axis) aggregate across arms.
    noise_pred, noise = _reshape_arms(noise_pred, noise, num_arms)
    # MSE over just the wrist + hand action channels (tactile already stripped).
    core_mse = float(_mse(noise_pred, noise).item())

    if hand_action_mode == "joint_only":
        # Joint-only run: the WHOLE (non-tactile) action is the 22-D joint
        # block; there is no eef component, so all eef-side keys are NaN and the
        # hand keys cover the (stripped) joint tensor.
        out["eef"] = float("nan")
        out["eef_xyz"] = float("nan")
        out["eef_x"] = float("nan")
        out["eef_y"] = float("nan")
        out["eef_z"] = float("nan")
        out["eef_rot6d"] = float("nan")
        out["hand"] = core_mse
        out["joint"] = out["hand"]
        out["hand_joint"] = out["hand"]
        out["hand_xyz"] = float("nan")
        out["hand_rot6d"] = float("nan")
        return out

    if hand_action_mode == "fingertip_only":
        # Fingertip-only run: the WHOLE (non-tactile) action is the 45-D
        # fingertip block (5 fingers x 9-D = xyz(3) + rot6d(6)); there is NO eef
        # component.  All eef-side keys are NaN; the hand keys cover the stripped
        # tensor and are split into xyz vs rot6d like the regular fingertip
        # branch.
        out["eef"] = float("nan")
        out["eef_xyz"] = float("nan")
        out["eef_x"] = float("nan")
        out["eef_y"] = float("nan")
        out["eef_z"] = float("nan")
        out["eef_rot6d"] = float("nan")
        out["hand"] = core_mse
        out["joint"] = out["hand"]
        out["hand_joint"] = float("nan")
        hand_p = noise_pred
        hand_t = noise
        xyz_pred = torch.stack(
            [hand_p[..., f * _FT_PER_FINGER: f * _FT_PER_FINGER + _FT_PER_XYZ]
             for f in range(_NUM_FINGERS)],
            dim=-2,
        )
        xyz_true = torch.stack(
            [hand_t[..., f * _FT_PER_FINGER: f * _FT_PER_FINGER + _FT_PER_XYZ]
             for f in range(_NUM_FINGERS)],
            dim=-2,
        )
        rot_pred = torch.stack(
            [hand_p[..., f * _FT_PER_FINGER + _FT_PER_XYZ: (f + 1) * _FT_PER_FINGER]
             for f in range(_NUM_FINGERS)],
            dim=-2,
        )
        rot_true = torch.stack(
            [hand_t[..., f * _FT_PER_FINGER + _FT_PER_XYZ: (f + 1) * _FT_PER_FINGER]
             for f in range(_NUM_FINGERS)],
            dim=-2,
        )
        out["hand_xyz"] = float(_mse(xyz_pred, xyz_true).item())
        out["hand_rot6d"] = float(_mse(rot_pred, rot_true).item())
        return out

    if noise_pred.shape[-1] <= _EEF_DIM:
        # EEF-only run (hand_action_mode='none'): the (non-tactile) action is
        # exactly the 9-D wrist pose, so ``eef`` == ``core_mse`` but we still
        # split xyz vs rot6d so those curves are meaningful.  All hand-side keys
        # are NaN.
        out["eef"] = core_mse
        out["eef_xyz"] = float(
            _mse(
                noise_pred[..., :_EEF_XYZ_DIM], noise[..., :_EEF_XYZ_DIM]
            ).item()
        )
        for _axis_i, _axis_name in enumerate(("eef_x", "eef_y", "eef_z")):
            out[_axis_name] = float(
                _mse(
                    noise_pred[..., _axis_i:_axis_i + 1],
                    noise[..., _axis_i:_axis_i + 1],
                ).item()
            )
        out["eef_rot6d"] = float(
            _mse(
                noise_pred[..., _EEF_XYZ_DIM:_EEF_XYZ_DIM + _EEF_ROT6D_DIM],
                noise[..., _EEF_XYZ_DIM:_EEF_XYZ_DIM + _EEF_ROT6D_DIM],
            ).item()
        )
        out["hand"] = float("nan")
        out["joint"] = out["hand"]
        out["hand_joint"] = float("nan")
        out["hand_xyz"] = float("nan")
        out["hand_rot6d"] = float("nan")
        return out

    eef_p = noise_pred[..., :_EEF_DIM]
    eef_t = noise[..., :_EEF_DIM]
    hand_p = noise_pred[..., _EEF_DIM:]
    hand_t = noise[..., _EEF_DIM:]

    out["eef"] = float(_mse(eef_p, eef_t).item())
    out["eef_xyz"] = float(
        _mse(eef_p[..., :_EEF_XYZ_DIM], eef_t[..., :_EEF_XYZ_DIM]).item()
    )
    for _axis_i, _axis_name in enumerate(("eef_x", "eef_y", "eef_z")):
        out[_axis_name] = float(
            _mse(eef_p[..., _axis_i:_axis_i + 1], eef_t[..., _axis_i:_axis_i + 1]).item()
        )
    out["eef_rot6d"] = float(
        _mse(
            eef_p[..., _EEF_XYZ_DIM:_EEF_XYZ_DIM + _EEF_ROT6D_DIM],
            eef_t[..., _EEF_XYZ_DIM:_EEF_XYZ_DIM + _EEF_ROT6D_DIM],
        ).item()
    )

    out["hand"] = float(_mse(hand_p, hand_t).item())
    # Legacy alias — keep so old wandb panels named "joint" still work.
    out["joint"] = out["hand"]

    if hand_action_mode == "fingertip":
        # hand_p shape: (..., 45) = (..., 5, 9); break into xyz vs rot6d
        # while staying agnostic to leading dims (don't reshape, just slice
        # finger-by-finger so any leading shape stays preserved).
        # Stack per-finger slices along a NEW axis then reduce.
        xyz_pred = torch.stack(
            [hand_p[..., f * _FT_PER_FINGER : f * _FT_PER_FINGER + _FT_PER_XYZ]
             for f in range(_NUM_FINGERS)],
            dim=-2,
        )
        xyz_true = torch.stack(
            [hand_t[..., f * _FT_PER_FINGER : f * _FT_PER_FINGER + _FT_PER_XYZ]
             for f in range(_NUM_FINGERS)],
            dim=-2,
        )
        rot_pred = torch.stack(
            [hand_p[..., f * _FT_PER_FINGER + _FT_PER_XYZ
                       : (f + 1) * _FT_PER_FINGER]
             for f in range(_NUM_FINGERS)],
            dim=-2,
        )
        rot_true = torch.stack(
            [hand_t[..., f * _FT_PER_FINGER + _FT_PER_XYZ
                       : (f + 1) * _FT_PER_FINGER]
             for f in range(_NUM_FINGERS)],
            dim=-2,
        )
        out["hand_xyz"] = float(_mse(xyz_pred, xyz_true).item())
        out["hand_rot6d"] = float(_mse(rot_pred, rot_true).item())
        out["hand_joint"] = float("nan")
    elif hand_action_mode == "joint":
        out["hand_joint"] = out["hand"]
        out["hand_xyz"] = float("nan")
        out["hand_rot6d"] = float("nan")
    else:
        raise ValueError(
            f"unknown hand_action_mode={hand_action_mode!r}; expected "
            "'joint' or 'fingertip'"
        )
    return out


def _aux_targets(batch, device):
    """Collect the ``aux_<name>`` keys the dataset emits under aux_as_head.

    Returns ``{}`` for a dataset that keeps its auxiliary targets inside the
    diffusion vector, which is what makes the model's world-head branch inert
    for every pre-existing config.
    """
    return {
        k[len("aux_"):]: v.to(device)
        for k, v in batch.items()
        if k.startswith("aux_")
    }


def _split_motor_blocks(pred, tgt, motor_blocks, num_arms):
    """Slice the (tactile-stripped) motor part into per-block tensor pairs.

    The motor part is laid out per arm as the enabled blocks back to back:
    ``[arm0: action | state, arm1: action | state, ...]``.  ``_reshape_arms``
    puts every arm on a common trailing axis first, so one slice per block
    aggregates that block across arms -- both arms then share one weight per
    block, exactly as they already share one eef and one hand weight.

    Returns ``{block_name: (pred_slice, tgt_slice)}`` preserving block order.
    """
    blocks = list(motor_blocks) or ["action"]
    pred, tgt = _reshape_arms(pred, tgt, num_arms)
    d = pred.shape[-1]
    if d % len(blocks) != 0:
        raise ValueError(
            f"per-arm motor dim {d} not divisible by {len(blocks)} output "
            f"block(s) {blocks}"
        )
    per = d // len(blocks)
    return {
        b: (pred[..., i * per:(i + 1) * per], tgt[..., i * per:(i + 1) * per])
        for i, b in enumerate(blocks)
    }


def _weighted_action_loss(
    noise_pred: torch.Tensor,
    noise: torch.Tensor,
    hand_action_mode: str = "joint",
    tactile_action_dim: int = 0,
    weights: dict | None = None,
    num_arms: int = 1,
    motor_blocks=("action",),
) -> torch.Tensor:
    """Group-balanced diffusion MSE used for BACK-PROP (differentiable).

    The default ``mse_loss(noise_pred, noise)`` averages over EVERY action dim,
    so a group's pull on the gradient is proportional to its width.  With the
    predict_future_tactile layout ``[eef(9) | hand(22) | tactile(F)]`` and the
    100-D fsr stream the tactile block is 100/131 of the dims and dominates the
    loss.

    This helper instead computes a SEPARATE mean-squared-error per PRESENT
    group (``eef`` / ``hand`` / ``tactile``) -- each already a per-dim mean, so
    dimension-count independent -- then returns::

        sum_g (w_g / sum_g' w_g') * mse_g

    with ``w_g`` looked up from ``weights`` (default 1.0 per group).  Equal
    weights (1:1:1) therefore make every group contribute exactly ``1/3`` of
    the loss regardless of its width.  Group boundaries match
    ``_split_action_mse``; ``hand`` is the 22-D joint block in ``joint`` mode
    (``joint`` / ``fingertips`` are accepted as weight-key aliases for it).

    ``motor_blocks`` : the enabled output blocks, in target order -- e.g.
    ``("action",)`` (legacy), ``("state",)``, or ``("action", "state")``.  Each
    block is split into its own ``eef`` / ``hand`` sub-groups, so the group names
    become ``action_eef`` / ``action_hand`` / ``state_eef`` / ``state_hand``
    alongside ``tactile``.

    DEFAULT weights are chosen so that every top-level BLOCK carries the same
    share.  With one motor block the sub-groups default to 1.0 each, reproducing
    the historical ``eef : hand : tactile = 1 : 1 : 1``; with two motor blocks
    they default to 0.5 each, so ``action : state : tactile = 1/3 : 1/3 : 1/3``
    with the eef / hand halves splitting their block evenly.  Explicit weights
    always win, and the legacy key names (``eef`` / ``hand`` / ``joint`` /
    ``fingertips``) still resolve for the corresponding sub-group of EVERY
    block, so an existing config keeps behaving as it did.

    Returns a scalar tensor that keeps ``noise_pred``'s grad history, so it can
    be passed straight to ``accelerator.backward``.
    """
    weights = weights or {}
    blocks = list(motor_blocks) or ["action"]
    # One motor block -> keep the historical flat 1.0 default so existing
    # configs are numerically untouched.  Two -> halve the motor sub-groups so
    # each block still sums to the same share as the single tactile group.
    _motor_default = 1.0 if len(blocks) < 2 else 0.5

    def _gw(group: str) -> float:
        if group in weights:
            return float(weights[group])
        # ``action_hand`` -> try ``hand``, ``joint``, ``fingertips``; likewise
        # ``action_eef`` -> ``eef``.  Lets a legacy weight dict apply unchanged.
        base = group.split("_", 1)[1] if "_" in group else group
        aliases = (
            ("hand", "joint", "fingertips") if base == "hand" else (base,)
        )
        for a in aliases:
            if a in weights:
                return float(weights[a])
        return 1.0 if group == "tactile" else _motor_default

    pred = noise_pred
    tgt = noise
    groups: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    # Trailing FUTURE-tactile block (predict_future_tactile); strip it off so
    # the eef / hand split below only sees the wrist + hand action channels.
    if tactile_action_dim and tactile_action_dim > 0:
        groups["tactile"] = (
            pred[..., -tactile_action_dim:],
            tgt[..., -tactile_action_dim:],
        )
        pred = pred[..., :-tactile_action_dim]
        tgt = tgt[..., :-tactile_action_dim]

    # Split per output block (which also reshapes bimanual motor dims so each
    # group aggregates across arms), then per eef / hand inside each block.
    for name, (bp, bt) in _split_motor_blocks(
        pred, tgt, blocks, num_arms
    ).items():
        if hand_action_mode in ("joint_only", "fingertip_only"):
            # No eef part: the whole block is the hand part.
            groups[f"{name}_hand"] = (bp, bt)
        elif hand_action_mode == "none":
            # EEF-only action: no hand part.
            groups[f"{name}_eef"] = (bp, bt)
        else:
            groups[f"{name}_eef"] = (bp[..., :_EEF_DIM], bt[..., :_EEF_DIM])
            if bp.shape[-1] > _EEF_DIM:
                groups[f"{name}_hand"] = (bp[..., _EEF_DIM:], bt[..., _EEF_DIM:])

    w = {g: _gw(g) for g in groups}
    wsum = sum(w.values())
    if wsum <= 0:
        wsum = 1.0

    loss = None
    for g, (p, t) in groups.items():
        term = (w[g] / wsum) * _mse(p, t)
        loss = term if loss is None else loss + term
    if loss is None:
        # No groups resolved (should not happen); fall back to full MSE.
        loss = _mse(noise_pred, noise)
    return loss


def _detailed_hand_mse(
    noise_pred: torch.Tensor,
    noise: torch.Tensor,
    hand_action_mode: str = "joint",
    tactile_action_dim: int = 0,
    num_arms: int = 1,
) -> dict:
    """Fine-grained per-element MSE breakdown of the *hand* block.

    ``tactile_action_dim`` : trailing FUTURE-tactile block width to strip off
    the action before the hand split (predict_future_tactile), so the per-joint
    / per-finger curves are not contaminated by tactile channels.

    This is a bookkeeping-only metric (never backprop'd), complementing the
    coarse ``hand`` / ``hand_joint`` / ``hand_xyz`` / ``hand_rot6d`` numbers in
    ``_split_action_mse`` with a much finer split:

    * ``hand_action_mode == 'joint'``     : one curve per joint angle,
      ``joint_00 .. joint_{N-1}`` where ``N`` is the hand dim (normally 22).
    * ``hand_action_mode == 'fingertip'`` : per finger ``f`` (0..4) three
      curves -- ``finger{f}`` (full 9-D), ``finger{f}_xyz`` (3-D xyz),
      ``finger{f}_rot6d`` (6-D rot6d).
    * ``hand_action_mode == 'none'``      : empty dict (eef-only action).

    Keys are derived from the tensor's hand dim, so they stay correct if the
    joint count ever changes.  ``noise_pred`` / ``noise`` carry the full action
    (eef block + hand block); we slice off the leading ``_EEF_DIM`` eef dims.
    """
    out: dict = {}
    if hand_action_mode == "none":
        return out

    # Strip the trailing future-tactile block (if any) so it is not folded into
    # the per-joint / per-finger hand breakdown.
    if tactile_action_dim and tactile_action_dim > 0:
        noise_pred = noise_pred[..., :-tactile_action_dim]
        noise = noise[..., :-tactile_action_dim]

    # Bimanual: reshape motor to (..., num_arms, per_arm) so the per-joint /
    # per-finger curves aggregate across arms.  num_arms=1 is a no-op.
    noise_pred, noise = _reshape_arms(noise_pred, noise, num_arms)

    if hand_action_mode in ("joint_only", "fingertip_only"):
        # No eef block: the full (non-tactile) action tensor is the hand block.
        hand_p = noise_pred
        hand_t = noise
    else:
        if noise_pred.shape[-1] <= _EEF_DIM:
            return out
        hand_p = noise_pred[..., _EEF_DIM:]
        hand_t = noise[..., _EEF_DIM:]

    if hand_action_mode in ("joint", "joint_only"):
        n_joints = hand_p.shape[-1]
        for i in range(n_joints):
            out[f"joint_{i:02d}"] = float(
                _mse(hand_p[..., i:i + 1], hand_t[..., i:i + 1]).item()
            )
    elif hand_action_mode in ("fingertip", "fingertip_only"):
        for f in range(_NUM_FINGERS):
            base = f * _FT_PER_FINGER
            fp = hand_p[..., base:base + _FT_PER_FINGER]
            ft = hand_t[..., base:base + _FT_PER_FINGER]
            out[f"finger{f}"] = float(_mse(fp, ft).item())
            out[f"finger{f}_xyz"] = float(
                _mse(fp[..., :_FT_PER_XYZ], ft[..., :_FT_PER_XYZ]).item()
            )
            out[f"finger{f}_rot6d"] = float(
                _mse(fp[..., _FT_PER_XYZ:], ft[..., _FT_PER_XYZ:]).item()
            )
    else:
        raise ValueError(
            f"unknown hand_action_mode={hand_action_mode!r}; expected "
            "'joint', 'fingertip' or 'none'"
        )
    return out


def _wandb_detailed_items(detailed: dict, stem: str, tag_prefix: str = "") -> dict:
    """Wrap a ``_detailed_hand_mse`` dict into wandb keys.

    Mirrors ``_wandb_loss_items``' key convention so detailed per-joint /
    per-finger curves live in the same metric family, e.g.
    ``"epoch loss (joint_07)"`` or ``"teleop loss (ema, finger2_rot6d)"``.
    """
    return {f"{stem} ({tag_prefix}{tag})": v for tag, v in detailed.items()}


def _resolve_val_sets(val_cfg) -> list:
    """Normalize the ``validation`` config into a list of per-set specs.

    Each returned item is a dict with keys ``name`` / ``data_dirs`` /
    ``data_dirs_embodiment`` (maybe None) / ``max_episode`` (maybe None).

    Two yaml layouts are supported:

    * **Multi-set (new)** -- ``validation.datasets`` is a list, each entry
      having its own ``name`` / ``data_dirs`` / ``data_dirs_embodiment`` and an
      optional ``max_episode`` (falls back to the top-level
      ``validation.max_episode``)::

          validation:
            enabled: True
            eval_frequency: 2
            batch_size: 200
            datasets:
              - name: teleop
                data_dirs: ["../data/.../teleop"]
                data_dirs_embodiment: [teleop]
              - name: exo
                data_dirs: ["../data/.../exo"]
                data_dirs_embodiment: [exoskeleton]
                max_episode: 20

    * **Single-set (legacy)** -- ``validation.data_dirs`` /
      ``data_dirs_embodiment`` / ``max_episode`` directly on ``validation``.
      Treated as one set named ``"val"`` so existing wandb keys are unchanged.
    """
    sets = val_cfg.get("datasets", None)
    if sets is not None:
        out = []
        seen = set()
        for i, s in enumerate(sets):
            name = s.get("name", None) or f"val{i}"
            if name in seen:
                raise ValueError(
                    f"duplicate validation set name {name!r}; names must be unique"
                )
            seen.add(name)
            embs = s.get("data_dirs_embodiment", None)
            out.append(
                {
                    "name": str(name),
                    "data_dirs": list(s["data_dirs"]),
                    "data_dirs_embodiment": list(embs) if embs is not None else None,
                    "max_episode": s.get("max_episode", val_cfg.get("max_episode", None)),
                }
            )
        return out

    # Legacy single-set layout.
    if val_cfg.get("data_dirs", None) is not None:
        embs = val_cfg.get("data_dirs_embodiment", None)
        return [
            {
                "name": "val",
                "data_dirs": list(val_cfg.data_dirs),
                "data_dirs_embodiment": list(embs) if embs is not None else None,
                "max_episode": val_cfg.get("max_episode", None),
            }
        ]
    return []


def _build_val_dataset(
    cfg: DictConfig, val_set: dict, train_stats: dict, debug: bool
):
    """Instantiate one validation dataset and force its normalization stats to
    match ``train_stats``.

    The ``UVTADataset`` constructor already in-place normalizes every key in
    ``buffer.memory_buffer`` (except those in ``unnormal_list``) using its own
    stats.  We undo that and re-normalize using the training stats so the
    train and val tensors are on the same numerical scale.

    Args:
        cfg: full hydra config (the function reads ``cfg.dataset`` for the
            schema).
        val_set: a single per-set spec from ``_resolve_val_sets`` (keys:
            ``name`` / ``data_dirs`` / ``data_dirs_embodiment`` /
            ``max_episode``).
        train_stats: the ``stats`` dict from the training dataset.
        debug: if True, only load a handful of episodes (mirrors the training
            ``cfg.debug`` shortcut).
    """
    # Start from the training dataset spec and override what differs in val.
    val_dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    val_dataset_cfg["data_dirs"] = list(val_set["data_dirs"])
    val_dataset_cfg["optional_transforms"] = list(_VAL_DETERMINISTIC_TRANSFORMS)

    # Per-path embodiment for the val set.  This MUST be overridden when
    # the val set's embodiment differs from the train set (e.g. train on
    # exoskeleton+manus, eval on teleop) -- otherwise ring_overlay /
    # debug labels would mislabel val frames.  We require it explicitly
    # via the set's ``data_dirs_embodiment`` and validate the length.
    val_embs = val_set.get("data_dirs_embodiment", None)
    if val_embs is not None:
        val_embs = list(val_embs)
        if len(val_embs) != len(val_set["data_dirs"]):
            raise ValueError(
                f"validation set {val_set['name']!r}: data_dirs_embodiment has "
                f"length {len(val_embs)} but data_dirs has length "
                f"{len(val_set['data_dirs'])}; the two must be the same length."
            )
        val_dataset_cfg["data_dirs_embodiment"] = val_embs

    val_max_episode = val_set.get("max_episode", None)
    if debug:
        val_max_episode = 5
    val_dataset_cfg["max_episode"] = val_max_episode

    val_dataset = hydra.utils.instantiate(val_dataset_cfg)

    # Re-normalize the val buffer using the training stats so the network sees
    # the same scale on train and val.  The dataset has already normalized
    # itself with its own stats; we first undo that, then redo it with
    # ``train_stats``.
    own_stats = val_dataset.stats
    buf = val_dataset.buffer.memory_buffer
    for key, data in buf.items():
        if key in val_dataset.unnormal_list:
            continue
        if key not in own_stats or key not in train_stats:
            # e.g. ``hand_action`` is in unnormal_list when relative_hand_action
            # is True, so it never appears in stats.  Skip it.
            continue
        # Step 1: undo val-stats normalization back to the raw tensor.
        raw = unnormalize_data(data, own_stats[key])
        # Step 2: re-normalize with train stats.
        buf[key] = normalize_data(raw, train_stats[key])

    # Replace stats with the training dict (this is what ``__getitem__`` reads
    # for ``relative_pose`` / ``relative_hand_action`` normalization).
    val_dataset.stats = dict(train_stats)
    return val_dataset


_LOSS_KEYS = (
    "total",
    "tactile",               # only when predict_future_tactile is on
    "eef", "eef_xyz", "eef_x", "eef_y", "eef_z", "eef_rot6d",
    "hand", "joint",         # ``joint`` is the legacy alias of ``hand``.
    "hand_joint",            # only joint mode
    "hand_xyz", "hand_rot6d",  # only fingertip mode
)


def _wandb_loss_items(
    losses: dict,
    hand_action_mode: str,
    stem: str,
    tag_prefix: str = "",
):
    """Build a ``{wandb_key: value}`` dict for one loss group, keeping only the
    *non-redundant, non-NaN* curves.

    The wandb key for each component ``tag`` is ``f"{stem} ({tag_prefix}{tag})"``
    so callers fully control naming, e.g. ``stem="epoch loss"`` -> ``"epoch
    loss (eef)"``, or ``stem="teleop loss", tag_prefix="ema, "`` ->
    ``"teleop loss (ema, eef)"``.  This is what lets multiple named validation
    sets log to disjoint metric families.

    The EEF translation is split into per-axis curves (``eef_x`` / ``eef_y`` /
    ``eef_z``) plus the rotation curve (``eef_rot6d``); the aggregate
    ``eef_xyz`` is intentionally NOT logged (it is redundant with the three
    axes and still available for terminal summaries).

    Redundancy removed vs. the old logging:
      * the legacy ``joint`` alias is dropped entirely (it always equalled
        ``hand``);
      * in joint mode we log only ``hand`` (``hand_joint`` was identical and
        ``hand_xyz`` / ``hand_rot6d`` were NaN);
      * in fingertip mode we log ``hand`` + ``hand_xyz`` + ``hand_rot6d``
        (``hand_joint`` was NaN).
    """
    def key(tag):
        return f"{stem} ({tag_prefix}{tag})"

    # Future-tactile prediction curve (only present when
    # predict_future_tactile is on; NaN otherwise so we skip it).
    tactile_items = {}
    _tac = losses.get("tactile", float("nan"))
    if _tac == _tac:  # not NaN
        tactile_items = {key("tactile"): _tac}

    if hand_action_mode == "joint_only":
        # Joint-only: there is no eef block; log just the joint loss.
        return {key("hand"): losses["hand"], **tactile_items}

    if hand_action_mode == "fingertip_only":
        # Fingertip-only: no eef block; log the fingertip loss + xyz/rot6d
        # split (labelled ``fingertips*`` like the regular fingertip mode).
        return {
            key("fingertips"): losses["hand"],
            key("fingertips_xyz"): losses["hand_xyz"],
            key("fingertips_rot6d"): losses["hand_rot6d"],
            **tactile_items,
        }

    items = {
        key("eef"): losses["eef"],
        key("eef_x"): losses["eef_x"],
        key("eef_y"): losses["eef_y"],
        key("eef_z"): losses["eef_z"],
        key("eef_rot6d"): losses["eef_rot6d"],
        **tactile_items,
    }
    if hand_action_mode == "none":
        # EEF-only: no hand curves to log.
        pass
    elif hand_action_mode == "fingertip":
        # Label fingertip-mode hand losses as ``fingertips*`` so they are not
        # confused with the end-effector (eef / wrist) curves.
        items[key("fingertips")] = losses["hand"]
        items[key("fingertips_xyz")] = losses["hand_xyz"]
        items[key("fingertips_rot6d")] = losses["hand_rot6d"]
    else:
        # joint mode: ``hand`` already is the joint loss.
        items[key("hand")] = losses["hand"]
    return items


def _hand_summary(losses: dict, hand_action_mode: str, prec: int = 6) -> str:
    """One-line human-readable hand-loss summary for terminal logging."""
    if hand_action_mode == "none":
        return "hand=<eef-only>"
    if hand_action_mode in ("fingertip", "fingertip_only"):
        tag = "(only)" if hand_action_mode == "fingertip_only" else ""
        return (
            f"fingertips{tag}={losses['hand']:.{prec}f} "
            f"(xyz={losses['hand_xyz']:.{prec}f}, "
            f"rot6d={losses['hand_rot6d']:.{prec}f})"
        )
    if hand_action_mode == "joint_only":
        return f"joint(only)={losses['hand_joint']:.{prec}f}"
    return f"joint={losses['hand_joint']:.{prec}f}"


def _val_wandb_payload(
    name: str,
    losses: dict,
    recon: dict,
    recon_clip: dict,
    hand_action_mode: str,
    ema: bool = False,
    detailed: dict | None = None,
):
    """Build the full wandb payload for one validation set + one model variant.

    ``name`` is the validation set's name (e.g. ``"val"`` for the legacy single
    set, or ``"teleop"`` / ``"exo"`` for named sets).  All metric families are
    prefixed with ``name`` so different val sets never collide.  With
    ``name="val"`` and ``ema=False/True`` the keys reproduce exactly the
    pre-multi-set naming (``"val loss"``, ``"val recon loss (clip)"``,
    ``"val loss (ema)"``, ...), so old wandb panels keep working.
    """
    if ema:
        total_suffix = " (ema)"
        clip_total_suffix = " (ema clip)"
        tag_prefix = "ema, "
        clip_tag_prefix = "ema clip, "
    else:
        total_suffix = ""
        clip_total_suffix = " (clip)"
        tag_prefix = ""
        clip_tag_prefix = "clip, "

    payload = {
        f"{name} loss{total_suffix}": losses["total"],
        f"{name} recon loss{total_suffix}": recon["total"],
        f"{name} recon loss{clip_total_suffix}": recon_clip["total"],
    }
    payload.update(
        _wandb_loss_items(
            losses, hand_action_mode, stem=f"{name} loss", tag_prefix=tag_prefix
        )
    )
    payload.update(
        _wandb_loss_items(
            recon, hand_action_mode,
            stem=f"{name} recon loss", tag_prefix=tag_prefix,
        )
    )
    payload.update(
        _wandb_loss_items(
            recon_clip, hand_action_mode,
            stem=f"{name} recon loss", tag_prefix=clip_tag_prefix,
        )
    )
    if detailed:
        payload.update(
            _wandb_detailed_items(
                detailed, stem=f"{name} loss", tag_prefix=tag_prefix
            )
        )
    return payload


@torch.no_grad()
def _eval_loop(
    accelerator: Accelerator,
    eval_module: torch.nn.Module,
    val_dataloader,
    noise_scheduler,
    epoch: int,
    base_seed: int,
    hand_action_mode: str = "joint",
    log_detailed: bool = True,
    tactile_key: str = "fsr",
    tactile_action_dim: int = 0,
    num_arms: int = 1,
    camera_ids: tuple = (0,),
):
    """Run one pass over ``val_dataloader`` and return mean MSE losses.

    Returns ``(losses, recon_losses, recon_clip_losses, detailed_losses)``.
    The first three dicts have keys exactly ``_LOSS_KEYS`` (all float):
    ``losses`` holds the diffusion (noise-space) MSE; ``recon_losses`` holds
    the action-space reconstruction MSE (``x0_hat`` vs true action);
    ``recon_clip_losses`` is the same with ``x0_hat`` clamped to the DDIM
    clip_sample range.  ``detailed_losses`` is the fine-grained per-joint /
    per-fingertip (noise-space) MSE from ``_detailed_hand_mse`` (dynamic keys,
    empty for ``hand_action_mode == 'none'``).  The recon and detailed dicts
    are eval-only monitoring metrics and never affect optimization.  See
    ``_split_action_mse`` for the meaning of each coarse key.
    Per-batch noise + timesteps are drawn from a generator seeded with
    ``base_seed * 1_000_003 + epoch`` so the val loss curve is comparable
    across epochs and not dominated by random noise sampling.
    """
    was_training = eval_module.training
    eval_module.eval()
    accum: dict[str, list[float]] = {k: [] for k in _LOSS_KEYS}
    recon_accum: dict[str, list[float]] = {k: [] for k in _LOSS_KEYS}
    recon_clip_accum: dict[str, list[float]] = {k: [] for k in _LOSS_KEYS}
    # Fine-grained per-joint / per-fingertip MSE (noise space). Keys are
    # dynamic (depend on hand_action_mode) so use a defaultdict.
    detailed_accum: dict[str, list[float]] = defaultdict(list)

    g_cpu = torch.Generator(device="cpu")
    g_cpu.manual_seed(base_seed * 1_000_003 + epoch)

    num_train_timesteps = noise_scheduler.config.num_train_timesteps

    is_main = accelerator.is_local_main_process
    val_bar = tqdm(
        val_dataloader,
        desc=f"val epoch {epoch}",
        total=len(val_dataloader),
        disable=not is_main,
        leave=False,
        file=sys.stdout,
        dynamic_ncols=True,
    )
    for batch in val_bar:
        visual_observation = _gather_visual_obs(
            batch, camera_ids, accelerator.device
        )
        actions = batch["action"].to(accelerator.device)
        fsr = (
            batch[tactile_key].to(accelerator.device)
            if tactile_key in batch
            else None
        )
        proprioception = (
            batch["proprioception"].to(accelerator.device)
            if "proprioception" in batch
            else None
        )

        bsz = actions.shape[0]
        noise = torch.randn(actions.shape, generator=g_cpu).to(accelerator.device)
        timesteps = torch.randint(
            0, num_train_timesteps, (bsz,), generator=g_cpu
        ).long().to(accelerator.device)
        noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)

        _, noise_pred, _ = eval_module(
            noisy_actions=noisy_actions,
            timesteps=timesteps,
            proprioception=proprioception,
            fsr=fsr,
            visual_obs=visual_observation,
            noise=noise,
            return_noise_pred=True,
            aux_targets=_aux_targets(batch, accelerator.device),
        )
        parts = _split_action_mse(
            noise_pred, noise, hand_action_mode, tactile_action_dim, num_arms
        )
        for k in _LOSS_KEYS:
            accum[k].append(parts[k])
        if log_detailed:
            for k, v in _detailed_hand_mse(
                noise_pred, noise, hand_action_mode, tactile_action_dim, num_arms
            ).items():
                detailed_accum[k].append(v)
        # Action-space reconstruction MSE (eval-only monitoring metric),
        # unclipped + DDIM clip_sample-clamped.
        recon_parts, recon_clip_parts = _recon_loss_parts(
            noisy_actions, noise_pred, timesteps, actions,
            noise_scheduler, hand_action_mode, tactile_action_dim, num_arms,
        )
        for k in _LOSS_KEYS:
            recon_accum[k].append(recon_parts[k])
            recon_clip_accum[k].append(recon_clip_parts[k])
        if is_main:
            val_bar.set_postfix(
                {
                    "loss": f"{parts['total']:.4f}",
                    "eef": f"{parts['eef']:.4f}",
                    "hand": f"{parts['hand']:.4f}",
                    "recon": f"{recon_parts['total']:.4f}",
                    "recon_clip": f"{recon_clip_parts['total']:.4f}",
                },
                refresh=False,
            )
    val_bar.close()

    if was_training:
        eval_module.train()
    if not accum["total"]:
        nan_losses = {k: float("nan") for k in _LOSS_KEYS}
        return nan_losses, dict(nan_losses), dict(nan_losses), {}
    # ``nanmean`` so the not-applicable-for-this-mode keys (which are NaN)
    # don't poison the others.
    losses = {k: _nanmean(accum[k]) for k in _LOSS_KEYS}
    recon_losses = {k: _nanmean(recon_accum[k]) for k in _LOSS_KEYS}
    recon_clip_losses = {k: _nanmean(recon_clip_accum[k]) for k in _LOSS_KEYS}
    detailed_losses = {k: _nanmean(v) for k, v in detailed_accum.items()}
    return losses, recon_losses, recon_clip_losses, detailed_losses


def _evaluate_and_log_val_sets(
    accelerator: Accelerator,
    val_sets: list,
    model: torch.nn.Module,
    ema,
    cfg: DictConfig,
    val_cfg,
    noise_scheduler,
    epoch: int,
    hand_action_mode: str,
    log_payload: dict,
    log_detailed: bool = True,
    tactile_key: str = "fsr",
    tactile_action_dim: int = 0,
    num_arms: int = 1,
    camera_ids: tuple = (0,),
):
    """Evaluate every (name, dataloader) in ``val_sets`` and append metrics.

    For each set we evaluate the live (training) model and -- when EMA is
    enabled -- the EMA copy, writing per-set wandb entries into ``log_payload``
    (mutated in place) and printing a per-set terminal summary.  Metric
    families are prefixed with each set's ``name`` so multiple validation sets
    never collide.
    """
    use_ema = cfg.training.use_ema and bool(val_cfg.get("use_ema", True))
    base_seed = int(val_cfg.seed)

    def _print(tag_name, prefix, losses, recon, recon_clip):
        accelerator.print(
            f"[val:{tag_name}] epoch={epoch}  {prefix}total={losses['total']:.6f}  "
            f"recon={recon['total']:.6f} (clip={recon_clip['total']:.6f})  "
            f"eef={losses['eef']:.6f} "
            f"(x={losses['eef_x']:.6f}, y={losses['eef_y']:.6f}, "
            f"z={losses['eef_z']:.6f}, rot6d={losses['eef_rot6d']:.6f})  "
            f"{_hand_summary(losses, hand_action_mode)}",
            flush=True,
        )

    for name, val_dataloader in val_sets:
        accelerator.wait_for_everyone()
        losses, recon, recon_clip, detailed = _eval_loop(
            accelerator,
            eval_module=model,
            val_dataloader=val_dataloader,
            noise_scheduler=noise_scheduler,
            epoch=epoch,
            base_seed=base_seed,
            hand_action_mode=hand_action_mode,
            log_detailed=log_detailed,
            tactile_key=tactile_key,
            tactile_action_dim=tactile_action_dim,
            num_arms=num_arms,
            camera_ids=camera_ids,
        )
        log_payload.update(
            _val_wandb_payload(
                name, losses, recon, recon_clip, hand_action_mode,
                ema=False, detailed=detailed,
            )
        )
        _print(name, "", losses, recon, recon_clip)

        if use_ema:
            e_losses, e_recon, e_recon_clip, e_detailed = _eval_loop(
                accelerator,
                eval_module=ema.averaged_model.to(accelerator.device),
                val_dataloader=val_dataloader,
                noise_scheduler=noise_scheduler,
                epoch=epoch,
                base_seed=base_seed,
                hand_action_mode=hand_action_mode,
                log_detailed=log_detailed,
                tactile_key=tactile_key,
                tactile_action_dim=tactile_action_dim,
                num_arms=num_arms,
                camera_ids=camera_ids,
            )
            log_payload.update(
                _val_wandb_payload(
                    name, e_losses, e_recon, e_recon_clip, hand_action_mode,
                    ema=True, detailed=e_detailed,
                )
            )
            _print(name, "ema ", e_losses, e_recon, e_recon_clip)


SAMPLER_MODES = ("none", "uniform", "balanced_coverage")


def _resolve_sampler_mode(cfg):
    """Resolve the co-training sampler from the two knobs that can set it.

    ``training.sampler`` is the current one; ``training.uniform_sampling`` is the
    legacy boolean that ~40 older configs still carry.  Precedence is
    ``sampler`` > ``uniform_sampling``, but a config that sets BOTH to
    contradictory values is a mistake, not a preference -- raise rather than
    silently honour one and ignore the other.

    Returns ``(mode, used_legacy)``.
    """
    mode = OmegaConf.select(cfg, "training.sampler", default=None)
    legacy = OmegaConf.select(cfg, "training.uniform_sampling", default=None)

    if mode is None or str(mode).strip() == "":
        if legacy is None:
            return "none", False
        return ("uniform" if bool(legacy) else "none"), True

    mode = str(mode).lower()
    if mode not in SAMPLER_MODES:
        raise ValueError(
            f"training.sampler must be one of {SAMPLER_MODES}; got {mode!r}"
        )
    if legacy is not None:
        implied = "uniform" if bool(legacy) else "none"
        if implied != mode:
            raise ValueError(
                f"training.sampler={mode!r} contradicts the legacy "
                f"training.uniform_sampling={bool(legacy)} (which implies "
                f"{implied!r}).  Delete uniform_sampling from the config and "
                "keep sampler as the single source of truth."
            )
    return mode, False


def _build_cosine_floor_scheduler(
    optimizer,
    num_warmup_steps,
    num_cosine_steps,
    min_lr_ratio,
):
    """LambdaLR: linear warmup -> cosine decay to a floor -> hold the floor.

    Unlike ``diffusers.get_scheduler("cosine", ...)`` (which always decays to
    exactly 0 over ``num_training_steps``), this:

    * decays the LR over ``num_cosine_steps`` (the *expected* training length,
      e.g. 2000 epochs worth of steps) so the cosine actually anneals inside
      the range you train in, instead of staying near peak because the period
      was tied to a 10000-epoch upper bound;
    * never drops below ``min_lr_ratio * base_lr`` (a small floor so learning
      does not fully stop);
    * holds that floor if training happens to run past ``num_cosine_steps``.

    The multiplier returned is relative to the optimizer's base LR.
    """
    num_warmup_steps = max(0, int(num_warmup_steps))
    num_cosine_steps = max(1, int(num_cosine_steps))
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # linear warmup 0 -> 1
            return float(current_step) / float(max(1, num_warmup_steps))
        # progress through the cosine phase, clamped to [0, 1] so that steps
        # beyond the cosine horizon hold the floor.
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_cosine_steps - num_warmup_steps)
        )
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
        # map cosine [0,1] onto [min_lr_ratio, 1]
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "configs",
)


def _source_header(cfg_name: str) -> str:
    """The leading comment block of the yaml this run was launched from.

    ``OmegaConf.save`` serializes the parsed tree, so every comment is dropped --
    including the header naming the task and the demo counts.  Once a checkpoint
    has been copied to the inference machine that header is the only thing that
    says which ablation arm it came from, so it is re-attached by hand.

    Returns "" when the source cannot be found (e.g. a config composed from
    somewhere other than the usual directory); the caller then just saves the
    plain tree as before.
    """
    name = cfg_name if cfg_name.endswith(".yaml") else f"{cfg_name}.yaml"
    path = os.path.join(_CONFIG_DIR, name)
    if not os.path.isfile(path):
        return ""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                out.append(line.rstrip("\n"))
            elif line.strip() == "" and out:
                out.append("")
            else:
                break
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n" if out else ""


def _save_cfg_with_header(cfg, path: str, header: str) -> None:
    """``OmegaConf.save`` then put ``header`` back on top.

    Comments are inert to every yaml parser, so the deploy-side
    ``load_config`` still reads this file unchanged.
    """
    OmegaConf.save(cfg, path)
    if not header:
        return
    with open(path, encoding="utf-8") as fh:
        body = fh.read()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header + body)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="flip_book",
)
def train_diffusion_policy(cfg: DictConfig):
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps
    )
    # # create save dir
    output_dir = HydraConfig.get().runtime.output_dir
    resume = cfg.training.resume
    if resume:
        # overwrite the config with the checkpoint config
        accelerator.print("resume from checkpoint")
        model_path = cfg.training.model_path
        model_ckpt = cfg.training.model_ckpt
        cfg = load_config(model_path)
        # save the checkpoint config to the output dir
        OmegaConf.save(cfg, os.path.join(output_dir, ".hydra", "config.yaml"))
    if accelerator.is_local_main_process:
        # Build a deterministic, human-readable wandb run name so it is
        # trivial to map a wandb run back to (a) the yaml config used
        # and (b) the local hydra experiment directory.  Example:
        #   train_diffusion_policy_v22__2026-05-27_14-29-19
        try:
            cfg_name = HydraConfig.get().job.config_name
        except Exception:
            cfg_name = "train_diffusion_policy"
        run_name = f"{cfg_name}__{os.path.basename(output_dir)}"
        # Keep Hydra's .hydra/config.yaml, and also save an easy-to-spot
        # config copy named after the yaml that launched this run.
        OmegaConf.save(cfg, os.path.join(output_dir, f"{cfg_name}.yaml"))
        wandb.init(project=cfg.project_name, name=run_name)
        wandb.config.update(OmegaConf.to_container(cfg))
        accelerator.print("Logging dir", output_dir)
        accelerator.print("Wandb run name", run_name)
        ckpt_save_dir = os.path.join(output_dir, "checkpoints")
        state_save_dir = os.path.join(output_dir, "state")
        os.makedirs(ckpt_save_dir, exist_ok=True)
        os.makedirs(state_save_dir, exist_ok=True)

    # max_episode = 5 if cfg.debug else cfg.dataset.max_episode
    dataset = hydra.utils.instantiate(
        cfg.dataset, max_episode=5 if cfg.debug else cfg.dataset.max_episode
    )
    print("Total training samples:", len(dataset))
    # open a file for writing in binary mode
    if accelerator.is_local_main_process:
        # save training data statistics (min, max) for each dim
        print("saving stats...")
        stats = dataset.stats
        print("checking if normalize...")
        # The "hand" half of the action vector can be stored as either
        # joint angles or fingertip pose (xyz part, since rot6d uses
        # identity normalization).  Pick the stats key that actually
        # exists in the dataset's stats dict.
        if "fingertip_xyz_action" in stats:
            print(
                stats["fingertip_xyz_action"]["max"]
                - stats["fingertip_xyz_action"]["min"]
                > 5e-2
            )
        elif "relative_hand_action" in stats and cfg.dataset.relative_hand_action:
            print(
                stats["relative_hand_action"]["max"]
                - stats["relative_hand_action"]["min"]
                > 5e-2
            )
        elif "hand_action" in stats:
            print(stats["hand_action"]["max"] - stats["hand_action"]["min"] > 5e-2)
        else:
            # EEF-only action (hand_action_mode='none'): no hand stats exist.
            print("eef-only action: no hand stats to check")
        with open(os.path.join(output_dir, "stats.pickle"), "wb") as f:
            # write the dictionary to the file
            pickle.dump(stats, f)
    # create dataloader
    # ------------------------------------------------------------------
    # Co-training uniform sampling.
    #
    # By default the concatenated multi-dataset pool is sampled uniformly per
    # SAMPLE, which biases training toward whichever dataset has more frames.
    # When ``training.uniform_sampling=True`` we instead draw with a
    # WeightedRandomSampler so that EACH listed data_dir contributes equal
    # probability mass per epoch (i.e. two datasets of very different sizes /
    # embodiments are each seen ~50% of the time).  ``shuffle`` is ignored when
    # a sampler is active (PyTorch forbids passing both); the sampler already
    # randomizes order, and ``accelerator.prepare`` shards it across GPUs.
    # ------------------------------------------------------------------
    # ``training.sampler`` picks the scheme (``training.uniform_sampling: True``
    # is still honoured and means "uniform"):
    #   none              -- plain shuffle over the concatenated pool; the bigger
    #                        dataset dominates in proportion to its size.
    #   uniform           -- WeightedRandomSampler, each data_dir gets equal
    #                        probability mass.  WITH replacement, so per epoch a
    #                        large dataset is only partially covered (~51% for
    #                        light_skeleton) while a small one repeats.
    #   balanced_coverage -- draw max_d(n_d) indices from EVERY data_dir, built
    #                        from whole permutations: the largest dataset is seen
    #                        exactly once end to end, smaller ones are upsampled
    #                        with every sample guaranteed at least once.  The
    #                        epoch grows to max_d(n_d) * num_data_dirs.
    _sampler_mode, _legacy = _resolve_sampler_mode(cfg)
    if _legacy:
        accelerator.print(
            f"[sampler] training.sampler not set; using legacy "
            f"uniform_sampling -> {_sampler_mode!r}"
        )
    train_sampler = None
    if _sampler_mode == "balanced_coverage":
        train_sampler = dataset.build_balanced_coverage_sampler(
            seed=int(OmegaConf.select(cfg, "training.seed", default=0))
        )
        accelerator.print(
            "[balanced-coverage sampling] every sample of every data_dir is "
            "drawn at least once per epoch:\n"
            + dataset.describe_balanced_coverage()
        )
    elif _sampler_mode == "uniform":
        train_sampler = dataset.build_uniform_sampler(
            seed=int(OmegaConf.select(cfg, "training.seed", default=0))
        )
        _dir_idx = dataset.get_sample_data_dir_idx()
        _counts = np.bincount(_dir_idx, minlength=len(dataset.buffer.data_path))
        accelerator.print(
            "[uniform-sampling] enabled: each data_dir gets equal probability "
            f"per epoch.  per-data_dir sample counts={_counts.tolist()} "
            f"embodiments={list(getattr(dataset.buffer, 'embodiments', []))}"
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=(cfg.training.shuffle if train_sampler is None else False),
        sampler=train_sampler,
        pin_memory=cfg.training.pin_memory,
        persistent_workers=cfg.training.persistent_workers,
        drop_last=cfg.training.drop_last,
    )
    print("====================================")
    print("len of dataset", len(dataset))
    print("====================================")

    # ------------------------------------------------------------------
    # Optional held-out validation set.
    # ------------------------------------------------------------------
    val_cfg = cfg.get("validation", None)
    val_enabled = bool(val_cfg.enabled) if val_cfg is not None else False
    # ``val_sets`` is a list of ``(name, dataloader)`` -- one entry per
    # validation set (a single legacy set is named "val").
    val_sets: list = []
    if val_enabled:
        val_specs = _resolve_val_sets(val_cfg)
        if not val_specs:
            accelerator.print(
                "[val] validation.enabled=True but no validation sets found "
                "(set validation.datasets or validation.data_dirs); skipping val."
            )
        for spec in val_specs:
            name = spec["name"]
            accelerator.print(f"[val] building validation set {name!r} ...")
            vds = _build_val_dataset(
                cfg, spec, train_stats=dataset.stats, debug=cfg.debug
            )
            accelerator.print(
                f"[val:{name}] {len(vds)} samples loaded from {spec['data_dirs']}"
            )
            vdl = torch.utils.data.DataLoader(
                vds,
                batch_size=val_cfg.batch_size,
                num_workers=val_cfg.num_workers,
                shuffle=False,
                pin_memory=val_cfg.pin_memory,
                persistent_workers=val_cfg.persistent_workers,
                drop_last=False,
            )
            val_sets.append((name, vdl))

    sample_batch = next(iter(dataloader))
    for k, v in sample_batch.items():
        accelerator.print(k, v.shape)

    # Resolve which batch key carries the tactile (FSR / force) stream.  The
    # dataset stores it under its SOURCE field name (``dataset.fsr_source_key``,
    # default "fsr"), so stats.pickle and the deploy side can refer to it by
    # its real name.  When tactile is disabled the key simply isn't in the
    # batch, so every downstream use is guarded by ``tactile_key in batch``.
    tactile_key = str(cfg.dataset.get("fsr_source_key", "fsr"))
    accelerator.print(f"[tactile] tactile_key = {tactile_key!r}")

    # ------------------------------------------------------------------
    # Dry-run dimensions from the actual sample, then override the yaml.
    # This lets the user change ``obs_horizon`` / ``proprio_mode`` /
    # ``hand_action_mode`` freely without touching ``action_dim`` or
    # ``global_cond_dim`` by hand.
    #
    # global_cond_dim layout (mirrors the order in DiffusionPolicy.forward):
    #     [vision_emb (b, o*D)]
    #     [proprioception (b, o*P)]      # if present
    #     [fsr (b, o*F)]                 # if present
    # so the total is ``o*(D + P + F)`` with the appropriate streams.
    # ------------------------------------------------------------------
    # Cameras fed to the vision backbone.  For a bimanual setup this is BOTH
    # wrist cameras (chips_teleop: camera_0 = right wrist, camera_2 = left
    # wrist), stacked along the horizon axis and encoded by the SHARED backbone;
    # a single camera keeps the original behaviour.
    camera_ids = [int(c) for c in cfg.dataset.load_camera_ids]
    if not camera_ids:
        raise ValueError(
            "train_diffusion_policy needs at least one camera in "
            "dataset.load_camera_ids."
        )
    num_cameras = len(camera_ids)
    accelerator.print(f"[vision] camera_ids = {camera_ids}")

    inferred_action_dim = int(sample_batch["action"].shape[-1])
    _cam0_key = f"camera_{camera_ids[0]}"
    obs_horizon_runtime = int(sample_batch[_cam0_key].shape[1])  # (B, o, C, H, W)
    # Vision embedding dim: forward one frame through the backbone.
    _tmp_model = hydra.utils.instantiate(cfg.model)
    _tmp_model.eval()
    with torch.no_grad():
        _img = sample_batch[_cam0_key][:1, 0].to(next(_tmp_model.parameters()).device)
        _emb = _tmp_model.vision_backbone(_img)
        vision_emb_dim = int(_emb.shape[-1])
    del _tmp_model
    # Every listed camera is encoded (num_cameras * obs_horizon frames total).
    vision_total = num_cameras * obs_horizon_runtime * vision_emb_dim
    proprio_total = 0
    if "proprioception" in sample_batch:
        # (B, o, P) -> P per frame * obs_horizon frames
        proprio_per_frame = int(sample_batch["proprioception"].shape[-1])
        proprio_total = obs_horizon_runtime * proprio_per_frame
    fsr_total = 0
    if tactile_key in sample_batch:
        fsr_per_frame = int(sample_batch[tactile_key].shape[-1])
        fsr_total = obs_horizon_runtime * fsr_per_frame
    inferred_global_cond_dim = vision_total + proprio_total + fsr_total
    accelerator.print(
        "[dim-inference] obs_horizon=", obs_horizon_runtime,
        "  num_cameras=", num_cameras,
        "  vision_emb_dim=", vision_emb_dim,
        "  vision_total=", vision_total,
        "  proprio_total=", proprio_total,
        "  fsr_total=", fsr_total,
        "  -> global_cond_dim=", inferred_global_cond_dim,
        "  action_dim=", inferred_action_dim,
    )
    # Patch the cfg so the freshly-built model uses the right sizes.
    OmegaConf.update(cfg, "action_dim", inferred_action_dim, merge=True)
    OmegaConf.update(
        cfg,
        "model.diffusion_policy_head.global_cond_dim",
        inferred_global_cond_dim,
        merge=True,
    )
    OmegaConf.update(
        cfg, "model.diffusion_policy_head.input_dim", inferred_action_dim, merge=True
    )

    # ------------------------------------------------------------------
    # World heads (aux_as_head).  Sized from the batch the dataset actually
    # produced rather than from yaml, so the head cannot silently disagree with
    # the target.  Must run BEFORE the config is persisted below, otherwise the
    # deploy side rebuilds a model without these heads and the checkpoint load
    # fails on unexpected keys.  Weights default to 0.2: the auxiliary targets
    # are a means (shaping the trunk), not an end, and at weight 1 they compete
    # with the action for capacity -- which is what putting them in the diffusion
    # vector already cost us (2.35mm -> 3.14mm wrist RMS).
    # ------------------------------------------------------------------
    aux_shapes = {
        k[len("aux_"):]: (int(v.shape[1]), int(v.shape[2]))
        for k, v in sample_batch.items()
        if k.startswith("aux_")
    }
    _whw = OmegaConf.select(cfg, "training.world_head_weights", default=None)
    world_head_weights = (
        OmegaConf.to_container(_whw, resolve=True) if _whw is not None else {}
    )
    if aux_shapes:
        hidden = int(OmegaConf.select(cfg, "training.world_head_hidden", default=512))
        for n in aux_shapes:
            world_head_weights.setdefault(n, 0.2)
        # ``model.world_heads`` is absent from every config that does not use
        # them, so struct mode has to be opened to add the key.
        with open_dict(cfg):
            OmegaConf.update(
                cfg,
                "model.world_heads",
                {
                    n: {
                        "_target_": "uvta.diffusion_policy.diffusion_policy.WorldHead",
                        "cond_dim": inferred_global_cond_dim,
                        "horizon": h,
                        "dim": d,
                        "hidden": hidden,
                    }
                    for n, (h, d) in aux_shapes.items()
                },
                merge=True,
            )
            OmegaConf.update(
                cfg, "training.world_head_weights", world_head_weights, merge=True
            )
            # Mirror onto the model so the weights travel with the checkpoint's
            # config and the model's own startup print is truthful.
            OmegaConf.update(
                cfg, "model.world_head_weights", world_head_weights, merge=True
            )
        accelerator.print(
            "[world-heads] "
            + ", ".join(f"{n}: {h}x{d}" for n, (h, d) in aux_shapes.items())
            + f"  cond_dim={inferred_global_cond_dim}  hidden={hidden}"
            + f"  weights={world_head_weights}"
        )

    # Persist the dry-run-resolved dims back to the saved configs so that
    # deploy-side loaders (``load_diffusion_model`` -> ``load_config`` reads
    # ``.hydra/config.yaml``) rebuild a model whose ``global_cond_dim`` /
    # ``action_dim`` match the checkpoint weights.  Without this, the saved
    # yaml keeps the fallback ``global_cond_dim`` (e.g. 384) and loading the
    # checkpoint fails with a size mismatch.  Main process only.
    if accelerator.is_local_main_process:
        try:
            cfg_name = HydraConfig.get().job.config_name
        except Exception:
            cfg_name = "train_diffusion_policy"
        # Carry the source yaml's header comments across (OmegaConf drops them);
        # they are what identifies the arm on the inference machine.
        header = _source_header(cfg_name)
        hydra_cfg_path = os.path.join(output_dir, ".hydra", "config.yaml")
        if os.path.exists(os.path.dirname(hydra_cfg_path)):
            _save_cfg_with_header(cfg, hydra_cfg_path, header)
        _save_cfg_with_header(
            cfg, os.path.join(output_dir, f"{cfg_name}.yaml"), header
        )
        accelerator.print(
            "[dim-inference] saved resolved dims to config "
            f"(global_cond_dim={inferred_global_cond_dim}, "
            f"action_dim={inferred_action_dim})"
        )

    # Read once: which hand action layout are we training so that the loss
    # split ("hand_xyz" vs "hand_rot6d" vs "hand_joint") gets the right
    # interpretation downstream.  We mirror the same default-resolution
    # rule used by ``UVTADataset``: when ``hand_action_mode`` is unset in
    # yaml, the dataset picks ``"fingertip"`` for fingertip-style proprio
    # and ``"joint"`` otherwise.
    _ds_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    hand_action_mode = _ds_cfg.get("hand_action_mode") or (
        "fingertip"
        if str(_ds_cfg.get("proprio_mode", "joint")).startswith("fingertip")
        else "joint"
    )
    accelerator.print(f"[loss-split] hand_action_mode = {hand_action_mode!r}")

    # Whether to log the fine-grained per-joint / per-fingertip MSE breakdown
    # (``epoch loss (joint_07)`` etc.) to wandb for both train and val.  These
    # add ~22 (joint) or ~15 (fingertip) extra curves; toggle via yaml
    # ``training.log_detailed_loss`` (default on).
    log_detailed_loss = bool(cfg.training.get("log_detailed_loss", True))
    accelerator.print(f"[loss-split] log_detailed_loss = {log_detailed_loss}")

    # Width of the trailing FUTURE-tactile action block (predict_future_tactile).
    # 0 when the feature is off.  Threaded into every loss-split call so the
    # tactile channels are reported under ``tactile`` and excluded from the
    # eef / hand curves.
    tactile_action_dim = int(getattr(dataset, "tactile_action_dim", 0))
    accelerator.print(f"[loss-split] tactile_action_dim = {tactile_action_dim}")

    # Number of arms (bimanual = 2).  The action's MOTOR part is laid out
    # ``[arm0(eef|hand), arm1(eef|hand), ...]``; the loss-split helpers reshape
    # it to ``(..., num_arms, per_arm)`` so eef / hand / tactile stay balanced
    # groups aggregated across arms (num_arms=1 leaves single-arm math intact).
    num_arms = int(getattr(dataset, "num_arms", 1))
    accelerator.print(f"[loss-split] num_arms = {num_arms}")
    # Enabled OUTPUT blocks in target order (``predict_action`` /
    # ``predict_state``).  Each arm's motor part is these blocks back to back, so
    # the loss split needs the list to know where one ends and the next begins.
    motor_blocks = tuple(getattr(dataset, "motor_blocks", ("action",)))
    accelerator.print(f"[loss-split] motor blocks = {list(motor_blocks)}")

    # ------------------------------------------------------------------
    # Group-balanced diffusion loss (optional back-prop reweighting).
    #
    # The default back-prop loss is a plain MSE over the WHOLE action tensor,
    # so each group's pull on the gradient scales with its dimensionality.
    # With predict_future_tactile the action is [eef(9) | hand(22) |
    # tactile(F)]; for the 100-D fsr stream the tactile block is 100/131 of the
    # dims and dominates the loss.  When ``training.balanced_action_loss`` is
    # on, ``_weighted_action_loss`` averages each group's MSE separately and
    # combines them with ``training.action_loss_group_weights`` (renormalized to
    # sum to 1), so e.g. weights 1:1:1 make eef / hand / tactile each contribute
    # exactly 1/3 regardless of width.  These bookkeeping metrics
    # (``_split_action_mse``) are unaffected -- only the scalar that drives
    # ``accelerator.backward`` changes.
    # ------------------------------------------------------------------
    balanced_action_loss = bool(
        OmegaConf.select(cfg, "training.balanced_action_loss", default=False)
    )
    _alw = OmegaConf.select(cfg, "training.action_loss_group_weights", default=None)
    action_loss_group_weights = (
        OmegaConf.to_container(_alw, resolve=True) if _alw is not None else {}
    )
    accelerator.print(
        f"[loss-split] balanced_action_loss = {balanced_action_loss}  "
        f"weights = {action_loss_group_weights}"
    )

    # ------------------------------------------------------------------
    # Per-embodiment loss breakdown (co-training diagnostics).
    #
    # Each batch mixes embodiments (e.g. robot teleop + human skeleton).  When
    # ``training.log_per_embodiment_loss`` is on we additionally split every
    # per-component MSE by the sample's ``embodiment_id`` and log separate
    # wandb curves (``epoch loss (robot, eef)`` / ``epoch loss (human,
    # tactile)`` ...), so it is easy to see WHICH embodiment / component has
    # (not) converged.  Skipped automatically when there is only one
    # embodiment (the curves would just duplicate the aggregate ones).
    # ------------------------------------------------------------------
    log_per_embodiment_loss = bool(
        OmegaConf.select(cfg, "training.log_per_embodiment_loss", default=True)
    )
    embodiment_names: list[str] = []
    if log_per_embodiment_loss:
        try:
            dataset.get_sample_embodiment_ids()  # populates dataset.embodiment_names
            embodiment_names = list(getattr(dataset, "embodiment_names", []) or [])
        except Exception as exc:
            accelerator.print(f"[loss-split] per-embodiment loss disabled: {exc}")
            embodiment_names = []
        if len(embodiment_names) <= 1:
            # single embodiment -> identical to the aggregate curves; skip.
            embodiment_names = []
    accelerator.print(
        f"[loss-split] per-embodiment loss for embodiments = {embodiment_names}"
    )

    model = hydra.utils.instantiate(cfg.model)
    noise_scheduler = hydra.utils.instantiate(cfg.noise_scheduler)

    # --- normalization vs clip_sample cross-check -----------------------
    # Mean-std normalization without a sigma clamp produces action targets
    # outside [-1, 1], which ``clip_sample=True`` would silently truncate.
    # The dataset (``UVTADataset.__init__``) already auto-supplies a default
    # ``norm_clip_sigma`` when it is missing under ``meanstd`` and warns, so by
    # the time we get here that footgun is defused.  We surface a clear
    # banner of the effective config and flag the genuinely dangerous combo
    # (someone who deliberately wants raw z-score yet left clip_sample on).
    _norm_method = str(_ds_cfg.get("norm_method", "minmax"))
    _clip_sigma = _ds_cfg.get("norm_clip_sigma", None)
    _clip_sample = bool(getattr(noise_scheduler.config, "clip_sample", False))
    _effective_sigma = getattr(dataset, "norm_clip_sigma", _clip_sigma)
    if _norm_method == "meanstd" and not _clip_sigma and _clip_sample:
        accelerator.print(
            "[norm] NOTE: norm_method='meanstd' had no norm_clip_sigma in "
            f"yaml; the dataset auto-set norm_clip_sigma={_effective_sigma} "
            "to stay compatible with clip_sample=True.  Set it explicitly in "
            "yaml to silence this, or set clip_sample=False for raw z-score.",
            flush=True,
        )
    accelerator.print(
        f"[norm] method={_norm_method} clip_sigma(yaml)={_clip_sigma} "
        f"clip_sigma(effective)={_effective_sigma} "
        f"clip_percentile={_ds_cfg.get('norm_clip_percentile', None)} "
        f"clip_sample={_clip_sample}"
    )
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    num_update_steps_per_epoch = len(dataloader)
    max_train_steps = cfg.training.epochs * num_update_steps_per_epoch
    # The LR schedule's horizon should match the *expected* training length,
    # not the (much larger) ``epochs`` upper bound -- otherwise the cosine
    # barely decays inside the range you actually train in.  Configurable via
    # ``lr_scheduler_steps_epochs`` (default 2000); falls back to ``epochs``.
    lr_period_epochs = int(
        OmegaConf.select(cfg, "lr_scheduler_steps_epochs", default=cfg.training.epochs)
    )
    lr_min_ratio = float(OmegaConf.select(cfg, "lr_min_ratio", default=0.0))
    # ``num_update_steps_per_epoch`` is ``len(dataloader)`` taken BEFORE
    # ``accelerator.prepare``, i.e. the full-dataset batch count.  After
    # ``prepare``, the dataloader is sharded across ``num_processes`` (so each
    # process sees ~1/N batches) but the prepared scheduler is stepped N times
    # per global optimizer step -- the two cancel out, so the prepared
    # scheduler advances exactly ``num_update_steps_per_epoch`` steps per
    # epoch regardless of GPU count.  Therefore the cosine horizon must NOT be
    # multiplied by ``num_processes`` (doing so previously stretched the
    # anneal to ``period * num_gpus`` epochs, so on 4 GPUs the LR barely
    # decayed -- it was effectively an 8000-epoch cosine).
    cosine_steps = lr_period_epochs * num_update_steps_per_epoch
    warmup_steps = int(cfg.num_warmup_steps)
    if str(cfg.lr_scheduler) == "cosine":
        # Custom warmup -> cosine-to-floor -> hold-floor schedule so we get a
        # configurable annealing horizon and a non-zero LR floor.
        lr_scheduler = _build_cosine_floor_scheduler(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_cosine_steps=cosine_steps,
            min_lr_ratio=lr_min_ratio,
        )
        accelerator.print(
            f"[lr] cosine-floor: warmup={warmup_steps} steps, "
            f"anneal over {lr_period_epochs} epochs "
            f"({cosine_steps} steps), floor={lr_min_ratio:g}*base_lr, "
            f"epochs upper bound={cfg.training.epochs}, "
            f"num_processes={accelerator.num_processes}"
        )
    else:
        # Other scheduler types keep the diffusers helper for back-compat.
        lr_scheduler = get_scheduler(
            cfg.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=cosine_steps,
        )
    # Prepare everything with our `accelerator`.
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    if val_sets:
        val_sets = [
            (name, accelerator.prepare(vdl)) for name, vdl in val_sets
        ]
    if resume:
        model_state = os.path.join(model_path, "state", f"epoch_{model_ckpt}")
        accelerator.load_state(model_state)
        print("successfully loaded model from checkpoint!")

    ema = None
    if cfg.training.use_ema:
        ema_model = copy.deepcopy(accelerator.unwrap_model(model))
        ema = hydra.utils.instantiate(cfg.ema, model=ema_model)

    # ------------------------------------------------------------------
    # Progress reporting setup.
    #
    # 1.  Per-batch tqdm bar showing live loss (main process only, to
    #     avoid garbled output under multi-GPU).
    # 2.  Periodic ``accelerator.print`` line every ``log_interval`` batches
    #     so the bar's content is also captured in non-tty logs (e.g.
    #     redirected to a file).  Configurable via ``cfg.training.log_interval``
    #     (default 50).
    # 3.  End-of-epoch summary line with mean loss and wall-clock time.
    #
    # ``flush=True`` everywhere is essential under debug mode (debugpy) and
    # when stdout is captured, because by default Python line-buffers
    # stdout only in interactive ttys.
    # ------------------------------------------------------------------
    log_interval = int(cfg.training.get("log_interval", 50))
    is_main = accelerator.is_local_main_process
    num_batches_per_epoch = len(dataloader)
    if is_main:
        accelerator.print(
            f"[train] start.  epochs={cfg.training.epochs}  "
            f"batches/epoch={num_batches_per_epoch}  "
            f"log_interval={log_interval}",
            flush=True,
        )

    for epoch in range(cfg.training.epochs):
        # Per-component MSE buckets.  Layout matches ``_LOSS_KEYS`` keys
        # returned by ``_split_action_mse``.  We use one bucket per key and
        # average at the end of the epoch.
        epoch_buckets: dict[str, list[float]] = {k: [] for k in _LOSS_KEYS}
        # Action-space reconstruction MSE buckets (eval-only monitoring; never
        # backprop'd).  Same layout/keys as ``epoch_buckets``.  ``*_clip`` is
        # the DDIM clip_sample-clamped variant.
        recon_buckets: dict[str, list[float]] = {k: [] for k in _LOSS_KEYS}
        recon_clip_buckets: dict[str, list[float]] = {k: [] for k in _LOSS_KEYS}
        # Fine-grained per-joint / per-fingertip (noise-space) MSE buckets.
        # Keys are dynamic (depend on hand_action_mode) -> defaultdict.
        detailed_buckets: dict[str, list[float]] = defaultdict(list)
        # Group-balanced back-prop loss actually optimized this epoch (only
        # populated when balanced_action_loss is on; it differs from
        # ``epoch_buckets['total']`` which is always the plain full-tensor MSE).
        weighted_loss_bucket: list[float] = []
        # Per-world-head MSE this epoch (aux_as_head only).  Logged separately
        # from the action curves: these targets no longer live in the action
        # tensor, so ``_split_action_mse`` cannot see them.
        aux_loss_buckets: dict[str, list[float]] = defaultdict(list)
        # Per-embodiment component MSE buckets (co-training diagnostics).
        # ``emb_epoch_buckets[emb][key]`` mirrors ``epoch_buckets`` but only for
        # the samples of that embodiment; ``emb_weighted_bucket[emb]`` holds the
        # per-embodiment group-balanced loss (when balanced_action_loss is on).
        emb_epoch_buckets: dict[str, dict[str, list[float]]] = {
            emb: {k: [] for k in _LOSS_KEYS} for emb in embodiment_names
        }
        emb_weighted_bucket: dict[str, list[float]] = {
            emb: [] for emb in embodiment_names
        }
        epoch_start_time = time.time()

        # batch loop with progress bar (only on main process).  We use
        # tqdm.auto so the bar adapts to terminal vs notebook contexts.
        # disable=True silences ranks > 0 entirely.
        pbar = tqdm(
            dataloader,
            desc=f"epoch {epoch}",
            total=num_batches_per_epoch,
            disable=not is_main,
            leave=False,
            file=sys.stdout,
            dynamic_ncols=True,
        )
        for batch_idx, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                (visual_observation, actions, fsr, proprioception) = (
                    _gather_visual_obs(batch, camera_ids, accelerator.device),
                    batch["action"].to(accelerator.device),
                    batch[tactile_key].to(accelerator.device)
                    if tactile_key in batch
                    else None,
                    batch["proprioception"].to(accelerator.device)
                    if "proprioception" in batch
                    else None,
                )
                noise = torch.randn(actions.shape, device=accelerator.device)
                bsz = actions.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=accelerator.device,
                ).long()
                noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)

                loss, noise_pred, aux_losses = model(
                    noisy_actions=noisy_actions,
                    timesteps=timesteps,
                    proprioception=proprioception,
                    fsr=fsr,
                    visual_obs=visual_observation,
                    noise=noise,
                    return_noise_pred=True,
                    aux_targets=_aux_targets(batch, accelerator.device),
                )

                # Group-balanced back-prop loss: replace the model's plain
                # full-tensor MSE with a per-group-weighted MSE so the
                # high-dim tactile block cannot dominate the gradient (see
                # ``_weighted_action_loss``).  ``noise_pred`` still carries the
                # grad history, so the reweighted scalar backprops correctly.
                if balanced_action_loss:
                    loss = _weighted_action_loss(
                        noise_pred,
                        noise,
                        hand_action_mode,
                        tactile_action_dim,
                        action_loss_group_weights,
                        num_arms,
                        motor_blocks,
                    )
                    # Replacing the model's loss also dropped the world-head
                    # terms it had added, which would leave those heads without
                    # gradient.  Re-apply them with their configured weights.
                    for _n, _l in aux_losses.items():
                        loss = loss + float(world_head_weights.get(_n, 1.0)) * _l
                    weighted_loss_val = float(loss.detach().item())

                # optimize
                optimizer.zero_grad()
                accelerator.backward(loss)
                lr_scheduler.step()
                optimizer.step()
                # update ema
                if cfg.training.use_ema:
                    ema.step(accelerator.unwrap_model(model))
                # logging.  Backprop uses the standard total MSE; the
                # eef / hand / per-component MSEs are bookkeeping-only
                # metrics computed from the same noise-prediction tensor
                # so they cost no extra forward pass.
                with torch.no_grad():
                    parts = _split_action_mse(
                        noise_pred.detach(), noise, hand_action_mode,
                        tactile_action_dim, num_arms,
                    )
                    # Action-space reconstruction MSE (eval-only metric):
                    # invert the epsilon prediction back to x0 and compare to
                    # the true (clean) action.  Does NOT touch the backprop
                    # loss above.  ``recon_clip_parts`` clamps x0_hat to the
                    # DDIM clip_sample range (deploy-representative).
                    recon_parts, recon_clip_parts = _recon_loss_parts(
                        noisy_actions,
                        noise_pred.detach(),
                        timesteps,
                        actions,
                        noise_scheduler,
                        hand_action_mode,
                        tactile_action_dim,
                        num_arms,
                    )
                    detailed_parts = (
                        _detailed_hand_mse(
                            noise_pred.detach(), noise, hand_action_mode,
                            tactile_action_dim, num_arms,
                        )
                        if log_detailed_loss
                        else {}
                    )
                    # Per-embodiment component MSE: mask the batch by each
                    # embodiment id and reuse the same split so robot / human
                    # curves are directly comparable to the aggregate ones.
                    emb_parts: dict[str, tuple[dict, float | None]] = {}
                    if embodiment_names and "embodiment_id" in batch:
                        emb_ids = batch["embodiment_id"]
                        np_det = noise_pred.detach()
                        for eid, emb in enumerate(embodiment_names):
                            mask = (emb_ids == eid).to(np_det.device)
                            if not bool(mask.any()):
                                continue
                            np_e = np_det[mask]
                            no_e = noise[mask]
                            ep = _split_action_mse(
                                np_e, no_e, hand_action_mode,
                                tactile_action_dim, num_arms,
                            )
                            wl = (
                                float(
                                    _weighted_action_loss(
                                        np_e, no_e, hand_action_mode,
                                        tactile_action_dim,
                                        action_loss_group_weights,
                                        num_arms,
                                        motor_blocks,
                                    ).item()
                                )
                                if balanced_action_loss
                                else None
                            )
                            emb_parts[emb] = (ep, wl)
                for k in _LOSS_KEYS:
                    epoch_buckets[k].append(parts[k])
                    recon_buckets[k].append(recon_parts[k])
                    recon_clip_buckets[k].append(recon_clip_parts[k])
                for k, v in detailed_parts.items():
                    detailed_buckets[k].append(v)
                if balanced_action_loss:
                    weighted_loss_bucket.append(weighted_loss_val)
                for _n, _l in aux_losses.items():
                    aux_loss_buckets[_n].append(float(_l.detach().item()))
                for emb, (ep, wl) in emb_parts.items():
                    for k in _LOSS_KEYS:
                        emb_epoch_buckets[emb][k].append(ep[k])
                    if wl is not None:
                        emb_weighted_bucket[emb].append(wl)

                # ----- Per-batch progress feedback -----
                # The tqdm bar always shows the *current* batch's loss
                # decomposition; the periodic stdout print is captured in
                # non-tty logs (file redirection, debug capture, etc.).
                if is_main:
                    if hand_action_mode == "none":
                        postfix = {
                            "loss": f"{parts['total']:.4f}",
                            "eef_xyz": f"{parts['eef_xyz']:.4f}",
                            "eef_rot6d": f"{parts['eef_rot6d']:.4f}",
                            "recon": f"{recon_parts['total']:.4f}",
                            "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                        }
                    elif hand_action_mode == "fingertip":
                        postfix = {
                            "loss": f"{parts['total']:.4f}",
                            "eef": f"{parts['eef']:.4f}",
                            "hand": f"{parts['hand']:.4f}",
                            "recon": f"{recon_parts['total']:.4f}",
                            "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                        }
                    elif hand_action_mode == "fingertip_only":
                        postfix = {
                            "loss": f"{parts['total']:.4f}",
                            "fingertips": f"{parts['hand']:.4f}",
                            "recon": f"{recon_parts['total']:.4f}",
                            "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                        }
                    elif hand_action_mode == "joint_only":
                        postfix = {
                            "loss": f"{parts['total']:.4f}",
                            "joint": f"{parts['hand_joint']:.4f}",
                            "recon": f"{recon_parts['total']:.4f}",
                            "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                        }
                    else:
                        postfix = {
                            "loss": f"{parts['total']:.4f}",
                            "eef": f"{parts['eef']:.4f}",
                            "joint": f"{parts['hand_joint']:.4f}",
                            "recon": f"{recon_parts['total']:.4f}",
                            "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                        }
                    if tactile_action_dim > 0 and parts["tactile"] == parts["tactile"]:
                        postfix["tac"] = f"{parts['tactile']:.4f}"
                    # ``wloss`` is the group-balanced scalar actually optimized;
                    # ``loss`` above stays the plain full-tensor MSE for
                    # continuity with the unbalanced runs.
                    if balanced_action_loss:
                        postfix["wloss"] = f"{weighted_loss_val:.4f}"
                    pbar.set_postfix(postfix, refresh=False)
                    if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
                        # Cumulative mean over the epoch so far.
                        cum_total = _nanmean(epoch_buckets["total"])
                        cum_eef = _nanmean(epoch_buckets["eef"])
                        cum_hand = _nanmean(epoch_buckets["hand"])
                        wloss_str = (
                            f"wloss={_nanmean(weighted_loss_bucket):.5f} "
                            if balanced_action_loss
                            else ""
                        )
                        accelerator.print(
                            f"[train] epoch={epoch} "
                            f"batch={batch_idx + 1}/{num_batches_per_epoch} "
                            f"loss={cum_total:.5f} "
                            f"{wloss_str}"
                            f"eef={cum_eef:.5f} "
                            f"hand={cum_hand:.5f} "
                            f"lr={lr_scheduler.get_last_lr()[0]:.3e}",
                            flush=True,
                        )
        pbar.close()
        # ----- End-of-epoch summary -----
        # Wall-clock time, mean losses, and learning rate.  This row is
        # always printed (independent of log_interval) so even with a
        # large dataset you get at least one line per epoch.
        epoch_elapsed = time.time() - epoch_start_time
        # Optional held-out validation pass.
        # ``nanmean`` so NaN buckets (= component not applicable for the
        # current ``hand_action_mode``) don't poison the averaged value.
        epoch_means = {k: _nanmean(epoch_buckets[k]) for k in _LOSS_KEYS}
        recon_means = {k: _nanmean(recon_buckets[k]) for k in _LOSS_KEYS}
        recon_clip_means = {k: _nanmean(recon_clip_buckets[k]) for k in _LOSS_KEYS}
        detailed_means = {k: _nanmean(v) for k, v in detailed_buckets.items()}
        log_payload = {
            "epoch loss": epoch_means["total"],
            "recon loss": recon_means["total"],
            "recon loss (clip)": recon_clip_means["total"],
            "epoch time (s)": float(epoch_elapsed),
            "lr": float(lr_scheduler.get_last_lr()[0]),
        }
        # The group-balanced scalar that actually drove back-prop (only when
        # enabled).  ``epoch loss`` above remains the plain full-tensor MSE so
        # its curve stays comparable to the unbalanced runs.
        if balanced_action_loss and weighted_loss_bucket:
            log_payload["epoch loss (weighted)"] = _nanmean(weighted_loss_bucket)
        for _n, _b in aux_loss_buckets.items():
            if _b:
                log_payload[f"aux head {_n} loss"] = _nanmean(_b)
        log_payload.update(
            _wandb_loss_items(epoch_means, hand_action_mode, stem="epoch loss")
        )
        log_payload.update(
            _wandb_detailed_items(detailed_means, stem="epoch loss")
        )
        log_payload.update(
            _wandb_loss_items(recon_means, hand_action_mode, stem="recon loss")
        )
        log_payload.update(
            _wandb_loss_items(
                recon_clip_means, hand_action_mode,
                stem="recon loss", tag_prefix="clip, ",
            )
        )
        # Per-embodiment component curves (co-training diagnostics):
        # ``epoch loss (robot, eef)`` / ``epoch loss (human, tactile)`` ...
        # Same key convention as the aggregate curves but with a ``"{emb}, "``
        # tag prefix, so robot / human live in disjoint metric families.
        emb_means_all: dict[str, dict[str, float]] = {}
        for emb in embodiment_names:
            emb_means = {
                k: _nanmean(emb_epoch_buckets[emb][k]) for k in _LOSS_KEYS
            }
            emb_means_all[emb] = emb_means
            log_payload[f"epoch loss ({emb}, total)"] = emb_means["total"]
            log_payload.update(
                _wandb_loss_items(
                    emb_means, hand_action_mode,
                    stem="epoch loss", tag_prefix=f"{emb}, ",
                )
            )
            if balanced_action_loss and emb_weighted_bucket[emb]:
                log_payload[f"epoch loss ({emb}, weighted)"] = _nanmean(
                    emb_weighted_bucket[emb]
                )
        if is_main:
            # Terminal summary reads from ``epoch_means`` (which always holds
            # every component) so it is independent of which curves we choose
            # to send to wandb.
            accelerator.print(
                f"[train] epoch={epoch} DONE  "
                f"loss={epoch_means['total']:.5f}  "
                f"recon={recon_means['total']:.5f} "
                f"(clip={recon_clip_means['total']:.5f})  "
                f"eef={epoch_means['eef']:.5f} "
                f"(x={epoch_means['eef_x']:.5f}, "
                f"y={epoch_means['eef_y']:.5f}, "
                f"z={epoch_means['eef_z']:.5f}, "
                f"rot6d={epoch_means['eef_rot6d']:.5f})  "
                f"{_hand_summary(epoch_means, hand_action_mode, prec=5)}  "
                f"lr={log_payload['lr']:.3e}  "
                f"time={epoch_elapsed:.1f}s",
                flush=True,
            )
            if balanced_action_loss and weighted_loss_bucket:
                accelerator.print(
                    f"[train] epoch={epoch} weighted(backprop) "
                    f"loss={_nanmean(weighted_loss_bucket):.5f} "
                    f"(groups reweighted to {action_loss_group_weights})",
                    flush=True,
                )
            for emb in embodiment_names:
                em = emb_means_all.get(emb)
                if em is None:
                    continue
                tac_str = (
                    f" tactile={em['tactile']:.5f}"
                    if em["tactile"] == em["tactile"]  # not NaN
                    else ""
                )
                wl_str = (
                    f" weighted={_nanmean(emb_weighted_bucket[emb]):.5f}"
                    if balanced_action_loss and emb_weighted_bucket[emb]
                    else ""
                )
                accelerator.print(
                    f"[train] epoch={epoch} [{emb}]  "
                    f"total={em['total']:.5f}  "
                    f"eef={em['eef']:.5f}  "
                    f"{_hand_summary(em, hand_action_mode, prec=5)}"
                    f"{tac_str}{wl_str}",
                    flush=True,
                )
        if val_sets and (epoch % int(val_cfg.eval_frequency) == 0):
            _evaluate_and_log_val_sets(
                accelerator,
                val_sets=val_sets,
                model=model,
                ema=ema,
                cfg=cfg,
                val_cfg=val_cfg,
                noise_scheduler=noise_scheduler,
                epoch=epoch,
                hand_action_mode=hand_action_mode,
                log_payload=log_payload,
                log_detailed=log_detailed_loss,
                tactile_key=tactile_key,
                tactile_action_dim=tactile_action_dim,
                num_arms=num_arms,
                camera_ids=camera_ids,
            )

        if accelerator.is_local_main_process and not cfg.debug:
            wandb.log(log_payload)
        # Periodic milestones, PLUS the very last epoch: with
        # ``epochs``/``ckpt_frequency`` = 350/50 the loop ends at epoch 349, which
        # is not a multiple of 50, so without this the final 49 epochs of
        # training were never written to disk.
        is_final_epoch = epoch == cfg.training.epochs - 1
        if (epoch % cfg.training.ckpt_frequency == 0 and epoch > 0) or is_final_epoch:
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                ckpt_model = accelerator.unwrap_model(model)
                accelerator.save(
                    ckpt_model.state_dict(),
                    os.path.join(ckpt_save_dir, f"epoch_{epoch}.ckpt"),
                )
                accelerator.print(f"Saved checkpoint at epoch {epoch}.")
                if cfg.training.use_ema:
                    accelerator.save(
                        ema_model.state_dict(),
                        os.path.join(ckpt_save_dir, f"ema_epoch_{epoch}.ckpt"),
                    )
                    accelerator.print(f"Saved ema checkpoint at epoch {epoch}.")
                # also save the state
                accelerator.save_state(
                    output_dir=os.path.join(state_save_dir, f"epoch_{epoch}")
                )
                accelerator.print(f"Saved state checkpoint at epoch {epoch}.")
                if is_final_epoch:
                    accelerator.print(
                        f"Training finished: final checkpoint is epoch_{epoch} "
                        f"({cfg.training.epochs} epochs completed)."
                    )


if __name__ == "__main__":
    train_diffusion_policy()
