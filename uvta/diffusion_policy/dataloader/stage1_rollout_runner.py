"""Run a FROZEN stage-1 policy over a teleop dataset and pack its predictions.

One source of truth for two consumers:

  * ``scripts/gen_stage1_rollout_cache.py`` -- writes ONE draw to disk, which
    ``Stage1RolloutTeleopDataset`` then reads back.
  * ``Stage1OnlineRolloutDataset`` -- keeps the runner alive and re-draws every
    epoch, so stage 2 never sees the same frozen prediction twice.

Both need the identical observation pipeline, rot6d decode and packing, so a
divergence between them would be an experiment-invalidating bug rather than a
cosmetic one.

Packing.  Per anchor ``t`` the row is ARM-CONTIGUOUS over stage 1's FULL
``pred_horizon`` (``H1``), not the shorter ``action_horizon`` it executes:

    [arm0: eef_rel(6) | joint(22) | tactile(F) , arm1: ... ]   x H1

``eef_rel`` and ``joint`` are PHYSICAL (needed for the SE(3) composition);
``tactile`` stays on the NETWORK scale [-1, 1] because ``clip_sample`` already
bounds it and stage 2 consumes it unchanged.

Noise.  The initial noise is a deterministic function of ``(seed, sample
index)`` and NOT of batch size, worker count or DDP sharding.  A draw is
therefore reproducible, and a sharded online refresh reproduces a
single-process offline cache: verified bit-exact at equal world size, and equal
to within float32 kernel non-determinism across world sizes (measured mean
2e-7 rad, 99% of values bit-identical, worst element 3e-4 rad -- four orders of
magnitude below the genuine draw-to-draw spread).
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from .uvta_dataset import UVTADataset
from .diffusion_bc_dataset import normalize_data, unnormalize_data
from .replay_buffer import UVTAReplayBuffer

_JOINT_DIM = 22
_EEF_LEGACY = 6  # xyz + rotvec
# Mixed into the per-sample noise seed so different draws of the same anchor are
# uncorrelated; any large odd number works.
_NOISE_SALT = 1_000_003


def build_stage1_obs_dataset(model_cfg, data_dir, max_episode, verbose=True,
                             allow_unseen=False):
    """Stage 1's own dataset, restricted to ONE data_dir and made deterministic.

    Only the observation side matters here, and it must match what stage 1 saw
    in training -- hence stage 1's config verbatim, with two changes:
      * ``data_dirs`` narrowed to the dataset we are predicting on (its
        embodiment is taken from stage 1's config so tactile baseline /
        per-embodiment normalization behave identically);
      * random image augmentation replaced by the deterministic Resize +
        CenterCrop that deploy uses, so one anchor has one prediction per draw.
    """
    ds_cfg = model_cfg.dataset
    dirs = [str(d) for d in ds_cfg.data_dirs]
    embs = [str(e) for e in ds_cfg.data_dirs_embodiment]
    target = os.path.normpath(str(data_dir))

    match = None
    for d, e in zip(dirs, embs):
        # stage 1's paths are relative to the DexUMI/ working dir ("../data/...")
        if os.path.normpath(d).endswith(target) or target.endswith(
            os.path.normpath(d).lstrip("./").lstrip("../")
        ):
            match = (d, e)
            break
    if match is None:
        if not allow_unseen:
            raise ValueError(
                f"{data_dir!r} is not one of stage 1's data_dirs {dirs}. "
                "Predicting on a dataset stage 1 never trained on would silently "
                "be out of distribution; pass one of the listed dirs, or set "
                "allow_unseen=True if that is the point (held-out evaluation)."
            )
        # Held-out evaluation: being out of distribution is the measurement, not
        # a mistake.  The embodiment has to be borrowed from a training dir so
        # the tactile baseline and per-embodiment normalization behave as they
        # did in training -- take the first dir sharing this one's embodiment
        # class, defaulting to the first entry.
        emb_guess = next(
            (e for d, e in zip(dirs, embs)
             if os.path.basename(os.path.normpath(d)).split("_")[0]
             == os.path.basename(target).split("_")[0]),
            embs[0],
        )
        match = (dirs[0], emb_guess)
        if verbose:
            print(
                f"[stage1-obs] {data_dir!r} is NOT a training dir -- evaluating "
                f"held out, borrowing embodiment {emb_guess!r}."
            )
    cfg_dir, emb = match
    resolved_dir = str(data_dir)
    if not os.path.isdir(resolved_dir):
        raise ValueError(f"data_dir {resolved_dir!r} does not exist")
    if verbose:
        print(
            f"[stage1-runner] observations from {resolved_dir!r} "
            f"(embodiment={emb!r}, stage-1 config entry {cfg_dir!r})"
        )

    kwargs = dict(
        data_dirs=[resolved_dir],
        data_dirs_embodiment=[emb],
        load_camera_ids=[int(c) for c in ds_cfg.load_camera_ids],
        camera_resize_shape=(
            list(ds_cfg.camera_resize_shape) if ds_cfg.camera_resize_shape else None
        ),
        pred_horizon=int(ds_cfg.pred_horizon),
        obs_horizon=int(ds_cfg.obs_horizon),
        action_horizon=int(ds_cfg.action_horizon),
        down_sample_steps=int(ds_cfg.down_sample_steps),
        unnormal_list=[str(u) for u in ds_cfg.unnormal_list],
        # DETERMINISTIC: no RandomCrop / ColorJitter / blur / grayscale.
        optional_transforms=["Resize", "CenterCrop"],
        max_episode=max_episode,
        replay_buffer_cls=UVTAReplayBuffer,
    )
    for k in (
        "relative_hand_action", "norm_method", "norm_clip_sigma",
        "norm_clip_percentile", "proprio_mode", "hand_action_mode",
        # The output-block flags MUST be forwarded: they decide the action
        # target's width and composition.  Omitting them let the rebuilt dataset
        # fall back to the all-True defaults, so a checkpoint trained on 31 or 51
        # dims was paired with an 82-dim target -- silently wrong for the cache
        # generator, and a shape error at best.
        "predict_action", "predict_state", "predict_tactile",
        # ``aux_as_head`` belongs with them: it decides whether the auxiliary
        # targets ride IN the action tensor or leave it for a regression head.
        # Drop it and a head-routed checkpoint (31-D action) gets paired with an
        # in-trajectory target (51-D or 82-D) -- a reshape error if you are lucky.
        "aux_as_head",
        "predict_future_tactile", "action_from_next_state", "enable_fsr",
        "fsr_source_key", "fsr_binarize", "fsr_binary_cutoff",
        "fsr_baseline_correct", "fsr_baseline_frames",
        "fsr_baseline_embodiments", "fsr_baseline_first_frame_embodiments",
        "bgr2rgb", "share_joint_stats", "arms",
    ):
        if k in ds_cfg:
            v = ds_cfg[k]
            kwargs[k] = list(v) if hasattr(v, "__iter__") and not isinstance(v, str) else v
    # Single embodiment here -> per-embodiment normalization would recompute the
    # stats from this dir alone.  Every stat is overwritten with stage 1's saved
    # ones right after construction anyway, so force it off.
    kwargs["norm_per_embodiment"] = False
    if "ring_overlay" in ds_cfg:
        ro = ds_cfg.ring_overlay
        kwargs["ring_overlay"] = {
            "enabled": bool(ro.get("enabled", False)),
            "embodiments": [str(x) for x in ro.get("embodiments", [])],
            "mask_path": ro.get("mask_path", None),
            "rgb_path": ro.get("rgb_path", None),
            "bgr2rgb_assets": bool(ro.get("bgr2rgb_assets", True)),
        }
    return UVTADataset(**kwargs)


def apply_saved_stats(dataset, saved_stats, emb_hint="robot", verbose=True):
    """Re-normalize the buffer with STAGE 1'S SAVED stats.

    The dataset just normalized every stream with statistics recomputed from
    this single data_dir.  Stage 1, however, was trained with per-embodiment
    stats pooled over its full co-training mix, and its network only makes sense
    on inputs normalized that way.  So undo the local normalization and redo it
    with the saved stats.
    """
    per_emb = saved_stats.get("_per_embodiment", None)

    def pick(key):
        if per_emb and emb_hint in per_emb and key in per_emb[emb_hint]:
            return per_emb[emb_hint][key]
        return saved_stats.get(key)

    fixed, skipped = [], []
    for key, local in list(dataset.stats.items()):
        if key.startswith("_"):
            continue
        want = pick(key)
        if want is None:
            skipped.append(key)
            continue
        dataset.stats[key] = want
        buf = dataset.buffer.memory_buffer
        if key in buf and key not in dataset.unnormal_list:
            phys = unnormalize_data(np.asarray(buf[key], dtype=np.float32), local)
            buf[key] = normalize_data(phys, want).astype(np.float32)
            fixed.append(key)
    if verbose:
        print(f"[stage1-runner] re-normalized with stage-1 stats: {fixed}")
        if skipped:
            print(f"[stage1-runner] no saved stat for (left as-is): {skipped}")


def anchor_frames(dataset):
    """Absolute buffer frame index of each sample's anchor ``t``.

    Sample position ``j`` maps to buffer position
    ``buffer_start_idx + (j - sample_start_idx)``, so the anchor at sample
    position ``_anchor_index_in_sample`` sits at
    ``buffer_start_idx - sample_start_idx + anchor``.
    """
    idx = np.asarray(dataset.indices, dtype=np.int64)
    return idx[:, 0] - idx[:, 2] + int(dataset._anchor_index_in_sample)


def verify_anchor_mapping(dataset, t_abs, n=64, verbose=True):
    """Assert the derived anchor really is the sample's current-time frame.

    Compares the anchor pose taken straight from the buffer against the pose the
    sample slice puts at the anchor position, for a spread of samples.
    """
    from .diffusion_bc_dataset import sample_sequence

    prefix = dataset.arm_prefixes[0]
    pose = dataset.buffer.memory_buffer[f"{prefix}pose"]
    picks = np.linspace(0, len(dataset) - 1, min(n, len(dataset)), dtype=np.int64)
    worst = 0.0
    for i in picks:
        bs, be, ss, se = dataset.indices[i]
        ns = sample_sequence(
            {f"{prefix}pose": pose}, dataset._sample_length, bs, be, ss, se
        )
        got = ns[f"{prefix}pose"][dataset._anchor_index_in_sample]
        want = pose[t_abs[i]]
        worst = max(worst, float(np.abs(got - want).max()))
    if worst > 1e-6:
        raise RuntimeError(
            f"anchor mapping is wrong (max pose mismatch {worst:.3e}); refusing "
            "to produce predictions that would be misaligned with the episodes."
        )
    if verbose:
        print(
            f"[stage1-runner] anchor mapping verified on {len(picks)} samples "
            f"(err {worst:.1e})"
        )


def sample_noise(sample_indices, horizon, action_dim, seed):
    """Initial diffusion noise as a pure function of ``(seed, sample index)``.

    Seeding per sample rather than once per pass is what makes a draw invariant
    to batch size, worker count and DDP sharding: rank 3's slice of an 8-way
    refresh gets byte-identical noise to the same anchors in a single-process
    offline run, so the two pipelines can be diffed directly.
    """
    g = torch.Generator()
    out = torch.empty(len(sample_indices), horizon, action_dim)
    for row, idx in enumerate(sample_indices):
        g.manual_seed((int(seed) * _NOISE_SALT + int(idx)) % (2**63 - 1))
        out[row] = torch.randn(horizon, action_dim, generator=g)
    return out


class Stage1Runner:
    """A frozen stage 1 plus the observation stream it predicts on.

    Construction is the expensive part (model load + decoding every episode's
    images into RAM); ``predict`` is then cheap enough to call once per training
    epoch.
    """

    def __init__(
        self,
        stage1_model_path,
        stage1_ckpt,
        data_dir,
        max_episode=None,
        use_ema=None,
        device=None,
        batch_size=256,
        num_workers=8,
        verbose=True,
    ):
        from uvta.common.utility.file import read_pickle
        from uvta.common.utility.model import load_config, load_diffusion_model
        from uvta.real_env.real_policy import RealPolicy

        self.verbose = bool(verbose)
        self.stage1_model_path = os.path.abspath(str(stage1_model_path))
        self.stage1_ckpt = int(stage1_ckpt)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.cfg = load_config(self.stage1_model_path)
        self.use_ema = (
            bool(self.cfg.training.use_ema) if use_ema is None else bool(use_ema)
        )
        self.model, self.noise_scheduler = load_diffusion_model(
            self.stage1_model_path, self.stage1_ckpt, use_ema=self.use_ema
        )
        self.model.eval().to(self.device)

        # RealPolicy gives the exact deploy-side action-stat assembly and rot6d
        # decode, so a prediction here matches what the rollout would compute.
        self.policy = RealPolicy(
            self.stage1_model_path, self.stage1_ckpt, use_ema=self.use_ema
        )
        if not self.policy.predict_future_tactile:
            raise ValueError(
                "stage 1 must be trained with predict_future_tactile=True -- "
                "stage 2 needs its predicted tactile as input."
            )
        self.action_stats = self.policy.stats["action"]
        self.tactile_dim = int(self.policy.tactile_action_dim)     # all arms
        self.per_arm_tactile_dim = int(self.policy.per_arm_tactile_dim)
        self.num_arms = int(self.policy.num_arms)
        self.camera_ids = list(self.policy.camera_ids)

        saved_stats = read_pickle(
            os.path.join(self.stage1_model_path, "stats.pickle")
        )
        self.dataset = build_stage1_obs_dataset(
            self.cfg, data_dir, max_episode, verbose=self.verbose
        )
        apply_saved_stats(self.dataset, saved_stats, verbose=self.verbose)
        self.t_abs = anchor_frames(self.dataset)
        verify_anchor_mapping(self.dataset, self.t_abs, verbose=self.verbose)

        self.H1 = int(self.cfg.dataset.pred_horizon)
        self.action_dim = int(np.asarray(self.action_stats["min"]).size)
        self.tactile_key = str(self.cfg.dataset.get("fsr_source_key", "fsr"))
        self.per_arm_step = _EEF_LEGACY + _JOINT_DIM + self.per_arm_tactile_dim
        self.per_step = self.per_arm_step * self.num_arms

        self.eps_end = np.asarray(self.dataset.buffer.eps_end, dtype=np.int64)
        self.eps_start = np.concatenate([[0], self.eps_end[:-1]])
        self.episode_names = list(self.dataset.buffer.episode_names)
        self.num_frames = int(self.eps_end[-1])

        if self.verbose:
            print(
                f"[stage1-runner] samples={len(self.dataset)}  "
                f"episodes={len(self.episode_names)}  frames={self.num_frames}  "
                f"H1={self.H1}  arms={self.num_arms}  "
                f"tactile total={self.tactile_dim} per_arm={self.per_arm_tactile_dim}"
            )

    # ------------------------------------------------------------------

    def meta(self, seed):
        """Provenance recorded alongside a draw; validated by the consumers."""
        return {
            "stage1_model_path": self.stage1_model_path,
            "stage1_ckpt": self.stage1_ckpt,
            "use_ema": self.use_ema,
            "H1": self.H1,
            "eef_dim": _EEF_LEGACY,
            "joint_dim": _JOINT_DIM,
            # PER ARM; the packed row is arm-contiguous.
            "tactile_dim": self.per_arm_tactile_dim,
            "num_arms": self.num_arms,
            "arms": [
                str(a) for a in (getattr(self.cfg.dataset, "arms", None) or [""])
            ],
            "per_arm_step": self.per_arm_step,
            "tactile_key": self.tactile_key,
            # Tactile is on the NETWORK scale; consumers must not re-normalize.
            "tactile_normalized": True,
            "action_horizon": int(self.cfg.dataset.action_horizon),
            "per_step": self.per_step,
            "num_inference_steps": int(self.cfg.num_inference_steps),
            "seed": int(seed),
        }

    @torch.no_grad()
    def predict(self, seed, sample_indices=None):
        """One draw over ``sample_indices`` (default: every sample).

        Returns ``(len(sample_indices), H1 * per_step)`` float32, rows in the
        order of ``sample_indices``.
        """
        ds = self.dataset
        if sample_indices is None:
            sample_indices = np.arange(len(ds), dtype=np.int64)
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        subset = torch.utils.data.Subset(ds, sample_indices.tolist())
        loader = torch.utils.data.DataLoader(
            subset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=False, drop_last=False,
        )

        motor_out, tac_out = [], []
        t0, done, cursor = time.time(), 0, 0
        for batch in loader:
            B = batch["action"].shape[0]
            vis = torch.cat(
                [batch[f"camera_{c}"].float().to(self.device) for c in self.camera_ids],
                dim=1,
            ) if self.camera_ids else None
            prop = (
                batch["proprioception"].float().to(self.device)
                if "proprioception" in batch else None
            )
            fsr = (
                batch[self.tactile_key].float().to(self.device)
                if self.tactile_key in batch else None
            )
            traj = sample_noise(
                sample_indices[cursor:cursor + B], self.H1, self.action_dim, seed
            ).to(self.device)
            cursor += B
            traj = self.model.inference(
                proprioception=prop, fsr=fsr, visual_obs=vis, trajectory=traj,
                noise_scheduler=self.noise_scheduler,
                num_inference_steps=int(self.cfg.num_inference_steps),
            )
            naction = traj.detach().cpu().numpy()
            act = unnormalize_data(
                naction.reshape(-1, self.action_dim), self.action_stats
            ).reshape(B, self.H1, self.action_dim)
            if self.tactile_dim > 0:
                # Tactile stays on the NETWORK scale -> take it from ``naction``.
                tac_out.append(naction[:, :, -self.tactile_dim:].astype(np.float32))
                act = act[..., :-self.tactile_dim]
            motor_out.append(act.astype(np.float32))
            done += B
            if self.verbose and done % (self.batch_size * 20) < self.batch_size:
                el = time.time() - t0
                print(
                    f"[stage1-runner]   {done}/{len(subset)}  {el:.0f}s  "
                    f"eta {el / max(done, 1) * (len(subset) - done):.0f}s",
                    flush=True,
                )
        motor = np.concatenate(motor_out, axis=0)
        tac = np.concatenate(tac_out, axis=0) if tac_out else None
        return self._pack(motor, tac)

    def _pack(self, motor, tac):
        """``(N,H1,motor_dim)`` + ``(N,H1,F)`` -> flat arm-contiguous rows."""
        N = motor.shape[0]
        # Decode the whole (N*H1) stack in ONE call: _decode_arm_block is
        # vectorized over its leading axis, and 140k per-sample scipy calls per
        # epoch would otherwise cost more than the diffusion itself.
        dec = self.policy._decode_action_to_legacy(
            motor.reshape(N * self.H1, -1)
        ).reshape(N, self.H1, -1)
        if dec.shape[-1] % self.num_arms != 0:
            raise ValueError(
                f"decoded width {dec.shape[-1]} not divisible by "
                f"num_arms={self.num_arms}"
            )
        per = dec.shape[-1] // self.num_arms
        F = self.per_arm_tactile_dim
        blocks = []
        for a in range(self.num_arms):
            blk = dec[..., a * per:(a + 1) * per]
            parts = [
                blk[..., :_EEF_LEGACY],
                blk[..., _EEF_LEGACY:_EEF_LEGACY + _JOINT_DIM],
            ]
            if tac is not None:
                parts.append(tac[..., a * F:(a + 1) * F])
            blocks.append(np.concatenate(parts, axis=-1))
        return np.concatenate(blocks, axis=-1).reshape(
            N, self.H1 * self.per_step
        ).astype(np.float32)

    def scatter_to_episodes(self, packed):
        """Anchor-ordered rows -> one ``(T, H1*per_step)`` array per episode.

        Anchors are dense over an episode except for the tail frames that have
        no sample; those repeat the last prediction, matching the dataset's own
        last-frame-repeat padding.
        """
        out = []
        for s, e, name in zip(self.eps_start, self.eps_end, self.episode_names):
            T = int(e - s)
            sel = np.where((self.t_abs >= s) & (self.t_abs < e))[0]
            local = self.t_abs[sel] - s
            arr = np.zeros((T, packed.shape[1]), dtype=np.float32)
            seen = np.zeros(T, dtype=bool)
            arr[local] = packed[sel]
            seen[local] = True
            if not seen.all():
                last = int(np.where(seen)[0].max())
                arr[~seen] = arr[last]
                if self.verbose:
                    print(
                        f"[stage1-runner]   {name}: {int((~seen).sum())}/{T} "
                        f"tail frames padded from frame {last}"
                    )
            out.append(arr)
        return out

    def scatter_to_frames(self, packed):
        """Same as ``scatter_to_episodes`` but concatenated to ``(N, W)``.

        ``N`` is the buffer's total frame count, so the result is indexed by the
        SAME absolute frame ``t`` the stage-2 dataset uses.
        """
        return np.concatenate(self.scatter_to_episodes(packed), axis=0)
