import random

import numpy as np
import torch
from torchvision.transforms import v2

from .replay_buffer import ReplayBuffer

# Sentinel for "this knob was not supplied, inherit the general one".  Needed
# because ``None`` is itself a meaningful value (= absolute min/max).
_INHERIT = "__inherit__"

normalize_threshold = 5e-2
# Below this absolute range a dimension is treated as constant and left
# untouched to avoid divide-by-zero.  This is NOT a "skip if small" knob --
# it only guards genuinely constant channels (e.g. rot6d identity columns,
# which have range exactly 0).
normalize_eps = 1e-8


def _should_normalize_dim(stats, i):
    """Decide whether dimension ``i`` should be normalized.

    Scheme A (explicit mask):  when ``stats`` carries an ``"identity_mask"``
    (a per-dim boolean array), a dimension is normalized iff it is NOT
    flagged identity AND its spread is non-degenerate.  For min-max the spread
    is ``max-min``; for mean-std it is ``std``.  This makes "skip" an explicit
    decision (only rot6d identity channels are flagged) instead of silently
    dropping any dim whose range happens to be below ``normalize_threshold``.

    Legacy fallback:  when no mask is present (old stats.pickle), fall back to
    the historical range-threshold behaviour so previously-trained
    checkpoints keep loading unchanged.
    """
    mask = stats.get("identity_mask", None)
    if stats.get("method", "minmax") == "meanstd":
        spread = stats["std"][i]
        if mask is not None:
            return (not bool(mask[i])) and (spread > normalize_eps)
        return spread > normalize_eps
    rng = stats["max"][i] - stats["min"][i]
    if mask is not None:
        return (not bool(mask[i])) and (rng > normalize_eps)
    # Legacy threshold behaviour.
    return rng > normalize_threshold


def process_image(image, optional_transforms=[], resize_shape=(240, 240)):
    if isinstance(image, np.ndarray):
        if image.ndim not in [3, 4]:
            raise ValueError("Image must be 3D (H,W,C) or 4D (N,H,W,C)")
        image = torch.from_numpy(image)
        if image.ndim == 3:
            image = image.permute(2, 0, 1)  # (C,H,W)
        else:
            image = image.permute(0, 3, 1, 2)  # (N,C,H,W)

    transform_list = [
        v2.ToDtype(torch.float32, scale=True),  # convert to float32 and scale to [0,1]
    ]

    for transform_name in optional_transforms:
        if transform_name == "Resize":
            transform_list.append(v2.Resize(resize_shape))
        if transform_name == "RandomCrop":
            transform_list.append(v2.RandomCrop((224, 224)))
        elif transform_name == "CenterCrop":
            transform_list.append(v2.CenterCrop((224, 224)))

        if transform_name == "GaussianBlur":
            transform_list.append(
                v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.2)
            )
        if transform_name == "ColorJitter":
            transform_list.append(
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=0.64, contrast=0.32, saturation=0.32, hue=0.08
                        )
                    ],
                    p=0.8,
                )
            )
        if transform_name == "RandomGrayscale":
            transform_list.append(v2.RandomGrayscale(p=0.2))

    transform_list.append(
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    transforms = v2.Compose(transform_list)
    return transforms(image)


def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    indices = np.array(indices)
    return indices


def sample_sequence(
    train_data,
    sequence_length,
    buffer_start_idx,
    buffer_end_idx,
    sample_start_idx,
    sample_end_idx,
):
    """Slice + pad each key in ``train_data`` to length ``sequence_length``.

    Both branches return a *fresh* (owned) array so the result is safe to
    pass to ``__getitem__`` / DataLoader collate without risk of
    corrupting the underlying replay buffer.  The padded branch already
    allocates a new array; the no-pad ("fast") branch explicitly copies
    the slice instead of returning a view, which (a) protects the buffer
    from accidental in-place writes downstream and (b) avoids PyTorch's
    "given NumPy array is not writable" warning during ``torch.as_tensor``
    inside the DataLoader collate.

    The copy is cheap in practice: dataset-level transforms (image
    transforms, advanced-index gathers, ``np.concatenate``) already
    allocate fresh arrays before the result hits collate, so the only
    extra memory traffic introduced here is for the small non-image keys
    (``pose``, ``hand_action``, ``proprioception`` etc., all
    ``O(sample_length, <small dim>)``).  The image array, where copy cost
    would matter, is replaced wholesale by ``process_image`` downstream
    and never reaches collate as a numpy view anyway.
    """
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        else:
            data = sample.copy()
        result[key] = data
    return result


# normalize data
def get_data_stats(data, clip_percentile=None, method="minmax", clip_sigma=None):
    """Compute per-dim normalization stats for one data stream.

    ``method``
      * ``"minmax"`` (default): range-normalize each dim to [-1, 1] using
        min/max.  See ``clip_percentile``.
      * ``"meanstd"``: standardize each dim with ``(x - mean) / std`` (z-score).
        Output is unbounded; pass ``clip_sigma`` to clamp it to ``±clip_sigma``
        so it stays compatible with ``clip_sample=True``.

    ``clip_percentile`` (min-max only) makes the min/max robust to outliers:
      * ``None`` / ``0`` (default): absolute min/max (legacy; every value lands
        inside [-1, 1] exactly).
      * ``p`` in ``(0, 50)``: use the ``p`` / ``100-p`` percentiles as min/max;
        values beyond the band are clamped to ±1 by ``normalize_data``.

    ``clip_sigma`` (mean-std only): clamp the standardized value to
    ``[-clip_sigma, +clip_sigma]`` (e.g. 3 -> ±3σ) and rescale by
    ``1/clip_sigma`` so the kept range maps onto [-1, 1] (keeps the diffusion
    target inside the ``clip_sample`` range).  ``None`` -> no clamp/rescale
    (raw z-score; only safe with ``clip_sample=False``).

    Sparse channels and ``clip_percentile``: a channel that is 0 for more than
    ``(100 - clip_percentile)``% of frames has BOTH percentiles at 0, so its band
    collapses and ``_should_normalize_dim`` skips it -- its rare spikes then reach
    the network in RAW units.  There used to be an automatic re-base onto the
    absolute min/max for exactly this case; it was removed once tactile moved to
    ``fsr_norm_clip_percentile: null`` (absolute min/max, where the situation
    cannot arise).  The case is now only WARNED about, so a config that puts a
    sparse stream back on percentile clipping is loud instead of silent.

    ALL of ``min`` / ``max`` / ``mean`` / ``std`` are stored regardless of
    method (cheap, and lets diagnostics / ``_should_normalize_dim`` work
    uniformly).  ``stats["method"]`` records which transform to apply; the
    legacy dict (no ``method`` key) is treated as ``"minmax"``.
    """
    data = data.reshape(-1, data.shape[-1])
    mean = np.mean(data, axis=0).astype(data.dtype)
    std = np.std(data, axis=0).astype(data.dtype)
    if method not in ("minmax", "meanstd"):
        raise ValueError(f"norm method must be 'minmax' or 'meanstd'; got {method!r}")

    if method == "minmax":
        abs_mn = np.min(data, axis=0).astype(data.dtype)
        abs_mx = np.max(data, axis=0).astype(data.dtype)
        if clip_percentile is None or clip_percentile <= 0:
            mn = abs_mn
            mx = abs_mx
            clip_p = 0.0
        else:
            p = float(clip_percentile)
            if not (0.0 < p < 50.0):
                raise ValueError(
                    f"clip_percentile must be in (0, 50); got {clip_percentile!r}"
                )
            mn = np.percentile(data, p, axis=0).astype(data.dtype)
            mx = np.percentile(data, 100.0 - p, axis=0).astype(data.dtype)
            clip_p = p
            # Sparse channels whose percentile band collapsed even though they
            # DO have a real range.  These get skipped by normalize_data and
            # their spikes reach the network raw.  Nothing is silently rewritten
            # any more -- say so loudly and let the caller pick a clip_percentile
            # (or ``null``) that suits the stream.
            sparse = ((mx - mn) <= normalize_eps) & (
                (abs_mx - abs_mn) > normalize_eps
            )
            if np.any(sparse):
                idx = np.where(sparse)[0]
                print(
                    f"[get_data_stats] WARNING: clip_percentile={p} collapses "
                    f"channels {idx.tolist()} (too sparse: both percentiles are "
                    f"equal) while their absolute range is up to "
                    f"{float((abs_mx - abs_mn)[idx].max()):.4g}.  They will NOT "
                    "be normalized and their values will reach the network in raw "
                    "units.  Use clip_percentile=null (absolute min/max) for this "
                    "stream, e.g. fsr_norm_clip_percentile for tactile."
                )
    else:
        # meanstd: still record min/max for provenance / diagnostics.
        mn = np.min(data, axis=0)
        mx = np.max(data, axis=0)
        clip_p = 0.0

    cs = None if clip_sigma in (None, 0) else float(clip_sigma)
    if cs is not None and cs <= 0:
        raise ValueError(f"clip_sigma must be > 0; got {clip_sigma!r}")

    stats = {
        "min": mn,
        "max": mx,
        "mean": mean,
        "std": std,
        "identity_mask": np.zeros(mn.shape[0], dtype=bool),
        "clip_percentile": clip_p,
        "method": method,
        "clip_sigma": cs,
    }
    return stats


def normalize_data(data, stats):
    """Normalize ``data`` (N, D) per-dim according to ``stats['method']``.

    minmax  -> ``(x-min)/(max-min)*2 - 1``, then clamp to [-1, 1]
               (clamp only bites when min/max are percentile-clipped).
    meanstd -> ``(x-mean)/std``; if ``clip_sigma`` set, divide by clip_sigma
               and clamp to [-1, 1] so the kept ±σ band maps onto [-1, 1].
    """
    method = stats.get("method", "minmax")
    ndata = data.astype(np.float32, copy=True)
    for i in range(ndata.shape[1]):
        if not _should_normalize_dim(stats, i):
            continue
        if method == "meanstd":
            ndata[:, i] = (data[:, i] - stats["mean"][i]) / stats["std"][i]
            cs = stats.get("clip_sigma", None)
            if cs:
                ndata[:, i] = ndata[:, i] / cs
                np.clip(ndata[:, i], -1.0, 1.0, out=ndata[:, i])
        else:
            ndata[:, i] = (data[:, i] - stats["min"][i]) / (
                stats["max"][i] - stats["min"][i]
            )
            ndata[:, i] = ndata[:, i] * 2 - 1
            np.clip(ndata[:, i], -1.0, 1.0, out=ndata[:, i])
    return ndata


def degenerate_dims(stats):
    """Indices of the channels ``normalize_data`` will pass through untouched.

    These were constant in training, so their values are NOT rescaled and reach
    the network in raw units.  Used for diagnostics only -- nothing clamps them.
    """
    n = int(np.asarray(stats["min"]).size)
    return [i for i in range(n) if not _should_normalize_dim(stats, i)]


def unnormalize_data(ndata, stats):
    """Invert ``normalize_data`` according to ``stats['method']``.

    NOTE: the forward transform may clamp (percentile-clip or ±clip_sigma),
    so values that were clamped are not recoverable -- the inverse maps them
    to the band edge.  This matches the legacy behaviour for min-max.
    """
    method = stats.get("method", "minmax")
    # Work on a copy so the caller's input array is never mutated in-place.
    data = ndata.copy()
    for i in range(data.shape[1]):
        if not _should_normalize_dim(stats, i):
            continue
        if method == "meanstd":
            z = data[:, i]
            cs = stats.get("clip_sigma", None)
            if cs:
                z = z * cs
            data[:, i] = z * stats["std"][i] + stats["mean"][i]
        else:
            tmp = (data[:, i] + 1) / 2
            data[:, i] = tmp * (stats["max"][i] - stats["min"][i]) + stats["min"][i]
    return data


class DiffusionBCDataset(torch.utils.data.Dataset):
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
        replay_buffer_cls=ReplayBuffer,
        norm_clip_percentile=None,
        fsr_norm_clip_percentile=_INHERIT,
        norm_method="minmax",
        norm_clip_sigma=None,
        norm_per_embodiment=False,
        stats_primary_embodiment="robot",
        share_joint_stats=False,
        **replay_buffer_kwargs,
    ):
        # Normalization config (see ``get_data_stats``):
        #   norm_method        : "minmax" (default) | "meanstd"
        #   norm_clip_percentile: min-max outlier clipping (None/0 = absolute)
        #   norm_clip_sigma    : mean-std ±sigma clamp (None = raw z-score)
        #   norm_per_embodiment: when True (co-training across embodiments),
        #     stats are computed PER embodiment and every frame is normalized
        #     using its OWN embodiment's stats (so human / robot each get their
        #     own [-1,1] mapping).  The top-level ``self.stats[key]`` is filled
        #     from ``stats_primary_embodiment`` (default ``robot``) so the
        #     inference side -- which deploys on the robot -- un-normalizes
        #     with the robot stats.  Per-embodiment stats are also kept under
        #     ``self.stats['_per_embodiment'][emb][key]``.
        #   stats_primary_embodiment: which embodiment's stats populate the
        #     top-level keys (used by inference).  Only meaningful when
        #     ``norm_per_embodiment`` is True.
        #   fsr_norm_clip_percentile: TACTILE-ONLY override of
        #     ``norm_clip_percentile``.  Omit the key to inherit the global
        #     value; set it (including to ``null``) to give the tactile stream a
        #     different one.  Tactile is the one stream where the global 1% is a
        #     poor fit: the robot distribution is mostly zeros with rare spikes
        #     three orders of magnitude higher, so P99 lands far below the real
        #     peak and contact frames saturate at +1, throwing away force
        #     resolution.  ``null`` uses the absolute min/max instead, which
        #     keeps every value inside [-1,1] at the cost of letting the single
        #     largest spike define the top of the range.
        self.norm_method = norm_method
        self.norm_clip_percentile = norm_clip_percentile
        self.fsr_norm_clip_percentile = (
            norm_clip_percentile
            if isinstance(fsr_norm_clip_percentile, str)
            and fsr_norm_clip_percentile == _INHERIT
            else fsr_norm_clip_percentile
        )
        self.norm_clip_sigma = norm_clip_sigma
        self.norm_per_embodiment = bool(norm_per_embodiment)
        self.stats_primary_embodiment = str(stats_primary_embodiment)
        # When True, the obs-side joint-angle stream (``proprioception``) and
        # the action-side absolute joint-angle stream (``hand_action``) share
        # ONE set of normalization stats (min/max computed on the pooled
        # frames of both streams), so the same joint angle maps to the same
        # normalized value whether it appears as an observation or an action
        # target.  Under ``norm_per_embodiment`` the pooling is done PER
        # embodiment.  Only meaningful when ``hand_action`` is a normalized
        # buffer stream (i.e. NOT relative_hand_action, whose action stats are
        # computed separately in UVTADataset).
        self.share_joint_stats = bool(share_joint_stats)
        self.buffer = replay_buffer_cls(
            data_dirs,
            load_camera_ids,
            camera_resize_shape,
            max_episode=max_episode,
            **replay_buffer_kwargs,
        )
        self.seed = seed
        self.set_seed(self.seed)
        self.unnormal_list = unnormal_list
        episode_ends = self.buffer.eps_end
        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        # compute statistics and normalize data to [-1, 1]
        stats = dict()
        norm_kw = dict(
            clip_percentile=self.norm_clip_percentile,
            method=self.norm_method,
            clip_sigma=self.norm_clip_sigma,
        )
        # Tactile gets its own clip_percentile (see fsr_norm_clip_percentile).
        # ``_tactile_keys`` is the per-arm set of buffer keys it applies to.
        self._tactile_keys = self._resolve_tactile_keys()
        fsr_norm_kw = dict(norm_kw, clip_percentile=self.fsr_norm_clip_percentile)
        if self._tactile_keys and (
            self.fsr_norm_clip_percentile != self.norm_clip_percentile
        ):
            print(
                f"[norm] tactile {sorted(self._tactile_keys)} use "
                f"clip_percentile={self.fsr_norm_clip_percentile} "
                f"(everything else: {self.norm_clip_percentile})"
            )

        def kw_for(key):
            return fsr_norm_kw if key in self._tactile_keys else norm_kw
        # ------------------------------------------------------------------
        # Joint-angle stat sharing.
        #
        # When ``share_joint_stats`` is on, the obs-side joint stream
        # (``proprioception``) and the action-side ABSOLUTE joint stream
        # (``hand_action``) are normalized with ONE shared set of min/max
        # (computed on the pooled frames of both), so an identical joint angle
        # maps to the same normalized value in the observation and in the
        # action target.  Only applies when BOTH streams are present, share
        # the same feature dim, and are normalized buffer streams (i.e. NOT
        # ``relative_hand_action``, which is unnormalized in the buffer and
        # whose action stats are computed separately in UVTADataset).
        shared_joint_groups = self._resolve_shared_joint_key_groups()
        # Flat set of every key covered by a shared group -> skipped by the
        # generic per-key normalization loop below.
        shared_joint_keys = {k for g in shared_joint_groups for k in g}
        # ------------------------------------------------------------------
        # Stat ALIASES: ``(source, alias)`` pairs where ``alias`` is normalized
        # with ``source``'s statistics and contributes NOTHING to them.  Needed
        # for streams that must land on an existing scale without perturbing it:
        # pooling them in would change the mixture the percentiles are taken
        # over, which would silently move the other stream's normalization and
        # ruin any ablation that toggles the alias on and off.
        # ------------------------------------------------------------------
        stat_aliases = self._resolve_stat_alias_pairs()
        alias_keys = {a for _, a in stat_aliases}

        if not self.norm_per_embodiment:
            # ---- global (single-embodiment / legacy) normalization -------
            for group in shared_joint_groups:
                pooled = np.concatenate(
                    [self.buffer.memory_buffer[k] for k in group],
                    axis=0,
                )
                shared_stats = get_data_stats(pooled, **norm_kw)
                for k in group:
                    stats[k] = shared_stats
                    self.buffer.memory_buffer[k] = normalize_data(
                        self.buffer.memory_buffer[k], shared_stats
                    )
                print(
                    f"[norm] sharing joint stats across {group} (pooled min/max)"
                )
            for key, data in self.buffer.memory_buffer.items():
                if (
                    key in self.unnormal_list
                    or key in shared_joint_keys
                    or key in alias_keys
                ):
                    continue
                stats[key] = get_data_stats(data, **kw_for(key))
                self.buffer.memory_buffer[key] = normalize_data(data, stats[key])
            for src, alias in stat_aliases:
                stats[alias] = stats[src]
                self.buffer.memory_buffer[alias] = normalize_data(
                    self.buffer.memory_buffer[alias], stats[src]
                )
                print(f"[norm] {alias} normalized with {src}'s stats (alias)")
        else:
            # ---- per-embodiment normalization ----------------------------
            # Each frame is normalized with the stats of the embodiment that
            # produced it.  Top-level ``stats[key]`` is filled from
            # ``stats_primary_embodiment`` (default 'robot') so inference
            # un-normalizes with the robot stats; the full per-embodiment
            # breakdown is preserved under ``stats['_per_embodiment']``.
            frame_emb = self.buffer.frame_embodiment_labels()
            embodiments = sorted(set(frame_emb.tolist()))
            per_emb_stats: dict[str, dict] = {e: {} for e in embodiments}
            # boolean masks per embodiment over the frame axis.
            emb_masks = {e: (frame_emb == e) for e in embodiments}

            primary = self.stats_primary_embodiment
            if primary not in per_emb_stats:
                # Fall back to the first present embodiment so single-embodiment
                # runs (or a missing 'robot') still populate top-level stats.
                fallback = embodiments[0]
                print(
                    f"[norm] stats_primary_embodiment={primary!r} not present "
                    f"among {embodiments}; falling back to {fallback!r} for "
                    "top-level stats."
                )
                primary = fallback

            # First, the shared joint group (if any): pool both streams' frames
            # of each embodiment, compute one stat per embodiment, apply to
            # both streams.  Done before the generic loop so the loop can skip
            # these keys.
            for group in shared_joint_groups:
                for emb in embodiments:
                    mask = emb_masks[emb]
                    pooled = np.concatenate(
                        [self.buffer.memory_buffer[k][mask] for k in group],
                        axis=0,
                    )
                    emb_stats = get_data_stats(pooled, **norm_kw)
                    for k in group:
                        per_emb_stats[emb][k] = emb_stats
                for k in group:
                    data = self.buffer.memory_buffer[k]
                    normalized = data.astype(np.float32, copy=True)
                    for emb in embodiments:
                        normalized[emb_masks[emb]] = normalize_data(
                            data[emb_masks[emb]], per_emb_stats[emb][k]
                        )
                    self.buffer.memory_buffer[k] = normalized
                    stats[k] = per_emb_stats[primary][k]
                print(
                    f"[norm] sharing joint stats across {group} "
                    "(pooled per embodiment)"
                )

            for key, data in self.buffer.memory_buffer.items():
                if (
                    key in self.unnormal_list
                    or key in shared_joint_keys
                    or key in alias_keys
                ):
                    continue
                # Compute stats per embodiment on that embodiment's frames.
                normalized = data.astype(np.float32, copy=True)
                for emb in embodiments:
                    mask = emb_masks[emb]
                    emb_stats = get_data_stats(data[mask], **kw_for(key))
                    per_emb_stats[emb][key] = emb_stats
                    normalized[mask] = normalize_data(data[mask], emb_stats)
                self.buffer.memory_buffer[key] = normalized
                # Top-level stats key = the primary embodiment's stats.
                stats[key] = per_emb_stats[primary][key]
            # Aliases borrow the source's per-embodiment stats verbatim.
            for src, alias in stat_aliases:
                data = self.buffer.memory_buffer[alias]
                normalized = data.astype(np.float32, copy=True)
                for emb in embodiments:
                    mask = emb_masks[emb]
                    per_emb_stats[emb][alias] = per_emb_stats[emb][src]
                    normalized[mask] = normalize_data(
                        data[mask], per_emb_stats[emb][src]
                    )
                self.buffer.memory_buffer[alias] = normalized
                stats[alias] = per_emb_stats[primary][alias]
                print(f"[norm] {alias} normalized with {src}'s stats (alias)")
            stats["_per_embodiment"] = per_emb_stats
            stats["_primary_embodiment"] = primary
            print(
                f"[norm] per-embodiment normalization over {embodiments}; "
                f"top-level stats from primary={primary!r}"
            )

        self.indices = indices
        self.stats = stats
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def _resolve_tactile_keys(self) -> set:
        """Buffer keys holding the tactile stream (one per arm), or empty.

        Named ``{prefix}{fsr_source_key}`` -- e.g. ``fsr_region`` single-arm,
        ``left_fsr_region`` / ``right_fsr_region`` for a bimanual buffer.
        """
        if not getattr(self.buffer, "enable_fsr", False):
            return set()
        base = str(getattr(self.buffer, "fsr_source_key", "fsr"))
        prefixes = getattr(self.buffer, "arm_prefixes", [""])
        buf = self.buffer.memory_buffer
        return {f"{p}{base}" for p in prefixes if f"{p}{base}" in buf}

    def _resolve_stat_alias_pairs(self):
        """``(source_key, alias_key)`` pairs; the alias reuses source's stats.

        Empty by default.  Subclasses override to put a derived stream on an
        existing scale WITHOUT contributing to it -- see the class-level note in
        ``__init__`` on why contributing would move the source's normalization.
        """
        return []

    def _resolve_shared_joint_key_groups(self):
        """Return a list of buffer-key groups that EACH share one joint stat.

        A group is the obs joint stream + the ABSOLUTE action joint stream for
        one arm.  Single-arm data yields one group
        ``[("proprioception", "hand_action")]``; bimanual data yields one group
        per arm, e.g. ``[("left_proprioception", "left_hand_action"),
        ("right_proprioception", "right_hand_action")]`` (each arm pools/normalizes
        independently).  A group is skipped (and the list may be empty) unless
        ALL of these hold for it:
          * ``self.share_joint_stats`` is True;
          * both keys exist in ``buffer.memory_buffer``;
          * neither key is in ``unnormal_list`` -- this excludes the
            ``relative_hand_action=True`` case, where ``hand_action`` is
            unnormalized in the buffer (its relative action stats are computed
            separately by UVTADataset), so relative mode does NOT share;
          * both streams have the same trailing feature dim (both 22-D joints).
        """
        if not getattr(self, "share_joint_stats", False):
            return []
        prefixes = getattr(self.buffer, "arm_prefixes", [""])
        buf = self.buffer.memory_buffer
        groups = []
        for p in prefixes:
            keys = (f"{p}proprioception", f"{p}hand_action")
            if any(k not in buf or k in self.unnormal_list for k in keys):
                continue
            shapes = {buf[k].shape[1:] for k in keys}
            if len(shapes) != 1:
                print(
                    "[norm] share_joint_stats requested but streams have "
                    f"mismatched feature dims { {k: buf[k].shape[1:] for k in keys} }; "
                    f"NOT sharing group {keys}."
                )
                continue
            groups.append(keys)
        return groups

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def transform_images(self, images_arr):
        images_arr = images_arr.astype(np.float32)
        images_tensor = np.transpose(images_arr, (0, 3, 1, 2)) / 255.0  # (T,dim,h,w)
        return images_tensor

    def __len__(self):
        # all possible segments of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
        ) = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.buffer.memory_buffer,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )
        for camera_id in self.buffer.load_camera_ids:
            nsample[f"camera_{camera_id}"] = self.transform_images(
                nsample[f"camera_{camera_id}"][: self.obs_horizon, :]
            )

        # discard unused observations
        nsample["proprioception"] = nsample["proprioception"][: self.obs_horizon, :]

        return nsample
