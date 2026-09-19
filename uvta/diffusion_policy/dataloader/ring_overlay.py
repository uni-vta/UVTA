"""Camera-ring overlay for sim-to-real domain matching.

The real teleop / robot fisheye camera has a thin white reflective ring
around the lens (see `data/20260514_teleop` for examples).  The
exoskeleton (e.g. ``20260516_skeleton``) and manus-glove datasets are
captured with a slightly different setup that does NOT include this
ring, which causes a domain gap when training jointly with teleop.

This module makes it easy to paint the teleop ring back onto exoskeleton
and manus frames at data-loading time, so the network sees a consistent
visual prior across all embodiments.

The ring is described by two static assets (default in ``data/mask/``):

* ``ring_mask.png``    : (H, W) uint8, 0 = ignore, 255 = ring pixel
* ``ring_rgb_*.png``   : (H, W, 3) uint8, the actual ring texture (BGR
  if loaded with ``cv2.imread``; this helper converts to RGB by
  default).

Apply formula (per pixel):
    out[y, x] = ring_rgb[y, x]    if ring_mask[y, x] > 0
                input[y, x]       otherwise
"""

from __future__ import annotations

import os.path as osp
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RingOverlay:
    """A single (mask, rgb) ring asset, validated and ready to apply.

    The asset is loaded at its native resolution (typically the raw
    camera resolution like ``384x480``); when :py:meth:`apply` /
    :py:meth:`apply_batch` is called with a different image size, the
    asset is **lazily resized once and cached**.  This lets the same
    asset file be reused with any ``camera_resize_shape`` setting
    without manually pre-rendering one variant per resolution.

    Attributes
    ----------
    mask : (H, W) uint8 in {0, 255}, where 255 marks ring pixels.
    rgb  : (H, W, 3) uint8 in the same color order as the images we
           apply onto (RGB by default; constructor converts BGR -> RGB
           when ``bgr2rgb_assets=True``).
    """

    mask: np.ndarray
    rgb: np.ndarray

    def __post_init__(self):
        # Cache of (H, W) -> (mask_bool, rgb) resized to that shape.
        self._resized_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def _get_resized(self, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (mask_bool, rgb) sized to ``(H, W)``, computing once.

        Mask is resized with ``INTER_NEAREST`` to keep the binary
        boundary sharp; rgb is resized with ``INTER_AREA`` (good for
        shrinking, which is the common case 384x480 -> 240x240).
        """
        key = (int(H), int(W))
        if key not in self._resized_cache:
            if self.mask.shape == key:
                mask_resized = self.mask
                rgb_resized = self.rgb
            else:
                mask_resized = cv2.resize(
                    self.mask, (W, H), interpolation=cv2.INTER_NEAREST
                )
                rgb_resized = cv2.resize(
                    self.rgb, (W, H), interpolation=cv2.INTER_AREA
                )
            self._resized_cache[key] = (mask_resized > 0, rgb_resized)
        return self._resized_cache[key]

    @classmethod
    def from_paths(
        cls,
        mask_path: str,
        rgb_path: str,
        bgr2rgb_assets: bool = True,
    ) -> "RingOverlay":
        if not osp.exists(mask_path):
            raise FileNotFoundError(f"ring mask not found: {mask_path}")
        if not osp.exists(rgb_path):
            raise FileNotFoundError(f"ring rgb not found:  {rgb_path}")
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"failed to read ring mask: {mask_path}")
        rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        if rgb is None:
            raise RuntimeError(f"failed to read ring rgb:  {rgb_path}")

        # Mask -> single 2-D uint8.
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        # Binarize: anything > 0 counts.  Some upstream tools save 0/1
        # rather than 0/255; we accept both.
        mask = (mask > 0).astype(np.uint8) * 255

        # RGB -> 3-channel uint8.  If the asset has alpha (RGBA / BGRA)
        # we drop the alpha; the alpha already informed the binary mask.
        if rgb.ndim == 2:
            raise ValueError(
                f"ring rgb is single-channel; expected RGB/BGR/RGBA "
                f"(shape={rgb.shape}, path={rgb_path})"
            )
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        if bgr2rgb_assets:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        if mask.shape != rgb.shape[:2]:
            raise ValueError(
                f"ring mask shape {mask.shape} doesn't match ring rgb shape "
                f"{rgb.shape[:2]}"
            )

        return cls(mask=mask, rgb=rgb)

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply(self, image: np.ndarray, in_place: bool = True) -> np.ndarray:
        """Paint the ring onto a single ``(H, W, 3)`` uint8 image.

        Pixels where ``mask > 0`` get replaced by ``self.rgb``; all
        other pixels are kept unchanged.  The ring asset is resized
        (and cached) to ``(H, W)`` on the first call for that shape.

        When ``in_place=True`` we write directly into ``image`` and
        return it (no allocation); when False we return a fresh copy.
        """
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"image must be (H, W, 3) uint8, got shape {image.shape}"
            )
        H, W = image.shape[:2]
        m, rgb = self._get_resized(H, W)
        if not in_place:
            image = image.copy()
        image[m] = rgb[m]
        return image

    def apply_batch(
        self, images: np.ndarray, in_place: bool = True
    ) -> np.ndarray:
        """Paint the ring onto a batch ``(N, H, W, 3)`` uint8.

        Same semantics as :py:meth:`apply` but vectorized across the
        leading ``N`` axis using a broadcast assignment, which is much
        faster than a Python loop for large episode buffers.  Asset is
        resized to ``(H, W)`` on the first call and cached.
        """
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(
                f"images must be (N, H, W, 3) uint8, got shape {images.shape}"
            )
        N, H, W, _ = images.shape
        m, rgb = self._get_resized(H, W)
        if not in_place:
            images = images.copy()
        # ``images[:, m]`` is (N, K, 3) where K = m.sum(); ``rgb[m]``
        # is (K, 3) and broadcasts across the N axis.
        images[:, m] = rgb[m]
        return images


# ----------------------------------------------------------------------
# Convenience wrapper used by the dataset / replay buffer side.  Keeps
# the per-episode "should I overlay?" decision in one place.
# ----------------------------------------------------------------------

# Embodiment taxonomy (2026-07 simplification): every data_dir is now
# labelled by ONE of two coarse embodiments:
#   * ``robot`` : real robot / teleop capture (has the physical camera ring).
#   * ``human`` : human-worn captures -- the former ``exoskeleton``
#                 (skeleton) and ``manus`` glove data -- which lack the ring.
# The ring overlay therefore paints the ring onto ``human`` frames by
# default so the network sees a consistent visual prior across embodiments.
# The yaml can override this list explicitly under ``ring_overlay.embodiments``.
DEFAULT_RING_EMBODIMENTS = ("human",)
VALID_EMBODIMENTS = ("human", "robot")

# Back-compat: map the legacy fine-grained embodiment names onto the new
# coarse two-way taxonomy.  Old configs / stats.pickle that still say
# ``teleop`` / ``exoskeleton`` / ``manus`` keep working.
LEGACY_EMBODIMENT_ALIASES = {
    "teleop": "robot",
    "exoskeleton": "human",
    "manus": "human",
}


def canonical_embodiment(name: str) -> str:
    """Map any (possibly legacy) embodiment name to the canonical set.

    ``robot`` / ``human`` pass through unchanged; the legacy
    ``teleop`` / ``exoskeleton`` / ``manus`` names are aliased.  Unknown
    names raise so typos are caught early.
    """
    if name in VALID_EMBODIMENTS:
        return name
    if name in LEGACY_EMBODIMENT_ALIASES:
        return LEGACY_EMBODIMENT_ALIASES[name]
    raise ValueError(
        f"unknown embodiment {name!r}; valid choices are "
        f"{VALID_EMBODIMENTS} (legacy aliases: "
        f"{tuple(LEGACY_EMBODIMENT_ALIASES)})"
    )


# Repo root resolved at import time so the default ring assets stay
# locatable even after hydra rewrites the working directory to its
# experiment output folder.  ``ring_overlay.py`` lives at
# ``<repo>/DexUMI/dexumi/diffusion_policy/dataloader/ring_overlay.py``
# so going up 4 levels (dataloader -> diffusion_policy -> dexumi ->
# DexUMI -> <repo>) gets us to the repo root.
_REPO_ROOT = osp.normpath(osp.join(osp.dirname(osp.abspath(__file__)), *([".."] * 4)))
_DEFAULT_MASK_PATH = osp.join(_REPO_ROOT, "data/mask/default_ring/ring_mask.png")
_DEFAULT_RGB_PATH = osp.join(_REPO_ROOT, "data/mask/default_ring/ring_patch_rgb.png")


def _resolve_asset_path(path: str) -> str:
    """Resolve a ring-asset path so hydra's cwd rewrite doesn't break it.

    - Absolute paths are returned as-is.
    - Relative paths are first tried as-is (in case the caller is running
      from a sensible cwd), then re-rooted against ``_REPO_ROOT``.
    """
    if osp.isabs(path):
        return path
    if osp.exists(path):
        return path
    return osp.join(_REPO_ROOT, path)


def parse_overlay_cfg(overlay_cfg):
    """Normalize a hydra-ish ring_overlay dict into a ``(RingOverlay, set)``
    tuple, or ``(None, set())`` if disabled.

    ``overlay_cfg`` may be ``None``, an OmegaConf DictConfig, or a plain
    dict with these keys (all optional):

        enabled         : bool, default True
        embodiments     : list[str] subset of VALID_EMBODIMENTS
        mask_path       : str, default <repo>/data/mask/default_ring/ring_mask.png
        rgb_path        : str, default <repo>/data/mask/default_ring/ring_patch_rgb.png
        bgr2rgb_assets  : bool, default True

    Relative paths are resolved against the repo root so hydra's cwd
    rewrite (to ``dexumi/experiment/dp/<timestamp>/``) does not break
    asset loading.

    Returns
    -------
    overlay : RingOverlay or None
    embs    : set[str]  embodiments to apply the overlay to
    """
    if overlay_cfg is None:
        return None, set()
    # Allow passing in a DictConfig.
    if hasattr(overlay_cfg, "get"):
        get = overlay_cfg.get
    else:
        get = lambda k, d=None: overlay_cfg.get(k, d)  # noqa: E731

    enabled = bool(get("enabled", True))
    if not enabled:
        return None, set()

    embs = list(get("embodiments", DEFAULT_RING_EMBODIMENTS))
    # Canonicalize (accepts legacy teleop/exoskeleton/manus -> robot/human).
    embs = [canonical_embodiment(e) for e in embs]

    mask_path = _resolve_asset_path(str(get("mask_path", _DEFAULT_MASK_PATH)))
    rgb_path = _resolve_asset_path(str(get("rgb_path", _DEFAULT_RGB_PATH)))
    bgr2rgb_assets = bool(get("bgr2rgb_assets", True))

    overlay = RingOverlay.from_paths(mask_path, rgb_path, bgr2rgb_assets=bgr2rgb_assets)
    return overlay, set(embs)
