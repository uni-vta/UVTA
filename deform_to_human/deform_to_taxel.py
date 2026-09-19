#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standard robot-deform -> per-finger 20-taxel (fsr) converter.

This is the single source of truth for turning Sharpa fingertip *deform* maps
into the 100-D ``fsr`` vector used by the teleop datasets
(``20260709_teleop`` / ``20260711_teleop``) and by inference. Skeleton and
teleop ``fsr`` share this exact layout, so a value at column ``j`` always refers
to the same physical glove fingertip taxel.

Recipe (do NOT change without re-generating the datasets):
  * input   : per-finger deform GRAYSCALE map (240, 240), 0 = no contact.
              (a BGR/JPEG deform must be decoded to grayscale first; a grayscale
               replicated over 3 channels is equivalent to its mean over channels)
  * regions : ``region_label_<hand>_<canonical_finger>_<model>.npy`` (240,240),
              int values 0..19, -1 = background. thumb uses model ``thumb_hb1``,
              the other four fingers use ``general_hb1``.
  * reduce  : per region r, value = SUM of the grayscale pixels with label==r.
  * order   : region 0..19 == ascending glove point_id == payload index order;
              five fingers concatenated [thumb, index, middle, ring, little].
  * scaling : none (raw). Across-orientation: flip_across=False (verified).

Result layout (identical to skeleton fsr):
  fsr[  0: 20] thumb   (glove payload idx   0.. 19)
  fsr[ 20: 40] index   (glove payload idx  40.. 59)
  fsr[ 40: 60] middle  (glove payload idx  80.. 99)
  fsr[ 60: 80] ring    (glove payload idx 120..139)
  fsr[ 80:100] little  (glove payload idx 160..179)

Example (inference, single frame):
    from deform_to_taxel import DeformToTaxel
    conv = DeformToTaxel()                 # default right-hand mapping dir
    fsr = conv.frame(robot_deform_5x240x240)   # -> (100,) float32

Example (a whole clip):
    fsr = conv.batch(tactile)              # (T,5,240,240) -> (T,100) float32
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# fixed conventions
# ---------------------------------------------------------------------------
FINGERS = ["thumb", "index", "middle", "ring", "little"]        # fsr block order
CANONICAL = {                                                    # -> filename part
    "thumb": "thumb", "index": "index_finger", "middle": "middle_finger",
    "ring": "ring_finger", "little": "little_finger",
}
FINGER_MODEL = {f: ("thumb_hb1" if f == "thumb" else "general_hb1")
                for f in FINGERS}
TAXELS_PER_FINGER = 20
DEFORM_HW = (240, 240)

_HERE = Path(__file__).resolve().parent
DEFAULT_MAPPING_DIR = _HERE / "glove_sharpa_mapping" / "out"


def to_grayscale(deform: np.ndarray) -> np.ndarray:
    """Coerce a deform map to (H, W) grayscale intensity (0 = no contact).

    Accepts (H,W) directly, or (H,W,3)/(H,W,C) which is averaged over channels
    (a grayscale replicated over BGR maps to its own value).
    """
    a = np.asarray(deform)
    if a.ndim == 3:
        a = a.mean(axis=2)
    if a.ndim != 2:
        raise ValueError(f"deform must be (H,W) or (H,W,C), got {a.shape}")
    return a


class DeformToTaxel:
    """Convert Sharpa deform maps to the 100-D fsr taxel vector.

    Parameters
    ----------
    mapping_dir : region label directory (default: the repo's
        ``deform_to_human/glove_sharpa_mapping/out``).
    hand : "right" (default) or "left".
    reduce : "sum" (default, matches the datasets) | "mean" | "max".
    """

    def __init__(self, mapping_dir=DEFAULT_MAPPING_DIR, hand: str = "right",
                 reduce: str = "sum"):
        if reduce not in ("sum", "mean", "max"):
            raise ValueError(f"reduce must be sum|mean|max, got {reduce!r}")
        self.mapping_dir = Path(mapping_dir)
        self.hand = hand
        self.reduce = reduce
        self._aggs = []          # per finger: dict(valid, onehot, counts)
        for finger in FINGERS:
            tag = f"{hand}_{CANONICAL[finger]}_{FINGER_MODEL[finger]}"
            path = self.mapping_dir / f"region_label_{tag}.npy"
            if not path.exists():
                raise FileNotFoundError(f"region label not found: {path}")
            label = np.load(path)
            if label.shape != DEFORM_HW:
                raise ValueError(
                    f"{finger}: region label {label.shape} != {DEFORM_HW}")
            flat = label.reshape(-1)
            valid = np.nonzero(flat >= 0)[0]
            reg = flat[valid].astype(np.int64)
            if reg.size and reg.max() >= TAXELS_PER_FINGER:
                raise ValueError(f"{finger}: region id {reg.max()} out of range")
            onehot = np.zeros((valid.size, TAXELS_PER_FINGER), dtype=np.float64)
            onehot[np.arange(valid.size), reg] = 1.0
            counts = onehot.sum(axis=0)               # pixels per region
            self._aggs.append({"valid": valid, "onehot": onehot,
                               "counts": counts})

    # -- core ---------------------------------------------------------------
    def _reduce_finger(self, field_flat_2d: np.ndarray, agg) -> np.ndarray:
        """field_flat_2d: (N, H*W) -> (N, 20) reduced per region."""
        fv = field_flat_2d[:, agg["valid"]]           # (N, n_valid)
        if self.reduce == "max":
            # per region max via masked columns
            out = np.full((fv.shape[0], TAXELS_PER_FINGER), 0.0)
            reg = agg["onehot"].argmax(axis=1)
            for r in range(TAXELS_PER_FINGER):
                m = reg == r
                if m.any():
                    out[:, r] = fv[:, m].max(axis=1)
            return out
        vec = fv @ agg["onehot"]                      # (N, 20) sum
        if self.reduce == "mean":
            counts = np.where(agg["counts"] > 0, agg["counts"], 1.0)
            vec = vec / counts
        return vec

    def batch(self, tactile: np.ndarray) -> np.ndarray:
        """(T, 5, H, W) deform maps -> (T, 100) float32 fsr.

        Grayscale expected; (T,5,H,W,3) is averaged over the last axis.
        """
        a = np.asarray(tactile)
        if a.ndim == 5:                # (T,5,H,W,C) -> grayscale
            a = a.mean(axis=-1)
        if a.ndim != 4 or a.shape[1] != 5 or a.shape[2:] != DEFORM_HW:
            raise ValueError(
                f"expected (T,5,{DEFORM_HW[0]},{DEFORM_HW[1]}[,C]), got "
                f"{np.asarray(tactile).shape}")
        T = a.shape[0]
        cols = []
        for fi, agg in enumerate(self._aggs):
            field = a[:, fi].reshape(T, -1).astype(np.float64)
            cols.append(self._reduce_finger(field, agg))
        return np.concatenate(cols, axis=1).astype(np.float32)

    def frame(self, deform5) -> np.ndarray:
        """5 finger deform maps -> (100,) float32 fsr.

        ``deform5`` may be (5,H,W), (5,H,W,3), or a length-5 sequence of (H,W)/
        (H,W,3) maps in order [thumb, index, middle, ring, little].
        """
        maps = [to_grayscale(m) for m in deform5]
        if len(maps) != 5:
            raise ValueError(f"need 5 finger maps, got {len(maps)}")
        stack = np.stack(maps, axis=0)[None]          # (1,5,H,W)
        return self.batch(stack)[0]

    def finger(self, finger: str, deform) -> np.ndarray:
        """One finger's deform map -> (20,) float32 (ordered by glove point_id)."""
        if finger not in FINGERS:
            raise ValueError(f"finger must be one of {FINGERS}")
        fi = FINGERS.index(finger)
        field = to_grayscale(deform).reshape(1, -1).astype(np.float64)
        return self._reduce_finger(field, self._aggs[fi])[0].astype(np.float32)


# convenience: column j -> (finger, within_finger_idx, glove_point_id, payload)
def column_index():
    from_ = []
    base = {"thumb": 1, "index": 41, "middle": 81, "ring": 121, "little": 161}
    for finger in FINGERS:
        for k in range(TAXELS_PER_FINGER):
            pid = base[finger] + k
            from_.append((finger, k, pid, pid - 1))
    return from_


if __name__ == "__main__":
    conv = DeformToTaxel()
    # smoke test: a single hot pixel on the thumb tip region
    d = np.zeros((5, 240, 240), dtype=np.float32)
    d[0, 55, 150] = 200.0
    fsr = conv.frame(d)
    print("fsr shape", fsr.shape, "nonzero cols", np.nonzero(fsr)[0].tolist())
