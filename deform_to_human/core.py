"""Convert real Sharpa fingertip deform maps into human-hand glove tactile.

Direction: Sharpa deform map (dense per-pixel 3D displacement image, one per
finger) -> per-region scalar -> the matching human glove fingertip-20 taxel
value, for all five fingers, assembled into the 240-length glove payload.

It reuses the region label maps produced by
``glove_sharpa_mapping/build_region_map.py`` (per hand, per finger):
    region_label_{hand}_{finger}_{model}.npy   (H, W) int, -1 = background
    region_table_{hand}_{finger}_{model}.json  region <-> point_id / payload

Deform map conventions (see viz_collect/contact_utils.py):
    channel 0,1 = tangential shear x/y, neutral = 128
    channel 2   = normal depth,        neutral = 64
    stored value q relates to physical displacement diff by
        q = sign(diff) * sqrt(|diff| / 2.2e-4) + center
    so diff = sign(q - center) * (q - center)^2 * 2.2e-4
When a deform map is loaded from a JPEG/PNG with OpenCV (as the collector
stores it, BGR), channel index 2 is the normal-depth channel.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_MAPPING_DIR = REPO / "glove_sharpa_mapping" / "out"

N_POINTS = 240
DEFORM_CENTER = np.array([128.0, 128.0, 64.0])
DEFORM_SCALE = 2.2e-4  # meters, from deform_quantize

FINGERS = ["thumb", "index_finger", "middle_finger", "ring_finger", "little_finger"]
FINGER_MODEL = {
    "thumb": "thumb_hb1",
    "index_finger": "general_hb1",
    "middle_finger": "general_hb1",
    "ring_finger": "general_hb1",
    "little_finger": "general_hb1",
}

# accepted aliases -> canonical finger key
FINGER_ALIASES = {
    "thumb": "thumb", "1": "thumb",
    "index": "index_finger", "index_finger": "index_finger", "2": "index_finger",
    "middle": "middle_finger", "middle_finger": "middle_finger", "3": "middle_finger",
    "ring": "ring_finger", "ring_finger": "ring_finger", "4": "ring_finger",
    "little": "little_finger", "little_finger": "little_finger",
    "pinky": "little_finger", "5": "little_finger",
}

CHANNELS = ("gray", "depth", "depth_signed", "magnitude", "shear",
            "depth_phys", "magnitude_phys", "shear_phys", "raw2")


def canonical_finger(name: str) -> str:
    key = str(name).strip().lower()
    if key not in FINGER_ALIASES:
        raise ValueError(f"unknown finger {name!r}; use one of {sorted(set(FINGER_ALIASES))}")
    return FINGER_ALIASES[key]


def load_deform(path: str | Path) -> np.ndarray:
    """Load a deform map as an (H, W, 3) uint8 array.

    ``.npy`` is loaded as-is (assumed channel order [shear_x, shear_y, normal]).
    Image files are read with OpenCV (BGR), which round-trips the collector's
    save order so channel 2 is the normal-depth channel.
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        import cv2
        arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if arr is None:
            raise FileNotFoundError(f"could not read image {path}")
    arr = np.asarray(arr)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"deform map {path} must be (H, W, 3), got {arr.shape}")
    return arr[:, :, :3]


def _dequant(delta: np.ndarray) -> np.ndarray:
    """Invert deform_quantize: (value - center) -> physical displacement (m)."""
    return np.sign(delta) * (delta.astype(np.float64) ** 2) * DEFORM_SCALE


def intensity_field(deform: np.ndarray, channel: str = "depth") -> np.ndarray:
    """Reduce an (H, W, 3) deform map to a scalar intensity field (H, W)."""
    d = deform.astype(np.float64)
    dn = d[:, :, 2] - DEFORM_CENTER[2]          # normal, signed (quantized units)
    sh = d[:, :, :2] - DEFORM_CENTER[:2]        # shear x/y (quantized units)
    if channel == "gray":
        # magnitude map where 0 == no deformation (real Sharpa recordings store
        # the deform this way: a single-channel intensity replicated over BGR).
        return d.mean(axis=2)
    if channel == "depth":
        return np.abs(dn)
    if channel == "depth_signed":
        return dn
    if channel == "magnitude":
        return np.linalg.norm(d - DEFORM_CENTER, axis=2)
    if channel == "shear":
        return np.linalg.norm(sh, axis=2)
    if channel == "raw2":
        return d[:, :, 2]
    if channel == "depth_phys":
        return np.abs(_dequant(dn))
    if channel == "shear_phys":
        return np.linalg.norm(_dequant(sh), axis=2)
    if channel == "magnitude_phys":
        return np.linalg.norm(_dequant(d - DEFORM_CENTER), axis=2)
    raise ValueError(f"unknown channel {channel!r}; use one of {CHANNELS}")


def load_region(mapping_dir: str | Path, hand: str, finger: str
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (point_ids, payload_indices, label) for one hand/finger.

    ``label`` is (H, W) int with region index r in [0, 19] (order = sorted
    point_id) and -1 for background. ``point_ids[r]`` / ``payload_indices[r]``
    give the glove taxel for region r.
    """
    finger = canonical_finger(finger)
    model = FINGER_MODEL[finger]
    tag = f"{hand}_{finger}_{model}"
    label = np.load(Path(mapping_dir) / f"region_label_{tag}.npy")
    with open(Path(mapping_dir) / f"region_table_{tag}.json", "r", encoding="utf-8") as f:
        table = json.load(f)["regions"]
    order = sorted(table, key=lambda t: t["region"])
    ids = np.array([t["point_id"] for t in order], dtype=int)
    payloads = np.array([t["payload_index"] for t in order], dtype=int)
    return ids, payloads, label


def aggregate(field: np.ndarray, label: np.ndarray, n_regions: int,
              reduce: str = "mean") -> np.ndarray:
    """Reduce a per-pixel field to one scalar per region (length n_regions)."""
    if field.shape != label.shape:
        raise ValueError(
            f"deform field {field.shape} does not match region label {label.shape}; "
            f"they must be the same fingertip model / resolution.")
    out = np.zeros(n_regions, dtype=np.float64)
    for r in range(n_regions):
        m = label == r
        if not m.any():
            continue
        vals = field[m]
        if reduce == "max":
            out[r] = float(vals.max())
        elif reduce == "sum":
            out[r] = float(vals.sum())
        else:
            out[r] = float(vals.mean())
    return out


def finger_to_taxels(deform: np.ndarray, mapping_dir, hand: str, finger: str,
                     channel: str = "depth", reduce: str = "mean"
                     ) -> Dict[int, float]:
    """Convert one finger's deform map to {point_id: value} (20 entries)."""
    ids, _payloads, label = load_region(mapping_dir, hand, finger)
    field = intensity_field(deform, channel)
    vec = aggregate(field, label, len(ids), reduce)
    return {int(pid): float(v) for pid, v in zip(ids, vec)}


def deform_to_human(deform_maps: Dict[str, np.ndarray], hand: str,
                    mapping_dir: str | Path = DEFAULT_MAPPING_DIR,
                    channel: str = "depth", reduce: str = "mean",
                    gain: float = 1.0, clip: float | None = None
                    ) -> Tuple[np.ndarray, Dict[str, Dict[int, float]]]:
    """Convert up to five finger deform maps into human-hand tactile.

    Args:
        deform_maps: {finger_name: (H, W, 3) deform array}. Missing fingers are
            left at zero. Finger names accept aliases (e.g. "index").
        hand: "left" or "right" (selects the region maps).
        mapping_dir: folder with region_label_*/region_table_* files.
        channel: intensity channel (see CHANNELS).
        reduce: 'mean' | 'max' | 'sum' over each region's pixels.
        gain: multiply all values (e.g. to match the ~0..4095 glove scale).
        clip: optional upper clip after gain.

    Returns:
        payload (240,) float vector (glove ordering, non-fingertip taxels = 0),
        per_finger {finger: {point_id: value}}.
    """
    payload = np.zeros(N_POINTS, dtype=np.float64)
    per_finger: Dict[str, Dict[int, float]] = {}
    for name, deform in deform_maps.items():
        finger = canonical_finger(name)
        ids, payloads, label = load_region(mapping_dir, hand, finger)
        field = intensity_field(deform, channel)
        vec = aggregate(field, label, len(ids), reduce) * gain
        if clip is not None:
            vec = np.clip(vec, 0.0, clip)
        per_finger[finger] = {}
        for pid, pl, v in zip(ids, payloads, vec):
            payload[pl] = v
            per_finger[finger][int(pid)] = float(v)
    return payload, per_finger


def discover_deform_files(deform_dir: str | Path) -> Dict[str, Path]:
    """Find one deform file per finger inside a directory.

    Matches files whose stem starts with a finger name/alias, e.g.
    ``index.png``, ``index_finger.npy``, ``thumb_deform.jpg``.
    """
    deform_dir = Path(deform_dir)
    exts = (".npy", ".png", ".jpg", ".jpeg", ".bmp")
    found: Dict[str, Path] = {}
    for p in sorted(deform_dir.iterdir()):
        if p.suffix.lower() not in exts:
            continue
        stem = p.stem.lower()
        for alias, finger in FINGER_ALIASES.items():
            if stem == alias or stem.startswith(alias + "_") or stem.startswith(alias):
                found.setdefault(finger, p)
                break
    return found
