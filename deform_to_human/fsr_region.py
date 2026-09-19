#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single source of truth: reduce the 100-D ``fsr`` taxel vector to a 20-D
``fsr_region`` vector (each fingertip's 20 taxels -> 4 pooled regions).

The 100-D ``fsr`` is laid out (see ``deform_to_taxel.py``) as five finger blocks
of 20 taxels each, ordered ``[thumb, index, middle, ring, little]``, and within a
finger the 20 columns are the glove fingertip taxels in *ascending point_id*
order (payload order).  ``fsr_region`` pools each finger's 20 taxels into FOUR
regions and stores the MEAN of each region, giving ``5 fingers x 4 = 20`` values
laid out ``[thumb(A,B,C,D), index(A,B,C,D), ... little(A,B,C,D)]``.

Region definition (demonstrated by the user on the RIGHT middle finger and
verified to reproduce that spec exactly, see ``__main__``):

    Using the canonical fingertip frame (``along`` = tip..base, ``across`` =
    one side..other), the 20 taxels form a small pad:
      * A  "tip"  : the 5 tip-most taxels (largest ``along``).                 (5)
      * the remaining 15 form 3 rows x 5 columns; split by ``across`` into
        B  "side" : the 3 taxels on one edge (smallest ``across``),           (3)
        C  "mid"  : the middle 9 taxels,                                      (9)
        D  "side" : the 3 taxels on the other edge (largest ``across``).      (3)

All five fingers share the SAME within-finger offset pattern (they use the same
sensor module wiring), so a single offset set per hand is applied to every finger
block.

Side convention (B vs D) -- read this before comparing the two hands.  B is
always the edge at ``across == 0`` *in that hand's own canonical frame*, and
that frame is mirrored between hands, so the LEFT B/D offsets come out swapped
relative to RIGHT.  The convention is therefore MIRROR-symmetric, not
anatomy-symmetric: for two hands held as mirror images B_side plays the same
role, but B_side is NOT the same physical side of the finger on both hands.
Concretely, taking the direction toward the thumb module from the glove layout:

    finger   RIGHT B_side   RIGHT D_side   LEFT B_side   LEFT D_side
    index      thumb side     little side   little side    thumb side
    middle     thumb side     ~neutral      ~neutral       thumb side
    ring       thumb side     little side   little side    thumb side
    little     thumb side     little side   little side    thumb side

Note this differs from the 100-D ``fsr`` itself, whose column j is the same
glove point_id -- and hence the same physical side -- on both hands.  So do NOT
assume ``left_fsr_region[B]`` and ``right_fsr_region[B]`` cover matching
taxels; A_tip and C_mid do match, B_side and D_side are swapped.
``scripts/plot_region_side_convention.py`` renders this.

Everything downstream (dataset build ``add_fsr_region.py`` and live inference
``build_tactile_from_obs``) MUST import this module so the pooled layout is
identical between training data and deploy.
"""
from __future__ import annotations

import numpy as np

FINGERS = ["thumb", "index", "middle", "ring", "little"]  # fsr block order
TAXELS_PER_FINGER = 20
REGIONS_PER_FINGER = 4
FSR_DIM = len(FINGERS) * TAXELS_PER_FINGER            # 100
FSR_REGION_DIM = len(FINGERS) * REGIONS_PER_FINGER    # 20
REGION_NAMES = ["A_tip", "B_side", "C_mid", "D_side"]

# Within-finger taxel offsets (0..19, ascending point_id) that make up each of
# the 4 regions, per hand.  Order is [A_tip, B_side, C_mid, D_side].  These are
# hard-coded (fixed sensor wiring) and re-verified from the canonical layout in
# ``__main__`` / ``compute_region_offsets``.
# B/D are the two edges of the pad and they are MIRRORED between hands (see the
# side convention in the module docstring): the same offsets sit on the thumb
# side of the right hand and on the little side of the left hand.
REGION_OFFSETS = {
    "right": [
        [0, 1, 2, 3, 4],                       # A tip
        [5, 6, 7],                             # B side  <- thumb side
        [8, 10, 11, 13, 14, 15, 16, 18, 19],   # C mid
        [9, 12, 17],                           # D side  <- little side
    ],
    "left": [
        [0, 1, 2, 3, 4],                       # A tip
        [9, 12, 17],                           # B side  <- little side
        [8, 10, 11, 13, 14, 15, 16, 18, 19],   # C mid
        [5, 6, 7],                             # D side  <- thumb side
    ],
}


def _validate_offsets(offs) -> None:
    flat = sorted(o for grp in offs for o in grp)
    if flat != list(range(TAXELS_PER_FINGER)):
        raise ValueError(
            f"region offsets must partition 0..{TAXELS_PER_FINGER - 1} exactly; "
            f"got {flat}"
        )
    sizes = [len(g) for g in offs]
    if sizes != [5, 3, 9, 3]:
        raise ValueError(f"region sizes must be [5,3,9,3], got {sizes}")


for _h, _o in REGION_OFFSETS.items():
    _validate_offsets(_o)


def region_offsets(hand: str):
    """Return the 4 within-finger offset lists ``[A,B,C,D]`` for ``hand``."""
    key = str(hand).strip().lower()
    if key not in REGION_OFFSETS:
        raise ValueError(f"hand must be 'left' or 'right', got {hand!r}")
    return REGION_OFFSETS[key]


def reduce_fsr_to_region(fsr: np.ndarray, hand: str = "right") -> np.ndarray:
    """Pool a 100-D ``fsr`` into a 20-D ``fsr_region`` (region MEAN).

    Parameters
    ----------
    fsr : ``(..., 100)`` array (any leading batch/time shape).
    hand : ``"right"`` (default) or ``"left"``.

    Returns
    -------
    ``(..., 20)`` float32, laid out ``[thumb(A,B,C,D), ..., little(A,B,C,D)]``.
    """
    a = np.asarray(fsr, dtype=np.float32)
    if a.shape[-1] != FSR_DIM:
        raise ValueError(
            f"fsr last dim must be {FSR_DIM} (5 fingers x {TAXELS_PER_FINGER}); "
            f"got {a.shape}"
        )
    offs = region_offsets(hand)
    lead = a.shape[:-1]
    out = np.empty(lead + (FSR_REGION_DIM,), dtype=np.float32)
    for fi in range(len(FINGERS)):
        base = fi * TAXELS_PER_FINGER
        for ri, cols in enumerate(offs):
            idx = [base + c for c in cols]
            out[..., fi * REGIONS_PER_FINGER + ri] = a[..., idx].mean(axis=-1)
    return out


def region_labels() -> list[str]:
    """Human-readable label for each of the 20 output columns."""
    return [f"{f}_{REGION_NAMES[r]}"
            for f in FINGERS for r in range(REGIONS_PER_FINGER)]


# ---------------------------------------------------------------------------
# Verification: recompute the offsets straight from the canonical fingertip
# frame and confirm they equal the hard-coded constants + the user's demo.
# ---------------------------------------------------------------------------
def compute_region_offsets(hand: str, mapping_dir=None):
    """Recompute ``[A,B,C,D]`` offsets from the canonical along/across layout.

    Independent of the hard-coded ``REGION_OFFSETS`` (used to verify them).
    """
    import json
    import sys
    from pathlib import Path

    here = Path(__file__).resolve().parent
    if mapping_dir is None:
        mapping_dir = here / "glove_sharpa_mapping" / "out"
    mapping_dir = Path(mapping_dir)
    sys.path.insert(0, str(here))
    from core import load_region  # noqa: E402

    canon = json.load(
        open(mapping_dir / f"glove_fingertip_canonical_{hand}.json")
    )
    canon_key = {"thumb": "thumb", "index": "index_finger",
                 "middle": "middle_finger", "ring": "ring_finger",
                 "little": "little_finger"}
    per_finger = []
    for finger in FINGERS:
        ids = load_region(str(mapping_dir), hand, finger)[0]  # ascending pid
        sub = canon[canon_key[finger]]
        along = np.array([sub[str(int(i))]["along"] for i in ids])
        across = np.array([sub[str(int(i))]["across"] for i in ids])
        A = set(np.argsort(-along)[:5].tolist())          # 5 tip-most
        rest = [i for i in range(TAXELS_PER_FINGER) if i not in A]
        rs = sorted(rest, key=lambda i: across[i])
        B, D = set(rs[:3]), set(rs[-3:])
        C = set(rs[3:-3])
        per_finger.append([sorted(A), sorted(B), sorted(C), sorted(D)])
    return per_finger


if __name__ == "__main__":
    # 1) every finger of a hand must share the same offset pattern, and it must
    #    equal the hard-coded constants.
    for hand in ("right", "left"):
        pf = compute_region_offsets(hand)
        assert all(g == pf[0] for g in pf), f"{hand}: fingers differ"
        assert pf[0] == REGION_OFFSETS[hand], (
            f"{hand}: computed {pf[0]} != hardcoded {REGION_OFFSETS[hand]}"
        )
    # 2) RIGHT middle finger must match the user's demonstrated split.
    user_mid = {
        "A": [p - 81 for p in (81, 82, 83, 84, 85)],
        "B": [p - 81 for p in (86, 87, 88)],
        "C": [p - 81 for p in (89, 91, 92, 94, 95, 96, 97, 99, 100)],
        "D": [p - 81 for p in (90, 93, 98)],
    }
    r = REGION_OFFSETS["right"]
    assert r == [user_mid["A"], user_mid["B"], user_mid["C"], user_mid["D"]], r
    # 3) numeric sanity: region mean of a known vector.
    fsr = np.arange(FSR_DIM, dtype=np.float32)[None]        # (1,100)
    reg = reduce_fsr_to_region(fsr, "right")                # (1,20)
    assert reg.shape == (1, FSR_REGION_DIM)
    # thumb A = mean(fsr[0..4]) = mean(0,1,2,3,4) = 2.0
    assert abs(float(reg[0, 0]) - 2.0) < 1e-5, reg[0, 0]
    # index A = mean(fsr[20..24]) = mean(20,21,22,23,24) = 22.0
    assert abs(float(reg[0, 4]) - 22.0) < 1e-5, reg[0, 4]
    print("fsr_region self-test OK")
    print("labels:", region_labels())
    print("right offsets:", REGION_OFFSETS["right"])
    print("left  offsets:", REGION_OFFSETS["left"])
