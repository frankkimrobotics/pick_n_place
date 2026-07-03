"""Cup-region depth helper (fixed mask; cup is rigidly fixed on the wrist).

Definition (per the pinned spec):
  * true cup   = convex hull of the near-BLACK cup pixels (the rubber dome) in the bottom-center.
  * cup depth  = median D405 depth over that hull region.
  * object-approach region = a 12px band DIRECTLY ABOVE the hull top edge (image-up = smaller
    rows), where the object surface enters view as the cup descends.
  * contact    = the above-band depth falls to the cup(-tip) depth (gap -> ~0).

Build once (any frame), save, and reuse the FIXED masks so it reads the same pixels every frame
(no per-frame re-segmentation -> lighting-robust).
"""
import os
import numpy as np
import cv2

ROI = (350, 480, 285, 465)     # r0, r1, c0, c1 -- bottom-center where the cup sits in the D405 view
DEPTH_NEAR = 0.20              # the cup is the only thing this close to the wrist cam (lighting-robust)
V_BLACK = 95                   # brightness (max channel) below this = black rubber cup (loose; near-depth gates noise)
CUP_UP = 0                     # near+dark already reaches the rim (row ~396, depth-confirmed); above it is far bg
ABOVE_PX = 12                  # thickness of the object-approach band above the (extended) cup


def _extend_up(mask, px):
    out = mask.copy()
    for x in np.where(mask.any(axis=0))[0]:
        top = int(np.where(mask[:, x])[0].min()); out[max(0, top - px):top, x] = True
    return out


def build_cup_region(bgr, depth, above_px=ABOVE_PX, cup_up=CUP_UP):
    """Segment the black cup robustly = NEAR-depth AND dark pixels (near-depth rejects the table/
    objects; dark rejects the silver stem), convex-hull ALL of them (spans the full dome even if a
    highlight splits it), extend UP by cup_up to grab the rim. Returns (cup_mask, above_mask)."""
    H, W = bgr.shape[:2]
    V = bgr.max(axis=2).astype(np.int16)
    roi = np.zeros((H, W), bool); roi[ROI[0]:ROI[1], ROI[2]:ROI[3]] = True
    near = (depth > 0.03) & (depth < DEPTH_NEAR)
    seg = (near & (V < V_BLACK) & roi).astype(np.uint8)
    seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))   # bridge highlight gaps
    ys, xs = np.where(seg)
    if len(ys) < 50:
        return None, None
    hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))          # hull of ALL near+dark px = full dome
    hull_mask = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(hull_mask, hull, 1); hull_mask = hull_mask.astype(bool)
    cup_mask = _extend_up(hull_mask, cup_up)                                    # include the rim
    above = _extend_up(cup_mask, above_px) & ~cup_mask                          # band directly above the cup top
    return cup_mask, above


def save(path, hull, above, above_px=ABOVE_PX):
    np.savez(path, hull=hull, above=above, above_px=above_px)


def load(path):
    d = np.load(path); return d["hull"], d["above"]


def load_or_build(path, bgr=None, depth=None):
    """Load the fixed masks if present, else build from (bgr, depth) and save."""
    if os.path.exists(path):
        return load(path)
    if bgr is None or depth is None:
        return None, None
    cup, above = build_cup_region(bgr, depth)
    if cup is not None:
        save(path, cup, above)
    return cup, above


def read_depths(depth, hull, above):
    """(cup_depth, surf_depth) in metres: cup = median over hull, surf = median over above-band."""
    cd = depth[hull]; cd = cd[(cd > 0.03) & (cd < 0.50)]
    ad = depth[above]; ad = ad[(ad > 0.03) & (ad < 0.80)]
    cup_d = float(np.median(cd)) if len(cd) >= 20 else None
    surf_d = float(np.median(ad)) if len(ad) >= 8 else None
    return cup_d, surf_d
