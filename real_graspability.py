#!/usr/bin/env python3
"""real_graspability :: score candidate grasp points on REAL ZED depth with the same
rim-conformance test the sim uses (rim_seal.py). Lets the robot skip objects a flat
suction cup physically cannot seal (curved bottles, balls, tapered cups) instead of
nudging them around the table.

    from real_graspability import score_xy
    ok, dev = score_xy(P_base, x, y)      # P_base: HxWx3 ZED points in BASE frame
"""
import json, os
import numpy as np

RIM_R   = 0.0075      # measured cup lip radius (m)
N_RIM   = 8
MAX_DEV = 0.0020      # cup compliance; CALIBRATE against real picks
SAMP_R  = 0.0035      # neighbourhood radius for the local depth median (m)


def zed_points_base(xyz_cam, T_base_cam):
    """ZED XYZ measure (HxWx3, left-optical frame) -> base frame points."""
    P = xyz_cam.reshape(-1, 3)
    ok = np.isfinite(P).all(1)
    out = np.full_like(P, np.nan)
    out[ok] = (T_base_cam[:3, :3] @ P[ok].T).T + T_base_cam[:3, 3]
    return out.reshape(xyz_cam.shape)


def _z_at(P, x, y, r=SAMP_R):
    """median height of base-frame points within r of (x,y); nan if none."""
    Q = P.reshape(-1, 3)
    m = np.isfinite(Q).all(1)
    Q = Q[m]
    if not len(Q):
        return np.nan
    d2 = (Q[:, 0] - x) ** 2 + (Q[:, 1] - y) ** 2
    sel = Q[d2 < r * r]
    return float(np.median(sel[:, 2])) if len(sel) >= 3 else np.nan


def score_xy(P_base, x, y, rim_r=RIM_R, n=N_RIM, max_dev=MAX_DEV):
    """(sealable, deviation_m) for a downward grasp centred at (x, y)."""
    centre = _z_at(P_base, x, y)
    if not np.isfinite(centre):
        return False, float("inf")
    rim = []
    for k in range(n):
        th = 2 * np.pi * k / n
        z = _z_at(P_base, x + rim_r * np.cos(th), y + rim_r * np.sin(th))
        if not np.isfinite(z):
            return False, float("inf")        # rim overhangs an edge -> leak
        rim.append(z)
    spread  = float(max(rim) - min(rim))
    sagitta = float(abs(np.mean(rim) - centre))
    dev = max(spread, sagitta)
    return bool(dev <= max_dev), dev


def rank_candidates(P_base, cands, max_dev=MAX_DEV):
    """cands: [{'x','y',...}] -> same list + 'dev'/'sealable', best (flattest) first."""
    out = []
    for c in cands:
        ok, dev = score_xy(P_base, c["x"], c["y"], max_dev=max_dev)
        out.append({**c, "sealable": ok, "dev": dev})
    return sorted(out, key=lambda c: c["dev"])
