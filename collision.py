"""collision :: dependency-free OBB-vs-cuboid collision + clearance.

The carried-object collision volume is an oriented box (OBB); the place-box is
modelled as a set of axis-aligned wall cuboids. We test the carried OBB against
each wall with the Separating-Axis Theorem (exact boolean), and report a
conservative clearance by sampling the OBB surface when not in contact. This
avoids needing python-fcl while still giving a faithful carried-object safety
check for the transport leg.
"""
import numpy as np


def _box_axes_corners(T, dims):
    """Return (center, axes(3x3 cols), half(3), corners(8x3)) for a box pose T."""
    R = T[:3, :3]
    c = T[:3, 3]
    half = np.asarray(dims, float) / 2.0
    signs = np.array([[sx, sy, sz] for sx in (-1, 1)
                      for sy in (-1, 1) for sz in (-1, 1)], float)
    corners = c + (signs * half) @ R.T
    return c, R, half, corners


def _aabb_box(center, dims):
    T = np.eye(4); T[:3, 3] = np.asarray(center, float)
    return _box_axes_corners(T, dims)


def obb_vs_obb(TA, dimsA, TB, dimsB, eps=1e-6):
    """SAT collision test between two oriented boxes. Returns True if overlapping."""
    cA, RA, hA, _ = _box_axes_corners(TA, dimsA)
    cB, RB, hB, _ = _box_axes_corners(TB, dimsB)
    axesA = [RA[:, i] for i in range(3)]
    axesB = [RB[:, i] for i in range(3)]
    axes = list(axesA) + list(axesB)
    for a in axesA:
        for b in axesB:
            cr = np.cross(a, b)
            if np.linalg.norm(cr) > eps:
                axes.append(cr / np.linalg.norm(cr))
    d = cB - cA
    for ax in axes:
        ax = ax / (np.linalg.norm(ax) + 1e-12)
        rA = sum(hA[i] * abs(np.dot(ax, axesA[i])) for i in range(3))
        rB = sum(hB[i] * abs(np.dot(ax, axesB[i])) for i in range(3))
        if abs(np.dot(d, ax)) > rA + rB + eps:
            return False                      # separating axis found
    return True


def _surface_samples(T, dims, n=3):
    """Sample points on an OBB surface (corners + grid on each face)."""
    R = T[:3, :3]; c = T[:3, 3]; half = np.asarray(dims, float) / 2.0
    ts = np.linspace(-1, 1, n)
    pts = []
    for axis in range(3):
        for s in (-1, 1):
            u, v = [a for a in range(3) if a != axis]
            for a in ts:
                for b in ts:
                    p = np.zeros(3); p[axis] = s
                    p[u] = a; p[v] = b
                    pts.append(p * half)
    P = np.array(pts)
    return c + P @ R.T


def _point_to_aabb_dist(pts, center, dims):
    """Min distance from points to an axis-aligned box surface (0 if inside)."""
    half = np.asarray(dims, float) / 2.0
    d = np.abs(pts - center) - half
    outside = np.clip(d, 0, None)
    return np.linalg.norm(outside, axis=1)       # 0 inside, >0 outside


def obb_vs_walls(T_obb, dims_obb, walls):
    """Test carried OBB against a list of wall (center,dims) AABBs.

    Returns (collided, clearance_m). clearance is a conservative min distance
    from the OBB surface to the nearest wall when not colliding.
    """
    collided = False
    for center, dims in walls:
        TB = np.eye(4); TB[:3, 3] = np.asarray(center, float)
        if obb_vs_obb(T_obb, dims_obb, TB, dims):
            collided = True
            break
    if collided:
        return True, 0.0
    samp = _surface_samples(T_obb, dims_obb, n=4)
    best = np.inf
    for center, dims in walls:
        dd = _point_to_aabb_dist(samp, np.asarray(center, float), np.asarray(dims, float))
        best = min(best, float(dd.min()))
    return False, best
