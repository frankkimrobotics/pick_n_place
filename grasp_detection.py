"""grasp_detection :: find the 1 cm circular suction grasp point.

A single suction cup seals on a flat circular footprint of diameter
``C.SUCTION_DIAMETER`` (1 cm). Given the segmented object point cloud (base frame)
with per-point surface normals, we search for the best such patch:

For each candidate surface point we look at neighbours within the cup radius
(5 mm) and require:
  * planarity      — RMS distance to the local best-fit plane < FLAT_RMS_TOL
  * full support   — the disk is covered (enough neighbours, no big angular gap),
                     so the cup is not hanging over an edge
  * normal agreement — neighbour normals lie within a cone (consistent sealing face)

Candidates are scored by planarity, verticality (prefer top-down suction) and
centrality (prefer the middle of the face). The best one yields the grasp point,
its sealing normal, and a full tcp grasp pose (tcp +Z = -normal = into surface),
plus pre-grasp and lift poses.
"""
import numpy as np
from scipy.spatial import cKDTree

import config as C
from geometry import R_from_two_axes, R_to_quat_wxyz, angle_between


def _plane_rms(pts):
    """RMS distance of pts to their best-fit plane + plane normal."""
    c = pts.mean(0)
    U, Sg, Vt = np.linalg.svd(pts - c, full_matrices=False)
    n = Vt[-1]
    d = (pts - c) @ n
    return float(np.sqrt(np.mean(d ** 2))), n, c


def _angular_coverage(neigh_xy):
    """Largest angular gap (rad) of neighbour bearings about the patch centre.

    A small max-gap means the disk is fully surrounded (interior point); a large
    gap means we're at/near an edge.
    """
    if len(neigh_xy) < 4:
        return 2 * np.pi
    ang = np.sort(np.arctan2(neigh_xy[:, 1], neigh_xy[:, 0]))
    gaps = np.diff(np.concatenate([ang, ang[:1] + 2 * np.pi]))
    return float(gaps.max())


def detect_suction_grasp(pts_base, normals_base, up=np.array([0.0, 0.0, 1.0]),
                         verbose=True):
    """Return the best suction grasp, or None if no valid 1 cm patch exists.

    Result dict: point, normal, grasp_pose, pregrasp_pose, lift_pose, score,
    plane_rms, verticality, n_support, max_gap_deg, candidates(int).
    """
    pts = np.asarray(pts_base, float)
    nrm = np.asarray(normals_base, float)
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
    if len(pts) < 8:
        return None

    r = C.SUCTION_RADIUS
    tree = cKDTree(pts)
    # mean nearest-neighbour spacing -> expected support count for a full disk
    dq, _ = tree.query(pts[:: max(1, len(pts) // 500)], k=2)
    spacing = float(np.median(dq[:, 1])) if dq.size else r / 4
    min_support = max(8, int(0.55 * np.pi * r ** 2 / max(spacing, 1e-4) ** 2))
    cos_cone = np.cos(np.deg2rad(C.NORMAL_CONE_DEG))

    centroid = pts.mean(0)
    cand = []
    # consider points whose normal is "graspable" (mostly upward for top suction)
    grasp_ok = (nrm @ up) > np.cos(np.deg2rad(60.0))
    idxs = np.where(grasp_ok)[0]
    for i in idxs:
        p, n = pts[i], nrm[i]
        nb = tree.query_ball_point(p, r)
        if len(nb) < min_support:
            continue
        nb = np.asarray(nb)
        local = pts[nb]
        rms, plane_n, _ = _plane_rms(local)
        if rms > C.FLAT_RMS_TOL:
            continue
        if plane_n @ n < 0:
            plane_n = -plane_n
        # neighbour-normal consistency
        if np.mean((nrm[nb] @ plane_n) > cos_cone) < 0.85:
            continue
        # coverage: project neighbours onto the plane basis and check angular gap
        b1 = R_from_two_axes(plane_n)[:, 0]
        b2 = np.cross(plane_n, b1)
        rel = local - p
        nbxy = np.stack([rel @ b1, rel @ b2], axis=1)
        max_gap = _angular_coverage(nbxy)
        if max_gap > np.deg2rad(110):                # near an edge -> poor seal
            continue
        verticality = float(plane_n @ up)
        centrality = -np.linalg.norm(p[:2] - centroid[:2])
        score = (-200.0 * rms) + (2.0 * verticality) + (3.0 * centrality)
        cand.append(dict(point=p, normal=plane_n, score=score, plane_rms=rms,
                         verticality=verticality, n_support=len(nb),
                         max_gap_deg=np.rad2deg(max_gap)))

    if verbose:
        print(f"[grasp] {len(idxs)} upward pts, {len(cand)} valid 1cm patches "
              f"(min_support={min_support}, spacing={spacing*1e3:.1f}mm)")
    if not cand:
        return None

    best = max(cand, key=lambda d: d["score"])
    p, n = best["point"], best["normal"]
    from sim_planner import grasp_pose_from_point_normal
    grasp_pose = grasp_pose_from_point_normal(p, n)
    pregrasp_pose = grasp_pose.copy()
    pregrasp_pose[:3] = p + C.PREGRASP_STANDOFF * n          # back off along +normal
    lift_pose = grasp_pose.copy()
    lift_pose[:3] = p + np.array([0.0, 0.0, C.LIFT_HEIGHT])  # straight up
    best.update(grasp_pose=grasp_pose, pregrasp_pose=pregrasp_pose,
                lift_pose=lift_pose, candidates=len(cand))
    return best
