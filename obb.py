"""obb :: 3D oriented bounding box of the segmented object point cloud.

Used as the carried-object collision volume (a cuRobo ``Cuboid`` = pose + dims)
for collision-aware transport, and by the simulator's collision sweep.
"""
import numpy as np
import trimesh

from geometry import R_to_quat_wxyz, inv_T, T_to_pose


def compute_obb(pts_base, pad=0.0):
    """Oriented bounding box of Nx3 points.

    Returns dict: pose7 [x,y,z,qw,qx,qy,qz] (box centre in base), dims (3,),
    T_base_obb (4x4), corners (8,3).
    """
    pts = np.asarray(pts_base, float)
    # trimesh returns the transform that maps points INTO the OBB-aligned frame
    to_obb, extents = trimesh.bounds.oriented_bounds(pts)
    T_base_obb = inv_T(to_obb)                       # OBB frame -> base
    dims = np.asarray(extents, float) + 2 * pad
    pose7 = T_to_pose(T_base_obb)
    # 8 corners
    hx, hy, hz = dims / 2.0
    local = np.array([[sx * hx, sy * hy, sz * hz]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners = (T_base_obb[:3, :3] @ local.T).T + T_base_obb[:3, 3]
    return {
        "pose7": pose7,
        "dims": dims,
        "T_base_obb": T_base_obb,
        "corners": corners,
    }


def compute_obb_grounded(pts_base, support_z, pad=0.003):
    """OBB of a single-view (top) cloud, extended down to the support plane.

    A single eye-in-hand top view only sees the object's top face, so the raw OBB
    is degenerate in the view direction. Since the object was resting on the table
    at z=`support_z`, we augment the cloud with its projection onto that plane
    before fitting the OBB — yielding a box that faithfully bounds the *physical*
    object (top footprint extruded down to the table). A small `pad` adds margin.
    """
    pts = np.asarray(pts_base, float)
    floor = pts.copy()
    floor[:, 2] = support_z
    aug = np.vstack([pts, floor])
    return compute_obb(aug, pad=pad)


def obb_mesh(obb):
    """A trimesh box for the OBB (for collision sweep / visualisation)."""
    m = trimesh.creation.box(extents=obb["dims"])
    m.apply_transform(obb["T_base_obb"])
    return m
