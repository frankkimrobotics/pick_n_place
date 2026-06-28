"""scene :: the simulated world geometry (trimesh).

Builds ground-truth meshes for: the table, the pick object (a low cylinder
"puck" with a clean flat top = a good suction target), and the open-top place
box. These meshes are used by:
  * perception.py  — synthetic eye-in-hand RGBD rendering (ray cast)
  * simulator.py   — carried-object collision sweep & final-pose evaluation

All meshes live in the base_link frame, metres.
"""
import numpy as np
import trimesh

import config as C
from geometry import make_T


# Stable integer ids encoded into the render mask
TABLE_ID, OBJECT_ID, BOX_ID = 1, 2, 3


def make_table():
    w, d, h = C.TABLE_DIMS
    m = trimesh.creation.box(extents=(w, d, h))
    # top face at TABLE_Z -> centre at TABLE_Z - h/2
    m.apply_translation([0.30, 0.10, C.TABLE_Z - h / 2.0])
    m.visual.face_colors = [150, 150, 155, 255]
    return m


def make_object(pose_xyz=None):
    """Low cylinder resting on the table; flat top is the suction surface.

    Returns (mesh, info) where info has the true top-centre, top normal and pose.
    """
    pose_xyz = C.OBJECT_POSE_XYZ if pose_xyz is None else np.asarray(pose_xyz, float)
    r, h = C.OBJECT_RADIUS, C.OBJECT_HEIGHT
    m = trimesh.creation.cylinder(radius=r, height=h, sections=64)
    # cylinder is centred at origin with axis +Z; base sits on table => centre at z=table+h/2
    cz = pose_xyz[2] + h / 2.0
    T = make_T(np.eye(3), [pose_xyz[0], pose_xyz[1], cz])
    m.apply_transform(T)
    m.visual.face_colors = [200, 90, 60, 255]
    top_centre = np.array([pose_xyz[0], pose_xyz[1], pose_xyz[2] + h])
    info = {
        "pose_xyz": np.asarray(pose_xyz, float),
        "radius": r, "height": h,
        "top_centre": top_centre,
        "top_normal": np.array([0.0, 0.0, 1.0]),
        "T_base_obj": make_T(np.eye(3), [pose_xyz[0], pose_xyz[1], cz]),  # obj centre frame
        "centre": np.array([pose_xyz[0], pose_xyz[1], cz]),
    }
    return m, info


def make_box(box_xyz=None):
    """Open-top bin (4 walls + bottom) as a single mesh, plus geometry info."""
    box_xyz = C.BOX_POSE_XYZ if box_xyz is None else np.asarray(box_xyz, float)
    bx, by, bz = box_xyz
    W, D, H = C.BOX_INNER
    t = C.BOX_WALL
    parts = []

    def slab(ext, ctr):
        b = trimesh.creation.box(extents=ext)
        b.apply_translation(ctr)
        return b

    zc = bz + H / 2.0
    parts.append(slab((W + 2 * t, D + 2 * t, t), [bx, by, bz - t / 2.0]))      # cavity floor
    parts.append(slab((t, D + 2 * t, H), [bx - (W / 2 + t / 2), by, zc]))      # -x wall
    parts.append(slab((t, D + 2 * t, H), [bx + (W / 2 + t / 2), by, zc]))      # +x wall
    parts.append(slab((W, t, H), [bx, by - (D / 2 + t / 2), zc]))             # -y wall
    parts.append(slab((W, t, H), [bx, by + (D / 2 + t / 2), zc]))             # +y wall
    # cosmetic riser: solid pedestal from the table up to the cavity floor so the
    # tall bin stands on the table instead of floating (only when floor is above table)
    riser_h = bz - C.TABLE_Z
    if riser_h > 1e-3:
        parts.append(slab((W + 2 * t, D + 2 * t, riser_h),
                          [bx, by, C.TABLE_Z + riser_h / 2.0]))
    m = trimesh.util.concatenate(parts)
    m.visual.face_colors = [70, 110, 200, 255]
    info = {
        "pose_xyz": box_xyz,
        "inner": (W, D, H),
        "wall": t,
        "centre_xy": np.array([bx, by]),
        "floor_z": bz,
        "rim_z": bz + H,
        # XY footprint half-extents of the inner opening
        "half_xy": np.array([W / 2.0, D / 2.0]),
    }
    return m, info


def build_scene(object_pose_xyz=None, box_xyz=None):
    """Assemble the world. Returns dict with meshes + per-object info + id map."""
    table = make_table()
    obj, obj_info = make_object(object_pose_xyz)
    box, box_info = make_box(box_xyz)
    return {
        "table": table,
        "object": obj,
        "box": box,
        "object_info": obj_info,
        "box_info": box_info,
        "ids": {"table": TABLE_ID, "object": OBJECT_ID, "box": BOX_ID},
        "meshes_by_id": {TABLE_ID: table, OBJECT_ID: obj, BOX_ID: box},
    }
