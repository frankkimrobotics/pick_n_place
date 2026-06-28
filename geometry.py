"""geometry :: pose / quaternion / transform helpers (numpy + scipy).

cuRobo uses **wxyz** quaternions and poses of the form [x,y,z, qw,qx,qy,qz].
scipy.spatial.transform.Rotation uses **xyzw** — all conversions are centralised
here so the wxyz/xyzw order is never mixed up elsewhere.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


# ---- quaternion <-> matrix (wxyz convention) ------------------------------ #
def quat_wxyz_to_R(q):
    """wxyz quaternion -> 3x3 rotation matrix."""
    w, x, y, z = q
    return R.from_quat([x, y, z, w]).as_matrix()


def R_to_quat_wxyz(M):
    """3x3 rotation matrix -> wxyz quaternion."""
    x, y, z, w = R.from_matrix(np.asarray(M)[:3, :3]).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


# ---- homogeneous transforms ----------------------------------------------- #
def make_T(Rm, t):
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def pose_to_T(pose):
    """pose [x,y,z, qw,qx,qy,qz] -> 4x4."""
    pose = np.asarray(pose, dtype=np.float64)
    return make_T(quat_wxyz_to_R(pose[3:7]), pose[0:3])


def T_to_pose(T):
    """4x4 -> pose [x,y,z, qw,qx,qy,qz]."""
    T = np.asarray(T)
    return np.concatenate([T[:3, 3], R_to_quat_wxyz(T[:3, :3])])


def inv_T(T):
    Rm = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = Rm.T
    Ti[:3, 3] = -Rm.T @ t
    return Ti


def transform_points(T, pts):
    """Apply 4x4 T to Nx3 points."""
    pts = np.asarray(pts, dtype=np.float64)
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


# ---- orientation construction -------------------------------------------- #
def R_from_two_axes(z_axis, x_hint=np.array([1.0, 0.0, 0.0])):
    """Build a right-handed rotation whose +Z is `z_axis`, with +X near `x_hint`.

    Used to orient the tcp so that its approach axis (tcp +Z) points along a
    desired direction (e.g. into the surface = -surface_normal).
    """
    z = np.asarray(z_axis, dtype=np.float64)
    z = z / (np.linalg.norm(z) + 1e-12)
    x_hint = np.asarray(x_hint, dtype=np.float64)
    if abs(np.dot(x_hint, z)) > 0.95:                 # hint nearly parallel to z
        x_hint = np.array([0.0, 1.0, 0.0])
    x = x_hint - np.dot(x_hint, z) * z
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)                # columns = axes


def look_at_R(eye, target, up=np.array([0.0, 0.0, 1.0])):
    """Rotation for a camera at `eye` looking at `target`, optical +Z = view dir.

    Optical frame: +Z forward (view), +X right, +Y down.
    """
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    z = target - eye
    z = z / (np.linalg.norm(z) + 1e-12)               # forward (+Z optical)
    up = np.asarray(up, float)
    x = np.cross(up, z)                               # +X right... but optical +Y is down
    if np.linalg.norm(x) < 1e-6:
        up = np.array([0.0, 1.0, 0.0]); x = np.cross(up, z)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)                                # +Y down
    return np.stack([x, y, z], axis=1)


def angle_between(a, b):
    """Angle (radians) between two vectors."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    c = np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12)
    return float(np.arccos(np.clip(c, -1.0, 1.0)))
