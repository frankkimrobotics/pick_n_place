"""perception :: synthetic eye-in-hand RGBD + segmentation + deprojection.

Because the real robot/D405/SAM-3 are offline, we *render* the eye-in-hand RGBD
from the known scene meshes (trimesh + embree ray casting), using the calibrated
D405 intrinsics and the camera pose  T_base_cam = FK_tcp(q) @ T_tcp_cam.

The rest of the perception path is the SAME as on real data:
  * a segmentation mask (here the ground-truth object id from the render; an
    optional SAM-3 ZMQ hook is provided for real images)
  * pinhole deprojection of masked depth -> camera-frame points
    (formula mirrored verbatim from ros2node/perception/object_pointclouds.py)
  * transform camera-frame points -> base frame

Camera optical frame convention: +X right, +Y down, +Z forward.
"""
import numpy as np

import config as C
from geometry import transform_points


# --------------------------------------------------------------------------- #
# Synthetic RGBD render (ray cast)
# --------------------------------------------------------------------------- #
def render_rgbd(T_base_cam, scene, Ks=None, w=None, h=None):
    """Render an RGBD image of `scene` from camera pose `T_base_cam`.

    Returns dict: rgb (h,w,3 u8), depth (h,w f32 metres, 0 = miss),
    mask (h,w int32 object-id), normals (h,w,3 f32 base-frame surface normal),
    K (3x3), T_base_cam (4x4).
    """
    if Ks is None:
        Ks, w, h = C.scaled_intrinsics()
    fx, fy = Ks[0, 0], Ks[1, 1]
    cx, cy = Ks[0, 2], Ks[1, 2]

    # Build per-pixel ray directions in the optical frame (+Z forward, +Y down).
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    dirs_cam = np.stack([(us - cx) / fx,
                         (vs - cy) / fy,
                         np.ones_like(us, dtype=np.float64)], axis=-1)  # (h,w,3)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
    dirs_base = (T_base_cam[:3, :3] @ dirs_cam.reshape(-1, 3).T).T
    origin = T_base_cam[:3, 3]
    origins = np.broadcast_to(origin, dirs_base.shape)

    depth = np.zeros((h * w,), np.float32)
    mask = np.zeros((h * w,), np.int32)
    normals = np.zeros((h * w, 3), np.float32)
    rgb = np.zeros((h * w, 3), np.uint8)
    zc_axis = T_base_cam[:3, 2]                      # camera +Z in base (for planar depth)

    light = C.SHADE_NORMAL_DIR / np.linalg.norm(C.SHADE_NORMAL_DIR)

    for oid, mesh in scene["meshes_by_id"].items():
        loc, idx_ray, idx_tri = mesh.ray.intersects_location(
            ray_origins=origins, ray_directions=dirs_base, multiple_hits=False)
        if len(idx_ray) == 0:
            continue
        # planar depth = projection of (hit - origin) onto camera +Z
        z = (loc - origin) @ zc_axis
        fn = mesh.face_normals[idx_tri]
        # orient normal towards camera
        towards = origin - loc
        flip = np.einsum("ij,ij->i", fn, towards) < 0
        fn = fn.copy(); fn[flip] = -fn[flip]
        base_col = np.asarray(mesh.visual.face_colors[idx_tri][:, :3], np.float32)
        shade = np.clip(0.35 + 0.65 * np.abs(fn @ light), 0, 1)[:, None]
        col = np.clip(base_col * shade, 0, 255).astype(np.uint8)

        # z-buffer: keep nearest hit per ray across meshes
        for k, ray_i in enumerate(idx_ray):
            zk = z[k]
            if zk <= 0:
                continue
            if depth[ray_i] == 0 or zk < depth[ray_i]:
                depth[ray_i] = zk
                mask[ray_i] = oid
                normals[ray_i] = fn[k]
                rgb[ray_i] = col[k]

    return {
        "rgb": rgb.reshape(h, w, 3),
        "depth": depth.reshape(h, w),
        "mask": mask.reshape(h, w),
        "normals": normals.reshape(h, w, 3),
        "K": Ks, "w": w, "h": h,
        "T_base_cam": T_base_cam,
    }


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
def segment_object(frame, object_id=None, backend="groundtruth", **kw):
    """Return a boolean object mask (h,w).

    backend='groundtruth' (default): use the render's object-id channel — exact and
    deterministic, the right choice for a closed-loop sim.
    backend='sam3': call the existing SAM-3 ZMQ server (needs pyzmq + a live image);
    kept as a drop-in hook for real RGB.
    """
    if backend == "groundtruth":
        oid = scene_object_id() if object_id is None else object_id
        return frame["mask"] == oid
    if backend == "sam3":
        return _segment_sam3(frame, **kw)
    raise ValueError(f"unknown segmentation backend: {backend}")


def scene_object_id():
    from scene import OBJECT_ID
    return OBJECT_ID


def _segment_sam3(frame, endpoint="tcp://127.0.0.1:5599", prompt="object",
                  timeout_ms=8000, max_instances=10):
    """Optional: query the existing SAM-3 server (protocol per sam3_server.py)."""
    import json, zmq
    rgb = frame["rgb"]
    h, w = rgb.shape[:2]
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0); s.setsockopt(zmq.RCVTIMEO, timeout_ms)
    s.setsockopt(zmq.SNDTIMEO, timeout_ms); s.connect(endpoint)
    hdr = {"h": h, "w": w, "encoding": "rgb8", "prompt": prompt,
           "max_instances": max_instances}
    s.send_multipart([json.dumps(hdr).encode(), np.ascontiguousarray(rgb).tobytes()])
    parts = s.recv_multipart()
    label = np.frombuffer(parts[1], dtype=np.uint8).reshape(h, w)
    return label > 0        # largest/any instance


# --------------------------------------------------------------------------- #
# Deprojection  (mirrors object_pointclouds.deproject_mask, pinhole model)
# --------------------------------------------------------------------------- #
def deproject(mask, frame, min_depth=None, max_depth=None):
    """Deproject masked depth into base-frame points (+ per-point normals).

    Returns dict: pts_base (N,3), pts_cam (N,3), normals_base (N,3), uv (N,2).
    """
    min_depth = C.DEPTH_MIN if min_depth is None else min_depth
    max_depth = C.DEPTH_MAX if max_depth is None else max_depth
    depth = frame["depth"]; K = frame["K"]; normals = frame["normals"]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    valid = mask & (depth > min_depth) & (depth < max_depth)
    vs, us = np.where(valid)
    z = depth[vs, us]
    x = (us.astype(np.float64) - cx) / fx * z       # X = (u-cx)/fx * Z
    y = (vs.astype(np.float64) - cy) / fy * z       # Y = (v-cy)/fy * Z
    pts_cam = np.stack([x, y, z], axis=1)           # colour optical frame

    T = frame["T_base_cam"]
    pts_base = transform_points(T, pts_cam)
    n_base = normals[vs, us]                         # already base-frame, cam-facing
    return {
        "pts_base": pts_base,
        "pts_cam": pts_cam,
        "normals_base": n_base,
        "uv": np.stack([us, vs], axis=1),
        "count": int(len(vs)),
    }
