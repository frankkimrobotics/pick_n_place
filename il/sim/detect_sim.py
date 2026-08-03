"""Vision object detection in the MuJoCo sim from the FIXED (D435) camera: render the segmentation,
take each object's centroid pixel, ray-cast through it to the table plane → base-frame grasp XY.
This is the sim analogue of the real SAM3→deproject path (perception.py), using the render's
object-id segmentation (perception.py's 'groundtruth' backend). Ray-to-plane sidesteps depth
convention issues and is exact for objects resting on the table.

  python3 il/sim/detect_sim.py    # self-test: detect vs ground-truth object XY
"""
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import config as C


def cam_K(fovy_deg, hw):
    H, W = hw; fy = 0.5 * H / np.tan(0.5 * np.radians(fovy_deg))
    return fy, fy, W / 2.0, H / 2.0                      # fx, fy, cx, cy


def detect_objects(model, data, objs, cam="top", table_z=-0.10, hw=(240, 320), min_px=25):
    """objs = [(name, qposadr, half_height)] from sim_mujoco_node. Returns [{id, xyz[base]}].
    Camera pose + fovy are read from the model (works for any named camera). 'top' (top-down) has no
    oblique-centroid bias — the right choice for accurate grasp targeting; 'fixed'/'wrist' are the
    obs cameras and carry the oblique bias real_grasp.py corrects with flat-patch detection."""
    import mujoco
    cid = model.camera(cam).id
    R_mj = np.array(data.cam_xmat[cid]).reshape(3, 3)
    R = R_mj @ np.diag([1.0, -1.0, -1.0])               # MuJoCo cam (-Z fwd) → optical (+Z fwd)
    origin = np.array(data.cam_xpos[cid])
    fx, fy, cx, cy = cam_K(float(model.cam_fovy[cid]), hw)
    r = mujoco.Renderer(model, hw[0], hw[1]); r.enable_segmentation_rendering()
    r.update_scene(data, camera=cam); seg = r.render()[:, :, 0]     # geom-id per pixel (−1 = none)
    dets = []
    for oi, (name, adr, hh) in enumerate(objs):
        try:
            gid = model.geom(f"g_object{oi}").id
        except Exception:
            continue
        m = seg == gid
        if int(m.sum()) < min_px:                        # not visible (parked / occluded)
            continue
        vs, us = np.where(m); u = us.mean(); v = vs.mean()
        d = np.array([(u - cx) / fx, (v - cy) / fy, 1.0]); d /= np.linalg.norm(d)  # optical ray
        db = R @ d
        if abs(db[2]) < 1e-6:
            continue
        t = (table_z + hh - origin[2]) / db[2]           # hit the object-top plane
        p = origin + t * db
        dets.append({"id": name, "geom": oi, "px": [float(u), float(v)],
                     "xyz": [float(p[0]), float(p[1]), float(table_z + hh)]})
    return dets


def _load_fixed_extrinsic():
    d = json.load(open(os.path.join(C.OUT_DIR, "extrinsics_d435_static.json")))
    return np.array(d["T_base_cam435"], float)


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import mujoco
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "mycobot_mpc")))
    from capture_cameras import build_capture_xml
    xml = os.path.join(C.OUT_DIR, "mujoco_sim", "robot_sim_capture.xml")
    build_capture_xml(os.path.join(C.OUT_DIR, "mujoco_sim", "robot_sim.xml"), xml)
    m = mujoco.MjModel.from_xml_path(xml); d = mujoco.MjData(m)
    # build objs list like the node does
    objs = []
    i = 0
    while True:
        try:
            adr = m.joint(f"obj{i}_free").qposadr[0]
        except Exception:
            break
        gid = m.geom(f"g_object{i}").id; sz = m.geom_size[gid]
        hh = float(sz[2] if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX else sz[1])
        objs.append((f"object{i}", int(adr), hh)); i += 1
    mujoco.mj_forward(m, d)
    adrs = dict((n, a) for n, a, _ in objs)
    for cam in ("top", "fixed"):
        dets = detect_objects(m, d, objs, cam=cam)
        errs = []
        for det in dets:
            gt = d.qpos[adrs[det["id"]]:adrs[det["id"]] + 2]
            errs.append(1000 * np.hypot(det["xyz"][0] - gt[0], det["xyz"][1] - gt[1]))
        print(f"[{cam:5}] {len(dets)} objects  mean_err={np.mean(errs):.1f}mm  max={np.max(errs):.1f}mm")
