"""Inject wrist (D405) + fixed (D435) cameras into robot_sim.xml so the sim renders the SAME two
RGB views the real robot observes. Poses come from the real calibrations:
  wrist  : T_TCP_CAM (tcp<-cam, OpenCV optical) → child of the `tcp` body.
  fixed  : T_base_cam435 (base<-cam, OpenCV)    → child of worldbody.
OpenCV optical (+Z fwd,+Y down) → MuJoCo camera (-Z fwd,+Y up) via a 180° flip about X.

  python3 il/sim/capture_cameras.py            # builds robot_sim_capture.xml + renders a test frame
"""
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import config as C

CV_TO_MJ = np.diag([1.0, -1.0, -1.0])          # OpenCV optical → MuJoCo camera axes


def mat2quat(R):
    m = np.asarray(R, float); t = np.trace(m)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1); w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s; y = (m[0, 2] - m[2, 0]) * s; z = (m[1, 0] - m[0, 1]) * s
    else:
        i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        if i == 0:
            s = 2 * np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s; y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
        elif i == 1:
            s = 2 * np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s; y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2 * np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s; y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
    return np.array([w, x, y, z])


def _cam_attr(T):
    p = T[:3, 3]; q = mat2quat(T[:3, :3] @ CV_TO_MJ)
    return (f'pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" '
            f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"')


def build_capture_xml(xml_in, xml_out, fovy_wrist=58.0, fovy_fixed=42.0, d435_json=None):
    T_wrist = np.array(C.T_TCP_CAM, float)
    d435_json = d435_json or os.path.join(C.OUT_DIR, "extrinsics_d435_static.json")
    T_fixed = np.array(json.load(open(d435_json))["T_base_cam435"], float)
    wrist = f'<camera name="wrist" {_cam_attr(T_wrist)} fovy="{fovy_wrist}"/>'
    fixed = f'<camera name="fixed" {_cam_attr(T_fixed)} fovy="{fovy_fixed}"/>'

    txt = open(xml_in).read()
    # wrist: insert as a child right after the `<body name="tcp" ...>` opening tag
    key = '<body name="tcp"'
    i = txt.index(key); j = txt.index('>', i) + 1
    txt = txt[:j] + "\n                                            " + wrist + txt[j:]
    # fixed: insert after <worldbody>
    k = txt.index('<worldbody>') + len('<worldbody>')
    txt = txt[:k] + "\n    " + fixed + txt[k:]
    os.makedirs(os.path.dirname(xml_out) or ".", exist_ok=True)
    open(xml_out, "w").write(txt)
    return xml_out


def render_test(xml_out, out_png):
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import mujoco, cv2
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "mycobot_mpc")))
    from joint_conventions import JOINT_NAMES
    m = mujoco.MjModel.from_xml_path(xml_out); d = mujoco.MjData(m)
    for n, v in zip(JOINT_NAMES, C.BASE_Q):
        d.qpos[m.joint(n).qposadr[0]] = float(v)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 240, 320)
    imgs = []
    for cam in ("wrist", "fixed"):
        r.update_scene(d, camera=cam); imgs.append(r.render())
    grid = np.concatenate(imgs, axis=1)          # side by side
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    cv2.imwrite(out_png, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    return [(cam, float(im.mean()), im.shape) for cam, im in zip(("wrist", "fixed"), imgs)], out_png


if __name__ == "__main__":
    xin = os.path.join(C.OUT_DIR, "mujoco_sim", "robot_sim.xml")
    xout = os.path.join(C.OUT_DIR, "mujoco_sim", "robot_sim_capture.xml")
    build_capture_xml(xin, xout)
    print("built", xout)
    stats, png = render_test(xout, os.path.join(C.OUT_DIR, "mujoco_sim", "capture_cams_test.png"))
    for cam, mean, shape in stats:
        print(f"  {cam}: mean_px={mean:.1f} shape={shape}  {'OK' if mean > 3 else 'BLANK?'}")
    print("wrote", png)
