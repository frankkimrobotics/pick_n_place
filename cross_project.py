#!/usr/bin/env python3
"""Task B: cross-project a grasp point between the fixed D435 and the wrist D405 (and vice
versa). Validates that all cameras share one base frame: a point detected in one camera,
lifted to base, must reproject onto the same object in the other camera.
Saves overlays + a base-agreement summary to outputs/taskB_calib/."""
import os, sys, socket, json, time
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc", "session_tools")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "ros2node", "perception")))
import config as C
from geometry import R_from_two_axes, R_to_quat_wxyz, make_T, quat_wxyz_to_R
from joint_conventions import linuxcnc_deg_to_rad, rad_to_linuxcnc_deg
from object_pointclouds import deproject_mask
from capture_and_plot import segment
from real_multi import detect_objects
from types import SimpleNamespace
import pyrealsense2 as rs
OUT = os.path.join(C.OUT_DIR, "taskB_calib"); os.makedirs(OUT, exist_ok=True)
WRIST = "218622271300"; D435 = "043422070101"; PI = "10.0.0.27"
DOWN = list(map(float, R_to_quat_wxyz(R_from_two_axes(np.array([0, 0, -1.0])))))
dargs = SimpleNamespace(xmin=0.15, xmax=0.60, ymin=-0.30, ymax=0.30, max_h=0.16, max_foot=0.22)

def rpc(d):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=40); s.sendall((json.dumps(d) + "\n").encode()); b = b""
    while not b.endswith(b"\n"): b += s.recv(65536)
    s.close(); return json.loads(b)
def send_chunk(m):
    k = socket.create_connection((PI, 9994), timeout=3); k.sendall((json.dumps(m) + "\n").encode()); k.close()
def read_q():
    s = socket.create_connection((PI, 9999), timeout=3); b = b""
    while b"\n" not in b: b += s.recv(4096)
    s.close(); return list(json.loads(b.split(b"\n")[0])["joints_deg"])
def qrad(qd): return [float(v) for v in linuxcnc_deg_to_rad(qd)]
def fk(qd): r = rpc({"type": "fk", "q": qrad(qd)}); return np.array(r["pos"][0]), np.array(r["quat"][0])
def goto(xyz, quat, V=18.0):
    r = rpc({"type": "plan_pose", "start_q": qrad(read_q()), "goal_pose": list(xyz) + list(quat), "max_attempts": 16})
    if not r.get("success"): return False
    traj = np.array(r["trajectory"]); traj[:, 5] = float(C.BASE_Q[5]); dt = r["dt"]
    peak = np.degrees(np.max(np.abs(np.diff(traj, axis=0)))) / dt
    sdt = dt * max(1.0, peak / V)
    send_chunk({"trajectory": [list(map(float, rad_to_linuxcnc_deg(w))) for w in traj], "traj_dt": sdt, "t_anchor": time.time() + 0.12})
    time.sleep(sdt * (len(traj) - 1) + 1.8); return True

def open_cam(serial, w, h):
    pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
    prof = pipe.start(cfg); scale = prof.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    it = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[it.fx, 0, it.ppx], [0, it.fy, it.ppy], [0, 0, 1.]])
    for _ in range(12): pipe.wait_for_frames(2000)
    return pipe, align, scale, K
def grab(pipe, align, scale):
    f = align.process(pipe.wait_for_frames(1000))
    rgb = np.asanyarray(f.get_color_frame().get_data())[:, :, ::-1].copy()
    depth = np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32) * scale
    return rgb, depth
def project(P, Tbc, K):
    Pc = np.linalg.inv(Tbc) @ np.array([P[0], P[1], P[2], 1.0])
    if Pc[2] <= 0.02: return None
    return int(round(K[0, 0] * Pc[0] / Pc[2] + K[0, 2])), int(round(K[1, 1] * Pc[1] / Pc[2] + K[1, 2]))
def biggest(objs, tgt=(0.4, 0.0)):
    return None if not objs else min(objs, key=lambda o: np.linalg.norm(np.asarray(o["centroid"][:2]) - np.asarray(tgt)))

def main():
    # move wrist to a table-viewing pose (J6 flattened inside goto)
    print("[B] moving wrist to table-view pose...")
    goto([0.35, 0.0, 0.30], DOWN, 18.0); time.sleep(0.6)
    qd = read_q(); pw, qw = fk(qd)
    Tbc_wrist = make_T(quat_wxyz_to_R(qw), pw) @ make_T(np.eye(3), [0, 0, C.CAM_TCP_Z_SHIFT]) @ C.T_TCP_CAM
    ext = json.load(open(os.path.join(C.OUT_DIR, "extrinsics_d435.json")))
    Tbc_d435 = np.array(ext["T_base_cam435"])

    pw_pipe, pw_al, pw_sc, Kw = open_cam(WRIST, 640, 480)
    d4_pipe, d4_al, d4_sc, Kd = open_cam(D435, 848, 480)
    rgb_w, dep_w = grab(pw_pipe, pw_al, pw_sc)
    rgb_d, dep_d = grab(d4_pipe, d4_al, d4_sc)
    pw_pipe.stop(); d4_pipe.stop()

    ow = biggest(detect_objects(rgb_w, dep_w, Kw, Tbc_wrist, segment, deproject_mask, dargs))
    od = biggest(detect_objects(rgb_d, dep_d, Kd, Tbc_d435, segment, deproject_mask, dargs))
    if ow is None or od is None:
        print("need an object visible in BOTH cameras; wrist:", ow is not None, "d435:", od is not None); return
    Pw = np.asarray(ow["centroid"]); Pd = np.asarray(od["centroid"])
    print(f"  object base-centroid  wrist->{np.round(Pw,3).tolist()}   d435->{np.round(Pd,3).tolist()}")
    print(f"  base agreement (should be small) = {np.linalg.norm(Pw - Pd)*1000:.1f} mm")

    # overlays: D435 point reprojected into wrist, and wrist point into D435
    vw = rgb_w.copy(); vd = rgb_d.copy()
    pw_self = project(Pw, Tbc_wrist, Kw); pw_cross = project(Pd, Tbc_wrist, Kw)   # wrist img: own(green) + d435->wrist(red)
    pd_self = project(Pd, Tbc_d435, Kd);  pd_cross = project(Pw, Tbc_d435, Kd)    # d435 img: own(green) + wrist->d435(red)
    for img, self_px, cross_px, txt in [(vw, pw_self, pw_cross, "WRIST D405"), (vd, pd_self, pd_cross, "fixed D435")]:
        if self_px: cv2.circle(img, self_px, 10, (0, 220, 0), 2); cv2.putText(img, "own", (self_px[0]+8, self_px[1]), 0, 0.5, (0,220,0), 2)
        if cross_px: cv2.drawMarker(img, cross_px, (255, 40, 40), cv2.MARKER_CROSS, 22, 2); cv2.putText(img, "cross-proj", (cross_px[0]+8, cross_px[1]+16), 0, 0.5, (255,40,40), 2)
        cv2.putText(img, txt, (10, 26), 0, 0.8, (255, 255, 0), 2)
    err_w = np.linalg.norm(np.subtract(pw_self, pw_cross)) if (pw_self and pw_cross) else -1
    err_d = np.linalg.norm(np.subtract(pd_self, pd_cross)) if (pd_self and pd_cross) else -1
    print(f"  reprojection error  in-wrist={err_w:.0f}px   in-d435={err_d:.0f}px")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
    ax[0].imshow(vw[:, :, ::-1]); ax[0].set_title(f"WRIST D405 — green=own detect, red=D435→wrist  (err {err_w:.0f}px)"); ax[0].axis("off")
    ax[1].imshow(vd[:, :, ::-1]); ax[1].set_title(f"fixed D435 — green=own detect, red=wrist→D435  (err {err_d:.0f}px)"); ax[1].axis("off")
    fig.suptitle(f"Task B cross-projection — base agreement {np.linalg.norm(Pw-Pd)*1000:.0f}mm (green & red should coincide)", fontsize=13)
    fig.tight_layout(); p = os.path.join(OUT, "cross_projection.png"); fig.savefig(p, dpi=110); print("saved", p)

if __name__ == "__main__":
    main()
