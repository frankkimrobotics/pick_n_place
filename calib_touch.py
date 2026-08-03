#!/usr/bin/env python3
"""One-time calibration of the cup-tip depth (tip_depth) used by visual_servo_touch.

The ring mask reports the surface reaching the "tip plane" a few mm too high, so touches
stop short. This tool descends the cup to a commanded tip z over the object and HOLDS
there, reporting the ring depth. Step it down (calling with smaller --to-z) until YOU see
it just touch; at that z the ring reading IS the true tip_depth -> paste it into config /
pass --ring-margin accordingly.

  # first call: detect the object, approach, hold just above it
  python3 calib_touch.py --to-z 0.078
  # then step down, reusing the printed xy (no re-detect):
  python3 calib_touch.py --xy 0.434,-0.012 --to-z 0.070
  python3 calib_touch.py --xy 0.434,-0.012 --to-z 0.066   ... until it just touches
"""
import argparse, json, os, socket, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")),
          os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc", "session_tools")),
          os.path.abspath(os.path.join(HERE, "..", "ros2node", "perception"))):
    sys.path.insert(0, p)
import config as C
import cup_region as CR
from geometry import R_from_two_axes, R_to_quat_wxyz, make_T, quat_wxyz_to_R
from joint_conventions import linuxcnc_deg_to_rad, rad_to_linuxcnc_deg
from object_pointclouds import deproject_mask
from capture_and_plot import segment
from real_multi import detect_objects
from types import SimpleNamespace
import pyrealsense2 as rs

PI = "10.0.0.27"; SER = "218622271300"; W, H = 640, 480
DOWN = list(map(float, R_to_quat_wxyz(R_from_two_axes(np.array([0, 0, -1.0])))))

def rpc(d):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=40); s.sendall((json.dumps(d) + "\n").encode()); b = b""
    while not b.endswith(b"\n"): b += s.recv(65536)
    s.close(); return json.loads(b)
def send_chunk(m):
    k = socket.create_connection((PI, 9994), timeout=3); k.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    k.sendall((json.dumps(m) + "\n").encode()); k.close()
def read_q():
    s = socket.create_connection((PI, 9999), timeout=3); b = b""
    while b"\n" not in b: b += s.recv(4096)
    s.close(); return list(json.loads(b.split(b"\n")[0])["joints_deg"])
def qrad(qd): return [float(v) for v in linuxcnc_deg_to_rad(qd)]
def fk(qd):
    r = rpc({"type": "fk", "q": qrad(qd)}); return np.array(r["pos"][0]), np.array(r["quat"][0])
def stream_to(goal_xyz, quat, Vmax):
    qd = read_q()
    r = rpc({"type": "plan_pose", "start_q": qrad(qd), "goal_pose": list(goal_xyz) + list(quat), "max_attempts": 14})
    if not r.get("success"): return False
    traj = np.array(r["trajectory"]); dt = r["dt"]
    peak = np.degrees(np.max(np.abs(np.diff(traj, axis=0)))) / dt if len(traj) > 1 else Vmax
    sdt = dt * max(1.0, peak / Vmax)
    td = [list(map(float, rad_to_linuxcnc_deg(wp))) for wp in traj]
    send_chunk({"trajectory": td, "traj_dt": sdt, "t_anchor": time.time() + 0.12})
    time.sleep(sdt * (len(traj) - 1) + 1.6)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xy", default="", help="object x,y (skip detect)")
    ap.add_argument("--to-z", type=float, required=True, help="commanded cup-tip z to hold at (m)")
    ap.add_argument("--v", type=float, default=3.0)
    args = ap.parse_args()

    pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(SER)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    prof = pipe.start(cfg); scale = prof.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    it = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[it.fx, 0, it.ppx], [0, it.fy, it.ppy], [0, 0, 1.]])
    for _ in range(12): pipe.wait_for_frames(2000)
    cm = np.load(os.path.join(C.OUT_DIR, "cup_mask.npz")); ring = cm["ring"]
    M_ct = make_T(np.eye(3), [0, 0, C.CAM_TCP_Z_SHIFT]) @ C.T_TCP_CAM
    tip_depth = float(np.linalg.inv(M_ct)[2, 3])

    if args.xy:
        obj_xy = np.array([float(v) for v in args.xy.split(",")])
    else:
        qd = read_q(); p, q = fk(qd)
        Tbc = make_T(quat_wxyz_to_R(q), p) @ make_T(np.eye(3), [0, 0, C.CAM_TCP_Z_SHIFT]) @ C.T_TCP_CAM
        f = align.process(pipe.wait_for_frames(1000))
        rgb = np.asanyarray(f.get_color_frame().get_data())
        depth = np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32) * scale
        dargs = SimpleNamespace(xmin=0.15, xmax=0.60, ymin=-0.30, ymax=0.30, max_h=0.16, max_foot=0.20)
        objs = detect_objects(rgb, depth, K, Tbc, segment, deproject_mask, dargs)
        if not objs: print("no object"); pipe.stop(); return
        o = min(objs, key=lambda o: np.linalg.norm(o["centroid"][:2] - np.array([0.4, 0.0])))
        obj_xy = np.array(o["centroid"][:2], float)
        print(f"detected object xy = [{obj_xy[0]:.3f},{obj_xy[1]:.3f}]  (pass --xy {obj_xy[0]:.3f},{obj_xy[1]:.3f})")

    print(f"moving cup tip to [{obj_xy[0]:.3f},{obj_xy[1]:.3f}, {args.to_z:.3f}] ...")
    if not stream_to([float(obj_xy[0]), float(obj_xy[1]), float(args.to_z)], DOWN, args.v):
        print("plan failed"); pipe.stop(); return
    time.sleep(0.8)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ys, xs = np.where(ring)
    cup_m, above_m = CR.load(os.path.join(C.OUT_DIR, "cup_region_fixed.npz"))   # fixed cup(hull)+above-band
    qd = read_q(); pcur, qcur = fk(qd); tip = pcur
    Tbc = make_T(quat_wxyz_to_R(qcur), pcur) @ make_T(np.eye(3), [0, 0, C.CAM_TCP_Z_SHIFT]) @ C.T_TCP_CAM
    rawd = []; basez = []; cup_ds = []; surf_ds = []
    for _ in range(20):
        f = align.process(pipe.wait_for_frames(1000))
        depth = np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32) * scale
        cd, sd = CR.read_depths(depth, cup_m, above_m)
        if cd is not None: cup_ds.append(cd)
        if sd is not None: surf_ds.append(sd)
        d = depth[ys, xs]; ok = (d > 0.05) & (d < 0.5)
        dd = d[ok]; uu = xs[ok]; vv = ys[ok]
        rawd.extend(dd.tolist())
        Xc = (uu - cx) * dd / fx; Yc = (vv - cy) * dd / fy
        Pb = np.stack([Xc, Yc, dd, np.ones(len(dd))], 1) @ Tbc.T
        basez.extend(Pb[:, 2].tolist())
    rawd = np.array(rawd); basez = np.array(basez)
    # CLEAN surface height: reject flying pixels via MAD, then the surface is the robust median base-z
    med = np.median(basez); mad = np.median(np.abs(basez - med)) + 1e-9
    clean = basez[np.abs(basez - med) < 3 * 1.4826 * mad]
    surf_z = float(np.median(clean))
    print("=" * 60)
    print(f"  HELD at tip z = {tip[2]:.4f}   (commanded {args.to_z:.3f})")
    print(f"  raw ring depth  p10/50/90 = {np.percentile(rawd,10):.3f}/{np.percentile(rawd,50):.3f}/"
          f"{np.percentile(rawd,90):.3f}  std={rawd.std()*1000:.0f}mm  (noise/flying-pixel spread)")
    print(f"  ring->base-z    p10/50/90 = {np.percentile(basez,10):.3f}/{np.percentile(basez,50):.3f}/"
          f"{np.percentile(basez,90):.3f}")
    print(f"  CLEAN surface z = {surf_z:.4f}  ({len(clean)}/{len(basez)} px kept)")
    print(f"  => gap  tip - surface = {(tip[2]-surf_z)*1000:+.1f} mm  (should match what you SEE)")
    # NEW cup-region signal (what multi_touch uses): cup(hull) vs above-band depth
    cup_d = float(np.median(cup_ds)) if cup_ds else None
    surf_d = float(np.median(surf_ds)) if surf_ds else None
    if cup_d is not None and surf_d is not None:
        print(f"  cup_region: cup_d={cup_d:.4f}  above_d={surf_d:.4f}  GAP(above-cup)={(surf_d-cup_d)*1000:+.1f} mm")
        print(f"     -> when it JUST TOUCHES, set multi_touch --contact-gap to this GAP value")
    print("=" * 60)
    pipe.stop()

if __name__ == "__main__":
    main()
