#!/usr/bin/env python3
"""Multi-object calibrated center-touch. Scans the table, then for EACH detected object:
fast approach -> measure a CLEAN surface height (rim-gapped offset ring, deproject+MAD, no
cup-edge shadow) -> slow descent to the calibrated touch depth -> gentle touch -> lift ->
next. Returns to base.

Calibration folded in (from calib_touch.py, 2026-07-03):
  * physical cup tip is ~19mm ABOVE the FK tip (cup compressed past the 0.135 TCP model),
    so we descend the FK tip to (surface_z - DESCEND_BELOW), DESCEND_BELOW ~ 0.019.
  * surface height from an OFFSET ring (cup mask dilated by a 6px GAP so the ring clears the
    rim shadow / flying pixels), deprojected to base-z with MAD outlier rejection.

  source /opt/ros/humble/setup.bash
  python3 multi_touch.py
"""
import argparse, json, os, socket, sys, time
import numpy as np
import cv2
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")),
          os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc", "session_tools")),
          os.path.abspath(os.path.join(HERE, "..", "ros2node", "perception"))):
    sys.path.insert(0, p)
import config as C
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
def stream_to(goal_xyz, quat, Vmax, settle=True):
    qd = read_q()
    r = rpc({"type": "plan_pose", "start_q": qrad(qd), "goal_pose": list(goal_xyz) + list(quat), "max_attempts": 16})
    if not r.get("success"): return False
    traj = np.array(r["trajectory"]); dt = r["dt"]
    traj[:, 5] = float(C.BASE_Q[5])   # Task C: hold J6 at base (symmetric cup) -> no excess wrist roll & J6~0
                                      # verified tip-pose-preserving to <0.001mm (taskC_verify.py)
    peak = np.degrees(np.max(np.abs(np.diff(traj, axis=0)))) / dt if len(traj) > 1 else Vmax
    sdt = dt * max(1.0, peak / Vmax)
    td = [list(map(float, rad_to_linuxcnc_deg(wp))) for wp in traj]
    send_chunk({"trajectory": td, "traj_dt": sdt, "t_anchor": time.time() + 0.12})
    if settle: time.sleep(sdt * (len(traj) - 1) + 1.6)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v-approach", type=float, default=26.0)
    ap.add_argument("--v-des", type=float, default=6.0)
    ap.add_argument("--v-touch", type=float, default=2.5)
    ap.add_argument("--standoff", type=float, default=0.05)
    ap.add_argument("--descend-below", type=float, default=0.019,
                    help="descend the FK tip this far below the measured surface (m); = TCP offset for a just-touch")
    ap.add_argument("--abs-floor", type=float, default=-0.03, help="SAFE absolute z stop (m), above the table -0.10")
    ap.add_argument("--scouts",
                    default="0.30,-0.15,0.30;0.30,0.0,0.30;0.30,0.15,0.30;0.43,-0.15,0.30;0.43,0.0,0.30;0.43,0.15,0.30",
                    help="';'-separated scout tip xyz poses to tile the table (wrist-cam FOV is small)")
    ap.add_argument("--gap-px", type=int, default=6); ap.add_argument("--ring-w", type=int, default=12)
    ap.add_argument("--dwell", type=float, default=1.2, help="hold-at-touch seconds")
    args = ap.parse_args()

    pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(SER)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    prof = pipe.start(cfg); scale = prof.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    it = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[it.fx, 0, it.ppx], [0, it.fy, it.ppy], [0, 0, 1.]])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    for _ in range(12): pipe.wait_for_frames(2000)

    cm = np.load(os.path.join(C.OUT_DIR, "cup_mask.npz")); cup = cm["mask"].astype(np.uint8)
    def dil(m, r): return cv2.dilate(m, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    inner = dil(cup, args.gap_px); outer = dil(cup, args.gap_px + args.ring_w)
    oring = (outer > 0) & (inner == 0)                  # thin annulus, gapped off the rim (no shadow)
    ry, rx = np.where(oring)
    dargs = SimpleNamespace(xmin=0.15, xmax=0.55, ymin=-0.28, ymax=0.28, max_h=0.13, max_foot=0.18)

    def fk_T(qd):
        p, q = fk(qd)
        return make_T(quat_wxyz_to_R(q), p) @ make_T(np.eye(3), [0, 0, C.CAM_TCP_Z_SHIFT]) @ C.T_TCP_CAM
    def grab():
        f = align.process(pipe.wait_for_frames(1000))
        rgb = np.asanyarray(f.get_color_frame().get_data())
        depth = np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32) * scale
        return rgb, depth
    def clean_surface_z(qd, depth):
        """base-z surface height under the cup from the offset ring: deproject + MAD reject flying px."""
        Tbc = fk_T(qd); d = depth[ry, rx]; ok = (d > 0.05) & (d < 0.6)
        if ok.sum() < 30: return None
        dd = d[ok]; uu = rx[ok]; vv = ry[ok]
        Xc = (uu - cx) * dd / fx; Yc = (vv - cy) * dd / fy
        bz = (np.stack([Xc, Yc, dd, np.ones(len(dd))], 1) @ Tbc.T)[:, 2]
        med = np.median(bz); mad = np.median(np.abs(bz - med)) + 1e-9
        return float(np.median(bz[np.abs(bz - med) < 3 * 1.4826 * mad]))

    # ---- multi-pose scout: tile the table, merge + dedupe detections ----
    scouts = [[float(v) for v in p.split(",")] for p in args.scouts.split(";")]
    seen = []
    for si, sc in enumerate(scouts):
        print(f"[scout {si+1}/{len(scouts)}] -> {sc}")
        if not stream_to(sc, DOWN, 22.0): continue
        time.sleep(0.6)
        qd = read_q(); Tbc = fk_T(qd); rgb, depth = grab()
        for o in detect_objects(rgb, depth, K, Tbc, segment, deproject_mask, dargs):
            if not (0.0 < o["hi"][2] < 0.12): continue              # sanity: reject phantoms
            c = np.array(o["centroid"][:2], float)
            if any(np.linalg.norm(c - u["xy"]) < 0.04 for u in seen): continue   # dedupe (<4cm = same)
            seen.append({"xy": c, "top": float(o["hi"][2])})
            print(f"    + object at [{c[0]:.3f},{c[1]:.3f}] top~{o['hi'][2]:.3f}")
    seen.sort(key=lambda u: (round(u["xy"][0], 2), u["xy"][1]))
    print(f"[scout] {len(seen)} unique object(s) total")
    if not seen: print("no objects."); pipe.stop(); return

    # ---- touch each ----
    for i, u in enumerate(seen):
        oxy = u["xy"]; otop = u["top"]
        print(f"\n=== object #{i} at [{oxy[0]:.3f},{oxy[1]:.3f}] (SAM3 top {otop:.3f}) ===")
        pre = [float(oxy[0]), float(oxy[1]), float(otop + args.standoff)]
        quat = DOWN                              # J6 is held at base inside stream_to (symmetric cup)
        if not stream_to(pre, quat, args.v_approach): print("   approach failed, skip"); continue
        time.sleep(0.8)
        print(f"   wrist J6 = {read_q()[5]:+.1f} deg  (held ~base)")
        # clean surface height from the offset ring, median over frames
        ss = []
        for _ in range(12):
            _, d = grab(); s = clean_surface_z(read_q(), d)
            if s is not None: ss.append(s)
        if not ss: print("   no surface reading, skip"); continue
        surf = float(np.median(ss))
        target = max(surf - args.descend_below, args.abs_floor)   # SAFE-clamped FK touch target
        print(f"   surface(offset-ring) = {surf:.4f}  (SAM3 {otop:.3f})  -> FK touch target {target:.4f}")
        # two-phase slow descent: to just above, then gently to target
        stream_to([float(oxy[0]), float(oxy[1]), float(surf + 0.008)], quat, args.v_des)
        stream_to([float(oxy[0]), float(oxy[1]), float(target)], quat, args.v_touch)
        tip = fk(read_q())[0]
        print(f"   >>> TOUCH: FK tip z={tip[2]:.4f} (physical ~{tip[2]+args.descend_below:.4f} = surface {surf:.3f})  "
              f"J6={read_q()[5]:+.1f}deg")
        time.sleep(args.dwell)
        stream_to([float(oxy[0]), float(oxy[1]), float(surf + 0.10)], quat, 12.0)   # lift for transit

    print("\n[done] touched all objects; returning to base")
    r = rpc({"type": "plan_joint", "start_q": qrad(read_q()), "goal_q": [float(v) for v in C.BASE_Q], "max_attempts": 12})
    if r.get("success"):
        traj = np.array(r["trajectory"]); dt = r["dt"]
        peak = np.degrees(np.max(np.abs(np.diff(traj, axis=0)))) / dt
        sdt = dt * max(1.0, peak / 22.0)
        send_chunk({"trajectory": [list(map(float, rad_to_linuxcnc_deg(wp))) for wp in traj],
                    "traj_dt": sdt, "t_anchor": time.time() + 0.12})
        time.sleep(sdt * (len(traj) - 1) + 1.6)
    pipe.stop()

if __name__ == "__main__":
    main()
