#!/usr/bin/env python3
"""Touch an object on the table THROUGH online_servo (streaming): detect with the fixed
D435 (object pose already in base), plan base -> over-object -> object top (cup-down), and
stream it slowly to the Pi servo. The compliant cup + slow descent make the open-loop touch
gentle. Returns to base after. Needs online_servo running + chunk_to_pi NOT required (streams
directly to :9994)."""
import json
import os
import socket
import sys
import time
import types

import numpy as np
import pyrealsense2 as rs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mycobot_mpc")))
import config as C
from geometry import R_from_two_axes, R_to_quat_wxyz
from joint_conventions import linuxcnc_deg_to_rad, rad_to_linuxcnc_deg
from real_multi import detect_objects
from capture_and_plot import segment
from object_pointclouds import deproject_mask

PI = "10.0.0.27"
MAX_DEG_S = 7.0
DOWN = list(R_to_quat_wxyz(R_from_two_axes(np.array([0, 0, -1.0]))))


def rpc(d):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=40)
    s.sendall((json.dumps(d) + "\n").encode()); b = b""
    while not b.endswith(b"\n"):
        b += s.recv(65536)
    s.close(); return json.loads(b)


def read_cur():
    s = socket.create_connection((PI, 9999), timeout=3); s.settimeout(2); b = b""
    try:
        while b"\n" not in b:
            b += s.recv(4096)
    except Exception:
        pass
    s.close(); return json.loads(b.split(b"\n")[0])["joints_deg"]


def stream_plan(traj_rad, dt, label):
    """Stream a planned trajectory as slices to online_servo :9994 at <= MAX_DEG_S."""
    deg = np.array([rad_to_linuxcnc_deg(w) for w in traj_rad])
    peak = float(np.abs(np.diff(deg, axis=0)).max()) / dt if len(deg) > 1 else MAX_DEG_S
    sdt = dt * max(1.0, peak / MAX_DEG_S)
    tt = np.arange(len(deg)) * sdt
    fine = np.column_stack([np.interp(np.arange(0, tt[-1] + 1e-9, 0.01), tt, deg[:, j]) for j in range(6)])
    print(f"  [{label}] streaming {len(fine)} pts over {len(fine)*0.01:.1f}s")
    sock = socket.create_connection((PI, 9994), timeout=3)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    t0 = time.time() + 0.3
    while True:
        now = time.time(); k = int((now + 0.05 - t0) / 0.01)
        if k >= len(fine):
            break
        if k < 0:
            time.sleep(0.05); continue
        sock.sendall((json.dumps({"trajectory": [r.tolist() for r in fine[k:k + 25]],
                                  "traj_dt": 0.01, "t_anchor": float(t0 + k * 0.01)}) + "\n").encode())
        time.sleep(0.1)
    sock.close(); time.sleep(2.5)                       # let the open-loop servo converge


def main():
    ext = json.load(open(os.path.join(C.OUT_DIR, "extrinsics_d435.json")))
    Tbc = np.array(ext["T_base_cam435"])
    pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device("043422070101")
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
    prof = pipe.start(cfg); align = rs.align(rs.stream.color)
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.]])
    for _ in range(15):
        pipe.wait_for_frames(2000)
    fr = align.process(pipe.wait_for_frames(2000))
    bgr = np.asanyarray(fr.get_color_frame().get_data())
    depth = np.asanyarray(fr.get_depth_frame().get_data()).astype(np.float32) * 0.001
    pipe.stop()
    args = types.SimpleNamespace(xmin=0.22, xmax=0.46, ymin=-0.22, ymax=0.22, max_h=0.13, max_foot=0.20)
    objs = detect_objects(bgr[:, :, ::-1].copy(), depth, K, Tbc, segment, deproject_mask, args)
    if not objs:
        print("no reachable object detected"); return
    o = max(objs, key=lambda o: len(o["pts"]))         # largest reachable object
    P = o["pts"]; cxy = P[:, :2].mean(0); top = float(P[:, 2].max())
    print(f"target object: centre [{cxy[0]:+.3f},{cxy[1]:+.3f}] top z={top:.3f} "
          f"({len(P)} pts, h={o['height']*100:.0f}cm)")

    cur = list(linuxcnc_deg_to_rad(read_cur()))
    pre = [float(cxy[0]), float(cxy[1]), top + 0.10]    # over the object
    touch = [float(cxy[0]), float(cxy[1]), top + 0.002] # cup tip ~ object top
    r1 = rpc({"type": "plan_pose", "start_q": cur, "goal_pose": pre + DOWN, "max_attempts": 12})
    if not r1.get("success"):
        print("plan to pregrasp FAILED"); return
    q_pre = [float(x) for x in r1["trajectory"][-1]]
    r2 = rpc({"type": "plan_pose", "start_q": q_pre, "goal_pose": touch + DOWN, "max_attempts": 12})
    if not r2.get("success"):
        print("plan to touch FAILED"); return
    print("plans OK -> streaming OVER the object, then DOWN to touch. Watch the arm.")
    stream_plan(np.array(r1["trajectory"]), float(r1["dt"]), "approach")
    stream_plan(np.array(r2["trajectory"]), float(r2["dt"]), "descend-to-touch")
    print("  >> at the object (touch). holding 1.5s ...")
    time.sleep(1.5)
    # return to base
    cur2 = list(linuxcnc_deg_to_rad(read_cur()))
    rb = rpc({"type": "plan_joint", "start_q": cur2, "goal_q": list(map(float, C.BASE_Q))})
    if rb.get("success"):
        stream_plan(np.array(rb["trajectory"]), float(rb["dt"]), "return-to-base")
    print("done.")


if __name__ == "__main__":
    main()
