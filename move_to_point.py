#!/usr/bin/env python3
"""Move the cup tip to an absolute base-frame point (cup pointing straight down) THROUGH
online_servo (slow streaming). Sweeps the in-plane wrist angle to pick the cleanest
(minimal joint swing) cup-down IK branch, FK-self-checks the tip lands on the goal, then
streams it slowly. For calibration checks: see where the cup PHYSICALLY lands vs the goal.
    python3 move_to_point.py X Y Z [--max-deg-s 8]"""
import argparse
import json
import os
import socket
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mycobot_mpc")))
import config as C
from geometry import R_from_two_axes, R_to_quat_wxyz
from joint_conventions import linuxcnc_deg_to_rad, rad_to_linuxcnc_deg

PI = "10.0.0.27"


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


def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def stream_plan(traj_rad, dt, max_deg_s):
    deg = np.array([rad_to_linuxcnc_deg(w) for w in traj_rad])
    peak = float(np.abs(np.diff(deg, axis=0)).max()) / dt if len(deg) > 1 else max_deg_s
    sdt = dt * max(1.0, peak / max_deg_s)
    tt = np.arange(len(deg)) * sdt
    fine = np.column_stack([np.interp(np.arange(0, tt[-1] + 1e-9, 0.01), tt, deg[:, j]) for j in range(6)])
    print(f"  streaming {len(fine)} pts over {len(fine)*0.01:.1f}s (<= {max_deg_s:.0f} deg/s). Watch the arm.")
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
    sock.close(); time.sleep(2.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz", nargs=3, type=float)
    ap.add_argument("--max-deg-s", type=float, default=8.0)
    ap.add_argument("--dry", action="store_true", help="plan + report only, do NOT move")
    a = ap.parse_args()
    x, y, z = a.xyz
    cur = list(linuxcnc_deg_to_rad(read_cur()))
    Rd = R_from_two_axes(np.array([0, 0, -1.0]))
    best = None
    for th in np.radians(np.arange(0, 360, 30)):
        quat = list(R_to_quat_wxyz(Rz(th) @ Rd))
        r = rpc({"type": "plan_pose", "start_q": cur, "goal_pose": [x, y, z] + quat, "max_attempts": 12})
        if r.get("success"):
            tr = np.array(r["trajectory"]); sw = float(np.degrees(np.abs(tr - tr[0]).max()))
            if best is None or sw < best[0]:
                best = (sw, r, float(np.degrees(th)))
    if best is None:
        print("NO cup-down plan found to that point"); return
    sw, r, th = best
    tr = np.array(r["trajectory"])
    fk = rpc({"type": "fk", "q": [float(v) for v in tr[-1]]}); p = fk["pos"][0]
    print(f"goal [{x},{y},{z}] cup-down -> cleanest branch: in-plane {th:.0f}deg, max joint swing {sw:.0f}deg")
    print(f"  planner FK at end-of-plan tcp = {[round(float(v),3) for v in p]}")
    print(f"  end joints_deg (LinuxCNC) = {[round(float(v),1) for v in rad_to_linuxcnc_deg(tr[-1])]}")
    if a.dry:
        print("  (dry run -- no motion)"); return
    stream_plan(tr, float(r["dt"]), a.max_deg_s)
    prev = None                                            # wait for the open-loop servo to fully settle
    for _ in range(25):
        cu = np.array(read_cur())
        if prev is not None and float(np.abs(cu - prev).max()) < 0.15:
            break
        prev = cu; time.sleep(0.3)
    final = read_cur()
    print(f"  final joints_deg = {[round(x,1) for x in final]}")
    ff = rpc({"type": "fk", "q": [float(v) for v in linuxcnc_deg_to_rad(final)]})["pos"][0]
    print(f"  planner FK at ACTUAL final tcp = {[round(float(v),3) for v in ff]}  (commanded [{x},{y},{z}])")


if __name__ == "__main__":
    main()
