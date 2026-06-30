#!/usr/bin/env python3
"""Stream a slow JOINT move to BASE_Q through online_servo (direct joint plan, no IK
branch surprises). Brings the arm back to base after a streaming test."""
import json
import os
import socket
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mycobot_mpc")))
import config as C
from joint_conventions import linuxcnc_deg_to_rad, rad_to_linuxcnc_deg

PI = "10.0.0.27"
MAX_DEG_S = 8.0


def rpc(d):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=30)
    s.sendall((json.dumps(d) + "\n").encode()); b = b""
    while not b.endswith(b"\n"):
        b += s.recv(65536)
    s.close(); return json.loads(b)


def read_stream():
    s = socket.create_connection((PI, 9999), timeout=3); s.settimeout(2); b = b""
    try:
        while b"\n" not in b:
            b += s.recv(4096)
    except Exception:
        pass
    s.close(); return json.loads(b.split(b"\n")[0])["joints_deg"]


def main():
    cur = list(linuxcnc_deg_to_rad(read_stream()))
    base = list(map(float, C.BASE_Q))
    r = rpc({"type": "plan_joint", "start_q": cur, "goal_q": base})
    if not r.get("success"):
        print("plan FAILED"); return
    deg = np.array([rad_to_linuxcnc_deg(w) for w in np.array(r["trajectory"])])
    dt = float(r["dt"])
    peak = float(np.abs(np.diff(deg, axis=0)).max()) / dt if len(deg) > 1 else MAX_DEG_S
    sdt = dt * max(1.0, peak / MAX_DEG_S)
    tt = np.arange(len(deg)) * sdt
    fine = np.column_stack([np.interp(np.arange(0, tt[-1] + 1e-9, 0.01), tt, deg[:, j]) for j in range(6)])
    print(f"returning to base: {len(fine)} pts over {len(fine)*0.01:.1f}s (~{MAX_DEG_S:.0f} deg/s)")
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
    sock.close(); time.sleep(1.5)
    final = read_stream()
    dev = float(np.degrees(np.abs(linuxcnc_deg_to_rad(final) - np.array(base))).max())
    print(f"final {[round(x,1) for x in final]}  dev from base {dev:.1f} deg")


if __name__ == "__main__":
    main()
