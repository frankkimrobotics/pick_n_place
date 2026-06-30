#!/usr/bin/env python3
"""SAFE first motion test of online_servo: plan a small slow motion and STREAM it as
continuous chunks (slices @ 10 Hz) directly to the Pi online_servo chunk port, watching
the :9999 feedback. 4 cm down, capped to ~MAX_DEG_S. No ROS/bridge needed for chunks."""
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
CHUNK_PORT = 9994
STREAM_PORT = 9999
MAX_DEG_S = 8.0
DOWN = list(R_to_quat_wxyz(R_from_two_axes(np.array([0, 0, -1.0]))))


def rpc(d):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=30)
    s.sendall((json.dumps(d) + "\n").encode()); b = b""
    while not b.endswith(b"\n"):
        b += s.recv(65536)
    s.close(); return json.loads(b)


def read_stream_once():
    s = socket.create_connection((PI, STREAM_PORT), timeout=3); s.settimeout(2); b = b""
    try:
        while b"\n" not in b:
            b += s.recv(4096)
    except Exception:
        pass
    s.close()
    return json.loads(b.split(b"\n")[0])["joints_deg"]


def main():
    cur_deg = read_stream_once()                      # LinuxCNC deg (actual)
    cur_rad = list(linuxcnc_deg_to_rad(cur_deg))
    fk = rpc({"type": "fk", "q": cur_rad}); p = fk["pos"][0]
    goal = [p[0], p[1], p[2] - 0.04]                  # 4 cm down, cup-down
    print(f"current tcp {[round(x,3) for x in p]} -> goal {[round(x,3) for x in goal]}")
    r = rpc({"type": "plan_pose", "start_q": cur_rad, "goal_pose": goal + DOWN, "max_attempts": 8})
    if not r.get("success"):
        print("plan FAILED"); return
    traj = np.array(r["trajectory"]); dt = float(r["dt"])
    deg = np.array([rad_to_linuxcnc_deg(w) for w in traj])           # LinuxCNC deg
    # slow uniform resample so peak per-joint speed <= MAX_DEG_S
    peak = float(np.abs(np.diff(deg, axis=0)).max()) / dt if len(deg) > 1 else MAX_DEG_S
    sdt = dt * max(1.0, peak / MAX_DEG_S)
    tt = np.arange(len(deg)) * sdt
    fine = np.column_stack([np.interp(np.arange(0, tt[-1] + 1e-9, 0.01), tt, deg[:, j]) for j in range(6)])
    T = len(fine) * 0.01
    print(f"streaming {len(fine)} pts over {T:.1f}s (~{MAX_DEG_S:.0f} deg/s). Watch the arm.")

    sock = socket.create_connection((PI, CHUNK_PORT), timeout=3)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    t0 = time.time() + 0.3
    clen = 25                                          # 0.25 s slices @ 10 Hz
    while True:
        now = time.time()
        k = int((now + 0.05 - t0) / 0.01)
        if k >= len(fine):
            break
        if k < 0:
            time.sleep(0.05); continue
        chunk = fine[k:k + clen]
        msg = {"trajectory": [row.tolist() for row in chunk], "traj_dt": 0.01,
               "t_anchor": float(t0 + k * 0.01)}
        sock.sendall((json.dumps(msg) + "\n").encode())
        time.sleep(0.1)
    sock.close()
    time.sleep(1.5)                                    # let the servo settle (it holds the last ref)
    final = read_stream_once()
    err = float(np.abs(linuxcnc_deg_to_rad(final) - traj[-1]).max())
    print(f"final joints_deg {[round(x,1) for x in final]}  reach err {np.degrees(err):.1f} deg "
          f"-> {'REACHED' if err < 0.05 else 'short (check)'}")


if __name__ == "__main__":
    main()
