#!/usr/bin/env python3
"""SAFE first hardware bring-up of the streaming layer: stream ONE small, slow welded
motion to the real robot through robot_weld_controller, watching feedback for smoothness
and following-error. No objects. ~4 cm down from base, <=10 deg/s. Returns to base after.

Run on domain 0 (real robot) with robot_weld_controller.py already running:
    python3 stream_hw_test.py
"""
import json
import os
import socket
import sys
import time

import numpy as np
import rclpy
from std_msgs.msg import String
from sensor_msgs.msg import JointState

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mycobot_mpc")))
import config as C
from geometry import R_from_two_axes, R_to_quat_wxyz
from joint_conventions import rad_to_linuxcnc_deg

MAX_DEG_S = 10.0
DOWN = list(R_to_quat_wxyz(R_from_two_axes(np.array([0, 0, -1.0]))))


def rpc(d):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=30)
    s.sendall((json.dumps(d) + "\n").encode()); b = b""
    while not b.endswith(b"\n"):
        b += s.recv(65536)
    s.close(); return json.loads(b)


def main():
    base = list(map(float, C.BASE_Q))
    fk = rpc({"type": "fk", "q": base}); p = fk["pos"][0]
    goal = [p[0], p[1], p[2] - 0.04]                       # 4 cm down, cup-down
    r = rpc({"type": "plan_pose", "start_q": base, "goal_pose": goal + DOWN, "max_attempts": 8})
    if not r.get("success"):
        print("plan FAILED"); return
    traj = np.array(r["trajectory"]); dt = float(r["dt"])
    peak = float(np.degrees(np.abs(np.diff(traj, axis=0)).max()) / dt) if len(traj) > 1 else MAX_DEG_S
    sdt = dt * max(1.0, peak / MAX_DEG_S)                  # slow to <= MAX_DEG_S
    td = [list(map(float, rad_to_linuxcnc_deg(w))) for w in traj]

    rclpy.init(); node = rclpy.create_node("stream_hw_test")
    pub = node.create_publisher(String, "/planner/weld_chunks", 10)
    fb = {"t": [], "pos": [], "last": time.time()}

    def on_fb(m):
        fb["t"].append(time.time()); fb["pos"].append([float(x) for x in m.position]); fb["last"] = time.time()
    node.create_subscription(JointState, "/mycobot/drive_feedback", on_fb, 100)
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.1)
    if not fb["t"]:
        print("no drive_feedback -- bridge down?"); rclpy.shutdown(); return

    dur = len(td) * sdt
    print(f"streaming 1 welded motion: {len(td)} wpts, ~{dur:.1f}s, peak {min(peak,MAX_DEG_S):.0f} deg/s "
          f"(4cm down). Watch the arm.")
    pub.publish(String(data=json.dumps(
        {"trajectory": td, "traj_dt": sdt, "target_deg": td[-1], "weld": True,
         "t_anchor": time.time() + 0.15})))
    target = traj[-1]
    t0 = time.time(); estop = False; reached = False
    while time.time() - t0 < dur + 4.0:
        rclpy.spin_once(node, timeout_sec=0.05)
        if time.time() - fb["last"] > 1.5:               # feedback stalled = likely e-stop / power-off
            estop = True; break
        if fb["pos"]:
            from joint_conventions import linuxcnc_deg_to_rad
            qr = linuxcnc_deg_to_rad(fb["pos"][-1])
            if np.abs(qr - target).max() < 0.03:
                reached = True; break
    # smoothness from feedback velocity
    if len(fb["t"]) > 5:
        from joint_conventions import linuxcnc_deg_to_rad
        P = np.array([linuxcnc_deg_to_rad(x) for x in fb["pos"]]); T = np.array(fb["t"])
        V = np.diff(P, axis=0) / np.maximum(np.diff(T)[:, None], 1e-3)
        pk = float(np.degrees(np.abs(V)).max())
        jrk = float(np.degrees(np.abs(np.diff(V, axis=0))).max())
        print(f"feedback: peak {pk:.0f} deg/s, max vel-step {jrk:.0f} deg/s")
    print("RESULT:", "E-STOP / feedback lost!" if estop else ("reached goal" if reached else "did not reach"))
    rclpy.shutdown()


if __name__ == "__main__":
    main()
