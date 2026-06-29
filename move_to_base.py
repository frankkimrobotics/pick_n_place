#!/usr/bin/env python3
"""Move the REAL robot to BASE_Q (domain 0): plan_joint via the socket planner, publish
ONE cmd/move at a velocity-capped dt, wait until settled. Safe, planned move."""
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
from joint_conventions import rad_to_linuxcnc_deg

MAX_DEG_S = 20.0


def rpc(d):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=30)
    s.sendall((json.dumps(d) + "\n").encode()); b = b""
    while not b.endswith(b"\n"):
        b += s.recv(65536)
    s.close(); return json.loads(b)


def main():
    rclpy.init(); node = rclpy.create_node("move_to_base")
    pub = node.create_publisher(String, "/mycobot/cmd/move", 10)
    q = {}
    node.create_subscription(JointState, "/joint_states", lambda m: q.update(p=list(m.position)), 10)
    t0 = time.time()
    while "p" not in q and time.time() - t0 < 5:
        rclpy.spin_once(node, timeout_sec=0.1)
    if "p" not in q:
        print("no /joint_states -- is the bridge up?"); return
    cur = [float(x) for x in q["p"]]
    base = list(map(float, C.BASE_Q))
    dev = float(np.degrees(np.abs(np.array(cur) - np.array(base))).max())
    print(f"current dev from base: {dev:.1f} deg")
    if dev < 2.0:
        print("already at base."); rclpy.shutdown(); return
    r = rpc({"type": "plan_joint", "start_q": cur, "goal_q": base})
    if not r.get("success"):
        print("plan to base FAILED:", r.get("status")); rclpy.shutdown(); return
    traj = np.array(r["trajectory"]); dt = float(r["dt"])
    peak = float(np.degrees(np.abs(np.diff(traj, axis=0)).max()) / dt) if len(traj) > 1 else MAX_DEG_S
    sdt = dt * max(1.0, peak / MAX_DEG_S)            # stretch so peak <= MAX_DEG_S
    td = [list(map(float, rad_to_linuxcnc_deg(w))) for w in traj]
    pub.publish(String(data=json.dumps(
        {"trajectory": td, "traj_dt": sdt, "target_deg": td[-1], "controller": "pid",
         "ramp_time": 0.3, "pos_gain": 1.0, "vff_scale": 1.0})))
    dur = len(td) * sdt
    print(f"moving to base: {len(td)} wpts, ~{dur:.1f}s, peak {min(peak, MAX_DEG_S):.0f} deg/s")
    t0 = time.time()
    while time.time() - t0 < dur + 2.5:
        rclpy.spin_once(node, timeout_sec=0.05)
        d = float(np.degrees(np.abs(np.array(q["p"]) - np.array(base))).max())
        if time.time() - t0 > dur - 0.3 and d < 2.0:
            break
    print(f"settled: dev {float(np.degrees(np.abs(np.array(q['p'])-np.array(base))).max()):.1f} deg")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
