#!/usr/bin/env python3
"""Mimic servo_touch --stream: plan a welded motion (socket planner) and publish it
as ONE weld chunk to /planner/weld_chunks (exactly what publish_motion does in stream
mode), then watch /joint_states until the goal tcp is reached. Validates the servo_touch
streaming-publish path through robot_weld_controller + sim_hal."""
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


def rpc(d):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=30)
    s.sendall((json.dumps(d) + "\n").encode()); b = b""
    while not b.endswith(b"\n"):
        b += s.recv(65536)
    s.close(); return json.loads(b)


def main():
    goal = [0.38, 0.12, 0.20, 0, 1, 0, 0]
    base = list(map(float, C.BASE_Q))
    r = rpc({"type": "plan_pose", "start_q": base, "goal_pose": goal, "max_attempts": 6})
    if not r.get("success"):
        print("plan failed"); return
    traj = np.array(r["trajectory"]); dt = float(r["dt"])
    td = [list(map(float, rad_to_linuxcnc_deg(w))) for w in traj]

    rclpy.init(); node = rclpy.create_node("stream_pub_test")
    pub = node.create_publisher(String, "/planner/weld_chunks", 10)
    q = {}
    node.create_subscription(JointState, "/joint_states", lambda m: q.update(p=list(m.position)), 10)
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.1)
    # publish_motion(--stream): ONE weld chunk for this welded motion
    pub.publish(String(data=json.dumps(
        {"trajectory": td, "traj_dt": dt, "target_deg": td[-1], "weld": True,
         "t_anchor": time.time() + 0.1})))
    print(f"published welded motion as 1 chunk ({len(td)} wpts @ dt={dt:.3f})")
    target = traj[-1]
    t0 = time.time()
    while time.time() - t0 < 20:
        rclpy.spin_once(node, timeout_sec=0.05)
        if "p" in q and np.abs(np.array(q["p"]) - target).max() < 0.03:
            print(f"REACHED goal config after {time.time()-t0:.1f}s (through weld-controller + 0.8s-lag hal)")
            break
    fk = rpc({"type": "fk", "q": [float(x) for x in q["p"]]})
    print("final tcp:", [round(x, 3) for x in fk["pos"][0]], " goal:", goal[:3])
    rclpy.shutdown()


if __name__ == "__main__":
    main()
