#!/usr/bin/env python3
"""cross_weld_test :: validate CROSS-SEGMENT welding through the streaming layer.

Welds a sequence of motion segments (e.g. lift -> place -> return) into ONE continuous
sweep with NON-ZERO junction velocities -- no settle between moves. cuRobo plans each
segment rest-to-rest, so each later segment is RE-TIMED to leave at the previous junction
velocity (retime_to_velocity) and anchored to overlap the previous before it decelerates;
the weld-controller blends them. Run against robot_weld_controller + sim_hal (domain 42).
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
from joint_conventions import rad_to_linuxcnc_deg
from traj_weld import retime_to_velocity

DOWN = [0, 1, 0, 0]
JFRAC = 0.80        # transition into the next segment at 80% of each (before it decelerates)
CRUISE = np.radians(35.0)


def rpc(d):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=30)
    s.sendall((json.dumps(d) + "\n").encode()); b = b""
    while not b.endswith(b"\n"):
        b += s.recv(65536)
    s.close(); return json.loads(b)


def resample(traj, dt_in, fine=0.01):
    t = np.arange(len(traj)) * dt_in
    tn = np.arange(0, t[-1] + 1e-9, fine)
    return np.column_stack([np.interp(tn, t, traj[:, j]) for j in range(traj.shape[1])])


def plan(q_start, seg):
    if seg[0] == "pose":
        r = rpc({"type": "plan_pose", "start_q": list(map(float, q_start)),
                 "goal_pose": list(seg[1]) + DOWN, "max_attempts": 6})
    else:
        r = rpc({"type": "plan_joint", "start_q": list(map(float, q_start)),
                 "goal_q": list(map(float, seg[1]))})
    return (np.array(r["trajectory"]), float(r["dt"])) if r.get("success") else (None, None)


def main():
    base = list(map(float, C.BASE_Q))
    segments = [("pose", [0.30, 0.00, 0.22]),     # lift
                ("pose", [0.20, 0.30, 0.18]),     # carry/place over box
                ("joint", base)]                  # return to base
    rclpy.init(); node = rclpy.create_node("cross_weld_test")
    pub = node.create_publisher(String, "/planner/weld_chunks", 10)
    for _ in range(15):
        rclpy.spin_once(node, timeout_sec=0.1)

    # 1) PLAN all segments first (planning is slow; do it before assigning wall-clock anchors)
    q_start = base; v_start = 0.0
    fines = []; junc_dts = []
    for i, seg in enumerate(segments):
        path, dt = plan(q_start, seg)
        if path is None:
            print(f"segment {i} plan FAILED"); rclpy.shutdown(); return
        if v_start > 1e-3:                         # re-time to LEAVE at the junction velocity
            fine, _ = retime_to_velocity(path, dt, v_start, CRUISE, 0.01, ramp=0.35)
        else:
            fine = resample(path, dt, 0.01)
        fines.append(fine)
        if i < len(segments) - 1:                  # junction at JFRAC through this segment
            idx = int(len(fine) * JFRAC)
            q_start = fine[idx]
            v_start = float(np.abs(fine[idx + 1] - fine[idx]).max()) / 0.01
            junc_dts.append(idx * 0.01)
            print(f"seg{i}: {len(fine)} wpts {len(fine)*0.01:.1f}s -> junction at {idx*0.01:.1f}s "
                  f"v_junc={np.degrees(v_start):.0f}deg/s")
        else:
            junc_dts.append(len(fine) * 0.01)
            print(f"seg{i}: {len(fine)} wpts {len(fine)*0.01:.1f}s (final, decel to rest)")

    # 2) PUBLISH all chunks on ONE future timeline (instant; the weld-controller welds them)
    t = time.time() + 0.20
    for fine, jdt in zip(fines, junc_dts):
        td = [list(map(float, rad_to_linuxcnc_deg(w))) for w in fine]
        pub.publish(String(data=json.dumps(
            {"trajectory": td, "traj_dt": 0.01, "target_deg": td[-1],
             "weld": True, "t_anchor": float(t)})))
        t += jdt
        rclpy.spin_once(node, timeout_sec=0.01)
    print("all segments published as ONE welded sweep (non-zero junctions)")
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.05)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
