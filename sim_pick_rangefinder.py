#!/usr/bin/env python3
"""sim_pick_rangefinder :: real-time pick-and-place in the MuJoCo twin, driven over the
4 ms-style streaming path — cuRobo (:9997) → 0.4 s / 10 Hz weld chunks → sim_mujoco_node
(welds + tracks) — with a RANGE FINDER at the cup tip for contact (no vision).

Per object: pregrasp over it → descend (stream chunks) watching /sim/tip_range → on
range < contact-threshold HOLD + suction ON → lift → carry over the bin → lower → release.

  source /opt/ros/humble/setup.bash ; ROS_DOMAIN_ID=42 python3 sim_pick_rangefinder.py
"""
import argparse
import json
import os
import socket
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, String

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C
from geometry import R_from_two_axes, R_to_quat_wxyz
from joint_conventions import rad_to_linuxcnc_deg

DOWN = list(R_to_quat_wxyz(R_from_two_axes(np.array([0, 0, -1.0]))))


class Orch(Node):
    def __init__(self, a):
        super().__init__("sim_pick_rangefinder")
        self.a = a
        self.q = None; self.rng = None; self.objs = []
        self.create_subscription(JointState, "/joint_states",
                                 lambda m: setattr(self, "q", np.array(m.position, float)), 20)
        self.create_subscription(Float64, "/sim/tip_range",
                                 lambda m: setattr(self, "rng", float(m.data)), 20)
        self.create_subscription(String, "/sim/objects", self._on_objs, 10)
        self.pub_cmd = self.create_publisher(String, "/mycobot/cmd/move", 10)
        self.pub_suc = self.create_publisher(Bool, "/mycobot/suction", 10)

    def _on_objs(self, m):
        try:
            self.objs = json.loads(m.data)
        except Exception:
            pass

    def rpc(self, d):
        s = socket.create_connection(("127.0.0.1", 9997), timeout=40)
        s.sendall((json.dumps(d) + "\n").encode()); b = b""
        while not b.endswith(b"\n"):
            b += s.recv(65536)
        s.close(); return json.loads(b)

    def plan(self, goal_pose=None, goal_q=None):
        cur = [float(x) for x in self.q]
        r = (self.rpc({"type": "plan_joint", "start_q": cur, "goal_q": goal_q}) if goal_q is not None
             else self.rpc({"type": "plan_pose", "start_q": cur, "goal_pose": goal_pose, "max_attempts": 14}))
        if not r.get("success"):
            return None
        traj = np.array(r["trajectory"], float); dt = float(r["dt"])
        t = np.arange(len(traj)) * dt; tn = np.arange(0.0, t[-1] + 1e-9, 0.01)
        return np.column_stack([np.interp(tn, t, traj[:, j]) for j in range(traj.shape[1])])

    def stream(self, fine, v_des, watch=False, label=""):
        """Slice the plan into 0.4 s chunks @10 Hz, publish as weld chunks. If watch, stop the
        instant the range finder reads < contact threshold (HOLD the current pose)."""
        if fine is None:
            self.get_logger().warn(f"  [{label}] plan FAILED"); return "fail"
        deg = np.array([rad_to_linuxcnc_deg(w) for w in fine])
        peak = float(np.abs(np.diff(deg, axis=0)).max()) / 0.01 if len(deg) > 1 else v_des
        sdt = 0.01 * max(1.0, peak / v_des); tt = np.arange(len(deg)) * sdt
        play = np.column_stack([np.interp(np.arange(0, tt[-1] + 1e-9, 0.01), tt, deg[:, j]) for j in range(6)])
        t0 = time.time() + 0.1
        while True:
            k = int((time.time() - t0) / 0.01)
            if watch and self.rng is not None and 0.0 < self.rng < self.a.contact and k > 5:
                self._hold(); self.get_logger().info(f"  [{label}] CONTACT range={self.rng*1000:.0f}mm -> HOLD")
                return "contact"
            if k >= len(play):
                if watch:
                    self._hold()
                return "floor" if watch else "done"
            if k >= 0:
                ch = [r.tolist() for r in play[k:k + 40]]
                self.pub_cmd.publish(String(data=json.dumps(
                    {"trajectory": ch, "traj_dt": 0.01, "target_deg": ch[-1], "weld": True,
                     "t_anchor": float(t0 + k * 0.01)})))
            time.sleep(0.1)

    def _hold(self):
        cur = rad_to_linuxcnc_deg(self.q).tolist()
        self.pub_cmd.publish(String(data=json.dumps(
            {"trajectory": [cur, cur], "traj_dt": 0.1, "target_deg": cur, "weld": True,
             "t_anchor": time.time()})))

    def suction(self, on):
        self.pub_suc.publish(Bool(data=bool(on))); time.sleep(0.5)

    def run(self):
        spin = threading.Thread(target=lambda: rclpy.spin(self), daemon=True); spin.start()
        t0 = time.time()
        while (self.q is None or not self.objs) and time.time() - t0 < 20:
            time.sleep(0.1)
        if self.q is None:
            self.get_logger().error("no /joint_states / objects"); return
        snap = list(self.objs)[: self.a.max_objects]
        bx, by = [float(v) for v in self.a.bin.split(",")]
        self.get_logger().info(f"picking {len(snap)} objects -> bin [{bx},{by}]")
        for i, o in enumerate(snap):
            ox, oy, oz = o["xyz"]
            self.get_logger().info(f"\n== object {i}: {o['name']} @ [{ox:.3f},{oy:.3f},{oz:.3f}] ==")
            if self.stream(self.plan(goal_pose=[ox, oy, oz + 0.075] + DOWN), 45, label="pregrasp") == "fail":
                continue
            time.sleep(0.5)
            self.rng = None                                             # drop stale range before the descent
            self.stream(self.plan(goal_pose=[ox, oy, -0.05] + DOWN), self.a.v_des, watch=True, label="descend")
            self.suction(True)                                              # range-finder contact -> grasp
            self.stream(self.plan(goal_pose=[ox, oy, 0.18] + DOWN), 45, label="lift")    # straight up, no twist
            # PLACE = swing J1 ONLY over the bin, keeping the lifted base-like shape (no wrist/elbow twist).
            # cuRobo plan_pose to the bin picks arbitrary IK branches (the twisting); we only borrow its J1.
            lift_q = self.q.copy()
            br = self.rpc({"type": "plan_pose", "start_q": [float(x) for x in lift_q],
                           "goal_pose": [bx, by, 0.16] + DOWN, "max_attempts": 14})
            carry_q = lift_q.copy()
            carry_q[0] = (float(np.array(br["trajectory"])[-1][0]) if br.get("success")
                          else lift_q[0] + np.arctan2(by, bx) - np.arctan2(oy, ox))    # bin bearing (J1)
            self.stream(self.plan(goal_q=[float(x) for x in carry_q]), 40, label="carry (J1 swing)")
            self.suction(False)                                             # release into bin
            time.sleep(0.5)
        self.stream(self.plan(goal_q=list(map(float, C.BASE_Q))), 45, label="home")
        time.sleep(1.0)
        self.get_logger().info("== done ==")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contact", type=float, default=0.008, help="range-finder contact threshold (m)")
    ap.add_argument("--v-des", type=float, default=14.0, help="descent speed (deg/s)")
    ap.add_argument("--bin", default="0.10,0.40", help="place bin centre x,y")
    ap.add_argument("--max-objects", type=int, default=5)
    a = ap.parse_args()
    rclpy.init()
    node = Orch(a)
    try:
        node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
