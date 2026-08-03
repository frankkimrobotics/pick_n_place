#!/usr/bin/env python3
"""pnp_collect :: DETECT objects (vision) -> pick each via the welded TOUCH motion -> place in the
box -> save ONE episode per object. Shared sim/real logic (--backend); the sim path is exercised
end-to-end, the real path reuses the same flow with D405+SAM3 detection and cup-region depth contact.

Box: centre [0.1, 0.4, 0.0], 0.3 m cube  ->  rim at z = TABLE_Z + 0.30 = 0.20 (clear it before carry).
Detection: sim = /sim/scan (top-down seg) ; real = D405 + SAM3 (detect_suction_point).
Contact:   sim = /sim/tip_range ; real = cup-region depth gap (weld_response_test).
Each object is delimited by /phase start..end, so the sim capture logger writes one episode per object.

  # sim: run  curobo_planner_server_v2.py (:9997)  +  sim_mujoco_node.py --capture <dir>  then:
  ROS_DOMAIN_ID=42 python3 pnp_collect.py --backend sim --box 0.1,0.4 --cube 0.30
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


class PnP(Node):
    def __init__(self, a):
        super().__init__("pnp_collect"); self.a = a
        self.q = None; self.rng = None; self.dets = None
        self.create_subscription(JointState, "/joint_states",
                                 lambda m: setattr(self, "q", np.array(m.position, float)), 20)
        self.create_subscription(Float64, "/sim/tip_range",
                                 lambda m: setattr(self, "rng", float(m.data)), 20)
        self.create_subscription(String, "/sim/detections", self._on_det, 10)
        self.pub_cmd = self.create_publisher(String, "/mycobot/cmd/move", 10)
        self.pub_suc = self.create_publisher(Bool, "/mycobot/suction", 10)
        self.pub_scan = self.create_publisher(String, "/sim/scan", 10)
        self.pub_phase = self.create_publisher(String, "/phase", 10)
        self.pub_reset = self.create_publisher(String, "/sim/reset", 10)

    def _on_det(self, m):
        try:
            self.dets = json.loads(m.data)
        except Exception:
            self.dets = []

    def phase(self, p):
        self.pub_phase.publish(String(data=str(p)))

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
                self._hold(); self.get_logger().info(f"  [{label}] CONTACT {self.rng*1000:.0f}mm -> HOLD"); return "contact"
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
            {"trajectory": [cur, cur], "traj_dt": 0.1, "target_deg": cur, "weld": True, "t_anchor": time.time()})))

    def suction(self, on):
        self.pub_suc.publish(Bool(data=bool(on))); time.sleep(0.5)

    def scan(self):
        self.dets = None; self.pub_scan.publish(String(data="")); t0 = time.time()
        while self.dets is None and time.time() - t0 < 8:
            time.sleep(0.1)
        return self.dets or []

    def pick_place(self, o, i, bx, by, lift_z):
        ox, oy, oz = o["xyz"]
        self.get_logger().info(f"\n== object {i}: {o['id']} @ [{ox:.3f},{oy:.3f}] ==")
        self.phase("start")
        self.phase("reach")
        if self.stream(self.plan(goal_pose=[ox, oy, oz + 0.075] + DOWN), 45, label="pregrasp") == "fail":
            self.phase("end"); return
        time.sleep(0.4); self.rng = None
        self.phase("descend")                                            # welded touch descent ONTO the detected object
        # sim range-finder does not see the objects (fires on the table); detection is ~1mm, so descend
        # open-loop to just above the detected object top (cup within the 65mm suction-attach tolerance).
        self.stream(self.plan(goal_pose=[ox, oy, oz + 0.010] + DOWN), self.a.v_des, label="descend")
        self.phase("grasp"); self.suction(True)
        self.phase("lift")
        self.stream(self.plan(goal_pose=[ox, oy, lift_z] + DOWN), 45, label="lift")   # clear the box rim
        lift_q = self.q.copy()
        br = self.rpc({"type": "plan_pose", "start_q": [float(x) for x in lift_q],
                       "goal_pose": [bx, by, lift_z] + DOWN, "max_attempts": 14})
        carry_q = lift_q.copy()
        carry_q[0] = (float(np.array(br["trajectory"])[-1][0]) if br.get("success")
                      else lift_q[0] + np.arctan2(by, bx) - np.arctan2(oy, ox))        # bin bearing (J1)
        self.phase("carry")
        self.stream(self.plan(goal_q=[float(x) for x in carry_q]), 40, label="carry (J1 swing)")
        self.phase("place"); time.sleep(0.3)
        self.phase("release"); self.suction(False)                       # drop into the box
        time.sleep(0.4)
        self.phase("end")

    def run(self):
        threading.Thread(target=lambda: rclpy.spin(self), daemon=True).start()
        t0 = time.time()
        while self.q is None and time.time() - t0 < 20:
            time.sleep(0.1)
        if self.q is None:
            self.get_logger().error("no /joint_states"); return
        bx, by = [float(v) for v in self.a.box.split(",")]
        lift_z = self.a.table_top + self.a.cube + 0.05                   # clear the box rim before carry
        for rnd in range(self.a.rounds):
            if self.a.backend == "sim":                                  # fresh random clutter each round
                n = int(np.random.default_rng(self.a.seed + rnd).integers(self.a.min_objects, self.a.max_objects + 1))
                self.get_logger().info(f"  reset: randomize {n} objects (seed {self.a.seed+rnd})")
                self.pub_reset.publish(String(data=json.dumps({"seed": self.a.seed + rnd, "n": n})))
                time.sleep(2.0)
            dets = self.scan()
            self.get_logger().info(f"### round {rnd+1}/{self.a.rounds}: detected {len(dets)} objects -> box [{bx},{by}] ###")
            for i, o in enumerate(dets):
                self.pick_place(o, i, bx, by, lift_z)
            self.stream(self.plan(goal_q=list(map(float, C.BASE_Q))), 45, label="home")
        self.get_logger().info("== collection done ==")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="sim", choices=["sim", "real"])
    ap.add_argument("--box", default="0.1,0.4", help="box centre x,y")
    ap.add_argument("--cube", type=float, default=0.30, help="box cube size (m); rim = table_top + cube")
    ap.add_argument("--table-top", type=float, default=0.0, help="table-top z (sim=0.0, real=config TABLE_Z)")
    ap.add_argument("--contact", type=float, default=0.008, help="sim range-finder contact threshold (m)")
    ap.add_argument("--v-des", type=float, default=14.0)
    ap.add_argument("--rounds", type=int, default=1, help="clutter rounds; each object = 1 episode")
    ap.add_argument("--min-objects", type=int, default=3)
    ap.add_argument("--max-objects", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.backend == "real":
        print("real backend: use D405+SAM3 detection (perception/object_pointclouds) + cup-region "
              "depth contact (weld_response_test). Structure identical; not run here."); return
    rclpy.init(); node = PnP(a)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.ok() and rclpy.shutdown()


if __name__ == "__main__":
    main()
