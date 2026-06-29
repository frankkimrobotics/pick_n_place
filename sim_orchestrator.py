#!/usr/bin/env python3
"""sim_orchestrator :: scripted pick-and-place over ROS2, in simulation.

The control loop, end to end over ROS2 nodes:
  - plans via the cuRobo planner NODE  (/mycobot/curobo/plan_request -> plan_result)
  - commands the MuJoCo robot NODE     (/mycobot/cmd/move, LinuxCNC deg)
  - reads feedback                     (/joint_states, URDF rad)
  - grasps via the planner-side attach (set_world box walls + attach object bbox) and
    a sim suction signal               (/mycobot/suction)

Sequence: base -> pregrasp -> grasp -> [suction+attach] -> lift -> place(into box)
          -> [release+detach] -> base.   Poses are scripted (no sim camera).

Run: source /opt/ros/humble/setup.bash; python3 sim_orchestrator.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C
from geometry import R_from_two_axes, R_to_quat_wxyz
from joint_conventions import JOINT_NAMES, rad_to_linuxcnc_deg

DOWN = list(R_to_quat_wxyz(R_from_two_axes(np.array([0, 0, -1.0]))))   # cup pointing down


class Orchestrator(Node):
    def __init__(self):
        super().__init__("sim_orchestrator")
        self.q = None
        self._results = {}
        self._rid = 0
        self.create_subscription(JointState, "/joint_states", self._on_js, 10)
        self.create_subscription(String, "/mycobot/curobo/plan_result", self._on_res, 10)
        self.pub_req = self.create_publisher(String, "/mycobot/curobo/plan_request", 10)
        self.pub_cmd = self.create_publisher(String, "/mycobot/cmd/move", 10)
        self.pub_suck = self.create_publisher(Bool, "/mycobot/suction", 10)

    # ---- io ----
    def _on_js(self, m):
        self.q = np.array(m.position, float)

    def _on_res(self, m):
        r = json.loads(m.data)
        self._results[r.get("id", "")] = r

    def _spin(self, dt=0.02):
        rclpy.spin_once(self, timeout_sec=dt)

    def wait_graph(self, timeout=30.0):
        """Wait until planner + mujoco nodes are discovered and feedback flows."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            self._spin()
            if (self.pub_req.get_subscription_count() > 0 and
                    self.pub_cmd.get_subscription_count() > 0 and self.q is not None):
                self.get_logger().info("graph ready (planner + mujoco + joint_states)")
                return True
        raise RuntimeError("graph not ready: missing planner/mujoco/joint_states")

    def get_q(self):
        while self.q is None:
            self._spin()
        return [float(x) for x in self.q]

    def rpc(self, req, timeout=45.0):
        self._rid += 1
        rid = str(self._rid)
        req["id"] = rid
        msg = String(); msg.data = json.dumps(req)
        self.pub_req.publish(msg)
        t0 = time.time(); resent = False
        while time.time() - t0 < timeout:
            self._spin()
            if rid in self._results:
                return self._results.pop(rid)
            if not resent and time.time() - t0 > timeout / 3:
                self.pub_req.publish(msg); resent = True       # re-send once (discovery)
        return {"success": False, "status": "rpc timeout", "id": rid}

    # ---- motion ----
    def execute(self, traj_rad, dt, label, tol=0.015):
        deg = [rad_to_linuxcnc_deg(np.array(w)).tolist() for w in traj_rad]
        cmd = {"trajectory": deg, "traj_dt": float(dt), "target_deg": deg[-1],
               "controller": "pid"}
        self.pub_cmd.publish(String(data=json.dumps(cmd)))
        target = np.array(traj_rad[-1], float)
        dur = len(traj_rad) * dt
        # wall-clock playback is slower than sim-time (osmesa render gates the loop),
        # so wait until actually reached, OR until motion settles, with a wide timeout.
        timeout = max(45.0, dur * 12.0)
        t0 = time.time(); last = None; still = 0
        while time.time() - t0 < timeout:
            self._spin()
            if self.q is None:
                continue
            dev = float(np.abs(self.q - target).max())
            if dev < tol:
                self.get_logger().info(f"[{label}] reached ({len(traj_rad)} wpts) dev={dev:.3f}")
                return True
            if last is not None and float(np.abs(self.q - last).max()) < 1e-4:
                still += 1
            else:
                still = 0
            last = self.q.copy()
            if still > 250 and time.time() - t0 > 5.0:        # stopped moving, not at target
                break
        self.get_logger().warn(f"[{label}] not reached (dev "
                               f"{float(np.abs(self.q - target).max()):.3f} rad)")
        return False

    def move_pose(self, goal_xyz, label, quat=None, attempts=20):
        r = self.rpc({"type": "plan_pose", "start_q": self.get_q(),
                      "goal_pose": list(goal_xyz) + (quat or DOWN), "max_attempts": attempts})
        if not r.get("success"):
            self.get_logger().error(f"[{label}] PLAN FAILED: {r.get('status')}")
            return False
        return self.execute(r["trajectory"], r["dt"], label)

    def move_joint(self, goal_q, label):
        r = self.rpc({"type": "plan_joint", "start_q": self.get_q(),
                      "goal_q": list(map(float, goal_q))})
        if not r.get("success"):
            self.get_logger().error(f"[{label}] PLAN FAILED: {r.get('status')}")
            return False
        return self.execute(r["trajectory"], r["dt"], label)

    def suction(self, on):
        self.pub_suck.publish(Bool(data=bool(on)))
        for _ in range(5):
            self._spin()


def box_walls(bx, by, bz, fp, h, t):
    half = fp / 2.0
    return [
        {"name": "box_floor", "dims": [fp+2*t, fp+2*t, t], "pose": [bx, by, bz-t/2, 1, 0, 0, 0]},
        {"name": "box_xm", "dims": [t, fp+2*t, h], "pose": [bx-(half+t/2), by, bz+h/2, 1, 0, 0, 0]},
        {"name": "box_xp", "dims": [t, fp+2*t, h], "pose": [bx+(half+t/2), by, bz+h/2, 1, 0, 0, 0]},
        {"name": "box_ym", "dims": [fp, t, h], "pose": [bx, by-(half+t/2), bz+h/2, 1, 0, 0, 0]},
        {"name": "box_yp", "dims": [fp, t, h], "pose": [bx, by+(half+t/2), bz+h/2, 1, 0, 0, 0]},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", default="0.30,0.0,0.06", help="object TOP xyz (grasp point)")
    ap.add_argument("--obj-size", default="0.06,0.06,0.06", help="object bbox dx,dy,dz")
    ap.add_argument("--box", default="0.20,0.30,0.0", help="place box bottom-centre xyz")
    ap.add_argument("--box-fp", type=float, default=0.20)
    ap.add_argument("--box-size", type=float, default=0.12, help="box height")
    ap.add_argument("--standoff", type=float, default=0.06)
    ap.add_argument("--release-clear", type=float, default=0.02)
    args = ap.parse_args()

    pick = [float(v) for v in args.pick.split(",")]
    odims = [float(v) for v in args.obj_size.split(",")]
    bx, by, bz = [float(v) for v in args.box.split(",")]
    rim = bz + args.box_size

    rclpy.init()
    node = Orchestrator()
    try:
        node.wait_graph()
        node.get_logger().info("=== sim pick-and-place start ===")

        pregrasp = [pick[0], pick[1], pick[2] + args.standoff]
        node.move_pose(pregrasp, "pregrasp")
        node.move_pose(pick, "grasp-descend")

        # GRASP: suction on + planner-side attach (object bbox at the grasp config)
        node.suction(True)
        node.rpc({"type": "set_world",
                  "cuboids": box_walls(bx, by, bz, args.box_fp, args.box_size, 0.01)})
        gq = node.get_q()
        oc = [pick[0], pick[1], pick[2] - odims[2] / 2.0]        # object centre
        ar = node.rpc({"type": "attach", "grasp_q": gq, "dims": odims,
                       "obj_pose": oc + [1, 0, 0, 0]})
        node.get_logger().info(f"attach -> {ar.get('success')} ({ar.get('n_managers')} mgrs)")

        # CARRY into the box, split so each segment is an easy plan (the planner-side
        # attach keeps the held object clear of the box throughout): lift at pick ->
        # traverse above the box rim -> descend into the box.
        over_z = min(0.42, rim + 0.06)
        node.move_pose([pick[0], pick[1], over_z], "lift")
        node.move_pose([bx, by, over_z], "over-box")
        place_z = rim - 0.01            # dip into the opening; wrist/camera stay above the rim
        node.move_pose([bx, by, place_z], "place-into-box")

        # RELEASE + detach
        node.suction(False)
        node.rpc({"type": "detach"})

        # RETURN to base
        node.move_joint(C.BASE_Q, "return-to-base")
        node.rpc({"type": "clear_world"})
        node.get_logger().info("=== sim pick-and-place DONE ===")
        time.sleep(1.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
