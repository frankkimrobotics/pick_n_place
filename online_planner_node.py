#!/usr/bin/env python3
"""online_planner_node :: an ONLINE (streaming) cuRobo planner that feeds the robot
short trajectory CHUNKS (dt=0.01) which the controller WELDS onto the current motion.

Reference behaviour (to match an ML planner): ~10 Hz inference, each producing a
16-20 waypoint chunk; the robot follows by welding chunk-to-chunk (no stops). Here the
chunk source is cuRobo. Two modes:

  --mode chunk  (Option B, receding-horizon replanning):
      Plan the FULL trajectory to the goal once (cuRobo MotionGen, the proper velocity
      profile), resample to dt=0.01, then STREAM ~0.2 s slices as chunks at plan-rate.
      Re-plan (and weld) only on a goal change / disturbance / cache exhaustion. Between
      replans every chunk comes from one consistent plan -> smooth, fast, momentum kept.

  --mode mpc    (Option A, reactive per-cycle replanning):
      Every cycle re-plan a short horizon from the PREDICTED junction state and re-time it
      to leave at the current per-joint speed (retime_to_velocity) so momentum is preserved
      across cuRobo's rest-to-rest output. (cuRobo also ships a native velocity-aware
      MpcSolver -- see online_planner_design.md -- this trajopt+retime path is the drop-in
      that reuses the existing socket planner.)

Both publish /mycobot/cmd/move with {"weld":true,"t_anchor":<abs s>,"traj_dt":0.01,...};
a weld-aware controller (sim_mujoco_node --weld, or robot_hal) tracks the welded reference.
The node keeps a MIRROR welder so it plans from the junction state it has already committed.

  source /opt/ros/humble/setup.bash
  python3 online_planner_node.py --mode chunk --goal 0.30,0.10,0.20
"""
import argparse
import json
import socket
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc"))):
    sys.path.insert(0, p)
import config as C
from geometry import R_from_two_axes, R_to_quat_wxyz
from joint_conventions import JOINT_NAMES, rad_to_linuxcnc_deg
from traj_weld import TrajectoryWelder, retime_to_velocity

DOWN = list(R_to_quat_wxyz(R_from_two_axes(np.array([0, 0, -1.0]))))


def resample(traj_rad, dt_in, fine_dt=0.01):
    traj = np.asarray(traj_rad, float)
    if len(traj) < 2:
        return traj
    t = np.arange(len(traj)) * dt_in
    tn = np.arange(0.0, t[-1] + 1e-9, fine_dt)
    return np.column_stack([np.interp(tn, t, traj[:, j]) for j in range(traj.shape[1])])


class OnlinePlanner(Node):
    def __init__(self, args):
        super().__init__("online_planner")
        self.a = args
        self.host, self.port = "127.0.0.1", 9997
        self.fine_dt = 0.01
        self.q = None
        self.welder = TrajectoryWelder(dof=6, fine_dt=self.fine_dt)   # mirror of controller
        self.cache = None            # (fine_plan_rad, t0_abs) for chunk mode
        self.create_subscription(JointState, "/joint_states", self._on_js, 20)
        self.pub = self.create_publisher(String, "/mycobot/cmd/move", 10)
        self.goal_pose = [float(x) for x in args.goal.split(",")] + DOWN if args.goal else None
        self.goal_q = [float(x) for x in args.goal_joint.split(",")] if args.goal_joint else None
        self._stop = threading.Event()

    # ---- io ----
    def _on_js(self, m):
        self.q = np.array(m.position, float)

    def rpc(self, d, timeout=30):
        s = socket.create_connection((self.host, self.port), timeout=timeout)
        s.sendall((json.dumps(d) + "\n").encode()); buf = b""
        while not buf.endswith(b"\n"):
            ch = s.recv(65536)
            if not ch: break
            buf += ch
        s.close(); return json.loads(buf)

    def plan_full(self, q_start):
        if self.goal_q is not None:
            r = self.rpc({"type": "plan_joint", "start_q": list(map(float, q_start)),
                          "goal_q": self.goal_q, "max_attempts": 6})
        else:
            r = self.rpc({"type": "plan_pose", "start_q": list(map(float, q_start)),
                          "goal_pose": self.goal_pose, "max_attempts": 6})
        if not r.get("success"):
            return None, None
        return np.array(r["trajectory"], float), float(r["dt"])

    # ---- streaming loop ----
    def run(self):
        while self.q is None and not self._stop.is_set():
            rclpy.spin_once(self, timeout_sec=0.05)
        t0 = time.time()
        self.welder.seed(self.q.copy(), t0 - 0.5)
        self.get_logger().info(f"seeded at q={np.round(self.q,3).tolist()}; mode={self.a.mode}")

        period = 1.0 / self.a.plan_hz
        chunk_len = max(2, int(round(self.a.horizon / self.fine_dt)))   # waypoints per chunk
        last_replan = 0.0
        while not self._stop.is_set():
            tick = time.time()
            rclpy.spin_once(self, timeout_sec=0.0)
            t_anchor = time.time() + self.a.commit
            q_anchor = self.welder.sample(t_anchor)
            v_anchor = self.welder.velocity(t_anchor)

            if self.a.mode == "chunk":
                need = (self.cache is None or
                        (time.time() - last_replan) > self.a.replan_period)
                if need:
                    plan, dt = self.plan_full(q_anchor)
                    if plan is not None:
                        self.cache = (resample(plan, dt, self.fine_dt), t_anchor)
                        last_replan = time.time()
                if self.cache is None:
                    time.sleep(period); continue
                fine, c_t0 = self.cache
                k = int(round((t_anchor - c_t0) / self.fine_dt))
                if k >= len(fine) - 1:                      # plan consumed -> hold goal
                    self._publish_hold(q_anchor, t_anchor);
                    if self.welder.reached(fine[-1], time.time(), tol=0.02):
                        self.get_logger().info("goal reached; streaming done"); break
                    self._sleep_to(tick + period); continue
                chunk = fine[k:k + chunk_len]
            else:  # mpc
                plan, dt = self.plan_full(q_anchor)
                if plan is None:
                    self._sleep_to(tick + period); continue
                # constant-cruise chunks: re-timing from the tiny current speed makes
                # rest-to-rest trajopt creep, so drive each chunk at cruise speed and let
                # the weld blend absorb the rest->cruise transition on the first chunk.
                dist = float(np.abs(np.asarray(plan[-1]) - q_anchor).max())   # rad to goal
                if dist < 0.025:
                    self.get_logger().info("goal reached (mpc); done"); break
                vcru = np.radians(self.a.cruise_deg) * min(1.0, max(0.12, dist / 0.4))  # decel near goal
                horizon_pts = max(2, int(self.a.horizon / max(dt, 1e-3)))
                seg, _ = retime_to_velocity(plan[:horizon_pts * 3], dt, vcru, vcru,
                                            self.fine_dt, ramp=self.a.ramp)
                chunk = seg[:chunk_len]

            self.welder.weld(chunk, self.fine_dt, t_anchor, blend=self.a.blend)
            self._publish_chunk(chunk, t_anchor)
            self._sleep_to(tick + period)

    def _publish_chunk(self, chunk_rad, t_anchor):
        deg = [rad_to_linuxcnc_deg(np.array(w)).tolist() for w in chunk_rad]
        msg = {"trajectory": deg, "traj_dt": self.fine_dt, "target_deg": deg[-1],
               "controller": "pid", "weld": True, "t_anchor": float(t_anchor)}
        self.pub.publish(String(data=json.dumps(msg)))

    def _publish_hold(self, q, t_anchor):
        self.welder.weld(np.vstack([q, q]), self.fine_dt, t_anchor, blend=self.a.blend)

    def _sleep_to(self, t_target):
        dt = t_target - time.time()
        if dt > 0:
            time.sleep(dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["chunk", "mpc"], default="chunk")
    ap.add_argument("--goal", default=None, help="goal tcp pose x,y,z (cup-down)")
    ap.add_argument("--goal-joint", default=None, help="goal joint config (6 rad)")
    ap.add_argument("--plan-hz", type=float, default=10.0)
    ap.add_argument("--horizon", type=float, default=0.20, help="chunk horizon (s)")
    ap.add_argument("--commit", type=float, default=0.20, help="anchor lead time (covers latency)")
    ap.add_argument("--blend", type=float, default=0.06, help="weld blend window (s)")
    ap.add_argument("--replan-period", type=float, default=1.5, help="chunk-mode refresh (s)")
    ap.add_argument("--cruise-deg", type=float, default=35.0, help="mpc cruise per-joint speed")
    ap.add_argument("--ramp", type=float, default=0.4, help="mpc retime ramp fraction")
    args = ap.parse_args()
    if not args.goal and not args.goal_joint:
        args.goal = "0.30,0.10,0.20"

    rclpy.init()
    node = OnlinePlanner(args)
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node._stop.set()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
