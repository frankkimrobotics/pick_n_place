#!/usr/bin/env python3
"""robot_weld_controller :: the desktop layer that lets the ONLINE streaming planner
drive the real robot through robot_hal (which only tracks a trajectory to target_deg
and has NO streaming/weld mode).

  online_planner --/planner/weld_chunks--> [this node] --/mycobot/cmd/move windows--> robot_hal
                                                ^ /mycobot/drive_feedback (drift correction)

What it does:
  1. Welds each incoming chunk onto a single live reference q_ref(t) (TrajectoryWelder).
  2. At stream_hz, samples q_ref over a short lookahead WINDOW at robot_hal's dt and sends
     it as ONE cmd/move (overlapping windows => robot_hal always has a smooth, advancing
     B-spline reference instead of being re-targeted by raw 10 Hz chunks).
  3. Forward-prediction: the window starts `lead` ahead of now so it is still future when
     robot_hal receives it; the planner's mirror welder predicts the same q_ref, so planning
     and streaming agree despite the ~0.8 s command->motion dead-time.
  4. Drift correction: compares drive_feedback (actual, lag-delayed) to q_ref(now-lag); if it
     drifts past --drift-tol it re-seeds q_ref to the measured state so the loop stays honest.

This node is the thing to harden on hardware; validate first against sim_hal_node (a robot_hal
proxy with the dead-time). NOTHING here is robot-specific beyond the cmd/feedback conventions.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc"))):
    sys.path.insert(0, p)
from joint_conventions import (JOINT_NAMES, linuxcnc_deg_to_rad,
                               rad_to_linuxcnc_deg)
from traj_weld import TrajectoryWelder


class WeldController(Node):
    def __init__(self, a):
        super().__init__("robot_weld_controller")
        self.a = a
        self.welder = TrajectoryWelder(dof=6, fine_dt=0.01)
        self.q_meas = None
        self.last_drift = 0.0
        self.create_subscription(String, a.chunk_topic, self._on_chunk, 20)
        self.create_subscription(JointState, "/mycobot/drive_feedback", self._on_fb, 50)
        self.pub = self.create_publisher(String, "/mycobot/cmd/move", 10)
        self.create_timer(1.0 / a.stream_hz, self._stream)
        self.get_logger().info(
            f"weld-controller up: chunks<-{a.chunk_topic}, windows->/mycobot/cmd/move "
            f"@{a.stream_hz:.0f}Hz win={a.window:.2f}s lead={a.lead:.2f}s lag={a.lag:.2f}s")

    # ---- inputs ----
    def _on_chunk(self, msg):
        try:
            c = json.loads(msg.data)
        except Exception:
            return
        deg = c.get("trajectory") or [c["target_deg"]]
        traj = np.array([linuxcnc_deg_to_rad(w) for w in deg])
        dt = float(c.get("traj_dt", 0.01))
        t_anchor = float(c.get("t_anchor", time.time() + self.a.lead))
        if self.welder.t is None:
            seed = self.q_meas if self.q_meas is not None else traj[0]
            self.welder.seed(seed, time.time() - 0.2)
        self.welder.weld(traj, dt, t_anchor, blend=self.a.blend)

    def _on_fb(self, msg):
        n = min(6, len(msg.position))
        self.q_meas = linuxcnc_deg_to_rad([float(x) for x in msg.position[:n]] + [0] * (6 - n))[:6]
        # drift: actual now reflects what was commanded ~lag ago
        if self.welder.t is not None:
            ref_then = self.welder.sample(time.time() - self.a.lag)
            if ref_then is not None:
                self.last_drift = float(np.abs(np.asarray(self.q_meas) - ref_then).max())
                if self.last_drift > self.a.drift_tol:
                    self.get_logger().warn(f"drift {math.degrees(self.last_drift):.1f}deg "
                                           f"> tol; re-seeding q_ref to measured")
                    self.welder.seed(np.asarray(self.q_meas), time.time() - 0.2)

    # ---- output: stream overlapping windows to robot_hal ----
    def _stream(self):
        if self.welder.t is None:
            return
        t_send = time.time() + self.a.lead
        n = max(2, int(self.a.window / self.a.hal_dt))
        ts = t_send + np.arange(n) * self.a.hal_dt
        traj_rad = [self.welder.sample(t) for t in ts]
        if any(q is None for q in traj_rad):
            return
        deg = [rad_to_linuxcnc_deg(np.asarray(q)).tolist() for q in traj_rad]
        cmd = {"trajectory": deg, "traj_dt": self.a.hal_dt, "target_deg": deg[-1],
               "controller": self.a.controller, "ramp_time": self.a.ramp_time,
               "pos_gain": self.a.pos_gain, "vff_scale": self.a.vff_scale}
        self.pub.publish(String(data=json.dumps(cmd)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-topic", default="/planner/weld_chunks")
    ap.add_argument("--stream-hz", type=float, default=12.0, help="window publish rate to robot_hal")
    ap.add_argument("--window", type=float, default=0.40, help="lookahead window length (s)")
    ap.add_argument("--hal-dt", type=float, default=0.02, help="window sample dt (robot_hal servo)")
    ap.add_argument("--lead", type=float, default=0.10, help="start the window this far ahead (comms)")
    ap.add_argument("--lag", type=float, default=0.80, help="command->motion dead-time (drift ref)")
    ap.add_argument("--blend", type=float, default=0.06)
    ap.add_argument("--drift-tol", type=float, default=0.12, help="rad; re-seed q_ref past this")
    ap.add_argument("--controller", default="pid")
    ap.add_argument("--ramp-time", type=float, default=0.05)
    ap.add_argument("--pos-gain", type=float, default=1.0)
    ap.add_argument("--vff-scale", type=float, default=1.0)
    args = ap.parse_args()
    rclpy.init()
    node = WeldController(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
