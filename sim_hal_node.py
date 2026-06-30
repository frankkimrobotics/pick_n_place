#!/usr/bin/env python3
"""sim_hal_node :: a robot_hal PROXY for validating the weld-controller before hardware.

Mimics the real robot_hal closely enough to test the streaming layer:
  - subscribes /mycobot/cmd/move, tracks the LATEST window (interpolated at its traj_dt
    from receive time), exactly like robot_hal re-targeting on each cmd;
  - applies the ~0.8 s command->motion DEAD-TIME (a delay line) + a first-order tracking
    lag, so feedback behaves like the real arm;
  - publishes /mycobot/drive_feedback (LinuxCNC deg pos + deg/s vel) and /joint_states
    (URDF rad) at 50 Hz.

If the weld-controller streams smooth, continuous motion through THIS (with the dead-time),
the same windows should track on the real robot_hal. Run on the sim ROS domain.
"""
import argparse
import collections
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
import config as C
from joint_conventions import (JOINT_NAMES, linuxcnc_deg_to_rad,
                               rad_to_linuxcnc_deg)


class SimHal(Node):
    def __init__(self, a):
        super().__init__("sim_hal")
        self.a = a
        self.q = np.array(C.BASE_Q, float)          # actual (rad)
        self.win = None                              # (traj_rad[N,6], dt, recv_t)
        self.delay = collections.deque(maxlen=4000)  # (t, cmd_rad) for the dead-time line
        self.create_subscription(String, "/mycobot/cmd/move", self._on_cmd, 20)
        self.pub_fb = self.create_publisher(JointState, "/mycobot/drive_feedback", 50)
        self.pub_js = self.create_publisher(JointState, "/joint_states", 10)
        self.dtl = 1.0 / 200.0
        self.create_timer(self.dtl, self._servo)            # 200 Hz servo
        self._last_fb = 0.0
        self._last_q = self.q.copy()
        self._last_fb_t = time.time()
        self.get_logger().info(f"sim_hal up (dead-time {a.lag:.2f}s, track tau {a.tau:.3f}s)")

    def _on_cmd(self, msg):
        import json
        try:
            c = json.loads(msg.data)
        except Exception:
            return
        deg = c.get("trajectory") or [c["target_deg"]]
        traj = np.array([linuxcnc_deg_to_rad(w) for w in deg])
        dt = float(c.get("traj_dt", 0.02))
        self.win = (traj, dt, time.time())

    def _commanded(self, now):
        if self.win is None:
            return self.q
        traj, dt, recv = self.win
        tau = min(max(now - recv, 0.0), (len(traj) - 1) * dt)   # play window from receive time
        k = tau / dt
        i = int(k); f = k - i
        if i >= len(traj) - 1:
            return traj[-1]
        return traj[i] * (1 - f) + traj[i + 1] * f

    def _servo(self):
        now = time.time()
        cmd = self._commanded(now)
        self.delay.append((now, cmd.copy()))
        # dead-time: the actuator only sees what was commanded `lag` seconds ago
        target = cmd
        for t, c in self.delay:
            if t >= now - self.a.lag:
                target = c
                break
        # first-order tracking toward the delayed command
        alpha = min(1.0, self.dtl / max(self.a.tau, 1e-3))
        self.q = self.q + (target - self.q) * alpha
        if now - self._last_fb >= 0.02:                         # 50 Hz feedback
            self._publish(now)
            self._last_fb = now

    def _publish(self, now):
        dt = max(now - self._last_fb_t, 1e-3)
        deg = rad_to_linuxcnc_deg(self.q)
        vel = (rad_to_linuxcnc_deg(self.q) - rad_to_linuxcnc_deg(self._last_q)) / dt
        stamp = self.get_clock().now().to_msg()
        fb = JointState(); fb.header.stamp = stamp; fb.name = list(JOINT_NAMES)
        fb.position = [float(x) for x in deg]; fb.velocity = [float(x) for x in vel]
        self.pub_fb.publish(fb)
        js = JointState(); js.header.stamp = stamp; js.name = list(JOINT_NAMES)
        js.position = [float(x) for x in self.q]
        self.pub_js.publish(js)
        self._last_q = self.q.copy(); self._last_fb_t = now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag", type=float, default=0.80, help="command->motion dead-time (s)")
    ap.add_argument("--tau", type=float, default=0.05, help="first-order tracking time const (s)")
    args = ap.parse_args()
    rclpy.init()
    node = SimHal(args)
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
