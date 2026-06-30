#!/usr/bin/env python3
"""sim_mujoco_node :: a MuJoCo "robot controller" ROS2 node that impersonates the
real MyCobot Pro 630 bridge+HAL, so the cuRobo control loop drives it in sim.

Mirrors the real robot's ROS2 interface exactly:
  subscribes  /mycobot/cmd/move        std_msgs/String  (JSON; waypoints LinuxCNC deg)
              /mycobot/suction         std_msgs/Bool    (sim grasp: attach object to tcp)
  publishes   /joint_states            sensor_msgs/JointState  position = URDF rad   @50Hz
              /mycobot/drive_feedback  sensor_msgs/JointState  pos LinuxCNC deg + vel deg/s @50Hz

The model is the joint-driven MJCF from sim_robot_mjcf.py. cmd/move trajectories are
played back kinematically (set qpos at traj_dt, mj_forward) — the planner already
guarantees dynamic feasibility, so ideal tracking is the right sim abstraction. The
held object follows the tcp while suction is on. Frames render offscreen to an mp4.

Run (system ROS2 python with mujoco pip):
    source /opt/ros/humble/setup.bash
    python3 sim_mujoco_node.py --video outputs/mujoco_sim/episode.mp4
"""
import argparse
import json
import os
import sys
import threading
import time

import numpy as np
import cv2
os.environ.setdefault("MUJOCO_GL", "osmesa")    # headless backend (EGL unavailable here)
import mujoco

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C
from joint_conventions import (JOINT_NAMES, linuxcnc_deg_to_rad,
                               rad_to_linuxcnc_deg)
from traj_weld import TrajectoryWelder


def _T(pos, mat):
    T = np.eye(4)
    T[:3, :3] = np.asarray(mat).reshape(3, 3)
    T[:3, 3] = pos
    return T


def _quat_wxyz(R):
    m = np.asarray(R, float)
    t = np.trace(m)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
    return np.array([w, x, y, z])


class MujocoRobot(Node):
    def __init__(self, xml, video=None, cam="iso", fb_hz=50.0, render_hz=20.0):
        super().__init__("sim_mujoco_robot")
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        self.jadr = [self.m.joint(n).qposadr[0] for n in JOINT_NAMES]
        self.tcp_sid = self.m.site("tcp").id
        self.obj_adr = self.m.joint("obj_free").qposadr[0]
        # start at BASE_Q (URDF rad), object at its model rest pose
        self.q = np.array(C.BASE_Q, float)
        self._apply(self.q)
        self._obj_rest = self.d.qpos[self.obj_adr:self.obj_adr + 7].copy()

        self.lock = threading.Lock()
        self._queue = []                  # list of (traj_rad[N,6], dt)
        self.welder = TrajectoryWelder(dof=6, fine_dt=0.01)   # online streaming reference
        self._weld = False
        self._attached = False
        self._rel = None                  # tcp->object relative transform when grasped
        self.fb_dt = 1.0 / fb_hz
        self.render_dt = 1.0 / render_hz

        self.pub_js = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_fb = self.create_publisher(JointState, "/mycobot/drive_feedback", 50)
        self.create_subscription(String, "/mycobot/cmd/move", self._on_cmd, 10)
        self.create_subscription(Bool, "/mycobot/suction", self._on_suction, 10)

        self.cam = cam
        self.video = video
        self._renderer = mujoco.Renderer(self.m, 480, 640) if video else None  # osmesa: keep small/fast
        self._vw = None
        self._stop = threading.Event()
        self._sim = threading.Thread(target=self._loop, daemon=True)
        self._sim.start()
        self.get_logger().info(
            f"sim MuJoCo robot up (nq={self.m.nq}); waiting for /mycobot/cmd/move")

    # ---- model helpers (only the sim thread touches self.d) ----
    def _apply(self, q):
        for a, v in zip(self.jadr, q):
            self.d.qpos[a] = float(v)
        if getattr(self, "_attached", False) and self._rel is not None:
            mujoco.mj_forward(self.m, self.d)
            Ttcp = _T(self.d.site_xpos[self.tcp_sid], self.d.site_xmat[self.tcp_sid])
            Tobj = Ttcp @ self._rel
            self.d.qpos[self.obj_adr:self.obj_adr + 3] = Tobj[:3, 3]
            self.d.qpos[self.obj_adr + 3:self.obj_adr + 7] = _quat_wxyz(Tobj[:3, :3])
        mujoco.mj_forward(self.m, self.d)

    # ---- ROS callbacks (spin thread) ----
    def _on_cmd(self, msg):
        try:
            c = json.loads(msg.data)
        except Exception:
            return
        deg = c.get("trajectory") or [c["target_deg"]]
        traj = np.array([linuxcnc_deg_to_rad(w) for w in deg])
        dt = float(c.get("traj_dt", c.get("duration", 2.0) / max(len(traj), 1)))
        if c.get("weld"):                          # streaming chunk -> weld onto live ref
            with self.lock:
                if self.welder.t is None:
                    self.welder.seed(self.q.copy(), time.time() - 0.2)
                self.welder.weld(traj, dt, float(c.get("t_anchor", time.time())), blend=0.06)
                self._weld = True
            return
        with self.lock:
            self._queue.append((traj, dt))
        self.get_logger().info(f"cmd: {len(traj)} wpts @ dt={dt:.3f}")

    def _on_suction(self, msg):
        with self.lock:
            want = bool(msg.data)
            if want and not self._attached:
                Ttcp = _T(self.d.site_xpos[self.tcp_sid], self.d.site_xmat[self.tcp_sid])
                op = self.d.qpos[self.obj_adr:self.obj_adr + 3]
                oq = self.d.qpos[self.obj_adr + 3:self.obj_adr + 7]
                Tobj = _T(op, _quat_to_mat(oq))
                self._rel = np.linalg.inv(Ttcp) @ Tobj
                self.get_logger().info("suction ON: object attached to tcp")
            elif not want and self._attached:
                self.get_logger().info("suction OFF: object released")
            self._attached = want
            if not want:
                self._rel = None

    # ---- sim + feedback + render loop (single thread owns self.d) ----
    def _loop(self):
        t_fb = t_rd = 0.0
        last_q = self.q.copy()
        last_fb_t = time.time()
        while not self._stop.is_set():
            if self._weld:                              # streaming mode: track the welded ref
                now = time.time()
                with self.lock:
                    qs = self.welder.sample(now)
                    if qs is not None:
                        self.q = qs.copy(); self._apply(self.q)
                if now - t_fb >= self.fb_dt:
                    self._publish_fb(last_q, now - last_fb_t)
                    last_q = self.q.copy(); last_fb_t = now; t_fb = now
                if self._renderer is not None and now - t_rd >= self.render_dt:
                    self._grab(); t_rd = now
                time.sleep(0.005)
                continue
            with self.lock:
                job = self._queue.pop(0) if self._queue else None
            if job is not None:
                traj, dt = job
                for wp in traj:
                    if self._stop.is_set():
                        break
                    with self.lock:
                        self.q = wp.copy()
                        self._apply(self.q)
                    now = time.time()
                    if now - t_fb >= self.fb_dt:
                        self._publish_fb(last_q, now - last_fb_t)
                        last_q = self.q.copy(); last_fb_t = now; t_fb = now
                    if self._renderer is not None and now - t_rd >= self.render_dt:
                        self._grab(); t_rd = now
                    time.sleep(dt)
            else:
                now = time.time()
                if now - t_fb >= self.fb_dt:
                    self._publish_fb(last_q, now - last_fb_t)
                    last_q = self.q.copy(); last_fb_t = now; t_fb = now
                if self._renderer is not None and now - t_rd >= 0.1:   # idle: slow frames
                    self._grab(); t_rd = now
                time.sleep(0.005)

    def _publish_fb(self, last_q, dt):
        now = self.get_clock().now().to_msg()
        with self.lock:
            q = self.q.copy()
        js = JointState(); js.header.stamp = now; js.name = list(JOINT_NAMES)
        js.position = [float(x) for x in q]                       # URDF rad
        self.pub_js.publish(js)
        deg = rad_to_linuxcnc_deg(q)
        vel = (rad_to_linuxcnc_deg(q) - rad_to_linuxcnc_deg(last_q)) / max(dt, 1e-3)
        fb = JointState(); fb.header.stamp = now; fb.name = list(JOINT_NAMES)
        fb.position = [float(x) for x in deg]                     # LinuxCNC deg
        fb.velocity = [float(x) for x in vel]                     # deg/s
        self.pub_fb.publish(fb)

    def _grab(self):
        try:
            self._renderer.update_scene(self.d, camera=self.cam)
            img = self._renderer.render()                          # RGB
        except Exception:
            return
        if self._vw is None:
            h, w = img.shape[:2]
            os.makedirs(os.path.dirname(self.video) or ".", exist_ok=True)
            self._vw = cv2.VideoWriter(self.video, cv2.VideoWriter_fourcc(*"mp4v"),
                                       30.0, (w, h))
        self._vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def close(self):
        self._stop.set()
        self._sim.join(timeout=2.0)
        if self._vw is not None:
            self._vw.release()
            self.get_logger().info(f"video saved -> {self.video}")


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=os.path.join(C.OUT_DIR, "mujoco_sim", "robot_sim.xml"))
    ap.add_argument("--video", default=os.path.join(C.OUT_DIR, "mujoco_sim", "episode.mp4"))
    ap.add_argument("--camera", default="iso")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    rclpy.init()
    node = MujocoRobot(args.xml, video=None if args.no_video else args.video, cam=args.camera)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
