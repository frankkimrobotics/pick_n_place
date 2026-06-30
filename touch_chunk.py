#!/usr/bin/env python3
"""touch_chunk :: touch + return driven by cuRobo through 0.1 s SLIDING-WINDOW chunks fed to
the 4 ms online_servo, with a D405 vision contact-stop (dome shift on cup compression).

  cuRobo (:9997) plans the full trajectory -> resample dt=0.01 -> every 0.1 s emit a 0.4 s
  chunk (sliding window) on /planner/weld_chunks -> chunk_to_pi -> :9994 -> online_servo welds
  + follows @250 Hz. During the descent the D405 dome shift is watched; on contact a HOLD is
  sent (online_servo freezes q_ref). Then the return is streamed the same way.

Records to a run dir (joints, chunks, phase, contact signals, D405 + D435 RGB) AND an mcap
rosbag (EpisodeBag: joints + D405 frames + phases). Plot with plot_touch.py.

  source /opt/ros/humble/setup.bash ; python3 touch_chunk.py 0.40 -0.05 0.07
"""
import argparse
import json
import os
import socket
import sys
import threading
import time

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C
from geometry import R_from_two_axes, R_to_quat_wxyz
from joint_conventions import rad_to_linuxcnc_deg
from servo_touch import GapMonitor

DOWN = list(R_to_quat_wxyz(R_from_two_axes(np.array([0, 0, -1.0]))))


def rpc(d):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=40)
    s.sendall((json.dumps(d) + "\n").encode()); b = b""
    while not b.endswith(b"\n"):
        b += s.recv(65536)
    s.close(); return json.loads(b)


class D435(threading.Thread):
    """Second RealSense (fixed D435) RGB grabber."""
    def __init__(self, serial="043422070101", w=848, h=480):
        super().__init__(daemon=True)
        self.serial, self.w, self.h = serial, w, h
        self.lock = threading.Lock(); self.stop_evt = threading.Event()
        self._rgb = None; self.ok = False

    def run(self):
        import pyrealsense2 as rs
        pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, 30)
        pipe.start(cfg); self.ok = True
        try:
            for _ in range(8):
                pipe.wait_for_frames(2000)
            while not self.stop_evt.is_set():
                try:
                    f = pipe.wait_for_frames(1000)
                except RuntimeError:
                    continue
                c = f.get_color_frame()
                if c:
                    with self.lock:
                        self._rgb = np.asanyarray(c.get_data())          # BGR
        finally:
            try:
                pipe.stop()
            except Exception:
                pass

    def latest(self):
        with self.lock:
            return None if self._rgb is None else self._rgb.copy()


class TouchChunk(Node):
    def __init__(self, a):
        super().__init__("touch_chunk")
        self.a = a
        self.chunk_pub = self.create_publisher(String, "/planner/weld_chunks", 10)
        self.q = None; self.create_subscription(JointState, "/joint_states", self._on_js, 50)
        self.joints = []            # (t, q_rad)
        self.chunks = []            # (t, phase, [deg])
        self.events = []            # (t, kind, detail)
        self.signals = []           # (t, domey, base_dome, cupd_mm, gap_mm, shift)
        self.phase = "init"
        # cameras
        cup = np.load(a.cup); rim = cup["rim"].astype(bool); ann = cup["ring"].astype(bool)
        dome = cup["dome"].astype(bool) if "dome" in cup.files else cup["mask"].astype(bool)
        self.mon = GapMonitor(a.serial, rim.shape[1], rim.shape[0], rim, ann, dome); self.mon.start()
        self.d435 = D435(); self.d435.start()
        t0 = time.time()
        while not self.mon.ok and time.time() - t0 < 8:
            time.sleep(0.2)
        self.run_dir = os.path.join(a.out, f"touch_{a.stamp}")
        os.makedirs(os.path.join(self.run_dir, "d405"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "d435"), exist_ok=True)
        self._frame_stop = threading.Event()
        self._fi = 0

    def _on_js(self, m):
        self.q = np.array(m.position, float)
        self.joints.append((time.time(), [float(x) for x in m.position]))

    def set_phase(self, p):
        self.phase = p
        self.events.append((time.time(), "phase", p))
        self.get_logger().info(f"=== phase: {p} ===")

    # frame + signal recorder (~15 Hz)
    def _frame_loop(self):
        while not self._frame_stop.is_set():
            t = time.time()
            r = self.mon.latest_rgbd()
            _, domey = self.mon.latest_cupd(); cupd, _ = self.mon.latest_cupd()
            gap, _ = self.mon.latest_gap()
            self.signals.append((t, None if domey is None else float(domey),
                                 self.base_dome, None if cupd is None else float(cupd),
                                 None if gap is None else float(gap), self.phase))
            if r is not None:
                cv2.imwrite(os.path.join(self.run_dir, "d405", f"{self._fi:04d}.jpg"),
                            cv2.cvtColor(r[0], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
            d4 = self.d435.latest()
            if d4 is not None:
                cv2.imwrite(os.path.join(self.run_dir, "d435", f"{self._fi:04d}.jpg"), d4,
                            [cv2.IMWRITE_JPEG_QUALITY, 70])
            self.events.append((t, "frame", self._fi)); self._fi += 1
            time.sleep(0.066)

    # plan + sliding-window chunk stream
    def plan(self, goal_pose=None, goal_q=None):
        cur = [float(x) for x in self.q]
        if goal_q is not None:
            r = rpc({"type": "plan_joint", "start_q": cur, "goal_q": goal_q})
        else:
            r = rpc({"type": "plan_pose", "start_q": cur, "goal_pose": goal_pose, "max_attempts": 14})
        if not r.get("success"):
            return None
        traj = np.array(r["trajectory"], float); dt = float(r["dt"])
        t = np.arange(len(traj)) * dt
        tn = np.arange(0.0, t[-1] + 1e-9, 0.01)
        return np.column_stack([np.interp(tn, t, traj[:, j]) for j in range(traj.shape[1])])

    def stream(self, fine, phase, v_des, contact=False):
        """Sliding-window: every 0.1 s emit the next 0.4 s slice; optionally watch contact."""
        self.set_phase(phase)
        deg = np.array([rad_to_linuxcnc_deg(w) for w in fine])
        peak = float(np.abs(np.diff(deg, axis=0)).max()) / 0.01 if len(deg) > 1 else v_des
        sdt = 0.01 * max(1.0, peak / v_des)                          # slow to <= v_des
        tt = np.arange(len(deg)) * sdt
        play = np.column_stack([np.interp(np.arange(0, tt[-1] + 1e-9, 0.01), tt, deg[:, j]) for j in range(6)])
        t0 = time.time() + 0.15
        domes = []
        while True:
            now = time.time(); k = int((now - t0) / 0.01)
            if contact:
                _, domey = self.mon.latest_cupd()
                if domey is not None:
                    if self.base_dome is None:                        # baseline over the first ~1.5s free descent
                        domes.append(domey)                          # one sample per 0.1s loop iteration
                        if len(domes) >= 15:
                            self.base_dome = float(np.median(domes))
                            self.get_logger().info(f"  armed: base_dome={self.base_dome:.0f}px")
                    elif (self.base_dome - domey) >= self.a.dome_shift:
                        self.chunk_pub.publish(String(data=json.dumps({"hold": True})))
                        self.events.append((now, "contact", f"dome_shift={self.base_dome-domey:.0f}px"))
                        self.get_logger().info(f"  CONTACT dome+{self.base_dome-domey:.0f}px -> HOLD")
                        return "contact"
            if k >= len(play):
                # do NOT hold-freeze: the welder clamps to the last welded point (the floor),
                # so the lagging cup keeps converging down to the commanded floor (into the object).
                return "floor" if contact else "done"
            if k >= 0:
                ch = [r.tolist() for r in play[k:k + 40]]            # 0.4 s chunk
                self.chunk_pub.publish(String(data=json.dumps(
                    {"trajectory": ch, "traj_dt": 0.01, "t_anchor": float(t0 + k * 0.01)})))
                self.chunks.append((now, phase, ch[0]))
            time.sleep(0.1)                                          # 10 Hz sliding window

    def run(self):
        from episode_bag import EpisodeBag
        while self.q is None:
            rclpy.spin_once(self, timeout_sec=0.05)
        self.base_dome = None
        threading.Thread(target=self._frame_loop, daemon=True).start()
        jn = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        bag = EpisodeBag(self, jn, storage="mcap")
        try:
            bag.start(os.path.join(self.run_dir, "rosbag"), frame_getter=self.mon.latest_rgbd)
        except Exception as e:
            self.get_logger().warn(f"rosbag start failed ({e}); continuing"); bag = None
        x, y, top = self.a.xyz
        spin = threading.Thread(target=lambda: rclpy.spin(self), daemon=True); spin.start()
        try:
            if bag: bag.write_phase("approach")
            self.stream(self.plan(goal_pose=[x, y, top + 0.05] + DOWN), "approach_pregrasp", 10.0)
            time.sleep(1.5)
            if bag: bag.write_phase("descend")
            res = self.stream(self.plan(goal_pose=[x, y, top - 0.015] + DOWN), "descend", self.a.v_des, contact=True)
            time.sleep(2.5)                                  # let the lagging cup settle onto/into the object
            if bag: bag.write_phase("contact")
            self.set_phase("contact_dwell"); time.sleep(self.a.dwell)
            if bag: bag.write_phase("return")
            self.stream(self.plan(goal_q=list(map(float, C.BASE_Q))), "return", 10.0)
            time.sleep(2.0)
            self.set_phase("done")
        finally:
            self._frame_stop.set(); time.sleep(0.3)
            if bag: bag.stop()
            self._save()
            self.mon.stop_evt.set(); self.d435.stop_evt.set()

    def _save(self):
        d = self.run_dir
        jt = np.array([r[0] for r in self.joints]); jq = np.array([r[1] for r in self.joints])
        np.savez(os.path.join(d, "joints.npz"), t=jt, q=jq)
        json.dump(self.chunks, open(os.path.join(d, "chunks.json"), "w"))
        json.dump(self.events, open(os.path.join(d, "events.json"), "w"))
        sig = [(t, dm, bd, cu, gp, ph) for (t, dm, bd, cu, gp, ph) in self.signals]
        json.dump(sig, open(os.path.join(d, "signals.json"), "w"))
        json.dump({"xyz": self.a.xyz, "frames": self._fi}, open(os.path.join(d, "meta.json"), "w"))
        print(f"  run dir -> {d}: {len(self.joints)} joint samples, {self._fi} frames, "
              f"{len(self.chunks)} chunks, rosbag/ (mcap)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz", nargs=3, type=float, help="x y object_TOP_z")
    ap.add_argument("--v-des", type=float, default=4.0)
    ap.add_argument("--dome-shift", type=float, default=12.0, help="dome rise (px) = contact")
    ap.add_argument("--dwell", type=float, default=1.0)
    ap.add_argument("--serial", default="218622271300")
    ap.add_argument("--cup", default=os.path.join(C.OUT_DIR, "cup_mask.npz"))
    ap.add_argument("--out", default=os.path.join(C.OUT_DIR, "episodes"))
    ap.add_argument("--stamp", default=time.strftime("%Y%m%d_%H%M%S"))
    a = ap.parse_args()
    rclpy.init()
    node = TouchChunk(a)
    try:
        node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
