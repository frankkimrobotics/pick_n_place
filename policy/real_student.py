#!/usr/bin/env python3
"""real_student :: the DAgger RGBD student policy on the REAL Pro 630.

Mirrors the sim interface of rl/distill.py Student:
  im1 = wrist D405 RGBD 96x96 (4ch: RGB + depth/2m as uint8)
  im2 = fixed D435 RGBD 96x96
  prop = [q(6, URDF rad), qd(6, rad/s), sealed(1)]
  goal = [place_x, place_y, target_h]
  act  = tanh 7: dq (+-2 deg per 0.1 s tick) + suction logit

Transport: robot_hal pid (:9998) per-tick targets; state from :9999.
Suction: halcmd digital_out00 over ssh, 3-tick release hysteresis
(mirrors the env). sealed proxy = commanded suction state.

SAFETY: dry-run default (--exec to move); --slow N stretches time N x
(deltas divided by N); per-tick delta clamp; URDF joint boxes
(|j2|<=70, |j3|<=145 deg, limits-5deg); start must be within 15 deg of
home; SIGINT/any exception -> suction OFF, motion stops (pid holds).

    PYTHONPATH=~/librealsense/build/release python3 policy/real_student.py \
        --ckpt ~/pnp_rl/distill1/student_final.pt --goal 0.30,-0.10 [--exec]
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.abspath(os.path.join(ROOT, "..", "mycobot_mpc"))):
    sys.path.insert(0, p)
from joint_conventions import linuxcnc_deg_to_rad, rad_to_linuxcnc_deg  # noqa: E402
import config as C                                                      # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "rl"))
from distill import Student                                             # noqa: E402

PI = "10.0.0.27"
WRIST_SER, FIXED_SER = "218622271300", "043422070101"
SUCTION_PIN = "pro600.digital_out00"
IMG = 96
DQ_MAX = np.radians(2.0)          # sim per-tick delta
TICK = 0.1                        # 10 Hz decisions
ELBOW = (70.0, 145.0)             # URDF deg


class ShmCam:
    """Read frames published by rs_shm_server.py (system python3)."""
    def __init__(self, name):
        self.path = f"/dev/shm/rs_{name}.npy"
        self.latest = None
        self.on = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        import numpy as _np
        while self.on:
            try:
                self.latest = _np.load(self.path)
            except Exception:
                pass
            time.sleep(0.03)

    def stop(self):
        self.on = False


class Cam:
    """Continuous aligned RGBD grabber -> latest 4ch 96x96 uint8."""
    def __init__(self, serial, w, h, fps):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        prof = self.pipe.start(cfg)
        self.scale = prof.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.latest = None
        self.on = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.on:
            try:
                f = self.align.process(self.pipe.wait_for_frames(4000))
                rgb = np.asanyarray(f.get_color_frame().get_data())
                dep = np.asanyarray(f.get_depth_frame().get_data()).astype(
                    np.float32) * self.scale
                rgb = cv2.resize(rgb, (IMG, IMG), interpolation=cv2.INTER_AREA)
                d8 = (np.clip(cv2.resize(dep, (IMG, IMG),
                                         interpolation=cv2.INTER_NEAREST),
                              0, 2.0) * 127.5).astype(np.uint8)
                self.latest = np.concatenate(
                    [rgb, d8[..., None]], -1).transpose(2, 0, 1)   # (4,96,96)
            except Exception:
                time.sleep(0.1)

    def stop(self):
        self.on = False
        try:
            self.pipe.stop()
        except Exception:
            pass


class Robot:
    def __init__(self, execute):
        self.execute = execute
        self.suction = 0
        self.samples = []
        self.cmds = []
        self.on = True
        self._q = None
        threading.Thread(target=self._stream, daemon=True).start()
        t0 = time.time()
        while self._q is None and time.time() - t0 < 5:
            time.sleep(0.05)
        if self._q is None:
            raise RuntimeError("no state from :9999")

    def _stream(self):
        while self.on:
            try:
                s = socket.create_connection((PI, 9999), timeout=2)
                buf = b""
                while self.on:
                    buf += s.recv(4096)
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        try:
                            d = json.loads(line)
                            self._q = list(map(float, d["joints_deg"]))
                            self.samples.append((time.time(), *self._q))
                        except Exception:
                            pass
            except Exception:
                time.sleep(0.3)

    def q_deg(self):
        return list(self._q)

    def send(self, tgt_deg, dur):
        self.cmds.append((time.time(), *tgt_deg, dur))
        if not self.execute:
            return
        k = socket.create_connection((PI, 9998), timeout=2)
        k.sendall((json.dumps({"target_deg": [float(v) for v in tgt_deg],
                               "duration": float(dur),
                               "controller": "pid"}) + "\n").encode())
        try:
            k.settimeout(1)
            k.recv(256)
        except Exception:
            pass
        k.close()

    def set_suction(self, onoff):
        onoff = int(onoff)
        if onoff == self.suction:
            return
        self.suction = onoff
        print(f"  [suction] -> {onoff}")
        if not self.execute:
            return
        subprocess.run(["ssh", "-o", "BatchMode=yes", f"pi@{PI}",
                        f"halcmd setp {SUCTION_PIN} {onoff}"],
                       timeout=6, capture_output=True)

    def stop(self):
        self.on = False


def check_box(q_rad):
    d = np.degrees(q_rad)
    if abs(d[0]) > 60 or abs(d[1]) > ELBOW[0] or abs(d[2]) > ELBOW[1]:
        raise RuntimeError(f"joint box violated: {np.round(d, 1)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser(
        "~/pnp_rl/distill1/student_final.pt"))
    ap.add_argument("--goal", default="0.30,-0.10", help="place target x,y (m)")
    ap.add_argument("--target_h", type=float, default=0.0)
    ap.add_argument("--exec", action="store_true")
    ap.add_argument("--slow", type=float, default=3.0,
                    help="time stretch: deltas / slow, tick * slow")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/real_student"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    st = Student().to(dev)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    st.load_state_dict(ck["student"])
    st.eval()
    gx, gy = map(float, a.goal.split(","))
    goal = torch.tensor([[gx, gy, a.target_h]], dtype=torch.float32,
                        device=dev)

    print("[cam] opening wrist D405 + fixed D435 ...")
    cam1 = ShmCam("wrist")
    cam2 = ShmCam("fixed")          # frames from rs_shm_server.py
    t0 = time.time()
    while (cam1.latest is None or cam2.latest is None) and time.time() - t0 < 15:
        time.sleep(0.2)
    if cam1.latest is None or cam2.latest is None:
        sys.exit("camera stream failed")
    print("[cam] both streaming")

    bot = Robot(a.exec)
    qd0 = np.array(bot.q_deg())
    home = np.array(rad_to_linuxcnc_deg(np.asarray(C.START_Q, float)))
    if np.max(np.abs(qd0 - home)) > 15:
        sys.exit(f"start too far from home: {np.round(qd0, 1)} vs {np.round(home, 1)}")
    print(f"[state] joints {np.round(qd0, 1)} (home ok)")

    stop_flag = {"stop": False}
    signal.signal(signal.SIGINT, lambda *_: stop_flag.update(stop=True))

    q_ref = np.array(linuxcnc_deg_to_rad(qd0), float)     # URDF rad reference
    q_prev = q_ref.copy()
    want_off = 0
    sealed_proxy = 0
    tick = TICK * a.slow
    print(f"[run] {a.steps} steps @ {1/tick:.1f} Hz effective "
          f"({'EXEC' if a.exec else 'DRY'}), goal ({gx:.2f},{gy:.2f})")
    try:
        for t in range(a.steps):
            if stop_flag["stop"]:
                print("[stop] SIGINT")
                break
            ts = time.time()
            im1 = torch.tensor(cam1.latest[None], device=dev).float() / 255.0
            im2 = torch.tensor(cam2.latest[None], device=dev).float() / 255.0
            q_now = np.array(linuxcnc_deg_to_rad(bot.q_deg()), float)
            qd_est = (q_now - q_prev) / tick
            q_prev = q_now
            prop = torch.tensor([[*q_now, *qd_est, float(sealed_proxy)]],
                                dtype=torch.float32, device=dev)
            with torch.no_grad():
                act = torch.tanh(st(im1, im2, prop, goal))[0].cpu().numpy()
            dq = act[:6] * DQ_MAX / a.slow
            q_ref = q_ref + dq
            # anti-windup: the reference may never run away from the
            # measured state (a stalled arm wound the ref into the joint
            # box on the first attempt; robot_hal's integral wound too)
            q_ref = np.clip(q_ref, q_now - np.radians(5), q_now + np.radians(5))
            check_box(q_ref)
            tgt = rad_to_linuxcnc_deg(q_ref)
            bot.send(tgt, tick)
            # suction with 3-tick release hysteresis (matches env)
            want = act[6] > 0
            if want:
                want_off = 0
                bot.set_suction(1)
                sealed_proxy = 1
            elif sealed_proxy:
                want_off += 1
                if want_off >= 3:
                    bot.set_suction(0)
                    sealed_proxy = 0
            if t % 10 == 0:
                print(f"  t={t:3d} q={np.round(np.degrees(q_now),1)} "
                      f"suction={sealed_proxy}")
            dt = time.time() - ts
            if dt < tick:
                time.sleep(tick - dt)
    finally:
        bot.set_suction(0)
        cam1.stop()
        cam2.stop()
        S = np.array(bot.samples) if bot.samples else np.zeros((0, 7))
        Cm = np.array(bot.cmds) if bot.cmds else np.zeros((0, 8))
        np.savez(os.path.join(a.out, "run_log.npz"), samples=S, cmds=Cm)
        bot.stop()
        print(f"[log] {len(S)} samples, {len(Cm)} cmds -> {a.out}/run_log.npz")
    print("[done]")


if __name__ == "__main__":
    main()
