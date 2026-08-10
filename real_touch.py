#!/usr/bin/env python3
"""real_touch :: touch the detected object with the (unpowered) suction cup
and return home -- cuRobo-planned, robot_hal-executed, fully logged.

Sequence: read q -> plan_joint HOME -> detect object (fixed D435, base
frame) -> plan_pose HOVER (top + standoff, cup straight down, J6 held) ->
plan_pose TOUCH (FK tip = object top - DESCEND_BELOW, the calib_touch
2026-07-03 cup-compression offset) at touch speed -> dwell -> retreat to
HOVER -> plan_joint HOME.  NO suction at any point.

Transport = robot_hal pid segments (:9998 {"target_deg","duration",
"controller":"pid"}), state stream :9999 sampled continuously in a
background thread.  Every command and every stream sample is logged;
exit writes an npz + a per-joint command-vs-response plot.

SAFETY: dry-run by default (--exec required for motion); per-segment
velocity caps (approach/touch); elbow box |j2|<=70 |j3|<=145 deg checked
on every waypoint; absolute z floor; refuses to start if any planner /
robot service is down.

    python3 real_touch.py                # dry-run: plan + print only
    python3 real_touch.py --exec         # move (slow)
    python3 real_touch.py --exec --obj 0.35,0.05,0.045   # skip detection
"""
import argparse
import json
import os
import socket
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc"))):
    sys.path.insert(0, p)
import config as C
from geometry import R_from_two_axes, R_to_quat_wxyz
from joint_conventions import linuxcnc_deg_to_rad, rad_to_linuxcnc_deg

PI = "10.0.0.27"
DOWN = list(map(float, R_to_quat_wxyz(R_from_two_axes(np.array([0, 0, -1.0])))))
DESCEND_BELOW = 0.019          # calib_touch: FK tip this far below surface = just-touch
ABS_FLOOR = -0.03              # never command FK tip below this (table at -0.10)
ELBOW = (70.0, 145.0)          # |j2|, |j3| limits (deg, LinuxCNC frame)


def rpc(d):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=60)
    s.sendall((json.dumps(d) + "\n").encode())
    b = b""
    while not b.endswith(b"\n"):
        b += s.recv(65536)
    s.close()
    return json.loads(b)


def read_q_deg():
    s = socket.create_connection((PI, 9999), timeout=3)
    b = b""
    while b"\n" not in b:
        b += s.recv(4096)
    s.close()
    return list(json.loads(b.split(b"\n")[0])["joints_deg"])


class Logger:
    """Continuous :9999 sampler + command record."""
    def __init__(self):
        self.samples = []          # (t, 6 joints_deg)
        self.cmds = []             # (t, 6 target_deg, duration)
        self.on = True
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
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
                            self.samples.append(
                                (time.time(), *map(float, d["joints_deg"])))
                        except Exception:
                            pass
                s.close()
            except Exception:
                time.sleep(0.3)

    def log_cmd(self, tgt, dur):
        self.cmds.append((time.time(), *map(float, tgt), float(dur)))

    def stop(self):
        self.on = False
        time.sleep(0.3)


def check_box(traj_rad):
    """Elbow box in the URDF/rad frame (the cuRobo clip frame)."""
    t = np.degrees(np.asarray(traj_rad, float))
    if np.any(np.abs(t[:, 1]) > ELBOW[0] + 1e-4) or \
       np.any(np.abs(t[:, 2]) > ELBOW[1] + 1e-4):
        raise RuntimeError(
            f"elbow box violated (urdf deg): j2 max {np.abs(t[:,1]).max():.1f} "
            f"j3 max {np.abs(t[:,2]).max():.1f}")


def exec_traj(traj_rad, dt, vmax_deg, log, execute, label):
    """Clamp trajectory rate to vmax and walk :9998 pid sub-targets."""
    traj = np.asarray(traj_rad, float)
    traj[:, 5] = float(C.BASE_Q[5])                    # hold J6 (symmetric cup)
    peak = (np.degrees(np.max(np.abs(np.diff(traj, axis=0)))) / dt
            if len(traj) > 1 else vmax_deg)
    sdt = dt * max(1.0, peak / vmax_deg)
    td = [list(map(float, rad_to_linuxcnc_deg(wp))) for wp in traj]
    check_box(traj)
    SUB_T = 1.0
    step = max(1, int(round(SUB_T / sdt)))
    idx = list(range(step, len(td), step)) + [len(td) - 1]
    total = sdt * (len(td) - 1)
    print(f"[{label}] {len(td)} wps, {total:.1f}s (peak {peak:.1f} -> "
          f"{min(peak, vmax_deg):.1f} deg/s)")
    prev = 0
    for i in idx:
        dur = (i - prev) * sdt
        prev = i
        if not execute:
            continue
        log.log_cmd(td[i], dur)
        k = socket.create_connection((PI, 9998), timeout=3)
        k.sendall((json.dumps({"target_deg": td[i], "duration": float(dur),
                               "controller": "pid"}) + "\n").encode())
        try:
            k.settimeout(2)
            k.recv(256)
        except Exception:
            pass
        k.close()
        time.sleep(dur)
    if execute:
        time.sleep(0.8)


def plan_pose(goal_xyz, start_qd=None):
    qd = start_qd if start_qd is not None else read_q_deg()
    r = rpc({"type": "plan_pose", "start_q": [float(v) for v in linuxcnc_deg_to_rad(qd)],
             "goal_pose": [float(v) for v in goal_xyz] + DOWN, "max_attempts": 16})
    if not r.get("success"):
        raise RuntimeError(f"plan_pose to {goal_xyz} failed")
    return r["trajectory"], r["dt"]


def plan_joint(goal_q, start_qd=None):
    qd = start_qd if start_qd is not None else read_q_deg()
    r = rpc({"type": "plan_joint", "start_q": [float(v) for v in linuxcnc_deg_to_rad(qd)],
             "goal_q": [float(v) for v in goal_q], "max_attempts": 12})
    if not r.get("success"):
        raise RuntimeError("plan_joint failed")
    return r["trajectory"], r["dt"]


def detect_object():
    """Fixed-D435 detection in base frame via d435_detect (sam3 env)."""
    import re
    import subprocess
    cmd = [os.path.expanduser("~/miniconda3/envs/sam3/bin/python"),
           os.path.join(HERE, "d435_detect.py")]
    print("[detect]", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(r.stdout)
    m = re.search(r"\[0\] base XY \[([+-\d.]+),([+-\d.]+)\].*?TOP=([\d.+-]+)",
                  r.stdout)
    if not m:
        raise RuntimeError(f"no objects detected:\n{r.stdout}\n{r.stderr[-500:]}")
    ox, oy, top = map(float, m.groups())
    print(f"[detect] object at ({ox:.3f},{oy:.3f}) top z={top:.3f}")
    return ox, oy, top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec", action="store_true", help="actually move")
    ap.add_argument("--obj", default=None,
                    help="x,y,top_z override (skip camera detection)")
    ap.add_argument("--standoff", type=float, default=0.06)
    ap.add_argument("--v_approach", type=float, default=10.0)
    ap.add_argument("--v_touch", type=float, default=3.0)
    ap.add_argument("--dwell", type=float, default=1.5)
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/real_touch"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # --- service checks ---
    need = [("127.0.0.1", 9997, "curobo planner")]
    if a.exec:
        need += [(PI, 9999, "robot_hal stream"), (PI, 9998, "robot_hal cmd")]
    for host, port, name in need:
        try:
            socket.create_connection((host, port), timeout=3).close()
        except Exception:
            sys.exit(f"SERVICE DOWN: {name} ({host}:{port}) -- start it first")

    try:
        qd0 = read_q_deg()
    except Exception:
        if a.exec:
            raise
        qd0 = list(rad_to_linuxcnc_deg(np.asarray(C.START_Q, float)))
        print("[dry] robot stream down -- assuming START_Q for planning")
    print(f"[state] joints (lcnc deg): {np.round(qd0, 1)}")

    if a.obj:
        ox, oy, otop = map(float, a.obj.split(","))
    else:
        ox, oy, otop = detect_object()
    touch_z = max(otop - DESCEND_BELOW, ABS_FLOOR)
    print(f"[plan] hover ({ox:.3f},{oy:.3f},{otop + a.standoff:.3f}) -> "
          f"touch z {touch_z:.3f} (top {otop:.3f} - {DESCEND_BELOW})")

    log = Logger()
    t0 = time.time()
    qd_cur = qd0

    def last_deg(traj):
        return list(map(float, rad_to_linuxcnc_deg(np.asarray(traj[-1], float))))

    try:
        # 1 home
        traj, dt = plan_joint([float(v) for v in C.START_Q], qd_cur)
        exec_traj(traj, dt, a.v_approach, log, a.exec, "home")
        qd_cur = last_deg(traj)
        # 2 hover over object
        traj, dt = plan_pose([ox, oy, otop + a.standoff], qd_cur)
        exec_traj(traj, dt, a.v_approach, log, a.exec, "hover")
        qd_cur = last_deg(traj)
        # 3 slow descent to touch
        traj, dt = plan_pose([ox, oy, touch_z], qd_cur)
        exec_traj(traj, dt, a.v_touch, log, a.exec, "touch")
        qd_cur = last_deg(traj)
        print(f"[touch] dwell {a.dwell}s (no suction)")
        if a.exec:
            time.sleep(a.dwell)
        # 4 retreat + home
        traj, dt = plan_pose([ox, oy, otop + a.standoff], qd_cur)
        exec_traj(traj, dt, a.v_touch, log, a.exec, "retreat")
        qd_cur = last_deg(traj)
        traj, dt = plan_joint([float(v) for v in C.START_Q], qd_cur)
        exec_traj(traj, dt, a.v_approach, log, a.exec, "home2")
    finally:
        log.stop()
        S = np.array(log.samples) if log.samples else np.zeros((0, 7))
        Cm = np.array(log.cmds) if log.cmds else np.zeros((0, 8))
        np.savez(os.path.join(a.out, "touch_log.npz"), samples=S, cmds=Cm,
                 t0=t0)
        print(f"[log] {len(S)} samples, {len(Cm)} cmds -> {a.out}/touch_log.npz")
        if len(S) and len(Cm):
            plot(S, Cm, t0, a.out)
    print("[done]")


def plot(S, Cm, t0, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    ts = S[:, 0] - t0
    for j in range(6):
        ax = axes[j // 2, j % 2]
        ax.plot(ts, S[:, 1 + j], "-", lw=1.2, color="#4878CF",
                label="response (:9999)")
        # command staircase: target held over its duration
        cx, cy = [], []
        for row in Cm:
            tc, tgt, dur = row[0] - t0, row[1 + j], row[7]
            cx += [tc, tc + dur]
            cy += [tgt, tgt]
        ax.plot(cx, cy, "--", lw=1.2, color="#D65F5F", label="command (:9998)")
        ax.set_title(f"J{j + 1}", fontsize=10)
        ax.set_ylabel("deg")
        if j == 0:
            ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("t (s)")
    fig.suptitle("real_touch: command vs response (LinuxCNC deg)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "touch_cmd_vs_response.png"), dpi=110)
    print(f"[plot] {out}/touch_cmd_vs_response.png")


if __name__ == "__main__":
    main()
