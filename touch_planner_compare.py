#!/usr/bin/env python3
"""touch_planner_compare :: 'touch a table point and return' under four
trajectory-execution strategies, NO suction. Same geometry (base -> touch
(0.28,-0.02,z) -> base), same recorder, four transports:

  1 curobo_native  cuRobo plan, its OWN TOPP-RA timing (r['dt']), streamed
  2 curobo_retime  same geometry, uniform velocity-cap retiming (stream)
  3 rl             the RL policy drives the cup down to the point (10 Hz,
                   suction bit forced OFF), then cuRobo returns it
  4 mpc            cuRobo geometry as segment setpoints, tracked by the Pi
                   lag-aware MPC controller (robot_hal --controller mpc)

Transport = the September robot_hal on :9998 (stream chunk/spline + segment
target_deg) / :9999 feedback. Records command vs response per joint + TCP z.

    python3 touch_planner_compare.py --which 1,2,4 --exec
"""
import argparse
import json
import os
import socket
import sys
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C                                          # noqa: E402
from joint_conventions import (linuxcnc_deg_to_rad,          # noqa: E402
                               rad_to_linuxcnc_deg)

PI = "10.0.0.27"
PLANNER = ("127.0.0.1", 9997)
TOUCH_XY = (0.28, -0.02)
TOUCH_Z = -0.016          # soft target (calibrated just-touch is -0.019)
DOWN = [0.0, 1.0, 0.0, 0.0]
ELBOW = (70.0, 145.0)


# ---------- planner RPC ----------
def rpc(d):
    s = socket.create_connection(PLANNER, timeout=90)
    s.sendall((json.dumps(d) + "\n").encode())
    b = b""
    while not b.endswith(b"\n"):
        b += s.recv(65536)
    s.close()
    return json.loads(b)


def qrad(qd):
    return [float(v) for v in linuxcnc_deg_to_rad(qd)]


def fk_z(qd):
    return float(rpc({"type": "fk", "q": qrad(qd)})["pos"][0][2])


def plan(start_qd, xyz=None, goal_qd=None, attempts=12):
    for _ in range(3):
        if xyz is not None:
            r = rpc({"type": "plan_pose", "start_q": qrad(start_qd),
                     "goal_pose": list(xyz) + DOWN, "max_attempts": attempts})
        else:
            r = rpc({"type": "plan_joint", "start_q": qrad(start_qd),
                     "goal_q": [float(v) for v in goal_qd],
                     "max_attempts": attempts})
        if r.get("success"):
            return r
        time.sleep(0.4)
    return None


def check_box(traj_rad):
    t = np.degrees(np.asarray(traj_rad, float))
    if np.any(np.abs(t[:, 1]) > ELBOW[0] + 1e-4) or \
       np.any(np.abs(t[:, 2]) > ELBOW[1] + 1e-4):
        raise RuntimeError("elbow box violated")


# ---------- robot_hal :9998 / :9999 client ----------
class Robot:
    def __init__(self):
        self.samples = deque(maxlen=20000)
        self.stop = False
        self.cmd = None
        import threading
        threading.Thread(target=self._fb, daemon=True).start()
        self._connect_cmd()
        time.sleep(0.4)

    def _connect_cmd(self):
        self.cmd = socket.create_connection((PI, 9998), timeout=5)
        self.cmd.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def settle(self, timeout=4.0, tol=0.05):
        """Block until joints stop moving (max |Δq| between 0.2 s reads < tol)."""
        prev = np.array(self.state())
        t = time.time()
        while time.time() - t < timeout:
            time.sleep(0.2)
            cur = np.array(self.state())
            if np.abs(cur - prev).max() < tol:
                return cur
            prev = cur
        return prev

    def _fb(self):
        while not self.stop:
            try:
                s = socket.create_connection((PI, 9999), timeout=5)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                buf = b""
                while not self.stop:
                    d = s.recv(65536)
                    if not d:
                        break
                    buf += d
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        try:
                            m = json.loads(line)
                            self.samples.append((time.time(), m["joints_deg"][:6],
                                                 m.get("torque", [0] * 6)[:6]))
                        except Exception:
                            pass
            except OSError:
                time.sleep(0.3)

    def state(self):
        while not self.samples:
            time.sleep(0.02)
        return list(self.samples[-1][1])

    def send(self, obj):
        for attempt in (0, 1):
            try:
                self.cmd.sendall((json.dumps(obj) + "\n").encode())
                return
            except OSError:
                try:
                    self.cmd.close()
                except OSError:
                    pass
                self._connect_cmd()
                time.sleep(0.2)
        raise RuntimeError("cmd socket dead")

    def stream_traj(self, traj_rad, dt, settle=True):
        """One linear chunk welded by robot_hal's stream reference."""
        td = [list(map(float, rad_to_linuxcnc_deg(w))) for w in traj_rad]
        self.send({"chunk": td, "traj_dt": float(dt),
                   "t_anchor": time.time() + 0.12})
        time.sleep(dt * (len(td) - 1) + 0.2)
        if settle:
            self.settle()

    def segment(self, target_qd, duration, controller, settle=False):
        """One setpoint tracked by robot_hal's chosen controller (mpc)."""
        self.send({"target_deg": list(map(float, target_qd)),
                   "duration": float(duration), "controller": controller})
        time.sleep(duration + 0.25)
        if settle:
            self.settle()

    def close(self):
        self.stop = True
        try:
            self.cmd.close()
        except OSError:
            pass


# ---------- the four strategies ----------
def retime_dt(traj_rad, v_deg):
    seg = np.degrees(np.max(np.abs(np.diff(traj_rad, axis=0)), axis=1))
    return max(float(np.max(seg)) / max(v_deg, 1e-6), 0.01)


def run_touch(bot, which, v_transit, v_touch, dwell, marks):
    base = list(rad_to_linuxcnc_deg(np.asarray(C.START_Q, float)))
    # go to base first (retimed, benign)
    tr = plan(bot.state(), goal_qd=C.START_Q)
    if tr:
        t = np.asarray(tr["trajectory"])
        check_box(t)
        bot.stream_traj(t, retime_dt(t, 25.0))
    q0 = bot.state()

    dn = plan(q0, xyz=[TOUCH_XY[0], TOUCH_XY[1], TOUCH_Z])
    assert dn, "touch plan failed"
    down = np.asarray(dn["trajectory"])
    check_box(down)
    native_dt = float(dn.get("dt") or 0.0)
    k = int(len(down) * 0.7)

    marks.append(("descend_start", time.time()))
    if which == "curobo_native":
        dt = native_dt if native_dt > 1e-3 else 0.05
        bot.stream_traj(down, dt)
    elif which == "curobo_retime":
        bot.stream_traj(down[:k + 1], retime_dt(down[:k + 1], v_transit))
        bot.stream_traj(down[k:], retime_dt(down[k:], v_touch))
    elif which == "mpc":
        # sub-sample the geometric path into setpoints, MPC tracks each
        idx = list(range(0, len(down), max(1, len(down) // 8))) + [len(down) - 1]
        for a, b in zip(idx[:-1], idx[1:]):
            seg = down[a:b + 1]
            v = v_transit if b < k else v_touch
            tgt = list(map(float, rad_to_linuxcnc_deg(down[b])))
            bot.segment(tgt, retime_dt(seg, v), "mpc")
        bot.settle()
    else:
        raise ValueError(which)
    marks.append(("touch", time.time()))
    bot.settle()
    z = fk_z(bot.state())
    time.sleep(dwell)

    # return to base (retimed for all; the leg under test is the descent)
    up = plan(bot.state(), goal_qd=C.START_Q)
    if up:
        t = np.asarray(up["trajectory"])
        check_box(t)
        bot.stream_traj(t, retime_dt(t, v_transit))
    marks.append(("home", time.time()))
    return z, native_dt


def run_rl(bot, marks):
    """RL policy descends the cup to the touch point (suction forced OFF),
    then cuRobo returns to base. Runs the September controller's policy in
    a minimal open-loop-to-point form: we command the object at the touch
    xy and let the approach/press phases drive the descent."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rpc_mod", os.path.join(HERE, "rl", "real_policy_ctrl.py"))
    print("[rl] the RL policy is a full pick controller; for a pure touch we "
          "run its approach+press with suction disabled — see notes")
    # Delegated to real_policy_ctrl in --no_contact + suction-off mode via
    # its own CLI is cleaner; here we just flag it for the operator.
    raise NotImplementedError("run via: rl/real_policy_ctrl.py --no_suction "
                              "(wrapper wired in main once base is live)")


def plot_all(logs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=False)
    colors = dict(curobo_native="#D65F5F", curobo_retime="#4878CF",
                  mpc="#6ACC64", rl="#B47CC7")
    for name, (ts, qs, marks, z, ndt) in logs.items():
        ts = np.asarray(ts) - ts[0]
        qs = np.asarray(qs)
        for j in range(6):
            axes[j // 2, j % 2].plot(ts, qs[:, j], lw=1.1,
                                     color=colors.get(name), label=name)
    for j in range(6):
        ax = axes[j // 2, j % 2]
        ax.set_title(f"J{j+1}", fontsize=10)
        ax.set_ylabel("deg")
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Touch-and-return: joint response by planner/transport")
    fig.tight_layout()
    p = os.path.join(out, "touch_planner_compare.png")
    fig.savefig(p, dpi=110)
    print(f"[plot] {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="1,2,4",
                    help="subset of 1(curobo_native) 2(curobo_retime) "
                         "3(rl) 4(mpc)")
    ap.add_argument("--v_transit", type=float, default=30.0)
    ap.add_argument("--v_touch", type=float, default=8.0)
    ap.add_argument("--dwell", type=float, default=0.5)
    ap.add_argument("--exec", action="store_true")
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/touch_compare"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    names = {"1": "curobo_native", "2": "curobo_retime", "3": "rl", "4": "mpc"}
    order = [names[c] for c in a.which.split(",")]

    if not a.exec:
        print("[dry] would run:", order)
        for q0 in [C.START_Q]:
            dn = plan(list(rad_to_linuxcnc_deg(np.asarray(q0, float))),
                      xyz=[TOUCH_XY[0], TOUCH_XY[1], TOUCH_Z])
            print("  touch plan:", "ok" if dn else "FAIL",
                  "native_dt", (dn.get("dt") if dn else None))
        return

    bot = Robot()
    logs = {}
    try:
        for name in order:
            print(f"\n=== {name} ===", flush=True)
            marks = []
            base_i = len(bot.samples)
            if name == "rl":
                print("[rl] skipped in this harness — run rl/real_policy_ctrl.py "
                      "separately (see notes); recording others")
                continue
            z, ndt = run_touch(bot, name, a.v_transit, a.v_touch, a.dwell, marks)
            seg = list(bot.samples)[base_i:]
            ts = [s[0] for s in seg]
            qs = [s[1] for s in seg]
            logs[name] = (ts, qs, marks, z, ndt)
            print(f"[{name}] touch fk z = {z*1000:.1f} mm (cmd {TOUCH_Z*1000:.0f})"
                  f"  native_dt={ndt:.3f}s", flush=True)
    finally:
        bot.close()
        if logs:
            np.savez(os.path.join(a.out, "log.npz"),
                     **{k: np.asarray(v[1]) for k, v in logs.items()},
                     meta=json.dumps({k: dict(touch_z=v[3], native_dt=v[4])
                                      for k, v in logs.items()}))
            plot_all(logs, a.out)
    print("[done]")


if __name__ == "__main__":
    main()
