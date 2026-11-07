#!/usr/bin/env python3
"""real_touch_online :: cuRobo touch demo over the ONLINE chunk-streaming stack.

Transport = elerob_online.ini + online_servo.py: trajectory chunks to :9994
(latency_common.stream_at), feedback + torque recorded at native rate from
:9999 (StreamRecorder). Sequence: hover -> slow touch descend -> dwell ->
retreat -> base. No suction. Saves command + response and plots
angle/velocity per joint.

    python3 real_touch_online.py --obj 0.35,0.05,0.045          # dry (plans only)
    python3 real_touch_online.py --obj 0.35,0.05,0.045 --exec   # move
    python3 real_touch_online.py --exec                          # detect via D435
"""
import argparse
import json
import os
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C                                          # noqa: E402
from joint_conventions import linuxcnc_deg_to_rad           # noqa: E402
from latency_common import (CMD_LOG, StreamRecorder, close_sender, fk,  # noqa: E402
                            plan_joint, plan_pose, qrad, read_state,
                            stream_at)

DESCEND_BELOW = 0.019      # calib_touch: FK tip this far below top = just-touch
ABS_FLOOR = -0.03
ELBOW = (70.0, 145.0)
DETECT_BOX = dict(x=(0.22, 0.52), y=(-0.28, 0.28), z=(-0.02, 0.15))


def check_box(traj_rad):
    t = np.degrees(np.asarray(traj_rad, float))
    if np.any(np.abs(t[:, 1]) > ELBOW[0] + 1e-4) or \
       np.any(np.abs(t[:, 2]) > ELBOW[1] + 1e-4):
        raise RuntimeError(f"elbow box violated: j2max {np.abs(t[:,1]).max():.1f} "
                           f"j3max {np.abs(t[:,2]).max():.1f}")


def detect_object():
    import re
    import subprocess
    cmd = [os.path.expanduser("~/miniconda3/envs/sam3/bin/python"),
           os.path.join(HERE, "d435_detect.py")]
    print("[detect]", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(r.stdout)
    m = re.search(r"\[0\] base XY \[([-+\d.]+),([-+\d.]+)\].*?TOP=([-+\d.]+)",
                  r.stdout)
    if not m:
        raise RuntimeError(f"no objects detected:\n{r.stdout[-800:]}\n{r.stderr[-400:]}")
    ox, oy, top = map(float, m.groups())
    print(f"[detect] object at ({ox:.3f},{oy:.3f}) top z={top:.3f}")
    if not (DETECT_BOX["x"][0] <= ox <= DETECT_BOX["x"][1] and
            DETECT_BOX["y"][0] <= oy <= DETECT_BOX["y"][1] and
            DETECT_BOX["z"][0] <= top <= DETECT_BOX["z"][1]):
        raise RuntimeError(f"detection outside sanity box {DETECT_BOX} — "
                           "camera calib suspect (neck reprint), aborting")
    return ox, oy, top


def run_leg(label, traj, v_deg, execute):
    """Plan sanity + stream one leg; returns final joint state (lcnc deg)."""
    check_box(traj)
    n, dur = len(traj), None
    print(f"[{label}] {n} wps @ {v_deg} deg/s"
          + ("" if execute else "  (dry — not streamed)"))
    if execute:
        dt = stream_at(traj, v_deg)
        dur = dt * (n - 1)
    from joint_conventions import rad_to_linuxcnc_deg
    return list(map(float, rad_to_linuxcnc_deg(np.asarray(traj[-1], float))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec", action="store_true")
    ap.add_argument("--obj", default=None, help="x,y,top_z (skip detection)")
    ap.add_argument("--standoff", type=float, default=0.06)
    ap.add_argument("--v_approach", type=float, default=10.0)
    ap.add_argument("--v_touch", type=float, default=3.0)
    ap.add_argument("--dwell", type=float, default=1.5)
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/real_touch_online"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    for host, port, name in [("127.0.0.1", 9997, "curobo"),
                             ("10.0.0.27", 9999, "feedback"),
                             ("10.0.0.27", 9994, "online_servo chunks")]:
        try:
            socket.create_connection((host, port), timeout=3).close()
        except Exception:
            sys.exit(f"SERVICE DOWN: {name} ({host}:{port})")

    qd0, tq0 = read_state()
    print(f"[state] joints (lcnc deg): {np.round(qd0, 2)}")

    if a.obj:
        ox, oy, otop = map(float, a.obj.split(","))
    else:
        ox, oy, otop = detect_object()
    touch_z = max(otop - DESCEND_BELOW, ABS_FLOOR)
    print(f"[plan] hover ({ox:.3f},{oy:.3f},{otop + a.standoff:.3f}) -> touch z {touch_z:.3f}")

    rec = StreamRecorder()
    rec.start()
    t0 = time.time()
    qd = qd0
    try:
        traj = plan_pose(qrad(qd), [ox, oy, otop + a.standoff])
        if traj is None:
            sys.exit("hover plan failed")
        qd = run_leg("hover", traj, a.v_approach, a.exec)

        traj = plan_pose(qrad(qd), [ox, oy, touch_z])
        if traj is None:
            sys.exit("touch plan failed")
        qd = run_leg("touch", traj, a.v_touch, a.exec)
        if a.exec:
            print(f"[touch] dwell {a.dwell}s")
            time.sleep(a.dwell)

        traj = plan_pose(qrad(qd), [ox, oy, otop + a.standoff])
        if traj is not None:
            qd = run_leg("retreat", traj, a.v_touch, a.exec)
        traj = plan_joint(qrad(qd), [float(v) for v in C.BASE_Q])
        if traj is not None:
            qd = run_leg("base", traj, a.v_approach, a.exec)
    finally:
        time.sleep(0.5)
        rec.stop()
        close_sender()
        ts, qs, tqs, _ = rec.arrays()
        cmds = [(t, m) for t, m in CMD_LOG if "trajectory" in m]
        np.savez(os.path.join(a.out, "touch_online_log.npz"),
                 ts=ts, qs=qs, tqs=tqs, t0=t0,
                 cmd_json=np.array([json.dumps({"t": t, **m}) for t, m in cmds]))
        print(f"[log] {len(ts)} samples, {len(cmds)} chunks -> {a.out}")
        if len(ts) and cmds and a.exec:
            plot(ts, qs, tqs, cmds, t0, a.out)
    print("[done]")


def plot(ts, qs, tqs, cmds, t0, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = np.asarray(ts) - t0
    qs = np.asarray(qs)
    # command series from chunks: (t_anchor + k*dt, traj[k])
    ct, cq = [], []
    for _, m in cmds:
        for k, wp in enumerate(m["trajectory"]):
            ct.append(m["t_anchor"] + k * m["traj_dt"] - t0)
            cq.append(wp)
    ct = np.asarray(ct)
    cq = np.asarray(cq)
    o = np.argsort(ct)
    ct, cq = ct[o], cq[o]

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    for j in range(6):
        ax = axes[j // 2, j % 2]
        ax.plot(ts, qs[:, j], "-", lw=1.2, color="#4878CF", label="response")
        ax.plot(ct, cq[:, j], "--", lw=1.1, color="#D65F5F", label="command")
        ax.set_title(f"J{j+1}", fontsize=10)
        ax.set_ylabel("deg")
        if j == 0:
            ax.legend(fontsize=8)
    axes[2, 0].set_xlabel("t (s)")
    axes[2, 1].set_xlabel("t (s)")
    fig.suptitle("touch (online chunk stack): command vs response — joint angles")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "touch_angles.png"), dpi=110)

    fig2, axes2 = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    for j in range(6):
        ax = axes2[j // 2, j % 2]
        if len(ts) > 3:
            vr = np.gradient(qs[:, j], ts)
            ax.plot(ts, vr, "-", lw=1.0, color="#4878CF", label="response vel")
        if len(ct) > 3:
            vc = np.gradient(cq[:, j], ct)
            ax.plot(ct, vc, "--", lw=1.0, color="#D65F5F", label="command vel")
        ax.set_title(f"J{j+1}", fontsize=10)
        ax.set_ylabel("deg/s")
        if j == 0:
            ax.legend(fontsize=8)
    fig2.suptitle("touch (online chunk stack): command vs response — velocities")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out, "touch_vel.png"), dpi=110)
    print(f"[plot] {out}/touch_angles.png, touch_vel.png")


if __name__ == "__main__":
    main()
