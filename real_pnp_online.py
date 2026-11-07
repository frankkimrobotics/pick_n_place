#!/usr/bin/env python3
"""real_pnp_online :: suction pick-and-place over the ONLINE chunk stack.

Object pose given explicitly (from d435_detect_offline). Sequence:
hover -> press (touch-calib depth) -> suction ON -> lift -> high transit
(clears table objects; carried volume hangs below the cup) -> place descend
-> suction OFF -> retreat -> base. Chunks to :9994, suction via
`halcmd setp pro600.digital_out00` over ssh, everything recorded from
:9999 + CMD_LOG; suction events timestamped. Plots cmd vs response.

    python3 real_pnp_online.py --obj 0.355,0.077,0.042 --objh 0.07 \
        --place 0.42,-0.15 --exec
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C                                          # noqa: E402
from latency_common import (CMD_LOG, StreamRecorder, close_sender,  # noqa: E402
                            plan_joint, plan_pose, qrad, read_state,
                            stream_at)
from joint_conventions import rad_to_linuxcnc_deg           # noqa: E402

TOUCH_CALIB = 0.019        # FK tip this far below surface = just-touch
PRESS_EXTRA = 0.004        # press a little past just-touch for the seal
ABS_FLOOR = -0.03
ELBOW = (70.0, 145.0)
TRANSIT_Z = 0.22           # tcp altitude in transit (carried box clears 11cm objects)
PI = "10.0.0.27"

suction_events = []


def suction(on):
    subprocess.run(["ssh", f"pi@{PI}",
                    "halcmd unlinkp pro600.digital_out00 2>/dev/null; "
                    f"halcmd setp pro600.digital_out00 {1 if on else 0}"],
                   check=True, timeout=15)
    suction_events.append((time.time(), int(on)))
    print(f"[suction] {'ON' if on else 'OFF'}")


def check_box(traj_rad):
    t = np.degrees(np.asarray(traj_rad, float))
    if np.any(np.abs(t[:, 1]) > ELBOW[0] + 1e-4) or \
       np.any(np.abs(t[:, 2]) > ELBOW[1] + 1e-4):
        raise RuntimeError("elbow box violated")


def leg(label, qd, xyz, v, execute):
    traj = plan_pose(qrad(qd), list(xyz))
    if traj is None:
        raise RuntimeError(f"{label}: plan failed for {xyz}")
    check_box(traj)
    print(f"[{label}] {len(traj)} wps @ {v} deg/s -> {np.round(xyz,3)}"
          + ("" if execute else " (dry)"))
    if execute:
        stream_at(traj, v)
    return list(map(float, rad_to_linuxcnc_deg(np.asarray(traj[-1], float))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec", action="store_true")
    ap.add_argument("--obj", required=True, help="x,y,top_z")
    ap.add_argument("--objh", type=float, default=0.07, help="object height (m)")
    ap.add_argument("--press_extra", type=float, default=PRESS_EXTRA)
    ap.add_argument("--place", required=True, help="x,y place target")
    ap.add_argument("--v_transit", type=float, default=8.0)
    ap.add_argument("--v_stroke", type=float, default=3.0)
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/real_pnp_online"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ox, oy, otop = map(float, a.obj.split(","))
    px, py = map(float, a.place.split(","))
    press_z = max(otop - TOUCH_CALIB - a.press_extra, ABS_FLOOR)
    # place: same table plane -> object top ends at (otop - obj bottom offset)
    place_top = otop            # same table height assumption
    place_z = max(place_top - TOUCH_CALIB, ABS_FLOOR)

    for host, port, name in [("127.0.0.1", 9997, "curobo"),
                             (PI, 9999, "feedback"), (PI, 9994, "chunks")]:
        try:
            socket.create_connection((host, port), timeout=3).close()
        except Exception:
            sys.exit(f"SERVICE DOWN: {name}")

    qd, _ = read_state()
    print(f"[state] {np.round(qd,2)}")
    rec = StreamRecorder()
    rec.start()
    t0 = time.time()
    ok = False
    try:
        qd = leg("hover", qd, [ox, oy, otop + 0.06], a.v_transit, a.exec)
        qd = leg("press", qd, [ox, oy, press_z], a.v_stroke, a.exec)
        if a.exec:
            suction(True)
            time.sleep(1.2)                      # pump spin-up + seal
        qd = leg("lift", qd, [ox, oy, TRANSIT_Z], a.v_stroke, a.exec)
        qd = leg("transit", qd, [px, py, TRANSIT_Z], a.v_transit, a.exec)
        qd = leg("place", qd, [px, py, place_z], a.v_stroke, a.exec)
        if a.exec:
            suction(False)
            time.sleep(1.0)                      # release
        qd = leg("retreat", qd, [px, py, place_top + 0.10], a.v_stroke, a.exec)
        traj = plan_joint(qrad(qd), [float(v) for v in C.BASE_Q])
        if traj is not None:
            check_box(traj)
            if a.exec:
                stream_at(traj, a.v_transit)
        ok = True
    finally:
        if a.exec:
            try:
                suction(False)
            except Exception:
                pass
        time.sleep(0.5)
        rec.stop()
        close_sender()
        ts, qs, tqs, _ = rec.arrays()
        cmds = [(t, m) for t, m in CMD_LOG if "trajectory" in m]
        np.savez(os.path.join(a.out, "pnp_online_log.npz"),
                 ts=ts, qs=qs, tqs=tqs, t0=t0,
                 suction=np.array(suction_events),
                 cmd_json=np.array([json.dumps({"t": t, **m}) for t, m in cmds]))
        print(f"[log] {len(ts)} samples, {len(cmds)} chunks, ok={ok}")
        if len(ts) and cmds and a.exec:
            plot(ts, qs, tqs, cmds, t0, a.out)


def plot(ts, qs, tqs, cmds, t0, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = np.asarray(ts) - t0
    qs = np.asarray(qs)
    tqs = np.asarray(tqs)
    ct, cq = [], []
    for _, m in cmds:
        for k, wp in enumerate(m["trajectory"]):
            ct.append(m["t_anchor"] + k * m["traj_dt"] - t0)
            cq.append(wp)
    ct = np.asarray(ct)
    cq = np.asarray(cq)
    o = np.argsort(ct)
    ct, cq = ct[o], cq[o]
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True)
    for j in range(6):
        ax = axes[j // 2, j % 2]
        ax.plot(ts, qs[:, j], "-", lw=1.2, color="#4878CF", label="response")
        ax.plot(ct, cq[:, j], "--", lw=1.0, color="#D65F5F", label="command")
        for st, on in suction_events:
            ax.axvline(st - t0, color="#2ca02c" if on else "#888",
                       lw=1.0, ls=":")
        ax.set_title(f"J{j+1}", fontsize=10)
        ax.set_ylabel("deg")
        if j == 0:
            ax.legend(fontsize=8)
    axv = axes[3, 0]
    for j in range(6):
        axv.plot(ts, np.gradient(qs[:, j], ts), lw=0.9, label=f"J{j+1}")
    axv.set_title("response velocities", fontsize=10)
    axv.set_ylabel("deg/s")
    axv.set_xlabel("t (s)")
    axv.legend(fontsize=7, ncol=3)
    axt = axes[3, 1]
    for j in range(6):
        axt.plot(ts, tqs[:, j], lw=0.9, label=f"J{j+1}")
    axt.set_title("joint torque (:9999)", fontsize=10)
    axt.set_xlabel("t (s)")
    axt.legend(fontsize=7, ncol=3)
    fig.suptitle("REAL pick-and-place (online chunk stack) — green/gray dots = suction on/off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pnp_online.png"), dpi=110)
    print(f"[plot] {out}/pnp_online.png")


if __name__ == "__main__":
    main()
