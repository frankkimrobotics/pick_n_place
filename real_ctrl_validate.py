#!/usr/bin/env python3
"""real_ctrl_validate :: on-robot validation of the lag-aware MPC vs pid.

Stages (each gated on a 1-deg probe; arm homed between stages):
  1 kv-fit     : +5 deg J1 step, fit drive lag kv from velocity response;
                 warn if the deployed K (kv=40/s) is off by >30%.
  2 step bench : the sim bench cases {J1+20, J2-15, J2+10, J4+20} x
                 {pid, mpc}; metrics table vs sim predictions.
  3 multi      : all-joints simultaneous ~15 deg step, mpc vs pid
                 (coupling check the decoupled model ignores).

Every stream sample + command is logged; writes npz + a comparison plot.
Dry-run by default; --exec to move. SIGINT-safe (pid holds, hold-mode
robot_hal keeps servoing).

    python3 real_ctrl_validate.py --exec [--stages 1,2,3]
"""
import argparse
import json
import os
import socket
import threading
import time

import numpy as np

PI = "10.0.0.27"
HOME = np.array([0.0, -110.0, 80.0, -80.0, -90.0, 0.0])
ELBOW = (70.0, 145.0)          # URDF-frame |j2|,|j3| limits handled by caller poses
SIM_PRED = {                   # from the MuJoCo twin bench (2026-08-24)
    "pid": dict(overshoot="17-21%", settle="~3.0s"),
    "mpc": dict(overshoot="0.0%", settle="0.5-0.7s"),
}


class Log:
    def __init__(self):
        self.samples = []
        self.cmds = []
        self.on = True
        threading.Thread(target=self._run, daemon=True).start()

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
                            self.samples.append((time.time(),
                                                 *map(float, d["joints_deg"])))
                        except Exception:
                            pass
            except Exception:
                time.sleep(0.3)

    def stop(self):
        self.on = False


def read_q():
    s = socket.create_connection((PI, 9999), timeout=3)
    b = b""
    while b"\n" not in b:
        b += s.recv(4096)
    s.close()
    return np.array(json.loads(b.split(b"\n")[0])["joints_deg"])


def send(tgt, dur, controller, log=None, execute=True):
    if log is not None:
        log.cmds.append((time.time(), *map(float, tgt), dur, controller))
    if not execute:
        print(f"  [dry] {controller} -> {np.round(tgt,1)} over {dur}s")
        return
    k = socket.create_connection((PI, 9998), timeout=3)
    k.sendall((json.dumps({"target_deg": [float(v) for v in tgt],
                           "duration": float(dur),
                           "controller": controller}) + "\n").encode())
    try:
        k.settimeout(2)
        k.recv(256)
    except Exception:
        pass
    k.close()


def probe(execute):
    q0 = read_q()
    t = q0.copy()
    t[0] += 1.0
    send(t, 1.5, "pid", execute=execute)
    if not execute:
        return True
    time.sleep(2.0)
    ok = abs(read_q()[0] - q0[0]) > 0.4
    send(q0, 1.5, "pid", execute=execute)
    time.sleep(2.0)
    if not ok:
        print("!! PROBE FAILED — drives not responding. Aborting stage.")
    return ok


def go_home(execute, dur=8.0):
    send(HOME, dur, "pid", execute=execute)
    if execute:
        time.sleep(dur + 1.0)
        send(HOME, 3.0, "pid", execute=execute)
        time.sleep(4.0)


def metrics(ts, qs, joint, q0, tgt):
    step = tgt - q0
    m = np.abs(qs - q0) >= 0.9 * abs(step)
    rise = ts[np.argmax(m)] if m.any() else np.nan
    ovs = (np.max(np.abs(qs - q0)) - abs(step)) / abs(step) * 100
    settled = np.abs(qs - tgt) < 0.5
    st = np.nan
    for i in range(len(settled)):
        if settled[i:].all():
            st = ts[i]
            break
    return rise, ovs, st, abs(qs[-1] - tgt)


def slice_log(log, t0, t1, joint):
    S = np.array([r for r in log.samples if t0 <= r[0] <= t1])
    if not len(S):
        return np.zeros(0), np.zeros(0)
    return S[:, 0] - t0, S[:, 1 + joint]


def stage1_kvfit(log, execute):
    print("\n=== STAGE 1: kv identification (+5 deg J1, mpc) ===")
    if not probe(execute):
        return
    q0 = read_q() if execute else HOME.copy()
    tgt = q0.copy()
    tgt[0] += 5.0
    t0 = time.time()
    send(tgt, 3.0, "mpc", log, execute)
    if not execute:
        return
    time.sleep(3.5)
    ts, qs = slice_log(log, t0, t0 + 3.5, 0)
    if len(ts) < 20:
        print("  insufficient samples")
        return
    v = np.gradient(qs, ts)
    # command was clamp(-K0 e - K1 v) -> initially saturated at 40 deg/s;
    # fit v(t) = vmax (1 - exp(-kv t)) on the rising segment
    vmax = np.percentile(v, 95)
    ridx = (v > 0.15 * vmax) & (v < 0.9 * vmax) & (ts < ts[np.argmax(v)])
    if ridx.sum() >= 4:
        kv_fit = np.polyfit(ts[ridx], -np.log(1 - v[ridx] / max(vmax, 1e-6)), 1)[0]
        print(f"  fitted drive lag kv ~= {kv_fit:.1f}/s (deployed model: 40/s)")
        if not (25 <= kv_fit <= 60):
            print("  !! >30% off — recompute K via DARE before trusting stage 2")
    else:
        print("  rise too fast to fit at 100 Hz — kv >= ~60/s, model conservative")
    send(q0, 3.0, "pid", log, execute)
    time.sleep(3.5)


def stage2_bench(log, execute):
    print("\n=== STAGE 2: step bench pid vs mpc (sim-mirror) ===")
    cases = [(0, 20.0, "J1+20"), (1, -15.0, "J2-15"), (1, 10.0, "J2+10"),
             (3, 20.0, "J4+20")]
    print(f"  {'case':7s} {'ctrl':4s} {'rise':>6} {'ovs%':>7} {'settle':>7} "
          f"{'ss':>6}   sim: mpc ovs 0.0% settle 0.5-0.7s | pid ovs 17-21%")
    results = {}
    for joint, step, label in cases:
        for ctrl in ("pid", "mpc"):
            if not probe(execute):
                return results
            go_home(execute, 6.0)
            q0 = read_q() if execute else HOME.copy()
            tgt = q0.copy()
            tgt[joint] += step
            t0 = time.time()
            send(tgt, 6.0, ctrl, log, execute)
            if not execute:
                continue
            time.sleep(6.5)
            ts, qs = slice_log(log, t0, t0 + 6.5, joint)
            if len(ts) > 20:
                r, o, s, e = metrics(ts, qs, joint, q0[joint], tgt[joint])
                results[(label, ctrl)] = (ts, qs, (r, o, s, e))
                print(f"  {label:7s} {ctrl:4s} {r:6.2f} {o:7.1f} {s:7.2f} {e:6.2f}")
            go_home(execute, 6.0)
    return results


def stage3_multi(log, execute):
    print("\n=== STAGE 3: all-joints simultaneous step (coupling) ===")
    if not probe(execute):
        return
    go_home(execute, 6.0)
    tgt = HOME + np.array([15.0, 10.0, -12.0, 15.0, 10.0, 15.0])
    for ctrl in ("pid", "mpc"):
        q0 = read_q() if execute else HOME.copy()
        t0 = time.time()
        send(tgt, 6.0, ctrl, log, execute)
        if execute:
            time.sleep(6.5)
            worst = 0.0
            for j in range(6):
                ts, qs = slice_log(log, t0, t0 + 6.5, j)
                if len(ts) > 20:
                    _, o, _, _ = metrics(ts, qs, j, q0[j], tgt[j])
                    worst = max(worst, o)
            print(f"  {ctrl}: worst-joint overshoot {worst:.1f}%")
        go_home(execute, 6.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec", action="store_true")
    ap.add_argument("--stages", default="1,2,3")
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/ctrl_validate"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for host, port in ((PI, 9999), (PI, 9998)):
        try:
            socket.create_connection((host, port), timeout=3).close()
        except Exception:
            raise SystemExit(f"robot service {port} down — bring the stack up first")
    log = Log()
    time.sleep(1.0)
    results = None
    try:
        stages = a.stages.split(",")
        if "1" in stages:
            stage1_kvfit(log, a.exec)
        if "2" in stages:
            results = stage2_bench(log, a.exec)
        if "3" in stages:
            stage3_multi(log, a.exec)
        go_home(a.exec, 8.0)
    finally:
        log.stop()
        np.savez(os.path.join(a.out, "validate_log.npz"),
                 samples=np.array(log.samples) if log.samples else np.zeros((0, 7)))
        if results:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 2, figsize=(13, 8))
            for ax, label in zip(axes.flat, ["J1+20", "J2-15", "J2+10", "J4+20"]):
                for ctrl, col in (("pid", "#4878CF"), ("mpc", "#D65F5F")):
                    if (label, ctrl) in results:
                        ts, qs, met = results[(label, ctrl)]
                        ax.plot(ts, qs, color=col, lw=1.3,
                                label=f"{ctrl} (ovs {met[1]:.1f}%)")
                ax.set_title(label, fontsize=10)
                ax.legend(fontsize=8)
                ax.set_xlabel("t (s)")
                ax.set_ylabel("deg")
            fig.suptitle("REAL robot: pid vs lag-aware mpc step responses")
            fig.tight_layout()
            fig.savefig(os.path.join(a.out, "real_ctrl_bench.png"), dpi=110)
            print(f"\nplot -> {a.out}/real_ctrl_bench.png")
    print("[done]")


if __name__ == "__main__":
    main()
