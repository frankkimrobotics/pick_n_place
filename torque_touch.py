#!/usr/bin/env python3
"""Slow straight-down descent that monitors JOINT TORQUE from the :9999 stream and STOPS
on the contact-induced torque rise. Logs the full torque/z trace + plots it, so we can see
whether torque is a usable contact signal vs vision.

  python3 torque_touch.py 0.45 0.0 --floor -0.006 --thresh 0.05

Descends from the CURRENT pose straight down to `floor` (cup pointing down). Contact (table
or object) makes trise = sum_j |tau_j - baseline_j| jump; when it crosses --thresh we send
{"hold"} to freeze the welded reference. The z-floor is a hard backstop if torque never fires.
"""
import argparse, json, os, socket, sys, threading, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C
from geometry import R_from_two_axes, R_to_quat_wxyz
from joint_conventions import linuxcnc_deg_to_rad, rad_to_linuxcnc_deg

PI = "10.0.0.27"
DOWN = list(R_to_quat_wxyz(R_from_two_axes(np.array([0, 0, -1.0]))))


def rpc(d):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=40)
    s.sendall((json.dumps(d) + "\n").encode()); b = b""
    while not b.endswith(b"\n"):
        b += s.recv(65536)
    s.close(); return json.loads(b)


def read_cur():
    s = socket.create_connection((PI, 9999), timeout=3); s.settimeout(2); b = b""
    try:
        while b"\n" not in b:
            b += s.recv(4096)
    except Exception:
        pass
    s.close(); return list(json.loads(b.split(b"\n")[0])["joints_deg"])


def to_fine(traj, dt, v):
    deg = np.array([rad_to_linuxcnc_deg(w) for w in traj])
    peak = float(np.abs(np.diff(deg, axis=0)).max()) / dt if len(deg) > 1 else v
    sdt = dt * max(1.0, peak / v); tt = np.arange(len(deg)) * sdt
    return np.column_stack([np.interp(np.arange(0, tt[-1] + 1e-9, 0.01), tt, deg[:, j]) for j in range(6)])


class TorqueMon(threading.Thread):
    """Reads the :9999 stream, keeps the latest torque + joints, logs every sample."""
    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock(); self.stop = False
        self.torque = None; self.joints = None; self.log = []; self.jlog = []

    def run(self):
        s = socket.create_connection((PI, 9999), timeout=3); s.settimeout(1.0); buf = b""
        while not self.stop:
            try:
                buf += s.recv(8192)
            except Exception:
                continue
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                tq = d.get("torque"); jt = d.get("joints_deg")
                if tq is None:
                    continue
                now = time.time()
                with self.lock:
                    self.torque = list(tq); self.joints = list(jt)
                    self.log.append((now, list(tq)))
                    if jt is not None:
                        self.jlog.append((now, list(jt)))
        s.close()

    def speed(self):
        """max per-joint actual speed (deg/s) over the last ~0.1 s -> ~0 when the cup stalls."""
        with self.lock:
            if len(self.jlog) < 11:
                return 99.0
            (t1, j1) = self.jlog[-1]; (t0, j0) = self.jlog[-11]   # ~0.2 s window
        dt = t1 - t0
        if dt <= 1e-3:
            return 99.0
        return max(abs(a - b) for a, b in zip(j1, j0)) / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xy", nargs=2, type=float)
    ap.add_argument("--floor", type=float, default=-0.006, help="hard z backstop (table at 0)")
    ap.add_argument("--thresh", type=float, default=0.05, help="(unused) absolute threshold")
    ap.add_argument("--joint", type=int, default=3, help="0-based joint to watch for contact (3 = J4)")
    ap.add_argument("--delta", type=float, default=0.015, help="(monitor only) J4 jump")
    ap.add_argument("--stall", type=float, default=0.5, help="(plot only) actual-speed stall ref line")
    ap.add_argument("--fe-margin", type=float, default=1.5, dest="fe_margin",
                    help="following-error (deg) above its trailing baseline that flags contact")
    ap.add_argument("--v-des", type=float, default=1.5)
    ap.add_argument("--out", default=os.path.join(C.OUT_DIR, "torque_touch"))
    a = ap.parse_args()
    x, y = a.xy
    os.makedirs(a.out, exist_ok=True)

    mon = TorqueMon(); mon.start(); time.sleep(0.3)

    sock = socket.create_connection((PI, 9994), timeout=3)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    # baseline torque over ~1.2 s at the current (pre-contact) pose
    print("baselining torque (1.2 s)...")
    t0 = time.time(); base_samp = []
    while time.time() - t0 < 1.2:
        with mon.lock:
            if mon.torque is not None:
                base_samp.append(mon.torque)
        time.sleep(0.02)
    base = np.median(np.array(base_samp), axis=0)
    print(f"baseline tau = {np.round(base, 4).tolist()}  (n={len(base_samp)})")

    cur = read_cur()
    rd = rpc({"type": "plan_pose", "start_q": list(linuxcnc_deg_to_rad(cur)),
              "goal_pose": [x, y, a.floor] + DOWN, "max_attempts": 16})
    if not rd.get("trajectory"):
        print("plan failed:", rd.get("error")); return
    fine = to_fine(np.array(rd["trajectory"]), float(rd["dt"]), a.v_des)
    print(f"descend straight down to z={a.floor}, {len(fine)} pts @ {a.v_des} deg/s; thresh={a.thresh}")

    # stream with per-tick torque check -- JUMP detector (sharp rise above trailing min),
    # so the gradual pose-dependent gravity drift during the descent doesn't false-trigger.
    import collections
    fe_recent = collections.deque(maxlen=20)   # ~1 s trailing window of following error @ 20 Hz
    t_anchor = time.time() + 0.2
    contact = None; rec = []; moved = False; stall_cnt = 0
    while True:
        k = int((time.time() + 0.05 - t_anchor) / 0.01)
        if k >= len(fine):
            break
        if k >= 0:
            sock.sendall((json.dumps({"trajectory": [r.tolist() for r in fine[k:k + 25]],
                                      "traj_dt": 0.01, "t_anchor": float(t_anchor + k * 0.01)}) + "\n").encode())
        kk = min(max(k, 0), len(fine) - 1)
        with mon.lock:
            tq = mon.torque; act = mon.joints
        spd = mon.speed()
        if spd > 0.6:
            moved = True
        if tq is not None and act is not None and k >= 0:
            dtq = np.abs(np.array(tq) - base)
            sig = float(dtq[a.joint])           # J4 torque monitor (kept for the plot)
            frac = k / len(fine)
            fe = float(np.max(np.abs(fine[kk] - np.array(act))))   # following error (deg) = commanded - actual
            fe_recent.append(fe)
            fe_base = float(np.median(fe_recent)) if len(fe_recent) >= 10 else fe
            fe_exc = fe - fe_base
            rec.append((time.time(), frac, sig, spd, fe, fe_base))
            # PRIMARY contact = following-error GROWS: the cup hits the can and can't descend,
            # so the actual joints fall behind the commanded descent. Robust to brief pauses
            # (a pause halts the command too, so fe stays flat) and force/gravity-independent.
            if contact is None and moved and frac > 0.15 and fe_exc > a.fe_margin:
                stall_cnt += 1
            else:
                stall_cnt = 0
            if contact is None and stall_cnt >= 3:        # ~0.15 s sustained
                contact = (time.time(), frac, sig, fe)
                sock.sendall((json.dumps({"hold": True}) + "\n").encode())
                print(f"  >>> CONTACT (following-error) at frac={frac:.2f}: fe={fe:.2f} deg (base {fe_base:.2f}); J4 sig={sig:.4f}")
                break
        time.sleep(0.05)
    if contact is None:
        sock.sendall((json.dumps({"hold": True}) + "\n").encode())
        print("  reached z-floor without a stall (cup may not have reached the object)")
    # passive settle: hold + keep logging torque/joints so the press shows in the plot
    time.sleep(2.0)

    # retract to a safe height, then base
    cur = read_cur()
    ru = rpc({"type": "plan_pose", "start_q": list(linuxcnc_deg_to_rad(cur)),
              "goal_pose": [x, y, 0.12] + DOWN, "max_attempts": 14})
    if ru.get("trajectory"):
        f = to_fine(np.array(ru["trajectory"]), float(ru["dt"]), 8.0)
        ta = time.time() + 0.2
        while True:
            k = int((time.time() + 0.05 - ta) / 0.01)
            if k >= len(f):
                break
            if k >= 0:
                sock.sendall((json.dumps({"trajectory": [r.tolist() for r in f[k:k + 25]],
                                          "traj_dt": 0.01, "t_anchor": float(ta + k * 0.01)}) + "\n").encode())
            time.sleep(0.05)
    time.sleep(1.5); sock.close(); mon.stop = True; time.sleep(0.2)

    # plot: per-joint |tau-base| and trise vs time, mark contact
    L = mon.log
    if not L:
        print("no torque log"); return
    t = np.array([r[0] for r in L]); t -= t[0]
    T = np.array([r[1] for r in L])           # (N,6) torque
    dT = np.abs(T - base)
    # actual joint speed from the joint log (the stall signal)
    JL = mon.jlog
    jt_t = np.array([r[0] for r in JL]) - L[0][0]
    J = np.array([r[1] for r in JL])          # (M,6) joints deg
    spd = np.zeros(len(J))
    for i in range(10, len(J)):
        dt = JL[i][0] - JL[i - 10][0]
        spd[i] = max(abs(J[i] - J[i - 10])) / dt if dt > 1e-3 else 0.0
    R = np.array(rec) if rec else np.zeros((0, 6))    # (t, frac, sig, spd, fe, fe_base)
    rt = (R[:, 0] - L[0][0]) if len(R) else np.array([])
    np.savez(os.path.join(a.out, "log.npz"), t=t, T=T, base=base, jt_t=jt_t, J=J, spd=spd, rec=R)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tc = (contact[0] - L[0][0]) if contact is not None else None
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for j in range(6):
        ax[0].plot(t, dT[:, j], lw=1.0, label=f"|d tau_J{j+1}|")
    ax[0].set_ylabel("|tau - base|  (all 6)"); ax[0].legend(fontsize=8, ncol=6); ax[0].grid(alpha=0.3)
    ax[0].set_title("torque-touch on can [2]: per-joint torque, actual speed, following-error (contact detector)")
    ax[1].plot(jt_t, spd, "b-", lw=1.2, label="max actual joint speed (deg/s)")
    ax[1].set_ylabel("actual speed (deg/s)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    ax[1].set_ylim(0, max(2.0, spd.max() * 1.1))
    if len(R):
        ax[2].plot(rt, R[:, 4], "k-", lw=1.3, label="following error fe (deg)")
        ax[2].plot(rt, R[:, 5], "c-", lw=1.0, label="fe trailing baseline")
        ax[2].plot(rt, R[:, 5] + a.fe_margin, "r--", lw=1.0, label=f"trigger = base + {a.fe_margin}")
    ax[2].set_ylabel("following error (deg)"); ax[2].set_xlabel("time (s)"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    for axi in ax:
        if tc is not None:
            axi.axvline(tc, color="g", ls=":", lw=1.5)
    fig.savefig(os.path.join(a.out, "torque.png"), dpi=110, bbox_inches="tight")
    pk = dT.max(axis=0)
    print("peak |d tau| per joint:", [round(float(v), 4) for v in pk])
    print(f"saved {a.out}/torque.png")


if __name__ == "__main__":
    main()
