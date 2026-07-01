#!/usr/bin/env python3
"""Recalibrate the cup-tip Z offset against the TABLE (defined as z=0).

Descends slowly onto a clear table spot, detects FIRST contact from the J2 (shoulder)
torque jump -- validated as the cleanest contact signal -- and FKs the ACTUAL joints at
first-touch. That FK z is where the model thinks the tip is while it is physically on the
table; the gap to 0 is the tcp-Z error. Prints the suggested new tcp offset.

  python3 calib_z.py 0.45 0.0
"""
import argparse, collections, json, os, socket, sys, threading, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C
from geometry import R_from_two_axes, R_to_quat_wxyz
from joint_conventions import linuxcnc_deg_to_rad, rad_to_linuxcnc_deg

PI = "10.0.0.27"
DOWN = list(R_to_quat_wxyz(R_from_two_axes(np.array([0, 0, -1.0]))))
TCP_NOW = 0.145   # current modeled cup-tip offset below the flange (URDF)


def rpc(d):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=40)
    s.sendall((json.dumps(d) + "\n").encode()); b = b""
    while not b.endswith(b"\n"):
        b += s.recv(65536)
    s.close(); return json.loads(b)


def fk_z(q_deg):
    return float(rpc({"type": "fk", "q": [float(v) for v in linuxcnc_deg_to_rad(q_deg)]})["pos"][0][2])


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


class Mon(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock(); self.stop = False
        self.torque = None; self.joints = None; self.jlog = []

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
                with self.lock:
                    self.torque = list(tq); self.joints = list(jt)
                    self.jlog.append((time.time(), list(jt), list(tq)))
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xy", nargs=2, type=float)
    ap.add_argument("--floor", type=float, default=-0.04, help="deep enough to truly contact the table")
    ap.add_argument("--v-des", type=float, default=1.2)
    ap.add_argument("--tq-abs", type=float, default=0.08, dest="tq_abs",
                    help="absolute J2 |tau-base| that flags contact (pose-drift maxes ~0.05)")
    a = ap.parse_args()
    x, y = a.xy

    mon = Mon(); mon.start(); time.sleep(0.3)
    sock = socket.create_connection((PI, 9994), timeout=3); sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    print("baselining torque (1.2 s)...")
    t0 = time.time(); bs = []
    while time.time() - t0 < 1.2:
        with mon.lock:
            if mon.torque is not None:
                bs.append(mon.torque)
        time.sleep(0.02)
    base = np.median(np.array(bs), axis=0)
    print(f"baseline J2 = {base[1]:.4f}")

    cur = read_cur()
    rd = rpc({"type": "plan_pose", "start_q": list(linuxcnc_deg_to_rad(cur)),
              "goal_pose": [x, y, a.floor] + DOWN, "max_attempts": 16})
    if not rd.get("trajectory"):
        print("plan failed:", rd.get("error")); return
    fine = to_fine(np.array(rd["trajectory"]), float(rd["dt"]), a.v_des)
    print(f"descend to z={a.floor}, {len(fine)} pts @ {a.v_des} deg/s; J2-abs thresh={a.tq_abs}")

    t_anchor = time.time() + 0.2
    contact = None; moved = False; cnt = 0
    while True:
        k = int((time.time() + 0.05 - t_anchor) / 0.01)
        if k >= len(fine):
            break
        if k >= 0:
            sock.sendall((json.dumps({"trajectory": [r.tolist() for r in fine[k:k + 25]],
                                      "traj_dt": 0.01, "t_anchor": float(t_anchor + k * 0.01)}) + "\n").encode())
        with mon.lock:
            tq = mon.torque; jl_len = len(mon.jlog)
        if tq is not None:
            j2 = abs(tq[1] - base[1])
            frac = k / len(fine)
            if abs(tq[2] - base[2]) > 0.02 or j2 > 0.045:     # J2 or J3 moving = we've begun descending
                moved = True
            if contact is None and moved and frac > 0.3 and j2 > a.tq_abs:
                cnt += 1
            else:
                cnt = 0
            if contact is None and cnt >= 2:
                contact = (time.time(), frac, jl_len)
                sock.sendall((json.dumps({"hold": True}) + "\n").encode())
                print(f"  >>> FIRST CONTACT (J2 jump) at frac={frac:.2f}, J2 dev={j2:.4f}")
                break
        time.sleep(0.05)
    if contact is None:
        sock.sendall((json.dumps({"hold": True}) + "\n").encode())
        print("  no J2 contact detected -- floor may still be above the table; deepen --floor")
    time.sleep(1.2)

    # analyse: find the onset of the J2 rise and FK the ACTUAL joints there (first-touch = model tip on table)
    with mon.lock:
        JL = list(mon.jlog)
    if contact is not None and len(JL) > 20:
        j2series = np.array([abs(r[2][1] - base[1]) for r in JL])
        ci = contact[2]                                   # jlog index at trigger
        # walk back to where J2 first departs from the pre-descent noise (onset)
        onset = ci
        for i in range(min(ci, len(j2series) - 1), 0, -1):
            if j2series[i] < base[1] * 0 + 0.03:          # back to near-baseline deviation
                onset = i; break
        q_trig = JL[min(ci, len(JL) - 1)][1]
        q_onset = JL[onset][1]
        z_trig = fk_z(q_trig)
        z_onset = fk_z(q_onset)
        print("\n=== Z CALIBRATION RESULT ===")
        print(f"FK tip z at first-touch onset  : {z_onset*1000:+.1f} mm  (model thinks tip here while on table)")
        print(f"FK tip z at trigger (compressed): {z_trig*1000:+.1f} mm")
        print(f"table is the z=0 reference -> tip-Z error ~ {z_onset*1000:+.1f} mm")
        print(f"suggested new tcp offset = {TCP_NOW:.3f} + ({z_onset:+.3f}) = {TCP_NOW + z_onset:.4f} m "
              f"(was {TCP_NOW:.3f})")

    # retract
    cur = read_cur()
    ru = rpc({"type": "plan_pose", "start_q": list(linuxcnc_deg_to_rad(cur)),
              "goal_pose": [x, y, 0.12] + DOWN, "max_attempts": 14})
    if ru.get("trajectory"):
        f = to_fine(np.array(ru["trajectory"]), float(ru["dt"]), 8.0); ta = time.time() + 0.2
        while True:
            k = int((time.time() + 0.05 - ta) / 0.01)
            if k >= len(f):
                break
            if k >= 0:
                sock.sendall((json.dumps({"trajectory": [r.tolist() for r in f[k:k + 25]],
                                          "traj_dt": 0.01, "t_anchor": float(ta + k * 0.01)}) + "\n").encode())
            time.sleep(0.05)
    time.sleep(1.5); sock.close(); mon.stop = True


if __name__ == "__main__":
    main()
