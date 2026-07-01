#!/usr/bin/env python3
"""Descend slowly onto the object and, each frame, compare the D405 DEPTH of the cup DOME
region vs a 12-px ring AROUND the dome (the surrounding). Saves colorized depth overlays
(dome outlined cyan, ring outlined yellow) + a plot of dome-depth vs ring-depth (and their
difference) over the descent, so we can see whether this is a usable contact signal.

  python3 depth_compare.py 0.40 -0.05 0.055
"""
import argparse
import json
import os
import socket
import sys
import time

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C
from geometry import R_from_two_axes, R_to_quat_wxyz
from joint_conventions import linuxcnc_deg_to_rad, rad_to_linuxcnc_deg
from servo_touch import GapMonitor

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


def stream_plan(sock, fine, on_tick=None):
    t0 = time.time() + 0.2
    while True:
        k = int((time.time() + 0.05 - t0) / 0.01)
        if k >= len(fine):
            return
        if k >= 0:
            sock.sendall((json.dumps({"trajectory": [r.tolist() for r in fine[k:k + 25]],
                                      "traj_dt": 0.01, "t_anchor": float(t0 + k * 0.01)}) + "\n").encode())
        if on_tick:
            on_tick(k / len(fine))
        time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz", nargs=3, type=float)
    ap.add_argument("--into", type=float, default=0.015)
    ap.add_argument("--grow", type=int, default=6, help="grow the dome mask by this many px to cover the FULL cup")
    ap.add_argument("--ring-in", type=int, default=5, dest="ring_in", help="inner edge of the boundary ring (px OUTSIDE the cup)")
    ap.add_argument("--ring", type=int, default=12, help="outer edge of the boundary ring (px OUTSIDE the cup)")
    ap.add_argument("--v-des", type=float, default=3.0)
    ap.add_argument("--serial", default="218622271300")
    ap.add_argument("--cup", default=os.path.join(C.OUT_DIR, "cup_mask.npz"))
    ap.add_argument("--out", default=os.path.join(C.OUT_DIR, "depth_compare"))
    a = ap.parse_args()
    x, y, top = a.xyz; floor = top - a.into
    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.out):
        os.remove(os.path.join(a.out, f))

    cup = np.load(a.cup); rim = cup["rim"].astype(bool); ann = cup["ring"].astype(bool)
    dome0 = cup["dome"].astype(bool) if "dome" in cup.files else cup["mask"].astype(bool)
    Kel = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1,) * 2)
    dome = (cv2.dilate(dome0.astype(np.uint8), Kel(a.grow)) > 0)          # grow the dome to the FULL cup (~100mm)
    ring = ((cv2.dilate(dome.astype(np.uint8), Kel(a.ring)) > 0)
            & ~(cv2.dilate(dome.astype(np.uint8), Kel(a.ring_in)) > 0))   # ring OUTSIDE the cup (ring_in..ring px) = table/object
    mon = GapMonitor(a.serial, rim.shape[1], rim.shape[0], rim, ann, dome); mon.start()
    t0 = time.time()
    while not mon.ok and time.time() - t0 < 8:
        time.sleep(0.2)

    sock = socket.create_connection((PI, 9994), timeout=3); sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    rp = rpc({"type": "plan_pose", "start_q": list(linuxcnc_deg_to_rad(read_cur())),
              "goal_pose": [x, y, top + 0.05] + DOWN, "max_attempts": 14})
    print("pregrasp"); stream_plan(sock, to_fine(np.array(rp["trajectory"]), float(rp["dt"]), 8.0)); time.sleep(2.5)

    rd = rpc({"type": "plan_pose", "start_q": list(linuxcnc_deg_to_rad(read_cur())),
              "goal_pose": [x, y, floor] + DOWN, "max_attempts": 14})
    fine = to_fine(np.array(rd["trajectory"]), float(rd["dt"]), a.v_des)
    print(f"descend {top+0.05:.3f}->{floor:.3f}, {len(fine)} pts; comparing dome vs {a.ring}px ring")
    rec = []; saved = [0]

    def dmed(depth, m):
        d = depth[m]; d = d[(d > 0.02) & (d < 0.5)]
        return float(np.median(d)) if len(d) >= 8 else np.nan

    def on_tick(frac):
        r = mon.latest_rgbd()
        if r is None:
            return
        rgb, depth, _ = r
        dd, rr = dmed(depth, dome), dmed(depth, ring)
        z = top + 0.05 + frac * (floor - (top + 0.05))
        rec.append((time.time(), z, dd, rr))
        if saved[0] % 6 == 0:                                   # save an overlay every ~0.3s
            vis = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255.0 / 0.5), cv2.COLORMAP_JET)
            vis[~np.isfinite(depth) | (depth <= 0)] = (0, 0, 0)
            cyan = cv2.dilate(dome.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~dome
            yel = cv2.dilate(ring.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~ring
            vis[cyan] = (255, 255, 0); vis[yel] = (0, 255, 255)
            cv2.putText(vis, f"z~{z:.3f} dome={dd*1000:.0f}mm ring={rr*1000:.0f}mm d={(rr-dd)*1000:.0f}",
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imwrite(os.path.join(a.out, f"d{saved[0]//6:03d}.jpg"), vis)
        saved[0] += 1

    stream_plan(sock, fine, on_tick)
    sock.sendall((json.dumps({"hold": True}) + "\n").encode()); time.sleep(0.6)
    rb = rpc({"type": "plan_joint", "start_q": list(linuxcnc_deg_to_rad(read_cur())), "goal_q": list(map(float, C.BASE_Q))})
    stream_plan(sock, to_fine(np.array(rb["trajectory"]), float(rb["dt"]), 8.0)); time.sleep(2); sock.close()
    mon.stop_evt.set()

    rec = np.array([(t, z, dd, rr) for (t, z, dd, rr) in rec], float)
    np.save(os.path.join(a.out, "rec.npy"), rec)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t = rec[:, 0] - rec[0, 0]
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(t, rec[:, 1], "k-"); ax[0].axhline(top, color="orange", ls=":", label=f"object top {top}")
    ax[0].set_ylabel("tcp z (m)"); ax[0].legend(fontsize=8); ax[0].set_title("descent: dome vs ring depth")
    ax[1].plot(t, rec[:, 2] * 1000, "c-", label="dome depth (mm)")
    ax[1].plot(t, rec[:, 3] * 1000, "y-", label="ring depth (mm)")
    ax[1].plot(t, (rec[:, 3] - rec[:, 2]) * 1000, "r-", label="ring - dome (mm)")
    ax[1].set_ylabel("depth (mm)"); ax[1].set_xlabel("time (s)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.savefig(os.path.join(a.out, "plot.png"), dpi=110, bbox_inches="tight")
    print(f"saved {a.out}/plot.png + {saved[0]//6+1} depth overlays")


if __name__ == "__main__":
    main()
