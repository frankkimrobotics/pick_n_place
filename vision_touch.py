#!/usr/bin/env python3
"""Vision contact-stop TOUCH at a given base-frame xy (no object detection).

Streams a slow vertical descent through the redesigned online_servo (:9994) toward a
target above the table, while watching the D405 cup view (GapMonitor): the blue plunger
dot RISES when the cup compresses on contact (bluedy >= --blue-thresh), and the rim-gap
collapses (gap <= --gap-contact). On the FIRST contact signal it sends a HOLD to stop the
descent, dwells, then returns to base. The redesigned 4 ms controller's lower lag (~233 ms
vs the old ~0.8 s) is what makes this vision-stop viable instead of decel-to-rest.

    python3 vision_touch.py 0.40 -0.05 0.07      # x y object_TOP_z

Needs cup_mask.npz (+ blue_dot_mask.npz for the blue dot), the D405 free, planner :9997 up,
and the redesigned online_servo running on the Pi.
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
from servo_touch import GapMonitor                      # reuse the blue-dot + gap monitor

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


def fk(q_deg):
    return rpc({"type": "fk", "q": [float(x) for x in linuxcnc_deg_to_rad(q_deg)]})["pos"][0]


def to_fine(traj_rad, dt, v_des):
    deg = np.array([rad_to_linuxcnc_deg(w) for w in traj_rad])
    peak = float(np.abs(np.diff(deg, axis=0)).max()) / dt if len(deg) > 1 else v_des
    sdt = dt * max(1.0, peak / v_des)
    tt = np.arange(len(deg)) * sdt
    return np.column_stack([np.interp(np.arange(0, tt[-1] + 1e-9, 0.01), tt, deg[:, j]) for j in range(6)])


def stream(fine, mon=None, a=None):
    """Stream fine to :9994; if mon given, watch for contact and HOLD. Returns reason."""
    sock = socket.create_connection((PI, 9994), timeout=3)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    t0 = time.time() + 0.2
    base_dome = None
    nblue = 0
    try:
        while True:
            now = time.time(); k = int((now + 0.05 - t0) / 0.01)
            if mon is not None:
                bluey, bluedy = mon.latest_bluey()
                gap, _ = mon.latest_gap()
                _, domey = mon.latest_cupd()
                if base_dome is None and domey is not None and 0 <= k < 6:
                    base_dome = domey
                nblue = nblue + 1 if (bluedy is not None and bluedy >= a.blue_thresh) else 0
                hit = []
                if nblue >= a.blue_debounce:                             # PRIMARY: blue-dot compression, debounced
                    hit.append(f"blue+{bluedy:.0f}px x{nblue}")
                if base_dome is not None and domey is not None and (base_dome - domey) >= a.dome_shift:
                    hit.append(f"dome+{base_dome - domey:.0f}px")        # SECONDARY: dome rise (compression)
                # NOTE: gap is geometry-dependent/unreliable in this setup -> logged only, never a stop
                if k % 10 == 0:
                    bd = "--" if bluedy is None else f"{bluedy:+.0f}"
                    gp = "--" if gap is None else f"{gap*1000:.0f}"
                    print(f"    k={k:4d} blue_dy={bd}px gap={gp}mm")
                if hit and k > 8:
                    sock.sendall((json.dumps({"hold": True}) + "\n").encode())
                    return "contact:" + ",".join(hit)
            if k >= len(fine):
                if mon is not None:
                    sock.sendall((json.dumps({"hold": True}) + "\n").encode())
                return "floor" if mon is not None else "done"
            if k >= 0:
                sock.sendall((json.dumps({"trajectory": [r.tolist() for r in fine[k:k + 25]],
                                          "traj_dt": 0.01, "t_anchor": float(t0 + k * 0.01)}) + "\n").encode())
            time.sleep(0.05)
    finally:
        sock.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz", nargs=3, type=float, help="x y object_TOP_z (m, base frame)")
    ap.add_argument("--floor-below", type=float, default=0.02, help="descend to top-this if no contact (m)")
    ap.add_argument("--pregrasp-clear", type=float, default=0.05, help="pregrasp height above top before the descent (m)")
    ap.add_argument("--v-des", type=float, default=3.0, help="descent speed (deg/s) -- slow for gentle contact")
    ap.add_argument("--blue-thresh", type=float, default=5.0, help="blue-dot rise (px) = contact (above the ~2px noise)")
    ap.add_argument("--blue-debounce", type=int, default=3, help="consecutive frames over blue-thresh required")
    ap.add_argument("--blue-gate", type=float, default=0.040, help="only trust the blue dot when gap <= this (m)")
    ap.add_argument("--gap-contact", type=float, default=0.024, help="rim gap (m) at/below which = contact (PRIMARY)")
    ap.add_argument("--dome-shift", type=float, default=6.0, help="dome image rise (px) = contact")
    ap.add_argument("--dwell", type=float, default=0.8, help="hold at contact before returning (s)")
    ap.add_argument("--serial", default="218622271300")
    ap.add_argument("--cup", default=os.path.join(C.OUT_DIR, "cup_mask.npz"))
    ap.add_argument("--blue-dot", default=os.path.join(C.OUT_DIR, "blue_dot_mask.npz"))
    a = ap.parse_args()
    x, y, top = a.xyz
    floor = top - a.floor_below

    cup = np.load(a.cup); rim = cup["rim"].astype(bool); ann = cup["ring"].astype(bool)
    dome = cup["dome"].astype(bool) if "dome" in cup.files else cup["mask"].astype(bool)
    CH, CW = rim.shape
    mon = GapMonitor(a.serial, CW, CH, rim, ann, dome); mon.start()
    t0 = time.time()
    while not mon.ok and time.time() - t0 < 8:
        time.sleep(0.2)
    if not mon.ok:
        print("D405 monitor failed to start"); return
    time.sleep(0.5)

    # 1) pregrasp ABOVE the object (open-loop), so the monitored descent is straight down
    preg = top + a.pregrasp_clear
    cur0 = list(linuxcnc_deg_to_rad(read_cur()))
    rp = rpc({"type": "plan_pose", "start_q": cur0, "goal_pose": [x, y, preg] + DOWN, "max_attempts": 14})
    if not rp.get("success"):
        print("pregrasp plan FAILED"); mon.stop_evt.set(); return
    print(f"pregrasp -> [{x:.2f},{y:.2f},{preg:.3f}]")
    stream(to_fine(np.array(rp["trajectory"]), float(rp["dt"]), 8.0))
    time.sleep(3.0)                                       # arm settle + D405 auto-exposure/depth stabilize

    # grab the blue-dot template HERE (object scene + settled exposure -> matching holds during the descent)
    bl, gp0 = mon.latest_bluey(), mon.latest_gap()[0]
    if os.path.exists(a.blue_dot):
        bd = np.load(a.blue_dot); cyx = bd["centroid"]; r = mon.latest_rgbd()
        if r is not None:
            rgb = r[0]; cyc, cx = int(cyx[0]), int(cyx[1])
            templ = cv2.cvtColor(rgb[cyc - 18:cyc + 18, cx - 18:cx + 18], cv2.COLOR_RGB2GRAY)
            mon.set_bluedot_template(templ, [cx - 30, cyc - 50, cx + 30, cyc + 20], float(cyc))
            print(f"blue-dot template {templ.shape} set; base_y={cyc}; contact rise >= {a.blue_thresh:.0f}px")
    else:
        print("WARN: no blue_dot_mask.npz -> using gap/dome contact only")
    print(f"  pregrasp baseline: gap={None if gp0 is None else round(gp0*1000)}mm (expect a clean monotonic drop on descent)")

    # 2) monitored vertical descent
    cur = list(linuxcnc_deg_to_rad(read_cur()))
    p0 = fk(rad_to_linuxcnc_deg(np.array(cur)))
    r = rpc({"type": "plan_pose", "start_q": cur, "goal_pose": [x, y, floor] + DOWN, "max_attempts": 14})
    if not r.get("success"):
        print("descent plan FAILED"); mon.stop_evt.set(); return
    fine = to_fine(np.array(r["trajectory"]), float(r["dt"]), a.v_des)
    print(f"descend [{x:.2f},{y:.2f}] z {p0[2]:.3f}->{floor:.3f} (top {top:.2f}), "
          f"{len(fine)} pts ~{len(fine)*0.01:.1f}s @ {a.v_des} deg/s. Watch the arm.")

    reason = stream(fine, mon, a)
    qc = read_cur(); pc = fk(qc)
    print(f"  STOP [{reason}] at tcp z={pc[2]:.3f} (object top ~{top:.2f}, gap to top {(pc[2]-top)*1000:+.0f}mm)")
    time.sleep(a.dwell)

    cur2 = list(linuxcnc_deg_to_rad(read_cur()))
    rb = rpc({"type": "plan_joint", "start_q": cur2, "goal_q": list(map(float, C.BASE_Q))})
    if rb.get("success"):
        stream(to_fine(np.array(rb["trajectory"]), float(rb["dt"]), 8.0))
        time.sleep(2.0)
        print("  returned to base")
    mon.stop_evt.set(); mon.join(timeout=2)


if __name__ == "__main__":
    main()
