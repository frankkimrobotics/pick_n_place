#!/usr/bin/env python3
"""Diagnostic descent: go slowly from a pregrasp down to ~`--into` BELOW the object top
(no contact stop -- compliant cup absorbs it) while SAVING annotated D405 cup-view frames,
so we can SEE why blue-dot/dome/gap contact detection is flaky. Annotates each frame with
the blue-dot search box + matched dot, the base_y line, the dome ROI + its centroid, and
the live signal values. Frames -> outputs/vision_debug/. Returns to base."""
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


def fkz(q_deg):
    return rpc({"type": "fk", "q": [float(x) for x in linuxcnc_deg_to_rad(q_deg)]})["pos"][0][2]


def to_fine(traj_rad, dt, v_des):
    deg = np.array([rad_to_linuxcnc_deg(w) for w in traj_rad])
    peak = float(np.abs(np.diff(deg, axis=0)).max()) / dt if len(deg) > 1 else v_des
    sdt = dt * max(1.0, peak / v_des)
    tt = np.arange(len(deg)) * sdt
    return np.column_stack([np.interp(np.arange(0, tt[-1] + 1e-9, 0.01), tt, deg[:, j]) for j in range(6)])


def send(sock, fine, k):
    sock.sendall((json.dumps({"trajectory": [r.tolist() for r in fine[k:k + 25]],
                              "traj_dt": 0.01, "t_anchor": time.time() + 0.05}) + "\n").encode())


def annotate(rgb, mon, z, k):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bluey, bluedy = mon.latest_bluey(); gap, _ = mon.latest_gap(); _, domey = mon.latest_cupd()
    bscore = float(mon._bluescore)
    if mon.sbox is not None:
        sx0, sy0, sx1, sy1 = mon.sbox
        cv2.rectangle(img, (sx0, sy0), (sx1, sy1), (255, 0, 0), 1)               # blue search box (BLUE)
        if bluey is not None:
            cv2.circle(img, (int((sx0 + sx1) / 2), int(bluey)), 5, (0, 0, 255), -1)  # matched dot (RED)
    if mon.base_y is not None:
        cv2.line(img, (0, int(mon.base_y)), (img.shape[1], int(mon.base_y)), (0, 255, 255), 1)  # base_y (YELLOW)
    y0, y1, x0, x1 = mon.dbox
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 1)                       # dome ROI (GREEN)
    if domey is not None:
        cv2.line(img, (x0, int(domey)), (x1, int(domey)), (0, 180, 0), 1)        # dome centroid y
    t1 = f"k{k} z{z:.3f}"
    t2 = f"bdy={'--' if bluedy is None else round(bluedy)} bsc={bscore:.2f}"
    t3 = f"gap={'--' if gap is None else round(gap*1000)}mm domey={'--' if domey is None else round(domey)}"
    for i, t in enumerate((t1, t2, t3)):
        cv2.putText(img, t, (5, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return img


def stream_plan(sock, fine):
    t0 = time.time() + 0.2
    while True:
        k = int((time.time() + 0.05 - t0) / 0.01)
        if k >= len(fine):
            return
        if k >= 0:
            sock.sendall((json.dumps({"trajectory": [r.tolist() for r in fine[k:k + 25]],
                                      "traj_dt": 0.01, "t_anchor": float(t0 + k * 0.01)}) + "\n").encode())
        time.sleep(0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz", nargs=3, type=float, help="x y object_TOP_z")
    ap.add_argument("--into", type=float, default=0.010, help="descend to top-this (m), no stop")
    ap.add_argument("--pregrasp-clear", type=float, default=0.05)
    ap.add_argument("--v-des", type=float, default=3.0)
    ap.add_argument("--every", type=int, default=25, help="save a frame every N ticks (~10ms each)")
    ap.add_argument("--serial", default="218622271300")
    ap.add_argument("--cup", default=os.path.join(C.OUT_DIR, "cup_mask.npz"))
    ap.add_argument("--blue-dot", default=os.path.join(C.OUT_DIR, "blue_dot_mask.npz"))
    ap.add_argument("--out", default=os.path.join(C.OUT_DIR, "vision_debug"))
    a = ap.parse_args()
    x, y, top = a.xyz
    floor = top - a.into
    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.out):
        os.remove(os.path.join(a.out, f))

    cup = np.load(a.cup); rim = cup["rim"].astype(bool); ann = cup["ring"].astype(bool)
    dome = cup["dome"].astype(bool) if "dome" in cup.files else cup["mask"].astype(bool)
    CH, CW = rim.shape
    mon = GapMonitor(a.serial, CW, CH, rim, ann, dome); mon.start()
    t0 = time.time()
    while not mon.ok and time.time() - t0 < 8:
        time.sleep(0.2)
    if not mon.ok:
        print("D405 failed"); return
    time.sleep(0.5)

    # pregrasp, settle, grab template
    preg = top + a.pregrasp_clear
    rp = rpc({"type": "plan_pose", "start_q": list(linuxcnc_deg_to_rad(read_cur())),
              "goal_pose": [x, y, preg] + DOWN, "max_attempts": 14})
    if not rp.get("success"):
        print("pregrasp FAILED"); mon.stop_evt.set(); return
    sock = socket.create_connection((PI, 9994), timeout=3); sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"pregrasp -> [{x},{y},{preg:.3f}]")
    stream_plan(sock, to_fine(np.array(rp["trajectory"]), float(rp["dt"]), 8.0))
    time.sleep(3.0)
    if os.path.exists(a.blue_dot):
        bd = np.load(a.blue_dot); cyx = bd["centroid"]; r = mon.latest_rgbd()
        if r is not None:
            rgb = r[0]; cyc, cx = int(cyx[0]), int(cyx[1])
            templ = cv2.cvtColor(rgb[cyc - 18:cyc + 18, cx - 18:cx + 18], cv2.COLOR_RGB2GRAY)
            mon.set_bluedot_template(templ, [cx - 30, cyc - 50, cx + 30, cyc + 20], float(cyc))
            print(f"blue-dot template set; base_y={cyc}")

    # diagnostic descent (NO stop) -- save annotated frames
    rd = rpc({"type": "plan_pose", "start_q": list(linuxcnc_deg_to_rad(read_cur())),
              "goal_pose": [x, y, floor] + DOWN, "max_attempts": 14})
    if not rd.get("success"):
        print("descent FAILED"); mon.stop_evt.set(); return
    fine = to_fine(np.array(rd["trajectory"]), float(rd["dt"]), a.v_des)
    print(f"descend {preg:.3f}->{floor:.3f} (into {a.into*1000:.0f}mm), {len(fine)} pts ~{len(fine)*0.01:.1f}s. Saving frames.")
    t0 = time.time() + 0.2; saved = 0; last_save = -999
    while True:
        k = int((time.time() + 0.05 - t0) / 0.01)
        if k >= len(fine):
            sock.sendall((json.dumps({"hold": True}) + "\n").encode()); break
        if k >= 0:
            send(sock, fine, k)
        if k - last_save >= a.every and k >= 0:
            r = mon.latest_rgbd()
            if r is not None:
                z = float(np.interp(k, [0, len(fine)], [preg, floor]))           # commanded z (approx)
                img = annotate(r[0], mon, z, k)
                cv2.imwrite(os.path.join(a.out, f"f{saved:03d}_k{k:04d}.jpg"), img)
                bdy = mon.latest_bluey()[1]; cupd, domey = mon.latest_cupd()
                print(f"  f{saved:03d} k={k:4d} z~{z:.3f} bdy={'--' if bdy is None else round(bdy):>3} "
                      f"bsc={mon._bluescore:.2f} cupd={'--' if cupd is None else round(cupd*1000):>4}mm "
                      f"domey={'--' if domey is None else round(domey)}")
                saved += 1; last_save = k
        time.sleep(0.05)
    time.sleep(0.6)
    print(f"saved {saved} frames -> {a.out}; holding at floor")

    # return to base
    rb = rpc({"type": "plan_joint", "start_q": list(linuxcnc_deg_to_rad(read_cur())),
              "goal_q": list(map(float, C.BASE_Q))})
    if rb.get("success"):
        stream_plan(sock, to_fine(np.array(rb["trajectory"]), float(rb["dt"]), 8.0)); time.sleep(2)
        print("returned to base")
    sock.close(); mon.stop_evt.set(); mon.join(timeout=2)


if __name__ == "__main__":
    main()
