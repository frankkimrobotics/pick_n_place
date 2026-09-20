#!/usr/bin/env python3
"""d435_track :: live tabletop object tracker from the fixed D435 (depth only), published over UDP.

Runs in a python that has pyrealsense2 (system python3 with PYTHONPATH=~/librealsense/build/release);
the RL controller (mjwarp env) listens with --track. Each frame: depth -> points in the base frame via
outputs/extrinsics_d435.json -> points above the table plane inside the work area -> grid connected
components -> the largest low (top < 0.12 m) compact cluster = the object; its top plateau centre and
height are sent as JSON {"t", "cx", "cy", "top", "n"} to 127.0.0.1:9701 (one datagram per 4-frame median, ~7 Hz).

  PYTHONPATH=~/librealsense/build/release python3 rl/d435_track.py            # prints detections
"""
import json
import os
import socket
import sys
import time

import numpy as np
import pyrealsense2 as rs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PORT = 9701
SERIAL = "043422070101"
XR, YR = (0.20, 0.60), (-0.32, 0.32)      # work area in the base frame (bin at (0.10, 0.40) excluded by x)


def cluster_object(P):
    A = P[(P[:, 2] > 0.015) & (P[:, 0] > XR[0]) & (P[:, 0] < XR[1]) & (P[:, 1] > YR[0]) & (P[:, 1] < YR[1])]
    if len(A) < 50:
        return None
    cells = {}
    for p in A:
        cells.setdefault((int(p[0] // 0.02), int(p[1] // 0.02)), []).append(p)
    occ = {k for k, v in cells.items() if len(v) >= 4}
    seen, comps = set(), []
    for k in occ:
        if k in seen:
            continue
        st, comp = [k], []
        while st:
            c = st.pop()
            if c in seen or c not in occ:
                continue
            seen.add(c)
            comp.append(c)
            st += [(c[0] + i, c[1] + j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
        comps.append(np.concatenate([cells[c] for c in comp]))
    comps = [c for c in comps if len(c) > 300 and np.percentile(c[:, 2], 95) < 0.12]   # low compact object, not the arm
    if not comps:
        return None
    c = max(comps, key=len)
    top = float(np.percentile(c[:, 2], 95))
    T = c[c[:, 2] > top - 0.01]
    return dict(cx=float(T[:, 0].mean()), cy=float(T[:, 1].mean()), top=top, n=int(len(c)))


def main():
    ext = json.load(open(os.path.join(ROOT, "outputs", "extrinsics_d435.json")))
    Tbc = np.array(ext["T_base_cam435"])
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(SERIAL)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    prof = pipe.start(cfg)
    ds = prof.get_device().first_depth_sensor().get_depth_scale()
    intr = prof.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    v, u = np.mgrid[0:480, 0:640]
    u = u.astype(np.float32); v = v.astype(np.float32)
    NMED = 4                                     # temporal median: a single frame leaves holes on dark objects
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for _ in range(10):
        pipe.wait_for_frames()
    n = 0
    t_last = time.time()
    print("[track] up, publishing to udp://127.0.0.1:%d" % PORT, flush=True)
    while True:
        D = np.median(np.stack([np.asanyarray(pipe.wait_for_frames().get_depth_frame().get_data()).astype(np.float32) for _ in range(NMED)]), 0) * ds
        ok = (D > 0.2) & (D < 1.5)
        z = D[ok]; x = (u[ok] - intr.ppx) / intr.fx * z; y = (v[ok] - intr.ppy) / intr.fy * z
        P = np.stack([x, y, z, np.ones_like(z)], 1) @ Tbc.T
        det = cluster_object(P[:, :3])
        msg = dict(t=time.time(), **(det or dict(cx=None, cy=None, top=None, n=0)))
        sock.sendto(json.dumps(msg).encode(), ("127.0.0.1", PORT))
        n += 1
        if n % 20 == 0:
            hz = 20 / (time.time() - t_last); t_last = time.time()
            print(f"[track] {hz:.1f} Hz  {json.dumps(det) if det else 'no object'}", flush=True)


if __name__ == "__main__":
    main()
