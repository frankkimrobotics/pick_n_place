#!/usr/bin/env python3
"""zed_track :: live object tracker from a fixed ZED (USB 3: ZED 2 / 2i / Mini), a DROP-IN replacement
for `rl/rgb_track.py --mode depth`.

It publishes EXACTLY the same UDP message on the same port, so every consumer
(`rl/real_policy_ctrl.py --track`, `rl/run_pick_rand.sh`, `rl/success_monitor.py`) works unchanged:

    udp 127.0.0.1:9701   {"t", "cx", "cy", "top", "n", "src", "cands": [{cx, cy, top, n} x <= 8]}

Only ONE tracker may run at a time (they would both bind-free-send to the same port and the consumers
would interleave two different detections). Start this one *instead of* rgb_track.py.

Geometry / frames
-----------------
The ZED SDK is opened with

    coordinate_system = sl.COORDINATE_SYSTEM.IMAGE      # +X right, +Y down, +Z forward
    coordinate_units  = sl.UNIT.METER

which is the classic OPTICAL convention, i.e. the same one the RealSense extrinsics files use
("optical +X right +Y down +Z fwd"). `sl.MEASURE.XYZ` then gives, per pixel of the *rectified left*
image, a metric 3-D point in the LEFT CAMERA optical frame (invalid pixels are NaN/Inf).
`outputs/extrinsics_zed.json["T_base_camzed"]` is the 4x4 base <- ZED-left-optical transform written by
`rl/calib_zed_board.py`, so

    P_base = T_base_camzed[:3,:3] @ P_leftcam + T_base_camzed[:3,3]

There is no colour/depth alignment step (unlike rgb_track, which has to compose the RealSense
colour<-depth extrinsics): the ZED's XYZ measure is already registered to the left image pixel grid.

Detection (identical to rgb_track.py --mode depth)
--------------------------------------------------
1. Work-area ROI: the base-frame rectangle x in XR, y in YR, at z = 0 and z = --top, projected into the
   left image through the ZED's rectified left intrinsics; convex hull -> mask.
2. Per-pixel base-frame height from the XYZ measure. Table plane offset `h_table` = median height over
   the ROI (objects are a minority of the ROI, so the median is the table).
3. Object pixels = height > h_table + --zmin_obj (1 cm).
4. Morphological open, connected components >= --min_px, border-touching components rejected (the arm).
5. Each component's points are re-clustered on a 2 cm xy grid (8-neighbour, >= 150 points per cluster):
   one colour/height blob can be several touching objects plus shadow.
6. Per cluster: top = 90th percentile height, centre = mean xy of the points within 1.5 cm of the top.
   Reject top > --zmax (0.15 m, that is the arm), top < --min_top, or a centre outside the work area.
7. The largest cluster is the primary detection; up to 8 are published in "cands", largest first.

Tracker-bias convention (unchanged): this tracker publishes RAW base-frame coordinates. The measured
bias in `~/pnp_rl/tracker_bias.json` is subtracted by the CONSUMERS (run_pick*.sh / --track_bias), never
here. `--affine` is the optional wrist-camera-fitted xy correction, same as rgb_track.

    python3 rl/zed_track.py                        # publish at 30 fps
    python3 rl/zed_track.py --once --debug /tmp/zed.png
    python3 rl/zed_track.py --resolution HD1080 --depth NEURAL_PLUS --fps 15
"""
import argparse
import json
import os
import socket
import time

import cv2
import numpy as np
import pyzed.sl as sl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PORT = 9701
XR, YR = (0.18, 0.60), (-0.32, 0.32)          # work area in the base frame (identical to rgb_track.py)


def split_clusters(P, cell=0.02, min_pts=150):
    """Connected components of the xy occupancy grid (8-neighbour) of the points above the table; a gap of one
    empty 2 cm cell separates two objects. Returns the point subsets, largest first.
    (Verbatim from rl/rgb_track.py -- duplicated rather than imported because rgb_track imports
    pyrealsense2 at module level, which needs PYTHONPATH=~/librealsense/build/release.)"""
    keys = np.floor(P[:, :2] / cell).astype(np.int64)
    cells = {}
    for idx, k in enumerate(map(tuple, keys)):
        cells.setdefault(k, []).append(idx)
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
        pts = np.concatenate([cells[c] for c in comp])
        if len(pts) >= min_pts:
            comps.append(P[pts])
    comps.sort(key=lambda c: -len(c))
    return comps


def open_zed(a):
    z = sl.Camera()
    ip = sl.InitParameters()
    ip.camera_resolution = getattr(sl.RESOLUTION, a.resolution)
    ip.camera_fps = a.fps
    ip.depth_mode = getattr(sl.DEPTH_MODE, a.depth)
    ip.coordinate_units = sl.UNIT.METER
    ip.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE        # +X right, +Y down, +Z fwd (optical)
    ip.depth_minimum_distance = a.dmin
    ip.depth_maximum_distance = a.dmax
    ip.sdk_verbose = 0
    st = z.open(ip)
    if st != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(
            f"[zed_track] ZED open failed: {st}\n"
            "  * the ZED needs a USB 3.0 port (lsusb -t must show the 2b03 device at 5000M, not 480M)\n"
            "  * the ZED SDK needs the V4L2 uvcvideo driver: /dev/video* must exist.\n"
            "    This box blacklists it for librealsense (/etc/modprobe.d/blacklist-uvcvideo.conf):\n"
            "      sudo modprobe uvcvideo     # load it by hand, the blacklist only stops autoloading")
    info = z.get_camera_information()
    cc = info.camera_configuration
    lc = cc.calibration_parameters.left_cam                  # RECTIFIED left intrinsics (disto ~ 0)
    K = dict(fx=lc.fx, fy=lc.fy, cx=lc.cx, cy=lc.cy)
    print(f"[zed_track] {info.camera_model} sn {info.serial_number}  {cc.resolution.width}x{cc.resolution.height}"
          f" @{cc.fps:.0f} fps  depth={a.depth}  fx={lc.fx:.1f} fy={lc.fy:.1f} cx={lc.cx:.1f} cy={lc.cy:.1f}", flush=True)
    return z, K, int(cc.resolution.width), int(cc.resolution.height)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", default=os.path.join(ROOT, "outputs", "extrinsics_zed.json"),
                    help="base <- ZED-left-optical extrinsics (rl/calib_zed_board.py)")
    ap.add_argument("--resolution", default="HD720", choices=["HD720", "HD1080", "HD1200", "SVGA", "VGA"])
    ap.add_argument("--depth", default="NEURAL", choices=["NEURAL", "NEURAL_PLUS", "NEURAL_LIGHT", "ULTRA", "QUALITY", "PERFORMANCE"])
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dmin", type=float, default=0.3, help="ZED depth_minimum_distance (m)")
    ap.add_argument("--dmax", type=float, default=4.0, help="ZED depth_maximum_distance (m)")
    ap.add_argument("--conf", type=int, default=95, help="ZED confidence_threshold (lower = stricter, fewer points)")
    ap.add_argument("--texture_conf", type=int, default=100, help="ZED texture_confidence_threshold")
    ap.add_argument("--top", type=float, default=0.05, help="object height (m) used only to raise the ROI polygon")
    ap.add_argument("--min_top", type=float, default=0.012, help="reject clusters lower than this (shadows/plane noise)")
    ap.add_argument("--zmax", type=float, default=0.15, help="reject clusters whose top exceeds this (arm)")
    ap.add_argument("--zmin_obj", type=float, default=0.010, help="points below this height above the table are table/shadow")
    ap.add_argument("--min_px", type=int, default=0, help="min component area in px; 0 = auto (300 px scaled from rgb_track's 640x480)")
    ap.add_argument("--min_pts", type=int, default=150, help="min points per 2 cm-grid cluster")
    ap.add_argument("--zrange", type=float, nargs=2, default=[0.2, 3.0], help="valid camera-frame Z (m) for a pixel")
    ap.add_argument("--median", type=int, default=1, help="temporal median over N height maps (1 = off; NEURAL depth is dense)")
    ap.add_argument("--affine", default=os.path.expanduser("~/pnp_rl/tracker_affine.json"),
                    help="optional xy correction fitted against the wrist camera (rl/wrist_centre.py): [x y 1] @ A -> corrected xy")
    ap.add_argument("--known", type=float, nargs=3, default=None, help="debug: project this base-frame point into the image (magenta)")
    ap.add_argument("--debug", default=None, help="write an annotated image here (every 20th frame, or once)")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(a.ext):
        raise SystemExit(f"[zed_track] missing {a.ext} -- run rl/calib_zed_board.py first "
                         "(board at base (0.4, 0, 0)), or rl/calib_zed_board.py --provisional")
    ext = json.load(open(a.ext))
    Tbc = np.array(ext["T_base_camzed"], float)                      # base <- ZED left optical
    Tcb = np.linalg.inv(Tbc)
    if ext.get("provisional"):
        print(f"[zed_track] WARNING: {a.ext} is PROVISIONAL ({ext.get('provisional_note', '')}) -- "
              "xy/yaw are a guess, re-run rl/calib_zed_board.py with the board", flush=True)

    z, K, W, H = open_zed(a)
    fx, fy, cx0, cy0 = K["fx"], K["fy"], K["cx"], K["cy"]
    min_px = a.min_px or max(50, int(round(300 * (W * H) / (640 * 480))))

    # work-area polygon (base frame, z = 0 and z = top) -> image mask, exactly like rgb_track
    corners = [(XR[0], YR[0]), (XR[1], YR[0]), (XR[1], YR[1]), (XR[0], YR[1])]
    pts = []
    for zc in (0.0, a.top):
        for (x, y) in corners:
            pc = Tcb @ np.array([x, y, zc, 1.0])
            if pc[2] <= 1e-3:                                        # corner behind the camera
                continue
            pts.append((int(fx * pc[0] / pc[2] + cx0), int(fy * pc[1] / pc[2] + cy0)))
    roi = np.zeros((H, W), np.uint8)
    if len(pts) >= 3:
        hull = cv2.convexHull(np.array(pts, np.int32))
        cv2.fillConvexPoly(roi, hull, 255)
    else:
        hull = np.array([[[0, 0]], [[W - 1, 0]], [[W - 1, H - 1]], [[0, H - 1]]], np.int32)
        roi[:] = 255
    if (roi > 0).sum() < 0.01 * W * H:
        print(f"[zed_track] WARNING: the work area projects to only {(roi > 0).sum()} px -- "
              "the ZED is probably not aimed at the table, or the extrinsics are wrong", flush=True)

    A_aff = None
    if a.affine and os.path.exists(a.affine):
        A_aff = np.array(json.load(open(a.affine))["A"], float)      # (3, 2): [x, y, 1] @ A = corrected [x, y]
        print(f"[zed_track] xy affine correction from {a.affine}: A = {np.round(A_aff, 4).tolist()}", flush=True)

    rt = sl.RuntimeParameters()
    rt.confidence_threshold = a.conf
    rt.texture_confidence_threshold = a.texture_conf
    m_img, m_xyz = sl.Mat(), sl.Mat()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for _ in range(10):                                              # let AE/AWB and the depth net settle
        z.grab(rt)
    R, t = Tbc[:3, :3], Tbc[:3, 3]
    rz = R[2, :].astype(np.float32)                                  # base-z row, for the fast height map
    n, t_last, hist = 0, time.time(), []
    print(f"[zed_track] up, publishing to udp://127.0.0.1:{PORT}; min_px {min_px}; "
          f"ROI hull {hull.reshape(-1, 2).tolist()}", flush=True)
    try:
        while True:
            if z.grab(rt) != sl.ERROR_CODE.SUCCESS:
                continue
            z.retrieve_image(m_img, sl.VIEW.LEFT)
            z.retrieve_measure(m_xyz, sl.MEASURE.XYZ)
            col = m_img.get_data()[:, :, :3].copy()                  # BGRA -> BGR
            P = m_xyz.get_data()[:, :, :3]                           # (H, W, 3) in the LEFT optical frame, metres
            zc = P[:, :, 2]
            valid = np.isfinite(zc) & (zc > a.zrange[0]) & (zc < a.zrange[1]) & (roi > 0)
            # per-pixel base-frame HEIGHT (only the z row of Tbc is needed for the mask). Written out
            # elementwise on the full float32 array rather than P[valid] @ R[2] -- the fancy-index copy of
            # ~1 Mpx x 3 floats costs 90 ms/frame at HD720, this costs 5 ms.
            hgt = P[:, :, 0] * rz[0]
            hgt += P[:, :, 1] * rz[1]
            hgt += P[:, :, 2] * rz[2]
            hgt += np.float32(t[2])
            hgt[~valid] = np.nan
            if a.median > 1:
                hist.append(hgt)
                if len(hist) > a.median:
                    hist.pop(0)
                hgt = np.nanmedian(np.stack(hist), 0) if len(hist) > 1 else hgt
            # table plane offset: median height over the ROI (objects are a minority of the work area).
            # Sampled every 3rd pixel: 1.6 ms instead of 13 ms, and the median moves by < 0.1 mm.
            hs = hgt[::3, ::3][roi[::3, ::3] > 0]
            hs = hs[np.isfinite(hs)]
            h_table = float(np.median(hs)) if len(hs) else 0.0
            hrel = np.nan_to_num(hgt - h_table, nan=0.0)
            mask = ((hrel > a.zmin_obj) & (roi > 0)).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
            ncc, lab, stats, cents = cv2.connectedComponentsWithStats(mask)
            cands = []
            for i in range(1, ncc):
                x, y, w, h, area = stats[i]
                if area < min_px or x == 0 or y == 0 or x + w >= W or y + h >= H:
                    continue                                          # too small, or the arm coming in from a border
                m = (lab == i) & valid
                if m.sum() < 30:
                    continue
                Q = P[m]
                Q = Q[np.isfinite(Q).all(1)]
                if len(Q) < 30:
                    continue
                B = Q @ R.T + t                                       # -> base frame
                B[:, 2] -= h_table
                B = B[(B[:, 2] > a.zmin_obj) & (B[:, 2] < a.zmax + 0.05)]
                if len(B) < 30:
                    continue
                for C in split_clusters(B, cell=0.02, min_pts=a.min_pts):
                    top = float(np.percentile(C[:, 2], 90))
                    T = C[C[:, 2] > top - 0.015]
                    cand = dict(cx=float(T[:, 0].mean()), cy=float(T[:, 1].mean()), top=top, n=int(len(C)), src="depth")
                    if cand["top"] > a.zmax or cand["top"] < a.min_top:
                        continue
                    if not (XR[0] < cand["cx"] < XR[1] and YR[0] < cand["cy"] < YR[1]):
                        continue
                    cands.append((len(C), cand, i))
            if A_aff is not None:
                for _, c, _ in cands:
                    c["cx"], c["cy"] = (float(v) for v in np.array([c["cx"], c["cy"], 1.0]) @ A_aff)
            cands.sort(key=lambda c: -c[0])
            det = cands[0][1] if cands else None
            msg = dict(t=time.time(), **(det or dict(cx=None, cy=None, top=None, n=0)))
            msg["cands"] = [dict(cx=c[1]["cx"], cy=c[1]["cy"], top=c[1]["top"], n=c[1]["n"]) for c in cands[:8]]
            sock.sendto(json.dumps(msg).encode(), ("127.0.0.1", PORT))
            n += 1
            if a.debug and (a.once or n % 20 == 0):
                dbg = col.copy()
                cv2.polylines(dbg, [hull], True, (0, 255, 255), 2)
                if a.known is not None:
                    pc = Tcb @ np.array([*a.known, 1.0])
                    if pc[2] > 1e-3:
                        cv2.circle(dbg, (int(fx * pc[0] / pc[2] + cx0), int(fy * pc[1] / pc[2] + cy0)), 8, (255, 0, 255), 2)
                dbg[mask > 0] = (0.5 * dbg[mask > 0] + [0, 0, 127]).astype(np.uint8)
                for rank, (_, c, i) in enumerate(cands[:8]):
                    x, y, w, h, _ = stats[i]
                    col_ = (0, 255, 0) if rank == 0 else (0, 200, 255)
                    cv2.rectangle(dbg, (x, y), (x + w, y + h), col_, 2)
                    cv2.putText(dbg, f"#{rank} ({c['cx']:.3f},{c['cy']:.3f}) top {c['top']:.3f} n{c['n']}",
                                (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_, 1)
                cv2.putText(dbg, f"table h {h_table * 1000:+.0f} mm   {len(cands)} cand", (10, H - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imwrite(a.debug, dbg)
            if n % 20 == 0 or a.once:
                hz = (20 if not a.once else n) / max(1e-6, time.time() - t_last)
                t_last = time.time()
                print(f"[zed_track] {hz:.1f} Hz  table {h_table * 1000:+.1f} mm  "
                      f"{len(cands)} cand  {json.dumps(det) if det else 'no object'}", flush=True)
                if a.once:
                    for rank, (_, c, _) in enumerate(cands[:8]):
                        print(f"           #{rank}  x {c['cx']:+.4f}  y {c['cy']:+.4f}  top {c['top']:.4f}  n {c['n']}", flush=True)
            if a.once:
                break
    finally:
        z.close()


if __name__ == "__main__":
    main()
