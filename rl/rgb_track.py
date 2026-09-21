#!/usr/bin/env python3
"""rgb_track :: live object tracker from the fixed D435 COLOUR image (dark object on the white table),
published over UDP like d435_track.py (same message: {"t","cx","cy","top","n"} to 127.0.0.1:9701).

Why colour: the grey/dark objects give sparse, holey D435 depth (a single frame loses them entirely), but
they are a high-contrast blob against the white table in RGB. Segmentation: pixels darker than the table
(V < --vmax) inside the projected work-area polygon, morphological open, largest blob above --min_px.
3-D position: median of the valid depth pixels inside the blob (colour aligned to the depth frame, so the
depth extrinsics apply exactly); if fewer than 30 valid depths, intersect the blob-centroid ray with the
plane z = top/2 (top from --top, the object height from the initial scan/touch).
Arm rejection: blobs whose depth-derived height exceeds --zmax (0.15 m) or that touch the image border
are ignored; the controller additionally freezes tracking within 8 cm of the grasp point.

  PYTHONPATH=~/librealsense/build/release python3 rl/rgb_track.py --top 0.098 [--debug out.png] [--once]
"""
import argparse
import json
import os
import socket
import time

import cv2
import numpy as np
import pyrealsense2 as rs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PORT = 9701
SERIAL = "043422070101"
XR, YR = (0.18, 0.60), (-0.32, 0.32)


def split_clusters(P, cell=0.02, min_pts=150):
    """Connected components of the xy occupancy grid (8-neighbour) of the points above the table; a gap of one
    empty 2 cm cell separates two objects. Returns the point subsets, largest first."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=float, default=0.05, help="object height (m) for the plane fallback")
    ap.add_argument("--vmax", type=int, default=100, help="HSV value threshold: object pixels are darker than this (table V 116-157, grey cylinder 49-86, black base 20-68 on 2026-09-20)")
    ap.add_argument("--known", type=float, nargs=3, default=None, help="debug: project this base-frame point into the image (magenta)")
    ap.add_argument("--mode", default="depth", choices=["depth", "dark", "hue", "diff"], help="depth: pixels standing > zmin_obj above the table plane (lighting-independent; default); dark: V < vmax (grey/black objects); hue: H in [hue_lo, hue_hi] and S > smin (coloured objects); diff: Lab distance from the table's median colour > dthr (anything that is not table)")
    ap.add_argument("--dthr", type=float, default=18.0, help="Lab distance threshold for --mode diff")
    ap.add_argument("--min_top", type=float, default=0.012, help="reject depth-derived blobs lower than this (shadows); low objects read ~0.017 on sparse depth")
    ap.add_argument("--max_px", type=int, default=15000, help="reject blobs larger than this (a sheet of paper is ~25k px, objects 2-8k)")
    ap.add_argument("--hue", type=int, nargs=2, default=[15, 40], help="OpenCV hue range (0-179) for --mode hue; yellow ~ 20-35, red ~ 0-8/170-179, blue ~ 100-125")
    ap.add_argument("--smin", type=int, default=80, help="min saturation for --mode hue")
    ap.add_argument("--min_px", type=int, default=300)
    ap.add_argument("--zmax", type=float, default=0.15, help="reject blobs whose depth height exceeds this (arm)")
    ap.add_argument("--zmin_obj", type=float, default=0.010, help="depth points below this height are table/shadow and are excluded from the object clusters")
    ap.add_argument("--affine", default=os.path.expanduser("~/pnp_rl/tracker_affine.json"), help="optional xy correction fitted against the wrist camera (rl/wrist_centre.py): [x y 1] @ A -> corrected xy; applied to every candidate")
    ap.add_argument("--debug", default=None, help="write an annotated image here (every 20th frame, or once)")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    ext = json.load(open(os.path.join(ROOT, "outputs", "extrinsics_d435.json")))
    Tbc = np.array(ext["T_base_cam435"]); Tcb = np.linalg.inv(Tbc)
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(SERIAL)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    prof = pipe.start(cfg)
    align = rs.align(rs.stream.color)                       # depth -> colour frame (the colour image stays complete)
    ds = prof.get_device().first_depth_sensor().get_depth_scale()
    cprof, dprof = prof.get_stream(rs.stream.color).as_video_stream_profile(), prof.get_stream(rs.stream.depth).as_video_stream_profile()
    intr = cprof.get_intrinsics()
    fx, fy, cx0, cy0 = intr.fx, intr.fy, intr.ppx, intr.ppy
    e = cprof.get_extrinsics_to(dprof)                      # colour-frame point -> depth-frame point
    Tdc = np.eye(4); Tdc[:3, :3] = np.array(e.rotation).reshape(3, 3).T; Tdc[:3, 3] = e.translation
    Tbc = Tbc @ Tdc                                         # base <- colour camera
    Tcb = np.linalg.inv(Tbc)
    # work-area polygon (base frame, z = 0 and z = top) -> image mask
    corners = [(XR[0], YR[0]), (XR[1], YR[0]), (XR[1], YR[1]), (XR[0], YR[1])]
    pts = []
    for zc in (0.0, a.top):
        for (x, y) in corners:
            pc = Tcb @ np.array([x, y, zc, 1.0])
            pts.append((int(fx * pc[0] / pc[2] + cx0), int(fy * pc[1] / pc[2] + cy0)))
    hull = cv2.convexHull(np.array(pts, np.int32))
    roi = np.zeros((480, 640), np.uint8); cv2.fillConvexPoly(roi, hull, 255)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    A_aff = None
    if a.affine and os.path.exists(a.affine):
        A_aff = np.array(json.load(open(a.affine))["A"], float)          # (3, 2): [x, y, 1] @ A = corrected [x, y]
        print(f"[rgb_track] xy affine correction from {a.affine}: A = {np.round(A_aff, 4).tolist()}", flush=True)
    for _ in range(10):
        pipe.wait_for_frames()
    n, t_last = 0, time.time()
    dep_hist = []
    print(f"[rgb_track] up, publishing to udp://127.0.0.1:{PORT}; ROI hull {hull.reshape(-1, 2).tolist()}", flush=True)
    while True:
        fs = align.process(pipe.wait_for_frames())
        col = np.asanyarray(fs.get_color_frame().get_data())
        dep = np.asanyarray(fs.get_depth_frame().get_data()).astype(np.float32) * ds
        if a.mode == "depth":                                   # 3-frame temporal median: single-frame holes on dark objects
            dep_hist.append(dep)
            if len(dep_hist) > 3:
                dep_hist.pop(0)
            dep = np.median(np.stack(dep_hist), 0)
        hsv = cv2.cvtColor(col, cv2.COLOR_BGR2HSV)
        # per-pixel height above the table from depth (2026-09-20): objects are also anything that stands up,
        # and the table colour reference is taken from TABLE-LEVEL pixels only (the D435's lighting gradient made
        # half the table fail the colour test once the white balance drifted -> one giant merged blob)
        zv = (dep > 0.2) & (dep < 1.5) & (roi > 0)
        hgt = np.full(dep.shape, np.nan, np.float32)
        if zv.any():
            vv, uu = np.nonzero(zv); zz = dep[vv, uu]
            Pz = (np.stack([(uu - cx0) / fx * zz, (vv - cy0) / fy * zz, zz], 1) @ Tbc[:3, :3].T)[:, 2] + Tbc[2, 3]
            hgt[vv, uu] = Pz
        # table plane offset: the extrinsics put the table at -5 mm (median over the ROI, objects are a minority)
        h_table = float(np.nanmedian(hgt[roi > 0])) if zv.any() else 0.0
        hrel = np.nan_to_num(hgt - h_table, nan=0.0)
        up = hrel > a.zmin_obj
        table_px = (np.abs(np.nan_to_num(hgt - h_table, nan=1.0)) < 0.005)
        if a.mode == "depth":                                   # default since 2026-09-20: colour is lighting-dependent
            sel = up
        elif a.mode == "dark":
            sel = hsv[:, :, 2] < a.vmax
        elif a.mode == "diff":
            lab = cv2.cvtColor(col, cv2.COLOR_BGR2LAB).astype(np.float32)
            ref = lab[table_px] if table_px.sum() > 2000 else lab[roi > 0]
            med = np.median(ref.reshape(-1, 3), axis=0)
            sel = (np.linalg.norm(lab - med, axis=2) > a.dthr) | up
        else:
            h = hsv[:, :, 0]
            sel = (h >= a.hue[0]) & (h <= a.hue[1]) & (hsv[:, :, 1] > a.smin) & (hsv[:, :, 2] > 60)
        mask = (sel & (roi > 0)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
        ncc, lab, stats, cents = cv2.connectedComponentsWithStats(mask)
        det = None
        cands = []
        for i in range(1, ncc):
            x, y, w, h, area = stats[i]
            if area < a.min_px or x == 0 or y == 0 or x + w >= 640 or y + h >= 480:
                continue
            m = lab == i
            zs = dep[m]; zs = zs[(zs > 0.2) & (zs < 1.5)]
            if len(zs) >= 30:
                us, vs = np.nonzero(m)[1], np.nonzero(m)[0]
                sel = (dep[vs, us] > 0.2) & (dep[vs, us] < 1.5)
                z = dep[vs[sel], us[sel]]
                P = np.stack([(us[sel] - cx0) / fx * z, (vs[sel] - cy0) / fy * z, z, np.ones_like(z)], 1) @ Tbc.T
                # 2026-09-20 s3_03: a colour blob can be SEVERAL objects plus their shadows (13.7k px merged blob ->
                # centroid between two cans -> cup landed on an edge). Drop table-level points (shadows) and split
                # the rest into xy clusters on a 2 cm grid; each cluster becomes its own candidate.
                P[:, 2] -= h_table
                P = P[(P[:, 2] > a.zmin_obj) & (P[:, 2] < a.zmax + 0.05)]
                if len(P) < 30:
                    continue
                for C in split_clusters(P, cell=0.02, min_pts=150):
                    top = float(np.percentile(C[:, 2], 90))
                    T = C[C[:, 2] > top - 0.015]
                    cand = dict(cx=float(T[:, 0].mean()), cy=float(T[:, 1].mean()), top=top, n=int(len(C)), src="depth")
                    if cand["top"] > a.zmax or not (XR[0] < cand["cx"] < XR[1] and YR[0] < cand["cy"] < YR[1]) or cand["top"] < a.min_top:
                        continue
                    cands.append((len(C), cand, i))
                continue
            else:                                                       # plane fallback at z = top/2 through the centroid ray
                if area > a.max_px:                                     # colour-only giant region (paper sheet / table shading)
                    continue
                u0, v0 = cents[i]
                r = np.array([(u0 - cx0) / fx, (v0 - cy0) / fy, 1.0])
                o = Tbc[:3, 3]; d = Tbc[:3, :3] @ r
                s = (a.top / 2 - o[2]) / d[2]
                p = o + s * d
                cand = dict(cx=float(p[0]), cy=float(p[1]), top=a.top, n=int(area), src="plane")
            if cand["top"] > a.zmax or not (XR[0] < cand["cx"] < XR[1] and YR[0] < cand["cy"] < YR[1]):
                continue
            if cand["src"] == "depth" and cand["top"] < a.min_top:
                continue
            cands.append((area, cand, i))
        if A_aff is not None:
            for _, c, _ in cands:
                c["cx"], c["cy"] = (float(v) for v in np.array([c["cx"], c["cy"], 1.0]) @ A_aff)
        if cands:
            cands.sort(key=lambda c: -c[0])
            det = cands[0][1]
        msg = dict(t=time.time(), **(det or dict(cx=None, cy=None, top=None, n=0)))
        msg["cands"] = [dict(cx=c[1]["cx"], cy=c[1]["cy"], top=c[1]["top"], n=c[1]["n"]) for c in cands[:8]]   # all blobs, largest first
        sock.sendto(json.dumps(msg).encode(), ("127.0.0.1", PORT))
        n += 1
        if a.debug and (a.once or n % 20 == 0):
            dbg = col.copy()
            cv2.polylines(dbg, [hull], True, (0, 255, 255), 1)
            if a.known is not None:
                pc = Tcb @ np.array([*a.known, 1.0]); cv2.circle(dbg, (int(fx * pc[0] / pc[2] + cx0), int(fy * pc[1] / pc[2] + cy0)), 6, (255, 0, 255), 2)
            dbg[mask > 0] = (0.5 * dbg[mask > 0] + [0, 0, 127]).astype(np.uint8)
            if cands:
                i = cands[0][2]; x, y, w, h, _ = stats[i]
                cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(dbg, f"({det['cx']:.3f},{det['cy']:.3f}) top {det['top']:.3f} {det['src']}", (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(a.debug, dbg)
        if n % 20 == 0 or a.once:
            hz = 20 / (time.time() - t_last); t_last = time.time()
            print(f"[rgb_track] {hz:.1f} Hz  {json.dumps(det) if det else 'no object'}", flush=True)
        if a.once:
            break
    pipe.stop()


if __name__ == "__main__":
    main()
