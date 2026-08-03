#!/usr/bin/env python3
"""REAL detect+pick+place using ONLY the wrist D405 (no fixed D435, no board):
  * DETECT: move to a scan pose over the table, capture D405 RGBD, SAM3, deproject via the calibrated
    hand-eye  T_base_cam = FK(scan_q) @ shift @ T_TCP_CAM  -> object base positions (centroid).
  * TOUCH:  welded descent with the cup-region DEPTH contact (the method that worked earlier).
  * suction grasp -> lift -> J1-swing to box -> release -> save one episode.

  python3 real_pnp_wrist.py                 # dry run: scan + detect + plan (grasp motion gated)
  python3 real_pnp_wrist.py --execute        # real pick+place
"""
import argparse
import json
import os
import sys
import threading
import time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C
from latency_common import rpc, send_chunk, read_state, qrad, fk, plan_pose, plan_joint, stream_at, to_base, DOWN
from geometry import transform_points
from real_pnp_one import WristMon, carry_swing_q, welded_move
sys.path.insert(0, os.path.join(HERE, "il", "data"))
from episode_writer import EpisodeWriter
from il.real_detect_test import sam3_labels

WRIST = "218622271300"


def make_T(R, t):
    T = np.eye(4); T[:3, :3] = np.asarray(R); T[:3, 3] = np.asarray(t).ravel(); return T
def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
def fk_T(qd):
    r = rpc({"type": "fk", "q": qrad(qd)}); return make_T(quat_wxyz_to_R(np.array(r["quat"][0])), np.array(r["pos"][0]))


def capture_d405():
    import pyrealsense2 as rs
    pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(WRIST)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    prof = pipe.start(cfg); align = rs.align(rs.stream.color)
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    for _ in range(15): pipe.wait_for_frames(2000)
    f = align.process(pipe.wait_for_frames(2000))
    color = np.asanyarray(f.get_color_frame().get_data())
    depth = np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32) * scale
    it = f.get_color_frame().profile.as_video_stream_profile().intrinsics
    K = np.array([[it.fx, 0, it.ppx], [0, it.fy, it.ppy], [0, 0, 1.]])
    pipe.stop()
    return color, depth, K


def detect_wrist(scan_xyz, move, min_px=400, max_px=60000):
    """Move to scan pose (if move), capture D405, SAM3, deproject via hand-eye -> [{id,xyz,npx}] (base)."""
    if move:
        welded_move(scan_xyz + DOWN, v=30.0)
    else:                                                     # no-motion dry run: staying put
        r = rpc({"type": "plan_pose", "start_q": qrad(read_state()[0]), "goal_pose": scan_xyz + DOWN, "max_attempts": 16})
        print(f"    scan-pose plan -> {'OK' if r.get('success') else 'FAILED'} "
              f"(no-motion: detection deprojected from CURRENT pose, positions only valid at scan)")
        if not r.get("success"):
            return []
    time.sleep(1.0)
    q = read_state()[0]; T_tcp = fk_T(q)
    T_base_cam = T_tcp @ make_T(np.eye(3), [0, 0, C.CAM_TCP_Z_SHIFT]) @ np.array(C.T_TCP_CAM, float)
    dets = []
    for attempt in range(3):                                 # retry: transient bad-depth frame / SAM3 hiccup
        color, depth, K = capture_d405()
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        label = sam3_labels(rgb)
        ov = color.copy(); dets = []
        for i in [int(x) for x in np.unique(label) if x > 0]:
            m = (label == i) & (depth > 0.05) & (depth < 0.6)
            vs, us = np.where(m)
            if not (min_px < len(vs) < max_px):
                continue
            z = depth[vs, us]; x = (us - cx) / fx * z; y = (vs - cy) / fy * z
            pts = transform_points(T_base_cam, np.stack([x, y, z], 1))
            cxb, cyb = np.median(pts[:, :2], 0)              # centroid XY
            zb = pts[:, 2]; cz = float(np.percentile(zb, 85))  # object top z (base)
            top = zb[zb > np.percentile(zb, 80)]
            flat_mm = float(np.std(top) * 1000)              # top-face flatness (mm); low = suctionable
            dets.append({"id": i, "xyz": [float(cxb), float(cyb), cz], "npx": len(vs), "flat_mm": flat_mm})
            ov[m] = (0.5 * ov[m] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
            cv2.putText(ov, str(i), (int(us.mean()), int(vs.mean())), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(C.OUT_DIR, "wrist_detect_view.png"), ov)
        if dets:
            break
        print(f"    (attempt {attempt+1}: 0 dets, retrying)"); time.sleep(0.5)
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", default="0.1,0.4"); ap.add_argument("--cube", type=float, default=0.30)
    ap.add_argument("--scan", default="0.38,0.0,0.32", help="scan-pose tcp xyz (D405 views the table)")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--detect-only", action="store_true", help="move to scan + detect + report; NO grasp")
    ap.add_argument("--contact-gap", type=float, default=0.0)
    ap.add_argument("--v-des", type=float, default=4.0)
    ap.add_argument("--table-top", type=float, default=0.0)
    ap.add_argument("--max-px", type=int, default=12000, help="reject blobs bigger than this (merged/box)")
    ap.add_argument("--out", default=os.path.join(C.OUT_DIR, "real_episodes"))
    ap.add_argument("--suction-host", default="10.0.0.27"); ap.add_argument("--suction-user", default="pi")
    a = ap.parse_args()
    bx, by = [float(v) for v in a.box.split(",")]; rim = a.table_top + a.cube
    scan = [float(v) for v in a.scan.split(",")]

    move = a.execute or a.detect_only
    if move:
        to_base(30.0)
    print("[1] scan + detect from WRIST D405...")
    dets = detect_wrist(scan, move)
    reach = [d for d in dets if 0.26 < d["xyz"][0] < 0.50 and abs(d["xyz"][1]) < 0.20
             and -0.03 < d["xyz"][2] < 0.20 and 800 < d["npx"] < a.max_px]   # plausible table object
    reach.sort(key=lambda d: (d["flat_mm"], (d["xyz"][0] - 0.38) ** 2 + d["xyz"][1] ** 2))  # flattest, then central
    print(f"    {len(dets)} instances, {len(reach)} reachable+flat  (view -> outputs/wrist_detect_view.png)")
    for d in reach:
        print(f"    obj{d['id']}: [{d['xyz'][0]:+.3f},{d['xyz'][1]:+.3f},{d['xyz'][2]:+.3f}] "
              f"px={d['npx']} flat={d['flat_mm']:.1f}mm")
    if not reach:
        print("    no reachable object detected"); return
    o = reach[0]; ox, oy, oz = o["xyz"]
    pregrasp = [ox, oy, oz + 0.06] + DOWN; lift = [ox, oy, rim + 0.05] + DOWN
    print(f"[2] TARGET obj{o['id']} centroid @ [{ox:.3f},{oy:.3f}] top_z={oz:.3f}")
    if a.detect_only:
        print("[detect-only] arm parked at scan pose; no grasp. Inspect wrist_detect_view.png, then --execute.")
        return
    if not a.execute:
        for nm, gp in [("pregrasp", pregrasp), ("lift", lift)]:
            r = rpc({"type": "plan_pose", "start_q": qrad(read_state()[0]), "goal_pose": gp, "max_attempts": 16})
            print(f"    {nm:9} -> {'OK' if r.get('success') else 'FAILED'}")
        print("[dry-run] done. --execute to pick.")
        return

    try:
        import suction_test
    except Exception:
        suction_test = None
    mon = WristMon(); mon.start(); time.sleep(3.0)
    ep = EpisodeWriter(a.out, f"{int(time.time())}", source="real", meta_extra={"obj": o["id"], "src": "wrist"})
    suction = 0; rec_stop = threading.Event(); phase = ["start"]

    def record():
        while not rec_stop.is_set():
            wr = mon.rgb()
            if wr is not None:
                ep.add(time.time(), qrad(read_state()[0]), suction, wr, wr, phase=phase[0])
            time.sleep(0.1)
    threading.Thread(target=record, daemon=True).start()

    def sset(on):
        nonlocal suction; suction = int(on)
        if suction_test is not None:
            try: suction_test.set_pin(int(on), a.suction_host, a.suction_user)
            except Exception as e: print("  suction:", e)

    print("[3] pregrasp"); phase[0] = "reach"; welded_move(pregrasp, v=30.0)
    print("[4] welded touch descent (cup-region depth contact)"); phase[0] = "descend"
    z = oz + 0.06; z_floor = oz - 0.02; t_contact = None
    while z > z_floor and t_contact is None:
        z -= 0.003
        tr = plan_pose(qrad(read_state()[0]), [ox, oy, z] + DOWN)
        if tr is None: break
        tr[:, 5] = float(C.BASE_Q[5]); stream_at(tr, a.v_des, settle=False); time.sleep(0.35)
        g = mon.gap(); _, tq = read_state()
        if g is not None and g <= a.contact_gap:
            t_contact = time.time(); send_chunk({"hold": True}); print(f"    CONTACT gap {g*1000:.1f}mm z={fk(read_state()[0])[2]:.3f}"); break
        if abs(tq[2]) > 0.14: send_chunk({"hold": True}); print("    torque abort"); break
    print("[5] suction ON"); phase[0] = "grasp"; sset(True); time.sleep(0.8)
    print("[6] lift"); phase[0] = "lift"; welded_move(lift, v=30.0)
    print("[7] carry (J1-swing)"); phase[0] = "carry"
    lqr = qrad(read_state()[0]); cq = carry_swing_q(lqr, ox, oy, bx, by)
    ctr = plan_joint(lqr, [float(x) for x in cq])
    if ctr is not None:
        ctr[:, 5] = float(C.BASE_Q[5]); stream_at(ctr, 30.0)
    print("[8] release"); phase[0] = "release"; sset(False); time.sleep(0.8)
    phase[0] = "home"; to_base(30.0)
    rec_stop.set(); time.sleep(0.3); mon.stop_evt.set()
    print(f"[9] saved -> {ep.close(success=True)}")


if __name__ == "__main__":
    main()
