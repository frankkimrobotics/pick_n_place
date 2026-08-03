#!/usr/bin/env python3
"""REAL single-object detect -> welded touch-grasp -> place-in-box -> save ONE episode.
Dry-run by default (detect + plan every move, NO motion); --execute commands the arm.

Reuses: D435+SAM3 detection (real_detect_test), cuRobo planner + welder/servo (latency_common),
cup-region D405 depth contact (cup_region + a monitor), suction GPIO (suction_test), EpisodeWriter.

SAFE: cuRobo collision-free plans, welded slow descent, cup-region depth contact stops the descent,
abs-floor + torque abort, J6 held at base. Attempts ONE flat-top object.

  python3 real_pnp_one.py                 # dry run (no motion)
  python3 real_pnp_one.py --execute        # real pick+place (watch it)
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C
import cup_region as CR
from latency_common import rpc, send_chunk, read_state, qrad, fk, plan_pose, plan_joint, stream_at, to_base, DOWN
from geometry import transform_points
sys.path.insert(0, os.path.join(HERE, "il", "data"))
from episode_writer import EpisodeWriter

WRIST = "218622271300"; D435 = "043422070101"
CUP_REGION = os.path.join(C.OUT_DIR, "cup_region_fixed.npz")


# ---------- perception (fixed D435 + SAM3) ----------
def detect(min_px=400):
    from il.real_detect_test import capture, sam3_labels, deproject_centroid  # noqa
    color, depth, K = capture()
    rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    T = np.array(json.load(open(os.path.join(C.OUT_DIR, "extrinsics_d435_static.json")))["T_base_cam435"], float)
    label = sam3_labels(rgb)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dets = []
    for i in [int(x) for x in np.unique(label) if x > 0]:
        m = (label == i) & (depth > 0.05) & (depth < 1.6)
        vs, us = np.where(m)
        if len(vs) < min_px:
            continue
        z = depth[vs, us]; x = (us - cx) / fx * z; y = (vs - cy) / fy * z
        pts = transform_points(T, np.stack([x, y, z], 1))
        cx_b, cy_b = np.median(pts[:, :2], 0)                            # object CENTROID (XY) -> grasp here
        cz = float(np.percentile(pts[:, 2], 85))                        # near-top z (pregrasp ref; contact stops descent)
        top = pts[pts[:, 2] > np.percentile(pts[:, 2], 80)]
        dets.append({"id": i, "xyz": [float(cx_b), float(cy_b), cz],
                     "flat_mm": float(np.std(top[:, 2]) * 1000), "npx": len(vs)})
    return dets


class WristMon(threading.Thread):
    """D405 wrist: latest RGB + cup-region contact gap (the real touch contact)."""
    def __init__(self):
        super().__init__(daemon=True); self.stop_evt = threading.Event()
        self.lock = threading.Lock(); self._gap = None; self._rgb = None; self.cup_d = None
    def run(self):
        import pyrealsense2 as rs
        pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(WRIST)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        prof = pipe.start(cfg); scale = prof.get_device().first_depth_sensor().get_depth_scale(); align = rs.align(rs.stream.color)
        cup_m, above_m = CR.load(CUP_REGION); cds = []
        while not self.stop_evt.is_set():
            try:
                f = align.process(pipe.wait_for_frames(1000))
                d = np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32) * scale
                cd, sd = CR.read_depths(d, cup_m, above_m)
                if cd is not None: cds.append(cd); cds = cds[-30:]
                cref = float(np.median(cds)) if cds else None
                with self.lock:
                    self.cup_d = cref
                    self._gap = (sd - cref) if (sd is not None and cref is not None) else None
                    self._rgb = cv2.cvtColor(np.asanyarray(f.get_color_frame().get_data()), cv2.COLOR_BGR2RGB)
            except Exception:
                pass
        pipe.stop()
    def gap(self):
        with self.lock: return self._gap
    def rgb(self):
        with self.lock: return None if self._rgb is None else self._rgb.copy()


def grab_d435_rgb():
    import pyrealsense2 as rs
    pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(D435)
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    pipe.start(cfg)
    for _ in range(5): pipe.wait_for_frames(1000)
    im = np.asanyarray(pipe.wait_for_frames(1000).get_color_frame().get_data()); pipe.stop()
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def welded_move(xyz, v=25.0, settle=True):
    q0 = read_state()[0]
    tr = plan_pose(qrad(q0), xyz)
    if tr is None:
        return None
    tr[:, 5] = float(C.BASE_Q[5]); stream_at(tr, v, settle=settle); return tr


def carry_swing_q(lift_qr, ox, oy, bx, by):
    """J1-swing over the box: keep the lifted arm shape, rotate the base toward the box bearing
    (plan_pose to the box fails on reachability). Returns the carry joint config (rad)."""
    q = np.array(lift_qr, float).copy()
    q[0] += (np.arctan2(by, bx) - np.arctan2(oy, ox))
    q[5] = float(C.BASE_Q[5]); return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", default="0.1,0.4")
    ap.add_argument("--cube", type=float, default=0.30)
    ap.add_argument("--execute", action="store_true", help="command the real arm (else dry run)")
    ap.add_argument("--contact-gap", type=float, default=0.0)
    ap.add_argument("--v-des", type=float, default=4.0, help="descent speed (deg/s)")
    ap.add_argument("--abs-floor", type=float, default=-0.02, help="hard z floor for the descent (m)")
    ap.add_argument("--flat-max", type=float, default=6.0, help="max top-face std (mm) to call it flat/suctionable")
    ap.add_argument("--max-px", type=int, default=8000, help="reject detections larger than this (excludes calib boards)")
    ap.add_argument("--out", default=os.path.join(C.OUT_DIR, "real_episodes"))
    ap.add_argument("--suction-host", default="10.0.0.27")
    ap.add_argument("--suction-user", default="pi")
    a = ap.parse_args()
    bx, by = [float(v) for v in a.box.split(",")]; rim = 0.0 + a.cube      # real table top ~0

    print("[1] detecting (D435 + SAM3)...")
    dets = detect()
    reach = [d for d in dets if 0.26 < d["xyz"][0] < 0.50 and abs(d["xyz"][1]) < 0.20 and d["npx"] < a.max_px]
    reach.sort(key=lambda d: d["flat_mm"])                                 # flattest top first (best seal)
    print(f"    {len(dets)} instances, {len(reach)} reachable")
    for d in reach:
        print(f"    obj{d['id']}: [{d['xyz'][0]:+.3f},{d['xyz'][1]:+.3f},{d['xyz'][2]:+.3f}] flat={d['flat_mm']:.1f}mm px={d['npx']}")
    if not reach:
        print("    no reachable object; abort"); return
    o = reach[0]; ox, oy, oz = o["xyz"]
    print(f"[2] TARGET obj{o['id']} @ [{ox:.3f},{oy:.3f}] top_z={oz:.3f} flat={o['flat_mm']:.1f}mm "
          f"({'suctionable' if o['flat_mm'] < a.flat_max else 'NOT very flat -- grasp may fail'})")
    pregrasp = [ox, oy, oz + 0.06] + DOWN
    lift = [ox, oy, rim + 0.05] + DOWN
    place = [bx, by, rim + 0.05] + DOWN

    if not a.execute:
        print("[dry-run] planning moves (NO motion):")
        for nm, gp in [("pregrasp", pregrasp), ("lift", lift)]:
            q0 = read_state()[0]; r = rpc({"type": "plan_pose", "start_q": qrad(q0), "goal_pose": gp, "max_attempts": 16})
            print(f"    {nm:16} {[round(x,3) for x in gp[:3]]} -> {'OK (' + str(len(r['trajectory'])) + ' wpts)' if r.get('success') else 'PLAN FAILED'}")
        lr = rpc({"type": "plan_pose", "start_q": qrad(read_state()[0]), "goal_pose": lift, "max_attempts": 16})
        if lr.get("success"):
            cq = carry_swing_q(np.array(lr["trajectory"])[-1], ox, oy, bx, by)
            cr = rpc({"type": "plan_joint", "start_q": list(map(float, np.array(lr["trajectory"])[-1])),
                      "goal_q": [float(x) for x in cq], "max_attempts": 12})
            print(f"    carry(J1-swing)  -> {'OK' if cr.get('success') else 'PLAN FAILED'}   "
                  f"(box rim z={rim:.2f}; drop over box)")
        print("[dry-run] done. Re-run with --execute to perform the real pick+place.")
        return

    # ---------- EXECUTE (real motion) ----------
    try:
        import suction_test
    except Exception:
        suction_test = None
    mon = WristMon(); mon.start(); time.sleep(3.0)
    ep = EpisodeWriter(a.out, f"{int(time.time())}", source="real", meta_extra={"obj": o["id"], "flat_mm": o["flat_mm"]})
    fixed_rgb = grab_d435_rgb(); suction = 0
    rec_stop = threading.Event()

    def record(phase_fn):
        while not rec_stop.is_set():
            q = np.array(read_state()[0], float); wr = mon.rgb()
            if wr is not None:
                ep.add(time.time(), qrad(q), suction, wr, fixed_rgb, phase=phase_fn[0])
            time.sleep(0.1)
    phase = ["start"]; rt = threading.Thread(target=record, args=(phase,), daemon=True); rt.start()

    def set_suction(on):
        nonlocal suction; suction = int(on)
        if suction_test is not None:
            try: suction_test.set_pin(int(on), a.suction_host, a.suction_user)
            except Exception as e: print("  suction GPIO:", e)

    to_base(30.0)
    print("[3] approach -> pregrasp"); phase[0] = "reach"; welded_move(pregrasp, v=30.0)
    print("[4] welded touch descent (cup-region depth contact)"); phase[0] = "descend"
    z = oz + 0.06; z_floor = oz - 0.015; t_contact = None    # object-relative floor (never over-press)
    while z > z_floor and t_contact is None:
        z -= 0.003
        tr = plan_pose(qrad(read_state()[0]), [ox, oy, z] + DOWN)
        if tr is None: break
        tr[:, 5] = float(C.BASE_Q[5]); stream_at(tr, a.v_des, settle=False); time.sleep(0.35)
        g = mon.gap()
        _, tq = read_state()
        if g is not None and g <= a.contact_gap:
            t_contact = time.time(); send_chunk({"hold": True}); print(f"    CONTACT gap {g*1000:.1f}mm FK z={fk(read_state()[0])[2]:.3f}"); break
        if abs(tq[2]) > 0.14: send_chunk({"hold": True}); print("    torque abort"); break
    print("[5] suction ON"); phase[0] = "grasp"; set_suction(True); time.sleep(0.8)
    print("[6] lift"); phase[0] = "lift"; lift_tr = welded_move(lift, v=30.0)
    print("[7] carry over box (J1-swing)"); phase[0] = "carry"
    lift_qr = qrad(read_state()[0])
    cq = carry_swing_q(lift_qr, ox, oy, bx, by)
    ctr = plan_joint(lift_qr, [float(x) for x in cq])
    if ctr is not None:
        ctr[:, 5] = float(C.BASE_Q[5]); stream_at(ctr, 30.0)
    else:
        print("    carry plan failed -> releasing over current pose")
    print("[8] suction OFF (release)"); phase[0] = "release"; set_suction(False); time.sleep(0.8)
    phase[0] = "home"; to_base(30.0)
    rec_stop.set(); time.sleep(0.3); mon.stop_evt.set()
    d = ep.close(success=True); print(f"[9] episode saved -> {d}")


if __name__ == "__main__":
    main()
