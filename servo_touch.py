#!/usr/bin/env python3
"""servo_touch :: CONTINUOUS welded touch (no suction).

A background thread streams the D405 depth at camera rate and continuously computes the
contact gap on the FIXED cup ROI:
    gap = median(depth[dome top-curve region]) - median(depth[cup rim])
(the cup is rigid to the camera so the ROI pixels never move; only the depth values
change as the object approaches). The gap is median+EMA filtered to reject the rim
flying-pixel spikes.

The main thread sends ONE welded descent (approach -> straight-down, concatenated, sent
as a single non-blocking command so robot_hal's B-spline streams through the standoff
with no stop), watches the live gap, and sends a single HOLD at contact -- lag-
compensated (stop early by v*dead_time so the ~0.8 s follow delay doesn't overshoot).
Final contact is handed to the soft cup's compliance (no F/T sensor).

Needs cup_mask.npz. Run in the ROS env with the D405 on PYTHONPATH; planner+bridge+SAM3 up.
"""
import argparse, collections, json, os, sys, threading, time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")),
          os.path.abspath(os.path.join(HERE, "..", "ros2node", "perception"))):
    sys.path.insert(0, p)
import config as C
from geometry import R_from_two_axes, R_to_quat_wxyz, quat_wxyz_to_R, make_T
from real_multi import detect_objects


class GapMonitor(threading.Thread):
    """Owns the D405; streams aligned colour+depth; exposes latest RGBD and a filtered
    live gap on the fixed cup ROI."""
    def __init__(self, serial, w, h, rim, ann, dome, ema=0.4, medn=5):
        super().__init__(daemon=True)
        self.serial, self.w, self.h = serial, w, h
        self.rim, self.ann, self.dome, self.ema, self.medn = rim, ann, dome, ema, medn
        ys, xs = np.where(dome)
        self.dbox = (max(0, int(ys.min()) - 35), int(ys.max()) + 5,   # window, expanded UP for movement
                     max(0, int(xs.min()) - 10), int(xs.max()) + 10)
        self.lock = threading.Lock(); self.stop_evt = threading.Event()
        self._rgbd = None; self._gap = None; self._cupd = None; self._domey = None
        self._t = 0.0; self.n = 0; self.K = None; self.ok = False

    def run(self):
        import pyrealsense2 as rs
        pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, self.w, self.h, rs.format.z16, 30)
        prof = pipe.start(cfg); scale = prof.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        it = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[it.fx, 0, it.ppx], [0, it.fy, it.ppy], [0, 0, 1.]]); self.ok = True
        raw = collections.deque(maxlen=self.medn); gf = None
        try:
            for _ in range(10):
                pipe.wait_for_frames(2000)
            while not self.stop_evt.is_set():
                try:
                    f = align.process(pipe.wait_for_frames(1000))
                except RuntimeError:
                    continue
                c = f.get_color_frame(); d = f.get_depth_frame()
                if not c or not d:
                    continue
                depth = np.asanyarray(d.get_data()).astype(np.float32) * scale
                rgb = np.asanyarray(c.get_data())[:, :, ::-1].copy()
                rd = depth[self.rim]; rd = rd[(rd > 0.04) & (rd < 0.30)]
                ad = depth[self.ann]; ad = ad[(ad > 0.04) & (ad < 0.60)]
                g = None
                if len(rd) >= 8 and len(ad) >= 8:
                    raw.append(float(np.median(ad) - np.median(rd)))
                    gm = float(np.median(raw))                       # median rejects spikes
                    gf = gm if gf is None else self.ema * gm + (1 - self.ema) * gf
                    g = gf
                dd = depth[self.dome]; dd = dd[(dd > 0.04) & (dd < 0.30)]  # DOME depth (cup deformation)
                cupd = float(np.median(dd)) if len(dd) >= 20 else None
                # DOME image position: centroid-y of the dark dome in the window. When the cup
                # compresses on contact, the dome shifts UP -> domey decreases (follows the move).
                y0, y1, x0, x1 = self.dbox
                gray = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
                dys, dxs = np.where(gray < 80)
                domey = float(dys.mean() + y0) if len(dys) >= 40 else None
                with self.lock:
                    self._rgbd = (rgb, depth, self.K); self._gap = g; self._cupd = cupd
                    self._domey = domey; self._t = time.time(); self.n += 1
        finally:
            try:
                pipe.stop()
            except Exception:
                pass

    def latest_rgbd(self, timeout=2.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                r = self._rgbd
            if r is not None:
                return r
            time.sleep(0.02)
        return None

    def latest_gap(self):
        with self.lock:
            return self._gap, self._t

    def latest_cupd(self):
        with self.lock:
            return self._cupd, self._domey


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="218622271300")
    ap.add_argument("--max-h", type=float, default=0.13); ap.add_argument("--max-foot", type=float, default=0.16)
    ap.add_argument("--xmin", type=float, default=0.15); ap.add_argument("--xmax", type=float, default=0.55)
    ap.add_argument("--ymin", type=float, default=-0.28); ap.add_argument("--ymax", type=float, default=0.30)
    ap.add_argument("--cup", default=os.path.join(C.OUT_DIR, "cup_mask.npz"))
    ap.add_argument("--standoff", type=float, default=0.03, help="approach standoff above the surface (m)")
    ap.add_argument("--press", type=float, default=0.002, help="tiny press past the predicted surface (m)")
    ap.add_argument("--max-descend", type=float, default=0.010, help="HARD floor: never below surf - this (m)")
    ap.add_argument("--v-approach", type=float, default=16.0, help="fast free-space approach speed (deg/s)")
    ap.add_argument("--v-descend", type=float, default=3.0, help="(unused) legacy")
    ap.add_argument("--v-touch", type=float, default=2.0, help="smooth linear descent speed (deg/s)")
    ap.add_argument("--margin", type=float, default=0.004, help="linear over-descent past the surface (m); spring absorbs it")
    ap.add_argument("--cal-a", type=float, default=0.78)      # calibrated gap->dist: true = a*gap + b
    ap.add_argument("--cal-b", type=float, default=0.0105)
    ap.add_argument("--dome-shift", type=float, default=6.0, help="dome image shift (px) flagged as contact")
    ap.add_argument("--detect-tries", type=int, default=3)
    ap.add_argument("--dwell", type=float, default=1.2)
    args = ap.parse_args()

    import rclpy
    from std_msgs.msg import String
    from object_pointclouds import deproject_mask
    from capture_and_plot import segment
    from multiview_fuse import pick_res
    from perturb_loop import PlannerClient, RobotState, execute, scale_traj
    from joint_conventions import rad_to_linuxcnc_deg

    cup = np.load(args.cup); rim = cup["rim"].astype(bool); ann = cup["ring"].astype(bool)
    dome = cup["dome"].astype(bool) if "dome" in cup.files else cup["mask"].astype(bool)
    print(f"cup ROI: rim {int(rim.sum())}px, dome arc {int(ann.sum())}px, dome {int(dome.sum())}px")
    CW, CH = pick_res(args.serial)
    mon = GapMonitor(args.serial, CW, CH, rim, ann, dome); mon.start()
    t0 = time.time()
    while not mon.ok and time.time() - t0 < 8:
        time.sleep(0.2)
    if not mon.ok:
        print("ABORT: depth stream failed"); return

    pc = PlannerClient(); rclpy.init(); node = rclpy.create_node("servo_touch")
    pub = node.create_publisher(String, "/mycobot/cmd/move", 10); state = RobotState(node)
    track = {"ramp_time": 0.15, "pos_gain": 1.0, "vff_scale": 1.0}
    down = list(R_to_quat_wxyz(R_from_two_axes(np.array([0, 0, -1.0]))))

    def fk_T(q):
        r = pc.rpc({"type": "fk", "q": list(map(float, q))})
        return make_T(quat_wxyz_to_R(np.array(r["quat"][0])), np.array(r["pos"][0]))

    def fix_j6(t):
        t = np.array(t, float); t[:, 5] = t[0, 5]; return t

    def send_descent_nb(goal, vmax):
        """Plan current -> goal (straight down) and publish NON-BLOCKING at a slow vmax.
        The cuRobo trajectory decelerates to rest AT goal, so the over-press is bounded by
        goal (no overshoot past it); vision HOLD just stops it earlier."""
        q = state.get_q()
        r = pc.plan_pose(list(map(float, q)), list(map(float, goal)) + down, max_attempts=14)
        if not r.get("success"):
            return False
        traj = fix_j6(r["trajectory"]); sdt, _ = scale_traj(traj, r["dt"], vmax, 0.5)
        td = [list(map(float, rad_to_linuxcnc_deg(wp))) for wp in traj]
        pub.publish(String(data=json.dumps({"trajectory": td, "traj_dt": sdt, "target_deg": td[-1],
                                            "controller": "pid", **track})))
        return True

    def send_hold():
        q = state.get_q(); d = list(map(float, rad_to_linuxcnc_deg(q)))
        pub.publish(String(data=json.dumps({"trajectory": [d, d], "traj_dt": 0.1, "target_deg": d,
                                            "controller": "pid", **track})))

    def goto(xyz, label, vmax):
        q = state.get_q()
        r = pc.plan_pose(list(map(float, q)), list(map(float, xyz)) + down, max_attempts=14)
        if not r.get("success"):
            print(f"  [{label}] PLAN FAILED"); return False
        ex = execute(state, pub, fix_j6(r["trajectory"]), r["dt"], "pid", vmax, 2.0, label, track=track)
        return ex.get("ok")

    def to_base():
        r = pc.plan_joint(list(map(float, state.get_q())), list(map(float, C.BASE_Q)))
        if r.get("success"):
            execute(state, pub, fix_j6(r["trajectory"]), r["dt"], "pid", 22.0, 3.0, "to-base", track=track)

    try:
        to_base(); time.sleep(0.3)
        objs = []
        for _ in range(args.detect_tries):
            Tbc = fk_T(state.get_q()) @ make_T(np.eye(3), [0, 0, C.CAM_TCP_Z_SHIFT]) @ C.T_TCP_CAM
            r = mon.latest_rgbd()
            if r is None:
                continue
            rgb, depth, K = r
            for o in detect_objects(rgb, depth, K, Tbc, segment, deproject_mask, args):
                if all(np.linalg.norm(o["centroid"][:2] - a["centroid"][:2]) > 0.04 for a in objs):
                    objs.append(o)
        print(f"detected {len(objs)} object(s)")

        for i, o in enumerate(objs):
            P = o["pts"]; cxy = o["centroid"][:2].copy()
            col = P[np.linalg.norm(P[:, :2] - cxy, axis=1) < 0.02]
            surf = float(np.median(col[:, 2])) if len(col) > 20 else float(np.percentile(P[:, 2], 80))
            face = P[np.abs(P[:, 2] - surf) < 0.010]
            if len(face) > 20:
                cxy = face[:, :2].mean(0)
            floor_z = surf - args.max_descend         # HARD safety floor (backstop only)
            print(f"\n== object {i+1}: centre [{cxy[0]:.3f},{cxy[1]:.3f}] surf_est={surf:.3f} ==")

            # PHASE 1: approach to the 2 cm standoff (blocking)
            if not goto([cxy[0], cxy[1], surf + args.standoff], f"approach{i}", args.v_approach):
                continue
            # PHASE 2: read the CALIBRATED gap (reliable at this close range) -> surface; dome baseline
            time.sleep(0.3)
            gs, by = [], []
            for _ in range(15):
                g, _ = mon.latest_gap(); c, dy = mon.latest_cupd()
                if g is not None:
                    gs.append(g)
                if dy is not None:
                    by.append(dy)
                time.sleep(0.04)
            if len(gs) < 5:
                print("  no gap; skipping"); continue
            gapm = float(np.median(gs)); tcpz = float(fk_T(state.get_q())[2, 3])
            surf_cal = tcpz - (args.cal_a * gapm + args.cal_b)
            base_domey = float(np.median(by)) if by else None
            floor_z = max(0.015, surf_cal - 0.012)                # safety vs the CALIBRATED surface
            target = max(floor_z, surf_cal - args.margin)         # linear target, bounded compression
            print(f"  gap={gapm*1000:.0f}mm -> surf_cal={surf_cal:.3f}; linear descent to {target:.3f} "
                  f"(margin {args.margin*1000:.0f}mm)")
            # PHASE 3: ONE smooth linear decel-to-rest descent (from rest -> no overshoot, gentle),
            # camera MONITORS the contact point (gap->0 / dome shift) during the motion.
            if not send_descent_nb([cxy[0], cxy[1], target], args.v_touch):
                print("  descent plan FAILED"); continue
            contact_z = None; last_print = 0; t_start = time.time()
            while time.time() - t_start < 20.0:
                rclpy.spin_once(node, timeout_sec=0.02)
                g, _ = mon.latest_gap(); c, dy = mon.latest_cupd()
                q = state.get_q(); z = float(fk_T(q)[2, 3]) if q is not None else 9.0
                shift = abs(dy - base_domey) if (dy is not None and base_domey is not None) else 0.0
                now = time.time()
                if contact_z is None and ((g is not None and g <= 0.003) or shift >= args.dome_shift):
                    contact_z = z
                    print(f"  >> CONTACT (camera) at z={z:.3f}  gap={'--' if g is None else f'{g*1000:.0f}mm'}  "
                          f"domeshift={shift:.1f}px")
                if now - last_print > 0.25:
                    print(f"    z={z:.3f}  gap={'--' if g is None else f'{g*1000:.0f}mm'}  domeshift={shift:+.1f}px")
                    last_print = now
                if abs(z - target) < 0.0015:
                    break
                time.sleep(0.01)
            z = float(fk_T(state.get_q())[2, 3])
            comp = (contact_z - z) * 1000 if contact_z is not None else None
            print(f"  ended z={z:.3f}; contact_z={contact_z}; cup compression ~"
                  f"{'?' if comp is None else f'{comp:.1f}mm'}")
            time.sleep(args.dwell)
            goto([cxy[0], cxy[1], surf + args.standoff + 0.03], f"lift{i}", 10.0)

        to_base()
    finally:
        mon.stop_evt.set(); mon.join(timeout=3)
        print("== done ==")
        rclpy.shutdown()


if __name__ == "__main__":
    main()
