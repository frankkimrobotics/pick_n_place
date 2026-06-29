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
        self.rec = None                                          # set to [] to record (t, jpg, depth-jpg, gap)
        self.templ = None; self.sbox = None; self.base_y = None  # plunger-dot template tracker (contact signal)
        self._bluey = None; self._bluedy = None; self._bluescore = 0.0

    def set_bluedot_template(self, templ_gray, sbox, base_y):
        self.templ = templ_gray; self.sbox = [int(v) for v in sbox]; self.base_y = float(base_y)

    def latest_bluey(self):
        with self.lock:
            return self._bluey, self._bluedy                     # (current y, dy = base_y - y; +up = contact)

    def start_rec(self):
        with self.lock:
            self.rec = []

    def stop_rec(self):
        with self.lock:
            r = self.rec; self.rec = None
        return r or []

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
                # PLUNGER-DOT tracking by TEMPLATE MATCHING (normalized -> robust to the dimming as
                # the cup nears the object). The dot rises (y decreases) when the cup retracts on contact.
                bluey = bluedy = None; bscore = 0.0
                if self.templ is not None:
                    sx0, sy0, sx1, sy1 = self.sbox
                    gs = cv2.cvtColor(rgb[sy0:sy1, sx0:sx1], cv2.COLOR_RGB2GRAY)
                    th, tw = self.templ.shape
                    if gs.shape[0] >= th and gs.shape[1] >= tw:
                        res = cv2.matchTemplate(gs, self.templ, cv2.TM_CCOEFF_NORMED)
                        _, bscore, _, loc = cv2.minMaxLoc(res)
                        if bscore >= 0.40:
                            bluey = float(loc[1] + sy0 + th / 2.0); bluedy = self.base_y - bluey
                tnow = time.time()
                with self.lock:
                    self._rgbd = (rgb, depth, self.K); self._gap = g; self._cupd = cupd
                    self._domey = domey; self._t = tnow; self.n += 1
                    self._bluey = bluey; self._bluedy = bluedy; self._bluescore = bscore
                    recording = self.rec is not None
                if recording:
                    okc, cb = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
                    dcol = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=425.0), cv2.COLORMAP_JET)
                    okd, db = cv2.imencode(".jpg", dcol, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if okc and okd:
                        with self.lock:
                            if self.rec is not None:
                                self.rec.append((tnow, cb.tobytes(), db.tobytes(), g))
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
    ap.add_argument("--v-des", type=float, default=2.5, help="descent speed past pregrasp (deg/s); keep slow — a fast"
                    " descent slams objects taller than the (noisy) detection and false-triggers the blue-dot")
    ap.add_argument("--v-contact", type=float, default=2.0, help="SLOW speed near contact (deg/s)")
    ap.add_argument("--slow-frac", type=float, default=0.5, help="fraction of the descent that is the slow contact zone")
    ap.add_argument("--weld-dt", type=float, default=0.06, help="uniform traj_dt of the re-timed weld (s)")
    ap.add_argument("--weld-ramp", type=float, default=0.12, help="junction decel window in joint arc-length (rad)")
    ap.add_argument("--gap-contact", type=float, default=0.005, help="raw gap (m) at which to HOLD = contact")
    ap.add_argument("--blue-dot", default=os.path.join(C.OUT_DIR, "blue_dot_mask.npz"))
    ap.add_argument("--blue-thresh", type=float, default=3.0, help="blue-dot rise (px) = contact")
    ap.add_argument("--post-dwell", type=float, default=0.5, help="wait after contact before returning (s)")
    ap.add_argument("--return-vel", type=float, default=20.0, help="welded-return per-joint peak (deg/s)")
    # pick-and-place mode: at contact -> suction ON -> welded move to place pose -> release
    ap.add_argument("--pick-place", action="store_true", help="grasp+place each object instead of touch-only")
    ap.add_argument("--place", default="0.10,0.40,0.30", help="release pose x,y,z (cup tip over the bin)")
    ap.add_argument("--box", default="0.10,0.40,0.0", help="place box bottom-centre x,y,z")
    ap.add_argument("--box-size", type=float, default=0.25, help="box height; rim_z = box_z + this")
    ap.add_argument("--table-z", type=float, default=0.015, help="table top height (for carried-object hang extent)")
    ap.add_argument("--release-clear", type=float, default=0.05, help="object bottom clears the rim by this at release")
    ap.add_argument("--place-vel", type=float, default=20.0, help="welded place-move per-joint peak (deg/s)")
    ap.add_argument("--seal-dwell", type=float, default=0.5, help="suction-seal dwell at contact (s, the only v=0)")
    ap.add_argument("--record", action="store_true", help="record per-object RGBD+joints (adds save time; off for fast cycle)")
    ap.add_argument("--suction-host", default="10.0.0.27"); ap.add_argument("--suction-user", default="pi")
    ap.add_argument("--dome-shift", type=float, default=6.0, help="dome image shift (px) flagged as contact")
    ap.add_argument("--detect-tries", type=int, default=3)
    ap.add_argument("--detect-only", action="store_true", help="list detected objects and exit (no motion)")
    ap.add_argument("--flat-target", dest="flat_target", action="store_true", default=True,
                    help="aim the cup at the flattest sealable patch (suction-point detector) not the centroid")
    ap.add_argument("--no-flat-target", dest="flat_target", action="store_false")
    ap.add_argument("--angled", action="store_true", help="approach along the face normal (tilted) not straight down")
    ap.add_argument("--max-objects", type=int, default=1, help="touch at most N objects (largest first)")
    ap.add_argument("--pick-index", type=int, default=0, help="start at the Nth largest object (skip earlier)")
    ap.add_argument("--grasp-press", type=float, default=0.006, help="press past contact for the suction seal (m)")
    ap.add_argument("--dwell", type=float, default=1.2)
    ap.add_argument("--rec-dir", default=os.path.join(C.OUT_DIR, "touch_rec"),
                    help="record joint pos/vel + RGBD frames per object here")
    args = ap.parse_args()

    import rclpy
    from std_msgs.msg import String
    from sensor_msgs.msg import JointState
    from object_pointclouds import deproject_mask
    from capture_and_plot import segment
    from multiview_fuse import pick_res
    from perturb_loop import PlannerClient, RobotState, execute, scale_traj
    from joint_conventions import rad_to_linuxcnc_deg
    import suction_test
    from real_grasp import estimate_normals, detect_suction_point
    place_xyz = [float(v) for v in args.place.split(",")]
    box_xyz = [float(v) for v in args.box.split(",")]

    cup = np.load(args.cup); rim = cup["rim"].astype(bool); ann = cup["ring"].astype(bool)
    dome = cup["dome"].astype(bool) if "dome" in cup.files else cup["mask"].astype(bool)
    print(f"cup ROI: rim {int(rim.sum())}px, dome arc {int(ann.sum())}px, dome {int(dome.sum())}px")
    CH, CW = rim.shape                     # match the cup-mask resolution (D405 may default to 848x480)
    mon = GapMonitor(args.serial, CW, CH, rim, ann, dome); mon.start()
    t0 = time.time()
    while not mon.ok and time.time() - t0 < 8:
        time.sleep(0.2)
    if not mon.ok:
        print("ABORT: depth stream failed"); return

    pc = PlannerClient(); rclpy.init(); node = rclpy.create_node("servo_touch")
    pub = node.create_publisher(String, "/mycobot/cmd/move", 10); state = RobotState(node)
    track = {"ramp_time": 0.15, "pos_gain": 1.0, "vff_scale": 1.0}
    # joint recorder: /mycobot/drive_feedback carries measured position (deg) + velocity (deg/s) @50Hz
    jrec = []; jrec_on = {"v": False}
    def _on_drive(msg):
        if jrec_on["v"]:
            jrec.append((time.time(), [float(p) for p in msg.position], [float(v) for v in msg.velocity]))
    node.create_subscription(JointState, "/mycobot/drive_feedback", _on_drive, 50)
    down = list(R_to_quat_wxyz(R_from_two_axes(np.array([0, 0, -1.0]))))

    def fk_T(q):
        r = pc.rpc({"type": "fk", "q": list(map(float, q))})
        return make_T(quat_wxyz_to_R(np.array(r["quat"][0])), np.array(r["pos"][0]))

    def fix_j6(t):
        t = np.array(t, float); t[:, 5] = t[0, 5]; return t

    def send_descent_nb(goal, vmax, quat=None):
        """Plan current -> goal pose and publish NON-BLOCKING at a slow vmax. `quat` (wxyz) sets the
        approach orientation (default = straight-down `down`); pass a normal-aligned quat for angled.
        The cuRobo trajectory decelerates to rest AT goal; vision HOLD just stops it earlier."""
        q = state.get_q()
        r = pc.plan_pose(list(map(float, q)), list(map(float, goal)) + (quat or down), max_attempts=14)
        if not r.get("success"):
            return False
        traj = fix_j6(r["trajectory"]); sdt, _ = scale_traj(traj, r["dt"], vmax, 0.5)
        td = [list(map(float, rad_to_linuxcnc_deg(wp))) for wp in traj]
        pub.publish(String(data=json.dumps({"trajectory": td, "traj_dt": sdt, "target_deg": td[-1],
                                            "controller": "pid", **track})))
        return True

    def _retime(path, junc, v_app, v_des, dt_ref, ramp, v_end=None, end_frac=0.0):
        """Re-parametrize a concatenated joint path with a SMOOTH velocity profile: cruise at
        v_app, ramp down to v_des just before the junction, hold v_des, then (optionally) ramp
        down to v_end over the last `end_frac` of the post-junction segment so the final approach
        to contact is slow/gentle while the rest of the descent is fast. v_* in rad/s."""
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]  # arc-length (rad)
        if d[-1] < 1e-6:
            return path[:1].copy(), 0.0
        dj = d[junc]; de = d[-1]
        if v_end is None:
            v_end = v_des
        es = de - max(0.0, end_frac) * (de - dj)                     # start of the slow contact zone
        sg = np.linspace(0, de, 4000)
        vg = np.where(sg < dj - ramp, v_app,
              np.where(sg < dj, v_app + (v_des - v_app) * (sg - (dj - ramp)) / max(ramp, 1e-6),
              np.where(sg < es, v_des,
                       v_des + (v_end - v_des) * (sg - es) / max(de - es, 1e-6))))     # ramp to v_end near contact
        vg = np.maximum(vg, 1e-3)
        tg = np.r_[0.0, np.cumsum(np.diff(sg) / vg[:-1])]            # cumulative time along the path
        ts = np.arange(0, tg[-1], dt_ref)
        ss = np.interp(ts, tg, sg)                                   # arc-length at each uniform-time tick
        out = np.column_stack([np.interp(ss, d, path[:, j]) for j in range(6)])
        junc_t = float(np.interp(dj, sg, tg))                        # time when the path reaches the junction
        return out, junc_t

    def send_weld_descent(pregrasp, grasp, v_app_deg, v_des_deg, dt_ref, ramp, v_contact_deg=None, slow_frac=0.0, quat=None):
        """ONE velocity-continuous trajectory current -> pregrasp(non-zero v) -> grasp(decel-to-rest).
        Descends fast at v_des, slowing to v_contact over the last `slow_frac` of the descent.
        `quat` (wxyz) is the approach orientation for both poses (default straight-down)."""
        q = state.get_q()
        r1 = pc.plan_pose(list(map(float, q)), list(map(float, pregrasp)) + (quat or down), max_attempts=14)
        if not r1.get("success"):
            return None
        r2 = pc.plan_pose([float(x) for x in r1["trajectory"][-1]], list(map(float, grasp)) + (quat or down), max_attempts=14)
        if not r2.get("success"):
            return None
        p1 = np.array(r1["trajectory"]); p2 = np.array(r2["trajectory"])
        # cap the approach speed to cuRobo's torque-FEASIBLE native peak (never speed past its plan);
        # forcing a higher flat peak races motor-cmd past feedback -> following-error power-off.
        nap = float(np.degrees(np.max(np.abs(np.diff(p1, axis=0))) / r1["dt"])) if len(p1) > 1 else v_app_deg
        v_app_eff = min(v_app_deg, 0.9 * nap)
        path = np.vstack([p1, p2[1:]]); junc = len(p1) - 1
        vc = np.radians(v_contact_deg) if v_contact_deg is not None else None
        traj, junc_t = _retime(path, junc, np.radians(v_app_eff), np.radians(v_des_deg), dt_ref, ramp,
                               v_end=vc, end_frac=slow_frac)
        traj = fix_j6(traj)
        peak = float(np.abs(np.diff(traj, axis=0)).max()) / dt_ref   # rad/s, per-joint
        if np.degrees(peak) > 56.0:                                  # firmware ceiling guard
            print(f"  WELD peak {np.degrees(peak):.0f}deg/s > ceiling; aborting"); return None
        td = [list(map(float, rad_to_linuxcnc_deg(wp))) for wp in traj]
        pub.publish(String(data=json.dumps({"trajectory": td, "traj_dt": dt_ref, "target_deg": td[-1],
                                            "controller": "pid", **track})))
        return {"n": len(traj), "junc_t": junc_t, "T": len(traj) * dt_ref, "peak_degs": np.degrees(peak)}

    def send_weld_return(lift_xyz, target_vel, dt_ref, ramp):
        """ONE velocity-continuous return: current -> lift(non-zero v via-point) -> base(decel-to-rest).
        `target_vel` is the desired PER-JOINT peak (deg/s); the welded path is uniformly re-timed to hit
        it, capped to cuRobo's torque-feasible native peak, and run with a gentle accel ramp."""
        q = state.get_q()
        r1 = pc.plan_pose(list(map(float, q)), list(map(float, lift_xyz)) + down, max_attempts=14)
        if not r1.get("success"):
            return False
        r2 = pc.plan_joint([float(x) for x in r1["trajectory"][-1]], list(map(float, C.BASE_Q)))
        if not r2.get("success"):
            return False
        p1 = np.array(r1["trajectory"]); p2 = np.array(r2["trajectory"])
        nmv = float(np.degrees(np.max(np.abs(np.diff(p2, axis=0))) / r2["dt"])) if len(p2) > 1 else target_vel
        path = np.vstack([p1, p2[1:]]); junc = len(p1) - 1
        _rt, junc_t = _retime(path, junc, np.radians(10.0), np.radians(15.0), dt_ref, ramp)       # profile shape
        traj = fix_j6(_rt)
        peak0 = float(np.degrees(np.max(np.abs(np.diff(traj, axis=0)))) / dt_ref)                 # per-joint @ dt_ref
        target = min(target_vel, 0.9 * nmv)                     # per-joint peak target, torque-capped
        eff_dt = dt_ref * peak0 / max(target, 1.0)              # uniform re-time so per-joint peak == target
        td = [list(map(float, rad_to_linuxcnc_deg(wp))) for wp in traj]
        pub.publish(String(data=json.dumps({"trajectory": td, "traj_dt": eff_dt, "target_deg": td[-1],
                                            "controller": "pid", "ramp_time": 0.35, "pos_gain": 1.0, "vff_scale": 1.0})))
        dur = len(traj) * eff_dt; t0 = time.time(); dev = 99.0
        while time.time() - t0 < dur + 2.0:                      # poll until settled (don't false-fail on follow-lag)
            rclpy.spin_once(node, timeout_sec=0.02)
            if time.time() - t0 > dur - 0.3:
                dev = float(np.degrees(np.abs(state.get_q()[:6] - C.BASE_Q)).max())
                if dev < 6.0:
                    break
        print(f"  weld-return: {len(traj)} wpts ~{dur:.1f}s, lift@{junc_t*eff_dt/dt_ref:.1f}s, "
              f"peak {target:.0f}deg/s (native {nmv:.0f}), final dev {dev:.1f}deg {'OK' if dev < 8 else 'INCOMPLETE'}")
        return dev < 8.0

    def send_weld_place(lift_xyz, place_pose, target_vel, dt_ref, ramp):
        """ONE velocity-continuous carry: current -> lift(non-zero v via-point) -> place pose (Cartesian,
        decel-to-rest). Re-timed to a torque-feasible per-joint peak with a gentle accel ramp."""
        q = state.get_q()
        r1 = pc.plan_pose(list(map(float, q)), list(map(float, lift_xyz)) + down, max_attempts=14)
        if not r1.get("success"):
            return False
        r2 = pc.plan_pose([float(x) for x in r1["trajectory"][-1]], list(map(float, place_pose)) + down, max_attempts=14)
        if not r2.get("success"):
            return False
        p1 = np.array(r1["trajectory"]); p2 = np.array(r2["trajectory"])
        nmv = float(np.degrees(np.max(np.abs(np.diff(p2, axis=0))) / r2["dt"])) if len(p2) > 1 else target_vel
        path = np.vstack([p1, p2[1:]]); junc = len(p1) - 1
        _rt, junc_t = _retime(path, junc, np.radians(10.0), np.radians(15.0), dt_ref, ramp)
        traj = fix_j6(_rt)
        peak0 = float(np.degrees(np.max(np.abs(np.diff(traj, axis=0)))) / dt_ref)
        target = min(target_vel, 0.9 * nmv); eff_dt = dt_ref * peak0 / max(target, 1.0)
        td = [list(map(float, rad_to_linuxcnc_deg(wp))) for wp in traj]
        pub.publish(String(data=json.dumps({"trajectory": td, "traj_dt": eff_dt, "target_deg": td[-1],
                                            "controller": "pid", "ramp_time": 0.35, "pos_gain": 1.0, "vff_scale": 1.0})))
        dur = len(traj) * eff_dt; t0 = time.time(); err = 99.0
        while time.time() - t0 < dur + 2.0:
            rclpy.spin_once(node, timeout_sec=0.02)
            if time.time() - t0 > dur - 0.3:
                err = float(np.linalg.norm(fk_T(state.get_q())[:3, 3] - np.array(place_pose)))
                if err < 0.02:
                    break
        print(f"  weld-place: {len(traj)} wpts ~{dur:.1f}s, peak {target:.0f}deg/s (native {nmv:.0f}), "
              f"tcp err {err*1000:.0f}mm {'OK' if err < 0.03 else 'INCOMPLETE'}")
        return err < 0.03

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

    def _save_rec(base, idx, jr, frames, marks, info):
        import shutil
        d = os.path.join(base, f"obj{idx:02d}")
        if os.path.isdir(d):
            shutil.rmtree(d)
        fd = os.path.join(d, "frames"); os.makedirs(fd, exist_ok=True)
        t0 = marks.get("approach") or (jr[0][0] if jr else (frames[0][0] if frames else 0.0))
        if jr:
            np.savez(os.path.join(d, "joints.npz"),
                     t=np.array([r[0] for r in jr]) - t0,
                     pos=np.array([r[1][:6] for r in jr], float),
                     vel=np.array([r[2][:6] for r in jr], float))
        with open(os.path.join(d, "frames.jsonl"), "w") as fh:
            for k, (tf, cb, db, g) in enumerate(frames):
                open(os.path.join(fd, f"{k:04d}_c.jpg"), "wb").write(cb)
                open(os.path.join(fd, f"{k:04d}_d.jpg"), "wb").write(db)
                fh.write(json.dumps({"idx": k, "t": tf - t0,
                                     "gap": None if g is None else round(g, 4)}) + "\n")
        json.dump({"marks": {k: v - t0 for k, v in marks.items()}, **info,
                   "n_joint": len(jr), "n_frame": len(frames)},
                  open(os.path.join(d, "meta.json"), "w"), indent=2)
        print(f"  recorded -> {d}: {len(jr)} joint samples, {len(frames)} frames")

    try:
        to_base(); time.sleep(0.3)
        # grab the plunger-dot template at rest (current lighting) for robust template tracking
        if os.path.exists(args.blue_dot):
            bd = np.load(args.blue_dot); cyx = bd["centroid"]; rr = mon.latest_rgbd()
            if rr is not None and not np.isnan(cyx[0]):
                rgb0 = rr[0]; cx, cyc = int(round(cyx[0])), int(round(cyx[1])); hw = 18
                templ = cv2.cvtColor(rgb0[cyc - hw:cyc + hw, cx - hw:cx + hw], cv2.COLOR_RGB2GRAY)
                sbox = [cx - 30, cyc - 50, cx + 30, cyc + 20]    # search strip extends UP (dot rises on contact)
                mon.set_bluedot_template(templ, sbox, float(cyc))
                print(f"blue-dot template {templ.shape} set; base_y={cyc}, search {sbox}, "
                      f"contact when rise >= {args.blue_thresh:.0f}px")
            else:
                print("WARN: could not grab base frame for blue-dot template")
        else:
            print("WARN: no blue_dot_mask.npz — blue-dot contact disabled")
        objs = []; cam_base = None
        for _ in range(args.detect_tries):
            Tbc = fk_T(state.get_q()) @ make_T(np.eye(3), [0, 0, C.CAM_TCP_Z_SHIFT]) @ C.T_TCP_CAM
            cam_base = Tbc[:3, 3]                                 # camera position for normal orientation
            r = mon.latest_rgbd()
            if r is None:
                continue
            rgb, depth, K = r
            for o in detect_objects(rgb, depth, K, Tbc, segment, deproject_mask, args):
                if all(np.linalg.norm(o["centroid"][:2] - a["centroid"][:2]) > 0.04 for a in objs):
                    objs.append(o)
        objs.sort(key=lambda o: -len(o["pts"]))                  # largest (most prominent) first
        if args.detect_only:
            print(f"=== {len(objs)} objects detected ===")
            for j, o in enumerate(objs):
                print(f"  [{j}] centre [{o['centroid'][0]:+.2f},{o['centroid'][1]:+.2f}] "
                      f"h={o['height']*100:.0f}cm foot={o['foot']*100:.0f}cm flat={o['flatness']:.2f} n={o['n']}")
            return
        objs = objs[args.pick_index:args.pick_index + args.max_objects]
        print(f"detected objects; touching {len(objs)} (largest-first, from index {args.pick_index})")

        for i, o in enumerate(objs):
            P = o["pts"]; cxy = o["centroid"][:2].copy(); flat_ok = False
            nrm_v = np.array([0.0, 0.0, 1.0])                     # descent direction (vertical unless --angled)
            if args.flat_target and cam_base is not None:        # aim at the flattest sealable patch
                try:
                    nrm = estimate_normals(P, cam_base)
                    found = detect_suction_point(P, nrm, normal_cone_deg=45.0, select="central")
                    if found is not None:
                        cxy = np.asarray(found[0], float)[:2].copy(); flat_ok = True
                        nn = np.asarray(found[1], float); nn = nn / (np.linalg.norm(nn) + 1e-9)
                        if nn[2] < 0:
                            nn = -nn                              # outward (up-ish)
                        if args.angled:
                            nrm_v = nn                            # approach along the face normal
                        print(f"  flat suction point [{cxy[0]:.3f},{cxy[1]:.3f}] normal={np.round(nrm_v,2).tolist()}")
                except Exception as e:
                    print(f"  flat-target failed ({e}); using centroid")
            col = P[np.linalg.norm(P[:, :2] - cxy, axis=1) < 0.02]
            surf = float(np.median(col[:, 2])) if len(col) > 20 else float(np.percentile(P[:, 2], 80))
            if not flat_ok:                                      # centroid path: re-centre on the flat top
                face = P[np.abs(P[:, 2] - surf) < 0.010]
                if len(face) > 20:
                    cxy = face[:, :2].mean(0)
            # normal-aligned approach: cup oriented along -normal; pregrasp/grasp offset along the normal.
            # (vertical normal -> identical to the straight-down path; z-targets are the angled points' z.)
            sp3 = np.array([cxy[0], cxy[1], surf])
            q_n = list(R_to_quat_wxyz(R_from_two_axes(-nrm_v)))
            pregrasp3 = (sp3 + args.standoff * nrm_v).tolist()
            grasp3 = (sp3 - args.margin * nrm_v).tolist()
            standoff_z = pregrasp3[2]
            target = grasp3[2]
            floor_abs = max(0.012, float((sp3 - args.max_descend * nrm_v)[2]))
            print(f"\n== object {i+1}: centre [{cxy[0]:.3f},{cxy[1]:.3f}] surf={surf:.3f} "
                  f"normal={np.round(nrm_v,2).tolist()} -> grasp z {target:.3f} ==")

            # ONE velocity-continuous WELD: current -> pregrasp(non-zero v) -> expected grasp(decel-to-rest).
            # The pregrasp is a true via-point (no rest-to-rest stop); RGBD then fine-tunes the endpoint
            # via the gap-contact HOLD during the slow descent.
            marks = {}
            if args.record:
                jrec.clear(); jrec_on["v"] = True; mon.start_rec()           # RECORD pregrasp+contact
            marks["approach"] = t_send = time.time()
            wd = send_weld_descent(pregrasp3, grasp3,
                                   args.v_approach, args.v_des, args.weld_dt, args.weld_ramp,
                                   v_contact_deg=args.v_contact, slow_frac=args.slow_frac, quat=q_n)
            if wd is None:
                jrec_on["v"] = False; mon.stop_rec(); print("  weld plan FAILED"); continue
            marks["pregrasp"] = t_send + wd["junc_t"]                # via-point pass-through time (non-zero v)
            print(f"  weld: {wd['n']} wpts, ~{wd['T']:.1f}s, pregrasp@{wd['junc_t']:.1f}s, "
                  f"peak {wd['peak_degs']:.0f}deg/s")
            # (3) ride it down; gap is LOG-ONLY + a slow-overshoot safety stop if contact is seen
            #     well above target (object taller than detection thought).
            print("  welded decel-to-rest descent (gap = log/safety only):")
            contact_z = None; last_print = 0; t0 = time.time(); zhist = collections.deque(maxlen=8)
            blue_base2 = None                                    # blue-dot baseline re-captured at gate-open
            descent_target = target; grasp_cur = list(grasp3)   # descend-until-contact extends along the normal
            while time.time() - t0 < 35.0:
                rclpy.spin_once(node, timeout_sec=0.02)
                gap, _ = mon.latest_gap(); q = state.get_q()
                z = float(fk_T(q)[2, 3]) if q is not None else 9.0
                now = time.time(); zhist.append((now, z)); vz = 0.0
                if len(zhist) >= 4 and zhist[-1][0] > zhist[0][0]:
                    vz = max(0.0, (zhist[0][1] - zhist[-1][1]) / (zhist[-1][0] - zhist[0][0]))
                bluey, _ = mon.latest_bluey()
                past_pregrasp = now >= marks.get("pregrasp", now) and z < standoff_z + 0.012
                if past_pregrasp and bluey is not None and blue_base2 is None:
                    blue_base2 = bluey                                          # re-baseline (no contact yet)
                dy2 = (blue_base2 - bluey) if (blue_base2 is not None and bluey is not None) else None
                # blue-dot plunger rise is the ONLY contact trigger (gap under-reads near contact and
                # false-fires above the object; it stays as a log signal only). Endpoint + floor = fallback.
                blue_hit = dy2 is not None and dy2 >= args.blue_thresh
                if past_pregrasp and contact_z is None and blue_hit:
                    contact_z = z; marks["contact"] = time.time()
                    send_hold(); print(f"  >> CONTACT (blue-dot {dy2:+.1f}px) HOLD at z={z:.3f} "
                                       f"vz={vz*1000:.0f}mm/s"); break
                if now - last_print > 0.2:
                    print(f"    z={z:.3f}  vz={vz*1000:.0f}mm/s  gap={'--' if gap is None else f'{gap*1000:.0f}mm'}  "
                          f"blue_dy={'--' if dy2 is None else f'{dy2:+.1f}px'}"); last_print = now
                if abs(z - descent_target) < 0.002 and vz < 0.002:             # reached target without contact
                    nxt = np.array(grasp_cur) - 0.008 * nrm_v                  # DESCEND-UNTIL-CONTACT: step along -normal
                    if float(nxt[2]) > floor_abs:
                        grasp_cur = nxt.tolist(); descent_target = grasp_cur[2]
                        send_descent_nb(grasp_cur, args.v_contact, quat=q_n)
                        print(f"  no contact yet; extending descent -> z={descent_target:.3f}")
                    else:
                        send_hold(); print("  reached floor, no contact"); break
                if z <= floor_abs + 1e-3:
                    send_hold(); print("  hit hard floor"); break
                time.sleep(0.01)
            z = float(fk_T(state.get_q())[2, 3])
            print(f"  ended z={z:.3f}; contact_z(gap)={contact_z}")
            if args.record:                                    # optional: record + save the contact settle
                tdw = time.time()
                while time.time() - tdw < args.post_dwell:
                    rclpy.spin_once(node, timeout_sec=0.02)
                jrec_on["v"] = False; frames = mon.stop_rec()
                _save_rec(args.rec_dir, i, jrec, frames, marks, dict(
                    surf=float(surf), target=float(target),
                    contact_z=(None if contact_z is None else float(contact_z)),
                    cxy=[float(cxy[0]), float(cxy[1])]))
            Tc = fk_T(state.get_q())
            if args.pick_place:
                # brief settle of the contact HOLD, then press from REST to the seal depth (poll until reached)
                tset = time.time()
                while time.time() - tset < 0.35:
                    rclpy.spin_once(node, timeout_sec=0.02)
                Tc = fk_T(state.get_q())
                if args.grasp_press > 0:
                    press_pos = (Tc[:3, 3] - args.grasp_press * nrm_v).tolist()  # press along -normal
                    pz = press_pos[2]
                    send_descent_nb(press_pos, 1.2, quat=q_n)
                    tp = time.time()
                    while time.time() - tp < 1.6:
                        rclpy.spin_once(node, timeout_sec=0.02)
                        if abs(float(fk_T(state.get_q())[2, 3]) - pz) < 0.0015:
                            break
                    Tc = fk_T(state.get_q()); print(f"  grasp-press -> z={float(Tc[2, 3]):.3f}")
                # GRASP: suction ON (cup pressed on the object), seal dwell = the only v=0
                suction_test.set_pin(True, args.suction_host, args.suction_user); print("  suction ON; sealing")
                ts = time.time()
                while time.time() - ts < args.seal_dwell:
                    rclpy.spin_once(node, timeout_sec=0.02)
                # COLLISION-AWARE place: the carried object hangs ~(contact_z - table) below the tip,
                # so lift it straight up CLEAR of the box rim, carry it over the box at that height, and
                # release above the rim -> the object volume can't clip the box.
                down_extent = max(0.02, float(Tc[2, 3]) - args.table_z)
                rim_z = box_xyz[2] + args.box_size
                transit_z = min(0.42, rim_z + args.release_clear + down_extent)
                lift_xyz = [float(Tc[0, 3]), float(Tc[1, 3]), transit_z]          # straight up, clear of rim
                place_pose = [box_xyz[0], box_xyz[1], transit_z]                  # carry over the box, high
                print(f"  carried object hangs ~{down_extent*100:.0f}cm -> transit/release z={transit_z:.2f} "
                      f"(rim {rim_z:.2f})")
                # WELDED carry: contact -> lift(non-zero v) -> place pose over the bin
                send_weld_place(lift_xyz, place_pose, args.place_vel, args.weld_dt, args.weld_ramp)
                # RELEASE over the bin, let it drop
                suction_test.set_pin(False, args.suction_host, args.suction_user); print("  released")
                tr = time.time()
                while time.time() - tr < 0.3:
                    rclpy.spin_once(node, timeout_sec=0.02)
                # WELDED return to base from the place pose
                Tp = fk_T(state.get_q())
                send_weld_return([float(Tp[0, 3]), float(Tp[1, 3]), min(0.32, float(Tp[2, 3]) + 0.05)],
                                 args.return_vel, args.weld_dt, args.weld_ramp)
            else:
                # touch-only: WELDED return current -> lift(non-zero v) -> base (torque-feasible)
                lift_xyz = [float(Tc[0, 3]), float(Tc[1, 3]), min(0.24, float(Tc[2, 3]) + 0.10)]
                if not send_weld_return(lift_xyz, args.return_vel, args.weld_dt, args.weld_ramp):
                    print("  weld-return failed -> segmented fallback")
                    goto(lift_xyz, "lift", 12.0); to_base()

        to_base()
    finally:
        mon.stop_evt.set(); mon.join(timeout=3)
        print("== done ==")
        rclpy.shutdown()


if __name__ == "__main__":
    main()
