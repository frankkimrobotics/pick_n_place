#!/usr/bin/env python3
"""real_pipeline :: detect ALL tabletop objects, pick-and-place them one by one, and
RECORD the full data stream for every object's pick-and-place.

For each object, while it is being picked and placed, a single D405 owner streams
aligned colour+depth at 30 Hz and, for every frame, writes:
  * the RGB frame (frames/NNNNNN.jpg)
  * the joint angles at that instant
  * the suction switch state (0/1)
  * a wall-clock timestamp
all synchronised by frame index in states.jsonl. High-level events (suction on/off,
contact, lift, release ...) are also timestamped in events.jsonl.

Design:
  * ONE D405 owner: the Recorder thread keeps the realsense pipeline open and exposes
    the latest aligned (rgb, depth, K) frame. The pick/place logic reads that latest
    frame instead of opening its own capture (which would fight the recorder for the
    USB device). This is the key difference vs real_multi.py (which captures per call).
  * The Recorder samples joints from the shared RobotState (updated by execute()'s ROS
    spinning during motion) and the shared suction state -- so each saved frame carries
    a consistent (rgb, joints, suction, t).
  * Placement centres the OBJECT's centroid over the bin (not the tcp) and releases with
    a conservative clearance ABOVE the rim (no descent into the bin) so a large/tilted
    object can't clip the bin wall -- the failure seen when placing the brown box.

Reuses real_multi.detect_objects/grasp_held, real_grasp.estimate_normals/
detect_suction_point, collision.obb_vs_walls, sim_planner.box_wall_obstacles,
perturb_loop.PlannerClient/RobotState/execute, suction_test.

Run (ROS2 env, D405 on PYTHONPATH, planner+bridge+SAM3 up):
  source /opt/ros/humble/setup.bash
  PYTHONPATH=~/librealsense/build/release:$PYTHONPATH \
    python3 pick_and_place/real_pipeline.py --suction-host 10.0.0.27 --max-objects 8
"""
import argparse, json, os, sys, threading, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")),
          os.path.abspath(os.path.join(HERE, "..", "ros2node", "perception"))):
    sys.path.insert(0, p)
import config as C
from geometry import (R_from_two_axes, R_to_quat_wxyz, quat_wxyz_to_R, make_T,
                      transform_points)
from collision import obb_vs_walls
from sim_planner import box_wall_obstacles
from real_grasp import estimate_normals, detect_suction_point
from real_multi import detect_objects, grasp_held


class Recorder(threading.Thread):
    """Single D405 owner: streams aligned colour+depth at `fps`, keeps the latest frame
    for the pick/place logic, and (while a clip is open) saves every frame as a jpg plus
    a synchronised states.jsonl row {idx, t, joints, suction}."""
    def __init__(self, serial, w, h, fps, get_joints, get_suction):
        super().__init__(daemon=True)
        self.serial, self.w, self.h, self.fps = serial, w, h, fps
        self.get_joints, self.get_suction = get_joints, get_suction
        self.stop_evt = threading.Event()
        self.lock = threading.Lock()
        self._latest = None                # (rgb, depth_m, K, t_wall)
        self.K = None
        self.ok = False
        self.dropped = 0
        # clip state
        self.saving = False
        self.frame_dir = None
        self.manifest = None
        self.idx = 0

    def run(self):
        import pyrealsense2 as rs
        import cv2
        pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, self.fps)
        cfg.enable_stream(rs.stream.depth, self.w, self.h, rs.format.z16, self.fps)
        try:
            profile = pipe.start(cfg)
        except Exception as e:
            print(f"[recorder] D405 start failed: {e}"); return
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.0]])
        self.ok = True
        try:
            while not self.stop_evt.is_set():
                try:
                    frames = align.process(pipe.wait_for_frames(1000))
                except RuntimeError:
                    self.dropped += 1; continue
                c = frames.get_color_frame(); d = frames.get_depth_frame()
                if not c or not d:
                    continue
                bgr = np.asanyarray(c.get_data())
                depth = np.asanyarray(d.get_data()).astype(np.float32) * scale
                t = time.time()
                with self.lock:
                    self._latest = (bgr[:, :, ::-1].copy(), depth, self.K, t)
                if self.saving:
                    j = self.get_joints(); s = self.get_suction()
                    cv2.imwrite(os.path.join(self.frame_dir, f"{self.idx:06d}.jpg"), bgr)
                    self.manifest.write(json.dumps({
                        "idx": self.idx, "t": t,
                        "joints": None if j is None else [float(x) for x in j[:6]],
                        "suction": int(s)}) + "\n")
                    self.manifest.flush()
                    self.idx += 1
        finally:
            pipe.stop()

    def latest(self, fresh_after=None, timeout=2.0):
        """Most recent (rgb, depth, K, t). If fresh_after is given, wait for a frame
        captured after that wall-time (so detection uses a frame from the current pose)."""
        t0 = time.time(); L = None
        while time.time() - t0 < timeout:
            with self.lock:
                L = self._latest
            if L is not None and (fresh_after is None or L[3] > fresh_after):
                return L
            time.sleep(0.01)
        return L

    def start_clip(self, clip_dir):
        os.makedirs(os.path.join(clip_dir, "frames"), exist_ok=True)
        with self.lock:
            self.frame_dir = os.path.join(clip_dir, "frames")
            self.manifest = open(os.path.join(clip_dir, "states.jsonl"), "w")
            self.idx = 0
            self.saving = True
        return clip_dir

    def stop_clip(self):
        with self.lock:
            self.saving = False
            n = self.idx
            if self.manifest:
                self.manifest.close(); self.manifest = None
        return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="218622271300")
    ap.add_argument("--max-objects", type=int, default=8, help="safety cap on objects to clear")
    ap.add_argument("--max-h", type=float, default=0.13)
    ap.add_argument("--max-foot", type=float, default=0.16)
    ap.add_argument("--xmin", type=float, default=0.15); ap.add_argument("--xmax", type=float, default=0.50)
    ap.add_argument("--ymin", type=float, default=-0.28); ap.add_argument("--ymax", type=float, default=0.30)
    ap.add_argument("--box", default="0.1,0.4,0.0"); ap.add_argument("--box-size", type=float, default=0.25)
    ap.add_argument("--grasp-dz", type=float, default=0.005)
    ap.add_argument("--contact-steps", type=int, default=8)
    ap.add_argument("--contact-thresh", type=float, default=0.007)
    ap.add_argument("--margin", type=float, default=0.005)
    ap.add_argument("--seal-press", type=float, default=0.003)
    ap.add_argument("--pre-dist", type=float, default=0.06)
    ap.add_argument("--near-gap", type=float, default=0.01,
                    help="open-loop descend (using the base-pose depth) to this height above "
                         "the detected surface, THEN take one close-range depth update + press")
    ap.add_argument("--max-descend", type=float, default=0.03)
    ap.add_argument("--min-flat", type=float, default=0.05)
    ap.add_argument("--high-z", type=float, default=0.40)
    ap.add_argument("--rim-clear", type=float, default=0.05,
                    help="release the object this far ABOVE the bin rim (no descent into "
                         "the bin) so a large/tilted object can't clip the wall")
    ap.add_argument("--vmax", type=float, default=45.0)
    ap.add_argument("--contact-vmax", type=float, default=8.0)
    ap.add_argument("--suction-host", default=os.environ.get("ROBOT_IP", "").strip())
    ap.add_argument("--suction-user", default="pi")
    ap.add_argument("--dry-run", action="store_true", help="detect only, no motion/suction")
    args = ap.parse_args()
    BOX = np.array([float(v) for v in args.box.split(",")])
    rim_z = float(BOX[2]) + args.box_size

    import rclpy
    from std_msgs.msg import String
    from sensor_msgs.msg import JointState  # noqa: F401 (RobotState subscribes)
    from capture_and_plot import segment
    from object_pointclouds import deproject_mask
    from multiview_fuse import pick_res
    from perturb_loop import PlannerClient, RobotState, execute
    import suction_test

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(C.OUT_DIR, f"pipeline_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"== run dir: {run_dir} ==")

    pc = PlannerClient()
    rclpy.init(); node = rclpy.create_node("real_pipeline")
    pub = node.create_publisher(String, "/mycobot/cmd/move", 10); state = RobotState(node)
    track = {"ramp_time": 0.15, "pos_gain": 1.0, "vff_scale": 1.0}
    down = list(R_to_quat_wxyz(R_from_two_axes(np.array([0, 0, -1.0]))))
    CW, CH = pick_res(args.serial)
    suction = {"on": 0}

    def set_suction(v, ev=None):
        suction_test.set_pin(v, args.suction_host or None, args.suction_user)
        suction["on"] = int(v)
        if ev is not None:
            ev(f"suction_{'on' if v else 'off'}")

    rec = Recorder(args.serial, CW, CH, 30, state.get_q, lambda: suction["on"])
    rec.start()
    t0 = time.time()
    while not rec.ok and time.time() - t0 < 8:
        time.sleep(0.2)
    if not rec.ok:
        print("ABORT: D405 recorder failed to start"); rclpy.shutdown(); return
    print("recorder streaming.")

    # ---------- helpers (mirror real_multi, but read frames from the recorder) ----------
    def cap():
        L = rec.latest(fresh_after=time.time())          # force a frame from NOW
        if L is None:
            L = rec.latest()
        return L[0], L[1], L[2]

    def fk_T(q):
        r = pc.rpc({"type": "fk", "q": list(map(float, q))})
        return make_T(quat_wxyz_to_R(np.array(r["quat"][0])), np.array(r["pos"][0]))

    def fk_many(qs):
        r = pc.rpc({"type": "fk", "q": [list(map(float, q)) for q in qs]})
        return [make_T(quat_wxyz_to_R(np.array(r["quat"][i])), np.array(r["pos"][i]))
                for i in range(len(qs))]

    def cam_pose(q):
        return fk_T(q) @ make_T(np.eye(3), [0, 0, C.CAM_TCP_Z_SHIFT]) @ C.T_TCP_CAM

    def fix_j6(traj):
        t = np.array(traj, float); t[:, 5] = t[0, 5]; return t

    def goto(goal, label, vmax=None):
        q = state.get_q()
        r = pc.plan_pose(list(map(float, q)), goal, max_attempts=12)
        if not r.get("success"):
            print(f"  [{label}] PLAN FAILED {r.get('status')}"); return None
        ex = execute(state, pub, fix_j6(r["trajectory"]), r["dt"], "pid",
                     vmax or args.vmax, 2.0, label, track=track)
        print(f"  [{label}] {'OK' if ex.get('ok') else 'FAIL'} err={ex.get('reach_err'):.1f}")
        return r["trajectory"][-1] if ex.get("ok") else None

    def _descend_to(newz, label):
        q = state.get_q(); Tbt = fk_T(q)
        goal = [float(Tbt[0, 3]), float(Tbt[1, 3]), float(newz)] + down
        r = pc.plan_pose(list(map(float, q)), goal, max_attempts=8)
        if not r.get("success"):
            print(f"  [{label}] PLAN FAILED"); return False
        execute(state, pub, fix_j6(r["trajectory"]), r["dt"], "pid",
                args.contact_vmax, 1.0, label, track=track)
        return True

    def measure_gap():
        q = state.get_q(); Tbt = fk_T(q); Tbc = cam_pose(q)
        rgb, depth, K = cap()
        tip = Tbt[:3, 3]
        pcam = transform_points(np.linalg.inv(Tbc), tip[None])[0]; tip_z = float(pcam[2])
        u = int(K[0, 0] * pcam[0] / pcam[2] + K[0, 2]); v = int(K[1, 1] * pcam[1] / pcam[2] + K[1, 2])
        Hh, Ww = depth.shape; w = 60
        sub = depth[max(0, v - w):min(Hh, v + w), max(0, u - w):min(Ww, u + w)]
        valid = sub[sub > 0.02]; surf = valid[valid > tip_z + 0.012]
        gap = 0.0 if len(surf) < 25 else float(np.percentile(surf, 25)) - tip_z
        return gap, float(Tbt[2, 3])

    def to_base(label="to-base"):
        rb = pc.plan_joint(list(map(float, state.get_q())), list(map(float, C.BASE_Q)))
        if rb.get("success"):
            execute(state, pub, fix_j6(rb["trajectory"]), rb["dt"], "pid", 25.0, 3.0, label, track=track)

    # --------------------------- per-object pick & place ---------------------------
    def pick_place(obj, clip_dir):
        events = open(os.path.join(clip_dir, "events.jsonl"), "w")

        def ev(name, **kw):
            events.write(json.dumps({"t": time.time(), "event": name, **kw}) + "\n"); events.flush()
            print(f"  * {name}")

        ev("begin", centroid=[float(x) for x in obj["centroid"]])
        Tbc = cam_pose(state.get_q())
        nrm = estimate_normals(obj["pts"], Tbc[:3, 3])
        found = detect_suction_point(obj["pts"], nrm, normal_cone_deg=45.0,
                                     select="central", centroid=obj["centroid"])
        if found is None:
            ev("no_suction_patch"); events.close(); return "no_patch"
        P, n = found
        ev("grasp_point", P=[float(x) for x in P])

        # carried-OBB (object) in tcp frame, vertical grasp at Pc
        Pc = P - np.array([0, 0, args.grasp_dz])
        Rg = R_from_two_axes(np.array([0, 0, -1.0]))
        Tbt_g = make_T(Rg, Pc)
        pts_tcp = transform_points(np.linalg.inv(Tbt_g), obj["pts"])
        lo, hi = pts_tcp.min(0), pts_tcp.max(0)
        obb_dims = (hi - lo) + 0.02; T_tcp_obb = make_T(np.eye(3), (lo + hi) / 2)
        inner = args.box_size - 0.02
        walls = [(np.array(p[:3]), np.array(d)) for d, p in
                 box_wall_obstacles(BOX, (inner, inner, args.box_size), 0.01).values()]
        walls.append((np.array([0, 0, -0.12]), np.array([2, 2, 0.04])))

        def sweep_clear(traj):
            for T in fk_many(traj):
                if obb_vs_walls(T @ T_tcp_obb, obb_dims, walls)[0]:
                    return False
            return True

        # PICK: pre-grasp -> open-loop approach to ~1cm above (using the BASE-pose depth)
        #       -> ONE close-range depth update -> suction ON -> press to contact -> lift.
        pre = list(map(float, P + [0, 0, args.pre_dist])) + down
        if goto(pre, "pre-grasp") is None:
            ev("pregrasp_fail"); events.close(); return "pregrasp_fail"
        surf = float(P[2])                                  # surface height from the base view
        floor_z = surf - args.max_descend
        if not _descend_to(max(floor_z, surf + args.near_gap), "approach"):   # open-loop, ~1cm above
            ev("approach_fail"); set_suction(0, ev); events.close(); return "approach_fail"
        gap, tcpz = measure_gap()                           # close-range, low-noise depth update
        ev("near_update", gap=float(gap), tcp_z=float(tcpz))
        print(f"  near update: gap={gap*1000:.0f} mm at tcp z={tcpz:.3f}")
        set_suction(1, ev)                                  # vacuum on just before contact
        if gap <= 0.0 or gap > args.near_gap + 0.03:        # surface not seen / implausible -> base depth
            target = max(floor_z, surf - args.seal_press)
        else:
            target = max(floor_z, tcpz - (gap + args.seal_press))
        _descend_to(target, "contact"); ev("contact")
        time.sleep(0.4)
        q = state.get_q(); Tbt = fk_T(q)
        goto([float(Tbt[0, 3]), float(Tbt[1, 3]), float(Tbt[2, 3] + 0.12)] + down, "lift")
        ev("lifted")

        # VERIFY grasp via depth at the tip
        time.sleep(0.5)
        qv = state.get_q(); rgb2, depth2, K2 = cap()
        held, tipd, tipz = grasp_held(depth2, K2, cam_pose(qv), fk_T(qv))
        ev("grasp_check", held=bool(held) if held is not None else None, tip_depth=tipd)
        print(f"  GRASP CHECK held={held} (tip depth={tipd})")
        if held is not True:
            set_suction(0, ev); events.close(); return "not_held"

        # PLACE: centre the OBJECT centroid over the bin; release ABOVE the rim
        place_off = obj["centroid"][:2] - P[:2]
        tgt = BOX[:2] - place_off
        down_extent = float(Pc[2] - obj["lo"][2])
        release_z = max(rim_z + down_extent + args.rim_clear, rim_z + 0.06)
        high_z = max(args.high_z, release_z)
        print(f"  place: tgt={np.round(tgt,3).tolist()} release_z={release_z:.3f} (hang~{down_extent*100:.0f}cm)")
        for xyz, label in [(np.r_[tgt, high_z], "over-box"), (np.r_[tgt, release_z], "lower")]:
            q = state.get_q()
            r = pc.plan_pose(list(map(float, q)), list(map(float, xyz)) + down, max_attempts=12)
            traj = fix_j6(r["trajectory"]) if r.get("success") else None
            if traj is None or not sweep_clear(traj):
                ev("place_blocked", seg=label); print(f"  [{label}] plan/clear FAILED (held)")
                events.close(); return "place_blocked"
            execute(state, pub, traj, r["dt"], "pid", args.vmax, 2.0, label, track=track)
        set_suction(0, ev)                                # RELEASE -> drops into bin
        ev("released")
        time.sleep(0.4)
        events.close(); return "placed"

    # ------------------------------- main loop -------------------------------
    summary = []
    try:
        if not args.dry_run:
            to_base("home")
        for i in range(args.max_objects):
            time.sleep(0.4)
            q0 = state.get_q()
            if q0 is None:
                print("ABORT: no /joint_states"); break
            Tbc = cam_pose(q0)
            rgb, depth, K = cap()
            objs = detect_objects(rgb, depth, K, Tbc, segment, deproject_mask, args)
            objs = [o for o in objs if o["flatness"] >= args.min_flat]
            print(f"\n=== iteration {i+1}: {len(objs)} pickable object(s) ===")
            for o in sorted(objs, key=lambda d: -d["flatness"]):
                print(f"  centroid={np.round(o['centroid'],3).tolist()} h={o['height']:.3f} "
                      f"foot={o['foot']:.3f} flat={o['flatness']:.2f}")
            if not objs:
                print("no pickable objects left -> done."); break
            obj = max(objs, key=lambda d: (d["flatness"], d["n"]))
            if args.dry_run:
                print(f"DRY RUN: would pick @ {np.round(obj['centroid'],3).tolist()}")
                summary.append({"obj": i + 1, "centroid": obj["centroid"].tolist(), "result": "dry"})
                break
            clip_dir = os.path.join(run_dir, f"obj_{i+1:02d}")
            rec.start_clip(clip_dir)
            print(f"-- recording -> {clip_dir} --")
            res = pick_place(obj, clip_dir)
            nframes = rec.stop_clip()                         # recording window = base -> release
            print(f"-- object {i+1}: {res} ({nframes} frames recorded, {rec.dropped} drops total) --")
            json.dump({"obj": i + 1, "centroid": obj["centroid"].tolist(),
                       "result": res, "frames": nframes},
                      open(os.path.join(clip_dir, "meta.json"), "w"), indent=2)
            summary.append({"obj": i + 1, "centroid": obj["centroid"].tolist(),
                            "result": res, "frames": nframes})
            if res == "place_blocked":
                print("** object still HELD (place blocked) -- stopping; manual recovery needed **"); break
            to_base()                                         # return to base (NOT recorded)
            if res == "pregrasp_fail":
                print("stopping (planning failure)."); break
    finally:
        rec.stop_evt.set(); rec.join(timeout=3)
        json.dump({"run": run_dir, "objects": summary}, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)
        print(f"\n== pipeline done: {len(summary)} object(s). summary -> {run_dir}/summary.json ==")
        rclpy.shutdown()


if __name__ == "__main__":
    main()
