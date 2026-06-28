#!/usr/bin/env python3
"""Eye-in-hand D405 extrinsic (T_tcp_cam) recalibration with JOINT-SPACE viewpoint
sampling (dynamic, diverse views; no odd elbow; cable-safe J6).

Instead of solving Cartesian look-at poses (where cuRobo picks the elbow branch
stochastically -> odd 'extruded' poses, and only one tilt resolved naturally), we
SAMPLE joint configs directly within the NATURAL elbow ranges (shoulder/elbow bounded,
|J6|<=limit), keep those whose camera actually frames the board, and pick an
orientation-diverse subset (farthest-point on camera orientation -> real tilt/roll/pan
diversity). Each is reached by a joint-goal move, so the natural branch is guaranteed.

At each settled pose: detect the ChArUco board (cam<-board) + FK(tcp) (base<-tcp);
solve cv2.calibrateHandEye (eye-in-hand) -> T_tcp_cam. Quality = board-pose spread in
base across views; also prints spread under the CURRENT extrinsic as a sanity check.
Use --apply to patch config.py (only if spread is good).

Run in the ROS env with the D405 on PYTHONPATH; planner+bridge up; board FIXED at --target.
"""
import argparse, json, math, os, sys, time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")),
          os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc", "session_tools")),
          os.path.abspath(os.path.join(HERE, "..", "ros2node", "perception"))):
    sys.path.insert(0, p)
import config as C
from geometry import quat_wxyz_to_R, make_T
from calib_utils import make_board, detect_charuco, rt_to_T, R_to_quat, R_to_rpy_deg, avg_pose


def farthest_point(feats, n):
    """Greedy farthest-point subset (indices) maximizing spread in `feats` (row vectors)."""
    feats = np.asarray(feats, float)
    if len(feats) <= n:
        return list(range(len(feats)))
    sel = [int(np.argmax(np.linalg.norm(feats - feats.mean(0), axis=1)))]   # start at an extreme
    d = np.linalg.norm(feats - feats[sel[0]], axis=1)
    while len(sel) < n:
        i = int(np.argmax(d))
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(feats - feats[i], axis=1))
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="0.4,0,0.0", help="board CENTRE x,y,z (m) in base frame")
    ap.add_argument("--square", type=float, default=0.035)
    ap.add_argument("--marker", type=float, default=0.026)
    ap.add_argument("--serial", default="218622271300")
    ap.add_argument("--vmax", type=float, default=18.0)
    ap.add_argument("--max-poses", type=int, default=20)
    ap.add_argument("--n-sample", type=int, default=600, help="random joint configs to screen")
    ap.add_argument("--j6-limit", type=float, default=95.0)
    ap.add_argument("--min-corners", type=int, default=14)
    ap.add_argument("--max-reproj", type=float, default=0.7, help="drop detections worse than this (px)")
    # NATURAL-branch joint ranges (URDF deg): keep the good elbow, vary wrist for tilt/roll/pan
    ap.add_argument("--j1", default="-20,20"); ap.add_argument("--j2", default="-20,16")
    ap.add_argument("--j3", default="110,146"); ap.add_argument("--j4", default="-82,-22")
    ap.add_argument("--j5", default="-112,-68")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    target = np.array([float(v) for v in args.target.split(",")])

    import rclpy, pyrealsense2 as rs
    from std_msgs.msg import String
    from multiview_fuse import pick_res
    from perturb_loop import PlannerClient, RobotState, execute

    board = make_board(args.square, args.marker)
    pc = PlannerClient()
    rclpy.init(); node = rclpy.create_node("calib_handeye")
    pub = node.create_publisher(String, "/mycobot/cmd/move", 10); state = RobotState(node)
    track = {"ramp_time": 0.15, "pos_gain": 1.0, "vff_scale": 1.0}
    SH = make_T(np.eye(3), [0, 0, C.CAM_TCP_Z_SHIFT])

    def fk_T(q):
        r = pc.rpc({"type": "fk", "q": list(map(float, q))})
        return make_T(quat_wxyz_to_R(np.array(r["quat"][0])), np.array(r["pos"][0]))

    def fk_many(qs):
        r = pc.rpc({"type": "fk", "q": [list(map(float, q)) for q in qs]})
        return [make_T(quat_wxyz_to_R(np.array(r["quat"][i])), np.array(r["pos"][i]))
                for i in range(len(qs))]

    w, h = pick_res(args.serial)

    def open_stream():
        pp = rs.pipeline(); cf = rs.config(); cf.enable_device(args.serial)
        cf.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30); prof = pp.start(cf)
        it = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        Kk = np.array([[it.fx, 0, it.ppx], [0, it.fy, it.ppy], [0, 0, 1.]])
        dd = np.array(it.coeffs[:5], float)
        for _ in range(15):
            pp.wait_for_frames(2000)
        return pp, Kk, dd

    pipe, K, dist = open_stream()

    def grab():
        nonlocal pipe, K, dist
        for _ in range(2):
            for _ in range(6):
                try:
                    f = pipe.wait_for_frames(2000)
                except RuntimeError:
                    continue
                c = f.get_color_frame()
                if c:
                    return np.asanyarray(c.get_data()).copy()
            try:
                pipe.stop()
            except Exception:
                pass
            try:
                pipe, K, dist = open_stream()
            except Exception:
                return None
        return None

    def board_visible(Tbt):
        """Does the camera at this tcp pose frame the board well? (board centre + a ring of
        points must project inside the image with margin, in front of the camera)."""
        Tbc = Tbt @ SH @ C.T_TCP_CAM
        Tcb = np.linalg.inv(Tbc)
        ring = [[0, 0, 0]] + [[0.08 * math.cos(t), 0.08 * math.sin(t), 0]
                              for t in np.linspace(0, 2 * math.pi, 6, endpoint=False)]
        vis = 0
        for off in ring:
            P = Tcb @ np.r_[target + off, 1.0]
            if P[2] < 0.12 or P[2] > 0.55:           # too close/far/behind
                return False
            u = K[0, 0] * P[0] / P[2] + K[0, 2]; v = K[1, 1] * P[1] / P[2] + K[1, 2]
            if 50 <= u <= w - 50 and 40 <= v <= h - 40:
                vis += 1
        return vis >= 6                              # centre + >=5 ring points in frame

    def goto_joint(qg, label):
        r = pc.plan_joint(list(map(float, state.get_q())), list(map(float, qg)))
        if not r.get("success"):
            return False
        execute(state, pub, np.array(r["trajectory"]), r["dt"], "pid", args.vmax, 2.0, label, track=track)
        return True

    def sample(label):
        time.sleep(0.4)
        q = state.get_q()
        bgr = grab()
        if bgr is None:
            print(f"  [{label}] no frame"); return None
        det = detect_charuco(board, bgr, K, dist, min_corners=args.min_corners)
        if det is None or det["reproj_px"] > args.max_reproj:
            print(f"  [{label}] reject ({None if det is None else round(det['reproj_px'],2)}px)"); return None
        print(f"  [{label}] OK corners={det['n_corners']} reproj={det['reproj_px']:.2f}px")
        return {"label": label, "T_base_tcp": fk_T(q), "T_cam_target": rt_to_T(det["rvec"], det["tvec"]),
                "reproj": det["reproj_px"], "n_corners": det["n_corners"]}

    samples = []
    try:
        # ---- joint-space sampling within natural ranges; keep board-viewing; pick diverse ----
        rng = np.random.default_rng(0)
        rngs = [args.j1, args.j2, args.j3, args.j4, args.j5, f"{-args.j6_limit},{args.j6_limit}"]
        lo = np.radians([float(s.split(",")[0]) for s in rngs])
        hi = np.radians([float(s.split(",")[1]) for s in rngs])
        cq = lo + rng.random((args.n_sample, 6)) * (hi - lo)
        Tbts = fk_many(cq)
        viz = [(q, T) for q, T in zip(cq, Tbts) if board_visible(T)]
        print(f"board-viewing natural configs: {len(viz)} / {args.n_sample}")
        if len(viz) < 6:
            print("ABORT: too few board-viewing configs; widen --j* ranges or check --target"); return
        # orientation-diverse subset (farthest-point on camera quaternion, hemisphere-aligned)
        quats = []
        for _, T in viz:
            Tbc = T @ SH @ C.T_TCP_CAM
            qv = np.array(R_to_quat(Tbc[:3, :3])); quats.append(qv * np.sign(qv[0]))
        sel = farthest_point(quats, args.max_poses)
        chosen = [viz[i][0] for i in sel]
        # order to MINIMISE wrist travel between consecutive poses (less cable winding ->
        # fewer USB drops); greedy nearest-neighbour from base, weighting wrist joints.
        Wj = np.array([1, 1, 1, 1.5, 2.0, 2.0]); order = []; rem = list(range(len(chosen)))
        cur = np.array(C.BASE_Q, float)
        while rem:
            j = min(rem, key=lambda i: float(np.sum((Wj * (np.array(chosen[i]) - cur)) ** 2)))
            order.append(j); cur = np.array(chosen[j]); rem.remove(j)
        chosen = [chosen[i] for i in order]
        print(f"selected {len(chosen)} orientation-diverse views (ordered for min wrist travel)")

        goto_joint(C.BASE_Q, "base")
        s = sample("base")
        if s:
            samples.append(s)
        for k, q in enumerate(chosen):
            if not goto_joint(q, f"p{k}"):
                print(f"  [p{k}] joint plan failed -> skip"); continue
            time.sleep(0.4); qa = state.get_q(); bgr = grab()
            if bgr is None:                       # camera dropped -> salvage: stop, solve with collected
                print("  CAMERA DROPPED -> stop capturing, solving with samples so far"); break
            det = detect_charuco(board, bgr, K, dist, min_corners=args.min_corners)
            if det is None or det["reproj_px"] > args.max_reproj:
                print(f"  [p{k}] reject ({None if det is None else round(det['reproj_px'],2)}px)"); continue
            print(f"  [p{k}] OK corners={det['n_corners']} reproj={det['reproj_px']:.2f}px")
            samples.append({"label": f"p{k}", "T_base_tcp": fk_T(qa),
                            "T_cam_target": rt_to_T(det["rvec"], det["tvec"]),
                            "reproj": det["reproj_px"], "n_corners": det["n_corners"]})
        print(f"\ngood samples: {len(samples)}")
        if len(samples) < 5:
            print("ABORT: need >=5 detections"); return

        Rg = [s["T_base_tcp"][:3, :3] for s in samples]
        tg = [s["T_base_tcp"][:3, 3] for s in samples]
        Rt = [s["T_cam_target"][:3, :3] for s in samples]
        tt = [s["T_cam_target"][:3, 3] for s in samples]
        # orientation diversity actually achieved (max pairwise optical-axis angle)
        axes = [s["T_base_tcp"][:3, :3] @ C.T_TCP_CAM[:3, 2] for s in samples]
        ang = max(math.degrees(math.acos(np.clip(a @ b, -1, 1))) for a in axes for b in axes)
        print(f"camera optical-axis spread across views: {ang:.0f} deg (more = better conditioned)")

        boards_cur = [s["T_base_tcp"] @ C.T_TCP_CAM @ s["T_cam_target"] for s in samples]
        _, spread_cur = avg_pose(boards_cur)
        print(f"data check: board-spread under CURRENT extrinsic = {spread_cur:.1f} mm")

        methods = {"TSAI": cv2.CALIB_HAND_EYE_TSAI, "PARK": cv2.CALIB_HAND_EYE_PARK,
                   "HORAUD": cv2.CALIB_HAND_EYE_HORAUD, "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS}
        best = None
        for name, m in methods.items():
            try:
                Rcg, tcg = cv2.calibrateHandEye(Rg, tg, Rt, tt, method=m)
            except Exception as e:
                print(f"  {name}: failed ({e})"); continue
            T = np.eye(4); T[:3, :3] = Rcg; T[:3, 3] = tcg.flatten()
            _, spread = avg_pose([s["T_base_tcp"] @ T @ s["T_cam_target"] for s in samples])
            print(f"  {name:11s} cam_origin_in_tcp(mm)={np.round(tcg.flatten()*1000,1).tolist()} "
                  f"spread={spread:.1f}mm")
            if best is None or spread < best[2]:
                best = (name, T, spread)

        name, T_tcp_cam, spread = best
        print(f"\n== BEST: {name}  board-spread={spread:.1f}mm (current={spread_cur:.1f}mm, "
              f"axis-spread={ang:.0f}deg) ==")
        print("T_tcp_cam =\n", np.array2string(T_tcp_cam, precision=6, suppress_small=True))
        print("cam origin in tcp (mm):", np.round(T_tcp_cam[:3, 3] * 1000, 2).tolist())
        print("rpy zyx (deg):", np.round(R_to_rpy_deg(T_tcp_cam[:3, :3]), 2).tolist())
        cur = C.T_TCP_CAM
        dpos = np.linalg.norm(T_tcp_cam[:3, 3] - cur[:3, 3]) * 1000
        dang = np.degrees(np.arccos(np.clip((np.trace(cur[:3, :3].T @ T_tcp_cam[:3, :3]) - 1) / 2, -1, 1)))
        print(f"vs current: dpos={dpos:.1f}mm drot={dang:.2f}deg")

        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join(C.OUT_DIR, f"handeye_{stamp}.json")
        json.dump({"method": name, "spread_mm": spread, "spread_current_mm": spread_cur,
                   "axis_spread_deg": ang, "n_samples": len(samples), "T_tcp_cam": T_tcp_cam.tolist(),
                   "cam_origin_in_tcp_mm": (T_tcp_cam[:3, 3] * 1000).tolist(),
                   "square": args.square, "marker": args.marker,
                   "samples": [{"label": s["label"], "reproj": s["reproj"], "n_corners": s["n_corners"],
                                "T_base_tcp": s["T_base_tcp"].tolist(),
                                "T_cam_target": s["T_cam_target"].tolist()} for s in samples]},
                  open(out, "w"), indent=2)
        print(f"saved -> {out}")
        if args.apply and spread < 5.0:
            _patch_config(T_tcp_cam); print("config.py T_TCP_CAM updated.")
        elif args.apply:
            print(f"NOT applying: spread {spread:.1f}mm not clearly good enough.")
    finally:
        try:
            pipe.stop()
        except Exception:
            pass
        try:
            goto_joint(C.BASE_Q, "to-base")
        except Exception:
            pass
        rclpy.shutdown()


def _patch_config(T):
    import re
    cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
    src = open(cfg).read()
    rows = ",\n".join("    [" + ", ".join(repr(float(x)) for x in T[i]) + "]" for i in range(4))
    new = "T_TCP_CAM = np.array([\n" + rows + ",\n], dtype=np.float64)"
    open(cfg, "w").write(re.sub(r"T_TCP_CAM = np\.array\(\[.*?\], dtype=np\.float64\)", new, src,
                                count=1, flags=re.S))


if __name__ == "__main__":
    main()
