#!/usr/bin/env python3
"""calib_d435 :: solve the FIXED D435 extrinsic T_base_cam435 using the ALREADY
calibrated D405 (eye-in-hand, config.T_TCP_CAM) + robot. Both cameras view the SAME
static ChArUco board on the table:

    T_base_cam405 = FK(q) @ T_shift @ T_TCP_CAM          (known D405 hand-eye)
    T_base_board  = T_base_cam405 @ T_cam405_board       (D405 sees the board)
    T_base_cam435 = T_base_board @ inv(T_cam435_board)    (D435 sees the same board)

Averages over frames (and, with --poses, several arm views) and reports the spread
of T_base_board across views (the quality metric: a well-calibrated D405 + static
board collapses to one place). Does NOT re-solve the D405 -- that's taken as given.

Run in the ROS env (planner + bridge up; board on the table, in BOTH cameras' view):
  source /opt/ros/humble/setup.bash
  export PYTHONPATH=~/librealsense/build/release:$PYTHONPATH
  python3 calib_d435.py --square 0.035 --marker 0.026 --frames 15
  # add --poses captures/calib_poses_v2.json to average over several arm views
  # add --apply to write T_base_cam435 into outputs/extrinsics_d435.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import cv2
import rclpy
import pyrealsense2 as rs

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")),
          os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc", "session_tools")),
          os.path.abspath(os.path.join(HERE, "..", "ros2node", "perception"))):
    sys.path.insert(0, p)
import config as C
from std_msgs.msg import String
from perturb_loop import PlannerClient, RobotState, execute
from calib_utils import (make_board, detect_charuco, rt_to_T, posquat_to_T,
                         T_inv, avg_pose, R_to_rpy_deg, R_to_quat)


class Cam:
    def __init__(self, name, serial, w=1280, h=720, fps=15):
        self.name, self.serial = name, serial
        self.pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        self.prof = self.pipe.start(cfg)
        intr = self.prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]]
        self.dist = list(intr.coeffs)
        for _ in range(10):
            try: self.pipe.wait_for_frames(2000)
            except RuntimeError: pass

    def grab(self):
        for _ in range(3):
            try: fr = self.pipe.wait_for_frames(2000)
            except RuntimeError: continue
            c = fr.get_color_frame()
            if c: return np.asanyarray(c.get_data()).copy()
        return None

    def stop(self):
        try: self.pipe.stop()
        except Exception: pass


def _T_shift(z):
    T = np.eye(4); T[2, 3] = z; return T


def grab_board(cam, board, frames, settle=0.04):
    """Average T_cam_board over `frames` detections; returns (T, spread_mm, n, reproj)."""
    Ts, rep = [], []
    for _ in range(frames):
        bgr = cam.grab()
        if bgr is not None:
            det = detect_charuco(board, bgr, cam.K, cam.dist)
            if det is not None:
                Ts.append(rt_to_T(det["rvec"], det["tvec"])); rep.append(det["reproj_px"])
        time.sleep(settle)
    if not Ts:
        return None, None, 0, None
    T, sp = avg_pose(Ts)
    return T, sp, len(Ts), float(np.mean(rep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--square", type=float, default=0.035, help="board square side (m)")
    ap.add_argument("--marker", type=float, default=0.026, help="aruco marker side (m)")
    ap.add_argument("--d405", default="218622271300")
    ap.add_argument("--d435", default="043422070101")
    ap.add_argument("--frames", type=int, default=15, help="frames averaged per camera per view")
    ap.add_argument("--poses", default=None, help="optional arm poses json (avg T_base_board over views)")
    ap.add_argument("--views", type=int, default=1, help="# arm views via small Cartesian jogs (validate spread)")
    ap.add_argument("--max-vel-deg", type=float, default=18.0)
    ap.add_argument("--out", default=os.path.join(C.OUT_DIR, "extrinsics_d435.json"))
    ap.add_argument("--apply", action="store_true", help="write the result to --out")
    args = ap.parse_args()

    board = make_board(args.square, args.marker)
    print("opening cameras ...")
    d405 = Cam("d405", args.d405)
    d435 = Cam("d435", args.d435)
    pc = PlannerClient(); print("planner:", pc.rpc({"type": "ping"}).get("backend"))
    rclpy.init(); node = rclpy.create_node("calib_d435")
    pub = node.create_publisher(String, "/mycobot/cmd/move", 10)
    state = RobotState(node)
    track = {"ramp_time": 0.15, "pos_gain": 1.0, "vff_scale": 1.0}

    for _ in range(20):                          # let /joint_states arrive
        rclpy.spin_once(node, timeout_sec=0.05)
    q0 = state.get_q()
    fk0 = pc.rpc({"type": "fk", "q": list(map(float, q0))})
    p0, quat0 = fk0["pos"][0], fk0["quat"][0]
    # build the view list: explicit joint poses (--poses), or small Cartesian jogs
    # around the current camera-on-board pose (--views), keeping the board framed.
    if args.poses:
        views = [("joint", P) for P in json.load(open(args.poses))["poses"]]
    else:
        offs = [(0, 0), (0.04, 0.0), (-0.04, 0.0), (0.0, 0.04), (0.0, -0.04),
                (0.035, 0.035), (-0.035, 0.035), (0.0, 0.0, 0.03)]
        views = []
        for o in offs[:max(1, args.views)]:
            dz = o[2] if len(o) > 2 else 0.0
            views.append(("cart", [p0[0] + o[0], p0[1] + o[1], p0[2] + dz] + list(quat0)))

    base_board, cam435_board = [], []
    for i, (kind, V) in enumerate(views):
        if kind == "joint":
            r = pc.plan_joint(list(map(float, state.get_q())), list(map(float, V["q_rad"])))
            if not r.get("success"):
                print(f"[{i:02d}] joint plan failed, skip"); continue
            execute(state, pub, np.array(r["trajectory"]), r["dt"], "pid",
                    args.max_vel_deg, 3.0, V.get("name", str(i)), track=track); time.sleep(0.6)
        elif kind == "cart" and i > 0:           # i==0 is the current pose (no move)
            r = pc.plan_pose(list(map(float, state.get_q())), [float(x) for x in V])
            if not r.get("success"):
                print(f"[{i:02d}] cart plan failed, skip"); continue
            execute(state, pub, np.array(r["trajectory"]), r["dt"], "pid",
                    args.max_vel_deg, 3.0, f"view{i}", track=track); time.sleep(0.6)
        else:
            for _ in range(10):
                rclpy.spin_once(node, timeout_sec=0.05)

        q = state.get_q()
        fk = pc.rpc({"type": "fk", "q": list(map(float, q))})
        pos, quat = fk["pos"][0], fk["quat"][0]
        T_base_tcp = posquat_to_T(pos, quat)
        T_base_cam405 = T_base_tcp @ _T_shift(C.CAM_TCP_Z_SHIFT) @ C.T_TCP_CAM

        T405, sp405, n405, rep405 = grab_board(d405, board, args.frames)
        T435, sp435, n435, rep435 = grab_board(d435, board, args.frames)
        tag = "current" if i == 0 else (V.get("name", str(i)) if kind == "joint" else f"view{i}")
        print(f"[{i:02d}] {tag}: "
              f"D405={'%ddet spr%.1fmm rep%.2fpx' % (n405, sp405, rep405) if T405 is not None else 'NONE'}  "
              f"D435={'%ddet spr%.1fmm rep%.2fpx' % (n435, sp435, rep435) if T435 is not None else 'NONE'}")
        if T405 is not None:
            base_board.append(T_base_cam405 @ T405)
        if T435 is not None:
            cam435_board.append(T435)

    d405.stop(); d435.stop()

    if not base_board:
        print("\nABORT: the D405 never saw the board (point the arm camera at it)."); rclpy.shutdown(); return
    if not cam435_board:
        print("\nABORT: the D435 never saw the board (check its view / lighting)."); rclpy.shutdown(); return

    T_base_board, spread_bb = avg_pose(base_board)        # board in base (quality metric)
    T_cam435_board, spread_cb = avg_pose(cam435_board)
    T_base_cam435 = T_base_board @ T_inv(T_cam435_board)

    bp = T_base_board[:3, 3]
    cp = T_base_cam435[:3, 3]
    print("\n===== RESULT =====")
    print(f"board in base  T_base_board: xyz=[{bp[0]:.3f},{bp[1]:.3f},{bp[2]:.3f}]  "
          f"(spread over {len(base_board)} D405 view(s) = {spread_bb:.1f} mm)")
    print(f"D435 in base   T_base_cam435: xyz=[{cp[0]:.3f},{cp[1]:.3f},{cp[2]:.3f}]  "
          f"rpy={[round(x,1) for x in R_to_rpy_deg(T_base_cam435[:3,:3])]} deg")
    print(f"D435 board-view spread over {len(cam435_board)} frame-avg(s) = {spread_cb:.1f} mm")
    quality = "GOOD" if spread_bb < 6 else ("OK" if spread_bb < 12 else "POOR (add --poses views / better light)")
    print(f"calibration quality: {quality}")

    if args.apply:
        out = {"frame_convention": "optical +X right +Y down +Z fwd; T_base_cam = base<-cam",
               "board": {"square": args.square, "marker": args.marker, "dict": "DICT_4X4_50", "squares": [5, 7]},
               "T_base_cam435": T_base_cam435.tolist(),
               "T_base_board": T_base_board.tolist(),
               "spread_mm": {"base_board": spread_bb, "cam435_board": spread_cb},
               "n_d405_views": len(base_board)}
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"\nwrote -> {args.out}")
    else:
        print("\n(dry run; pass --apply to save extrinsics_d435.json)")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
