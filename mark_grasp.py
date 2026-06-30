#!/usr/bin/env python3
"""mark_grasp :: detect the object, compute the grasp (suction) point in the robot
BASE frame, and overlay that SAME 3D point projected into BOTH the D405 (eye-in-hand)
and D435 (fixed) color images. Saves a side-by-side plot.

If the marker lands on the object in BOTH views, the D405 hand-eye and the D435
extrinsic are mutually consistent; a base-frame bias shows up as the marker sitting
ON the object in the (self-consistent) D405 view but OFF it in the D435 view.

  source /opt/ros/humble/setup.bash
  export PYTHONPATH=~/librealsense/build/release:$PYTHONPATH
  python3 mark_grasp.py            # to-base scout, detect, overlay, save
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")),
          os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc", "session_tools")),
          os.path.abspath(os.path.join(HERE, "..", "ros2node", "perception"))):
    sys.path.insert(0, p)
import config as C
from std_msgs.msg import String
from geometry import make_T, quat_wxyz_to_R, R_to_quat_wxyz, R_from_two_axes
from perturb_loop import PlannerClient, RobotState, execute
from real_multi import detect_objects
from real_grasp import estimate_normals, detect_suction_point
from capture_and_plot import segment
from object_pointclouds import deproject_mask
from calib_utils import T_inv


class Cam:
    def __init__(self, serial, w=1280, h=720, fps=15):
        self.pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        self.prof = self.pipe.start(cfg); self.align = rs.align(rs.stream.color)
        intr = self.prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.]])
        for _ in range(15):
            try: self.pipe.wait_for_frames(2000)
            except RuntimeError: pass

    def grab(self):
        fr = self.align.process(self.pipe.wait_for_frames(2000))
        bgr = np.asanyarray(fr.get_color_frame().get_data()).copy()
        depth = np.asanyarray(fr.get_depth_frame().get_data()).astype(np.float32) * 0.001
        return bgr, depth

    def stop(self):
        try: self.pipe.stop()
        except Exception: pass


def project(T_base_cam, K, xyz_base):
    """Project a base-frame 3D point into the camera image -> (u,v), in_front."""
    pc = T_inv(T_base_cam) @ np.r_[np.asarray(xyz_base, float), 1.0]
    if pc[2] <= 1e-6:
        return None, False
    u = K[0, 0] * pc[0] / pc[2] + K[0, 2]
    v = K[1, 1] * pc[1] / pc[2] + K[1, 2]
    return (float(u), float(v)), True


def draw_marker(bgr, uv, color=(0, 0, 255), label="grasp"):
    img = bgr.copy()
    if uv is None:
        return img
    u, v = int(round(uv[0])), int(round(uv[1]))
    cv2.circle(img, (u, v), 16, color, 2)
    cv2.line(img, (u - 24, v), (u + 24, v), color, 2)
    cv2.line(img, (u, v - 24), (u, v + 24), color, 2)
    cv2.circle(img, (u, v), 3, color, -1)
    cv2.putText(img, label, (u + 20, v - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d405", default="218622271300"); ap.add_argument("--d435", default="043422070101")
    ap.add_argument("--no-scout", action="store_true", help="capture at current pose (don't to-base)")
    ap.add_argument("--xmin", type=float, default=0.15); ap.add_argument("--xmax", type=float, default=0.55)
    ap.add_argument("--ymin", type=float, default=-0.28); ap.add_argument("--ymax", type=float, default=0.30)
    ap.add_argument("--max-h", type=float, default=0.13); ap.add_argument("--max-foot", type=float, default=0.25)
    ap.add_argument("--out", default=os.path.join(C.OUT_DIR, "grasp_overlay.png"))
    args = ap.parse_args()

    ext = json.load(open(os.path.join(C.OUT_DIR, "extrinsics_d435.json")))
    T_base_cam435 = np.array(ext["T_base_cam435"])

    d405 = Cam(args.d405); d435 = Cam(args.d435)
    pc = PlannerClient(); print("planner:", pc.rpc({"type": "ping"}).get("backend"))
    rclpy.init(); node = rclpy.create_node("mark_grasp")
    pub = node.create_publisher(String, "/mycobot/cmd/move", 10)
    state = RobotState(node)
    track = {"ramp_time": 0.15, "pos_gain": 1.0, "vff_scale": 1.0}

    # 1) detect the object with the FIXED D435 (reliable; sees the whole table)
    bgr435, depth435 = d435.grab()
    rgb435 = bgr435[:, :, ::-1].copy()
    objs = detect_objects(rgb435, depth435, d435.K, T_base_cam435, segment, deproject_mask, args)
    if not objs:
        print("no object detected -- place an object on the table")
        d405.stop(); d435.stop(); rclpy.shutdown(); return
    objs.sort(key=lambda o: -len(o["pts"]))
    o = objs[0]; P = o["pts"]
    grasp = o["centroid"].copy(); how = "centroid"
    try:
        nrm = estimate_normals(P, T_base_cam435[:3, 3])
        found = detect_suction_point(P, nrm, normal_cone_deg=45.0, select="central")
        if found is not None:
            grasp = np.asarray(found[0], float); how = "suction-point"
    except Exception as e:
        print("suction-point failed:", e)
    print(f"object base centre [{o['centroid'][0]:+.3f},{o['centroid'][1]:+.3f}]  "
          f"grasp ({how}) base [{grasp[0]:+.3f},{grasp[1]:+.3f},{grasp[2]:.3f}]")

    # 2) move the arm so the D405 looks straight down over the object
    down = list(R_to_quat_wxyz(R_from_two_axes(np.array([0, 0, -1.0]))))
    z_view = min(0.42, max(0.20, float(grasp[2]) + 0.20))
    if not args.no_scout:
        for _ in range(20): rclpy.spin_once(node, timeout_sec=0.05)
        r = pc.plan_pose(list(map(float, state.get_q())),
                         [float(grasp[0]), float(grasp[1]), z_view] + down, max_attempts=12)
        if r.get("success"):
            execute(state, pub, np.array(r["trajectory"]), r["dt"], "pid", 18.0, 3.0, "look-at-obj", track=track)
        else:
            print("  (could not move D405 over the object; D405 view may not contain it)")
        time.sleep(0.5)
    for _ in range(20): rclpy.spin_once(node, timeout_sec=0.05)

    # 3) capture the D405 at this pose + its base<-cam transform; re-grab the D435
    q = state.get_q()
    fk = pc.rpc({"type": "fk", "q": list(map(float, q))})
    T_base_tcp = make_T(quat_wxyz_to_R(np.array(fk["quat"][0])), np.array(fk["pos"][0]))
    T_base_cam405 = T_base_tcp @ make_T(np.eye(3), [0, 0, C.CAM_TCP_Z_SHIFT]) @ C.T_TCP_CAM
    bgr405, depth405 = d405.grab()
    bgr435, depth435 = d435.grab()
    d405.stop(); d435.stop()

    # 4) project the SAME base grasp point into both images
    uv405, f405 = project(T_base_cam405, d405.K, grasp)
    uv435, f435 = project(T_base_cam435, d435.K, grasp)
    print(f"  -> D405 px {None if uv405 is None else [round(x) for x in uv405]} (in_front={f405})")
    print(f"  -> D435 px {None if uv435 is None else [round(x) for x in uv435]} (in_front={f435})")

    img405 = draw_marker(bgr405, uv405 if f405 else None)
    img435 = draw_marker(bgr435, uv435 if f435 else None)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].imshow(img405[:, :, ::-1]); ax[0].set_title(f"D405 (eye-in-hand) — grasp {how}"); ax[0].axis("off")
    ax[1].imshow(img435[:, :, ::-1]); ax[1].set_title("D435 (fixed) — same base point projected"); ax[1].axis("off")
    fig.suptitle(f"grasp point base=[{grasp[0]:.3f},{grasp[1]:.3f},{grasp[2]:.3f}]")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"saved -> {args.out}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
