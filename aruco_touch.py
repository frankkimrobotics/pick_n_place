#!/usr/bin/env python3
"""Hand-eye VALIDATION: detect each ArUco marker in the D405 RGBD, compute its 3D
position in the base frame (camera pose = FK(tcp) @ T_TCP_CAM, the just-calibrated
extrinsic), then move the suction tip to each marker (hover -> touch). If the tip lands
on the markers, the hand-eye + TCP calibration is correct.

Run in the ROS env with the D405 on PYTHONPATH; planner+bridge up; board fixed.
  PYTHONPATH=~/librealsense/build/release python3 pick_and_place/aruco_touch.py
"""
import argparse, os, sys, time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")),
          os.path.abspath(os.path.join(HERE, "..", "ros2node", "perception"))):
    sys.path.insert(0, p)
import config as C
from geometry import (R_from_two_axes, R_to_quat_wxyz, quat_wxyz_to_R, make_T, transform_points)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="218622271300")
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--max-markers", type=int, default=6, help="touch this many (spread across the board)")
    ap.add_argument("--touch-dz", type=float, default=0.0, help="stop this far ABOVE the marker (0 = touch)")
    ap.add_argument("--hover", type=float, default=0.05, help="hover height above each marker before descending")
    ap.add_argument("--vmax", type=float, default=18.0)
    ap.add_argument("--dwell", type=float, default=1.0, help="seconds to hold at each marker")
    ap.add_argument("--no-touch", action="store_true", help="only detect + report positions, no motion")
    args = ap.parse_args()

    import rclpy
    from std_msgs.msg import String
    from object_pointclouds import capture_aligned
    from multiview_fuse import pick_res
    from perturb_loop import PlannerClient, RobotState, execute

    pc = PlannerClient()
    rclpy.init(); node = rclpy.create_node("aruco_touch")
    pub = node.create_publisher(String, "/mycobot/cmd/move", 10); state = RobotState(node)
    track = {"ramp_time": 0.15, "pos_gain": 1.0, "vff_scale": 1.0}
    down = list(R_to_quat_wxyz(R_from_two_axes(np.array([0, 0, -1.0]))))

    def fk_T(q):
        r = pc.rpc({"type": "fk", "q": list(map(float, q))})
        return make_T(quat_wxyz_to_R(np.array(r["quat"][0])), np.array(r["pos"][0]))

    def cam_pose(q):
        return fk_T(q) @ make_T(np.eye(3), [0, 0, C.CAM_TCP_Z_SHIFT]) @ C.T_TCP_CAM

    def fix_j6(traj):
        t = np.array(traj, float); t[:, 5] = t[0, 5]; return t

    def goto(xyz, label, vmax=None):
        q = state.get_q()
        r = pc.plan_pose(list(map(float, q)), list(map(float, xyz)) + down, max_attempts=14)
        if not r.get("success"):
            print(f"  [{label}] PLAN FAILED"); return False
        ex = execute(state, pub, fix_j6(r["trajectory"]), r["dt"], "pid", vmax or args.vmax, 2.0, label, track=track)
        return ex.get("ok")

    def to_base():
        r = pc.plan_joint(list(map(float, state.get_q())), list(map(float, C.BASE_Q)))
        if r.get("success"):
            execute(state, pub, fix_j6(r["trajectory"]), r["dt"], "pid", 22.0, 3.0, "to-base", track=track)

    # ---- ensure at base, capture RGBD from there ----
    to_base(); time.sleep(0.3)
    q_act = state.get_q()
    Tbc = cam_pose(q_act)
    CW, CH = pick_res(args.serial)
    rgb, depth, K, _ = capture_aligned(args.serial, CW, CH, 30, 30)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # ---- detect ArUco markers ----
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, args.dict))
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    gray = cv2.cvtColor(rgb[:, :, ::-1], cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        print("no ArUco markers detected"); rclpy.shutdown(); return
    ids = ids.flatten()
    print(f"detected {len(ids)} markers: {sorted(ids.tolist())}")

    markers = []
    for c, mid in zip(corners, ids):
        quad = c[0]                                  # 4x2 image corners
        ctr = quad.mean(0); u, v = int(round(ctr[0])), int(round(ctr[1]))
        # median depth over the marker interior (robust to per-pixel noise)
        m = np.zeros(depth.shape, np.uint8)
        cv2.fillConvexPoly(m, quad.astype(np.int32), 1)
        dz = depth[(m > 0) & (depth > 0.05) & (depth < 1.0)]
        if len(dz) < 10:
            print(f"  marker {mid}: no valid depth, skip"); continue
        z = float(np.median(dz))
        Pcam = np.array([(ctr[0] - cx) / fx * z, (ctr[1] - cy) / fy * z, z])
        Pbase = transform_points(Tbc, Pcam[None])[0]
        markers.append({"id": int(mid), "P": Pbase, "uv": (u, v)})

    markers.sort(key=lambda d: d["id"])
    print("marker positions in base frame (m):")
    for mk in markers:
        print(f"  id {mk['id']:>2}: {np.round(mk['P'], 4).tolist()}")

    # annotated image
    vis = rgb[:, :, ::-1].copy()
    aruco.drawDetectedMarkers(vis, corners, ids)
    for mk in markers:
        cv2.putText(vis, f"{mk['id']}:{mk['P'][0]:.2f},{mk['P'][1]:.2f}", (mk["uv"][0] - 30, mk["uv"][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    os.makedirs(os.path.join(C.OUT_DIR, "realdbg"), exist_ok=True)
    cv2.imwrite(os.path.join(C.OUT_DIR, "realdbg", "aruco_detect.png"), vis)
    print(f"annotated -> {C.OUT_DIR}/realdbg/aruco_detect.png")

    if args.no_touch or not markers:
        rclpy.shutdown(); return

    # ---- select a spread-out subset (farthest-point on xy) and touch each ----
    sel = [markers[0]]
    while len(sel) < min(args.max_markers, len(markers)):
        rem = [m for m in markers if m not in sel]
        nxt = max(rem, key=lambda m: min(np.linalg.norm(m["P"][:2] - s["P"][:2]) for s in sel))
        sel.append(nxt)
    sel.sort(key=lambda d: (d["P"][0], d["P"][1]))
    print(f"\ntouching {len(sel)} markers (ids {[m['id'] for m in sel]}):")

    for mk in sel:
        P = mk["P"]
        print(f"  -> marker {mk['id']} at {np.round(P, 4).tolist()}")
        if goto([P[0], P[1], P[2] + args.hover], f"hover{mk['id']}"):
            goto([P[0], P[1], P[2] + args.touch_dz], f"touch{mk['id']}", vmax=4.0)
            time.sleep(args.dwell)
            goto([P[0], P[1], P[2] + args.hover], f"lift{mk['id']}", vmax=8.0)

    to_base()
    print("== done; tip visited each marker (validates hand-eye + TCP) ==")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
