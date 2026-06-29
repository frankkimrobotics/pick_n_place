#!/usr/bin/env python3
"""d435_detect :: localize tabletop objects in the ROBOT BASE frame using the FIXED
D435 + its calibrated extrinsic T_base_cam435 (calib_d435.py). The camera is fixed,
so there is NO FK in the chain -- Tbc is simply the constant T_base_cam435. Reuses
the SAM3 detect_objects pipeline, so detections come out directly in base_link.

  source /opt/ros/humble/setup.bash
  export PYTHONPATH=~/librealsense/build/release:$PYTHONPATH
  python3 d435_detect.py                 # SAM3 detect -> object base XY
  python3 d435_detect.py --board         # board self-check: D435->base vs saved T_base_board
"""
import argparse
import json
import os
import sys
import types

import numpy as np
import pyrealsense2 as rs

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")),
          os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc", "session_tools")),
          os.path.abspath(os.path.join(HERE, "..", "ros2node", "perception"))):
    sys.path.insert(0, p)
import config as C
from calib_utils import make_board, detect_charuco, rt_to_T


def open_d435(serial, w=1280, h=720, fps=15):
    pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    prof = pipe.start(cfg); align = rs.align(rs.stream.color)
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]]
    dist = list(intr.coeffs)
    for _ in range(15):
        try: pipe.wait_for_frames(2000)
        except RuntimeError: pass
    return pipe, align, K, dist


def grab(pipe, align):
    fr = align.process(pipe.wait_for_frames(2000))
    col, dep = fr.get_color_frame(), fr.get_depth_frame()
    bgr = np.asanyarray(col.get_data()).copy()
    depth_m = np.asanyarray(dep.get_data()).astype(np.float32) * 0.001   # mm -> m
    return bgr, depth_m


def load_extrinsics():
    path = os.path.join(C.OUT_DIR, "extrinsics_d435.json")
    ext = json.load(open(path))
    return np.array(ext["T_base_cam435"]), np.array(ext.get("T_base_board", np.eye(4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="043422070101")
    ap.add_argument("--board", action="store_true", help="board self-check vs saved T_base_board")
    ap.add_argument("--square", type=float, default=0.035)
    ap.add_argument("--marker", type=float, default=0.026)
    ap.add_argument("--xmin", type=float, default=0.15); ap.add_argument("--xmax", type=float, default=0.55)
    ap.add_argument("--ymin", type=float, default=-0.28); ap.add_argument("--ymax", type=float, default=0.30)
    ap.add_argument("--max-h", type=float, default=0.13); ap.add_argument("--max-foot", type=float, default=0.25)
    args = ap.parse_args()

    Tbc, T_base_board_saved = load_extrinsics()
    print(f"T_base_cam435 cam xyz = {[round(x,3) for x in Tbc[:3,3]]}")
    pipe, align, K, dist = open_d435(args.serial)
    try:
        bgr, depth = grab(pipe, align)
    finally:
        pass

    if args.board:
        board = make_board(args.square, args.marker)
        det = detect_charuco(board, bgr, K, dist)
        pipe.stop()
        if det is None:
            print("board not detected by D435"); return
        T_base_board = Tbc @ rt_to_T(det["rvec"], det["tvec"])
        e = np.linalg.norm(T_base_board[:3, 3] - T_base_board_saved[:3, 3]) * 1000
        print(f"D435 board->base : {[round(x,4) for x in T_base_board[:3,3]]}")
        print(f"saved T_base_board: {[round(x,4) for x in T_base_board_saved[:3,3]]}")
        print(f"consistency error : {e:.1f} mm  ({'OK' if e < 3 else 'check'})")
        return

    # SAM3 object detection, deprojected straight into base via the fixed extrinsic
    from real_multi import detect_objects
    from capture_and_plot import segment
    from object_pointclouds import deproject_mask
    pipe.stop()
    rgb = bgr[:, :, ::-1].copy()
    K = np.asarray(K, float)                          # deproject_mask indexes K[0,0]
    dargs = types.SimpleNamespace(xmin=args.xmin, xmax=args.xmax, ymin=args.ymin,
                                  ymax=args.ymax, max_h=args.max_h, max_foot=args.max_foot)
    objs = detect_objects(rgb, depth, K, Tbc, segment, deproject_mask, dargs)
    objs.sort(key=lambda o: -len(o["pts"]))
    print(f"=== D435 detected {len(objs)} object(s) in base ===")
    for j, o in enumerate(objs):
        c = o["centroid"]
        print(f"  [{j}] base XY [{c[0]:+.3f},{c[1]:+.3f}] z={c[2]:.3f} "
              f"foot={o['foot']*100:.0f}cm h={o['height']*100:.0f}cm n={len(o['pts'])}")


if __name__ == "__main__":
    main()
