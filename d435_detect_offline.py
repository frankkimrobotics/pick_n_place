#!/usr/bin/env python3
"""SAM3 object detection from a saved D435 frame (d435_capture.py) using the
fixed-camera extrinsic -- same output line format as d435_detect.py.
    ~/miniconda3/envs/sam3/bin/python d435_detect_offline.py --frame /tmp/d435_frame
"""
import argparse
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")),
          os.path.abspath(os.path.join(HERE, "..", "ros2node", "perception"))):
    sys.path.insert(0, p)
import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="/tmp/d435_frame")
    ap.add_argument("--xmin", type=float, default=0.15)
    ap.add_argument("--xmax", type=float, default=0.55)
    ap.add_argument("--ymin", type=float, default=-0.28)
    ap.add_argument("--ymax", type=float, default=0.30)
    ap.add_argument("--max-h", type=float, default=0.13)
    ap.add_argument("--max-foot", type=float, default=0.25)
    a = ap.parse_args()

    ext = json.load(open(os.path.join(C.OUT_DIR, "extrinsics_d435.json")))
    Tbc = np.array(ext["T_base_cam435"], float)
    bgr = np.load(os.path.join(a.frame, "bgr.npy"))
    depth = np.load(os.path.join(a.frame, "depth.npy"))
    ki = json.load(open(os.path.join(a.frame, "K.json")))
    K = np.array([[ki["fx"], 0, ki["ppx"]], [0, ki["fy"], ki["ppy"]],
                  [0, 0, 1.0]])

    from real_multi import detect_objects
    from capture_and_plot import segment
    from object_pointclouds import deproject_mask
    rgb = bgr[:, :, ::-1].copy()
    dargs = types.SimpleNamespace(xmin=a.xmin, xmax=a.xmax, ymin=a.ymin,
                                  ymax=a.ymax, max_h=a.max_h,
                                  max_foot=a.max_foot)
    objs = detect_objects(rgb, depth, K, Tbc, segment, deproject_mask, dargs)
    objs.sort(key=lambda o: -len(o["pts"]))
    print(f"=== D435 detected {len(objs)} object(s) in base ===")
    for j, o in enumerate(objs):
        c = o["centroid"]
        print(f"  [{j}] base XY [{c[0]:+.3f},{c[1]:+.3f}] "
              f"centroid_z={c[2]:.3f} TOP={o['hi'][2]:.3f} "
              f"base_z={o['lo'][2]:.3f} foot={o['foot']*100:.0f}cm "
              f"h={o['height']*100:.0f}cm n={len(o['pts'])}")


if __name__ == "__main__":
    main()
