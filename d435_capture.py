#!/usr/bin/env python3
"""Grab one aligned RGB+depth frame from the fixed D435 and save to files.
Runs under the system python3 with PYTHONPATH=~/librealsense/build/release
(the pip pyrealsense2 in the conda envs cannot enumerate devices).
    PYTHONPATH=~/librealsense/build/release python3 d435_capture.py --out /tmp/d435_frame
"""
import argparse
import json
import os

import numpy as np
import pyrealsense2 as rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="043422070101")
    ap.add_argument("--out", default="/tmp/d435_frame")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(a.serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
    prof = pipe.start(cfg)
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    for _ in range(12):
        pipe.wait_for_frames(6000)
    f = align.process(pipe.wait_for_frames(6000))
    bgr = np.asanyarray(f.get_color_frame().get_data())
    depth = np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32) * scale
    it = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    pipe.stop()
    np.save(os.path.join(a.out, "bgr.npy"), bgr)
    np.save(os.path.join(a.out, "depth.npy"), depth)
    json.dump({"fx": it.fx, "fy": it.fy, "ppx": it.ppx, "ppy": it.ppy},
              open(os.path.join(a.out, "K.json"), "w"))
    print(f"saved frame to {a.out} (depth median "
          f"{np.median(depth[depth > 0]):.3f} m)")


if __name__ == "__main__":
    main()
