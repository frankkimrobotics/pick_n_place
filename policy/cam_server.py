#!/usr/bin/env python3
"""cam_server :: publish wrist D405 (color+center depth) and fixed D435 color
to /dev/shm at ~15 Hz for real_rollout.py. Runs under SYSTEM python3 (the
only interpreter whose librealsense build sees these cameras).

Files (atomic rename):  /dev/shm/pnp_wrist.jpg  /dev/shm/pnp_fixed.jpg
                        /dev/shm/pnp_range.txt  (metres at image center)
    /usr/bin/python3 policy/cam_server.py
"""
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

WRIST_SER = "218622271300"
FIXED_SER = "043422070101"
SHM = "/dev/shm"


def start(ser, fps, depth=False):
    p = rs.pipeline()
    c = rs.config()
    c.enable_device(ser)
    c.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, fps)
    if depth:
        c.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, fps)
    p.start(c)
    return p


def put(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def main():
    wrist = start(WRIST_SER, 30, depth=True)
    fixed = start(FIXED_SER, 15)
    # publish wrist depth intrinsics once (for goal estimation deprojection)
    import json
    prof = wrist.get_active_profile().get_stream(rs.stream.depth)                .as_video_stream_profile().get_intrinsics()
    with open(f"{SHM}/pnp_wrist_intr.json", "w") as f:
        json.dump(dict(fx=prof.fx, fy=prof.fy, ppx=prof.ppx, ppy=prof.ppy,
                       w=prof.width, h=prof.height), f)
    print("[cam_server] running", flush=True)
    n = 0
    while True:
        f = wrist.wait_for_frames(2000)
        col = np.asanyarray(f.get_color_frame().get_data())
        dep = f.get_depth_frame()
        rng = dep.get_distance(212, 120) if dep else 0.0
        ok, buf = cv2.imencode(".jpg", col, [cv2.IMWRITE_JPEG_QUALITY, 95])
        put(f"{SHM}/pnp_wrist.jpg", buf.tobytes())
        put(f"{SHM}/pnp_range.txt", f"{rng:.4f}".encode())
        if dep and n % 5 == 0:
            dm = np.asanyarray(dep.get_data()).astype(np.uint16)
            put(f"{SHM}/pnp_wrist_depth.npy", dm.tobytes())
        f2 = fixed.wait_for_frames(2000)
        col2 = np.asanyarray(f2.get_color_frame().get_data())
        ok, buf2 = cv2.imencode(".jpg", col2, [cv2.IMWRITE_JPEG_QUALITY, 95])
        put(f"{SHM}/pnp_fixed.jpg", buf2.tobytes())
        n += 1
        if n % 150 == 0:
            print(f"[cam_server] {n} frames, range {rng:.3f} m", flush=True)


if __name__ == "__main__":
    main()
