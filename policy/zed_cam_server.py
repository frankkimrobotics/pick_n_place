#!/usr/bin/env python3
"""zed_cam_server :: publish the wrist D405 colour stream and the fixed ZED 2i LEFT image to
/dev/shm for policy/real_rollout_zed.py (the UMI-style policy trained on ~/pnp_dataset_zed).

Runs under SYSTEM python3: the only interpreter with both pyzed (ZED SDK, py3.8) and the
librealsense build (PYTHONPATH=~/librealsense/build/release).

Files (atomic rename, so a reader never sees a torn JPEG):
    /dev/shm/pnp_wrist.jpg    wrist D405 colour 424x240 BGR  (as the sim wrist_d405 render)
    /dev/shm/pnp_zed.jpg      ZED left image resized 1280x720 -> 432x240 BGR
                              (sim: 424x240 render with fovy 69.1 deg, written by imageio as 432x240;
                              the real HD720 left camera has v_fov 69.1 deg, so a plain resize gives the
                              same field of view -- no crop)
    /dev/shm/pnp_range.txt    wrist depth at the image centre (m), diagnostics only
    /dev/shm/pnp_cam_stamp.json {"wrist": t, "zed": t, "n": frames}

    PYTHONPATH=~/librealsense/build/release /usr/bin/python3 policy/zed_cam_server.py [--fps 15]
"""
import argparse
import io
import json
import os
import sys
import threading
import time

import cv2
import numpy as np

SHM = "/dev/shm"
WRIST_SER = "218622271300"


def put(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def jpg(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes()


class Stamps:
    def __init__(self):
        self.lock = threading.Lock()
        self.d = {"wrist": 0.0, "zed": 0.0, "n": 0}

    def set(self, key, t):
        with self.lock:
            self.d[key] = t
            self.d["n"] += 1
            put(f"{SHM}/pnp_cam_stamp.json", json.dumps(self.d).encode())


def wrist_loop(stamps, fps):
    import pyrealsense2 as rs
    p = rs.pipeline()
    c = rs.config()
    c.enable_device(WRIST_SER)
    c.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)
    c.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
    p.start(c)
    print("[cam] wrist D405 streaming", flush=True)
    n = 0
    while True:
        f = p.wait_for_frames(3000)
        col = np.asanyarray(f.get_color_frame().get_data())
        dep = f.get_depth_frame()
        rng = dep.get_distance(212, 120) if dep else 0.0
        put(f"{SHM}/pnp_wrist.jpg", jpg(col))
        put(f"{SHM}/pnp_range.txt", f"{rng:.4f}".encode())
        stamps.set("wrist", time.time())
        n += 1
        if n % 300 == 0:
            print(f"[cam] wrist {n} frames, centre range {rng:.3f} m", flush=True)


def zed_loop(stamps, fps):
    import pyzed.sl as sl
    z = sl.Camera()
    ip = sl.InitParameters()
    ip.camera_resolution = sl.RESOLUTION.HD720
    ip.camera_fps = 30
    ip.depth_mode = getattr(sl.DEPTH_MODE, os.environ.get('ZED_DEPTH', 'NEURAL'))
    ip.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    ip.coordinate_units = sl.UNIT.METER
    ip.sdk_verbose = 0
    st = z.open(ip)
    if st != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"[cam] ZED open failed: {st}")
    info = z.get_camera_information()
    cc = info.camera_configuration
    lc = cc.calibration_parameters.left_cam
    print(f"[cam] {info.camera_model} sn {info.serial_number} {cc.resolution.width}x{cc.resolution.height} "
          f"v_fov {lc.v_fov:.1f} deg (sim fovy 69.1)", flush=True)
    img = sl.Mat()
    xyz = sl.Mat()
    rt = sl.RuntimeParameters()
    n = 0
    period = 1.0 / fps
    t_next = time.time()
    while True:
        if z.grab(rt) != sl.ERROR_CODE.SUCCESS:
            time.sleep(0.005)
            continue
        now = time.time()
        if now < t_next:
            continue
        t_next = now + period
        z.retrieve_image(img, sl.VIEW.LEFT)
        a = img.get_data()[:, :, :3]
        small = cv2.resize(a, (432, 240), interpolation=cv2.INTER_AREA)
        put(f"{SHM}/pnp_zed.jpg", jpg(small))
        # publish a decimated XYZ map (base-frame conversion is the consumer's job) so
        # object detection / graspability scoring never has to take the camera away.
        if n % max(1, int(fps // 3)) == 0:
            z.retrieve_measure(xyz, sl.MEASURE.XYZ)
            P = np.ascontiguousarray(xyz.get_data()[::4, ::4, :3]).astype(np.float16)
            buf = io.BytesIO(); np.save(buf, P)
            put(f"{SHM}/pnp_zed_xyz.npy", buf.getvalue())
        stamps.set("zed", now)
        n += 1
        if n % 300 == 0:
            print(f"[cam] zed {n} frames", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=15.0, help="ZED publish rate (the policy samples at 10 Hz)")
    ap.add_argument("--no_wrist", action="store_true")
    a = ap.parse_args()
    stamps = Stamps()
    threads = [threading.Thread(target=zed_loop, args=(stamps, a.fps), daemon=True)]
    if not a.no_wrist:
        threads.append(threading.Thread(target=wrist_loop, args=(stamps, a.fps), daemon=True))
    for t in threads:
        t.start()
    print("[cam] running", flush=True)
    while True:
        time.sleep(1.0)
        if not all(t.is_alive() for t in threads):
            print("[cam] a camera thread died -> exit", flush=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
