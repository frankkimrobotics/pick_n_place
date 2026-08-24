#!/usr/bin/env python3
"""Publish wrist D405 + fixed D435 as 4ch 96x96 uint8 frames to /dev/shm.
System python3 (working librealsense build); policy processes read the npy.
    PYTHONPATH=~/librealsense/build/release python3 policy/rs_shm_server.py
"""
import os, threading, time
import numpy as np, cv2
import pyrealsense2 as rs

IMG = 96

def serve(serial, w, h, fps, name):
    pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    prof = pipe.start(cfg)
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    n = 0
    while True:
        try:
            f = align.process(pipe.wait_for_frames(4000))
            rgb = np.asanyarray(f.get_color_frame().get_data())
            dep = np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32) * scale
            rgb = cv2.resize(rgb, (IMG, IMG), interpolation=cv2.INTER_AREA)
            d8 = (np.clip(cv2.resize(dep, (IMG, IMG), interpolation=cv2.INTER_NEAREST), 0, 2.0) * 127.5).astype(np.uint8)
            out = np.concatenate([rgb, d8[..., None]], -1).transpose(2, 0, 1).copy()
            tmp = f"/dev/shm/rs_{name}.tmp.npy"
            np.save(tmp, out)
            os.replace(tmp, f"/dev/shm/rs_{name}.npy")
            n += 1
            if n % 100 == 1:
                print(f"[{name}] {n} frames, depth ch mean {out[3].mean():.0f}", flush=True)
        except Exception as e:
            print(f"[{name}] {str(e)[:60]}", flush=True)
            time.sleep(0.3)

threading.Thread(target=serve, args=("218622271300", 424, 240, 30, "wrist"), daemon=True).start()
threading.Thread(target=serve, args=("043422070101", 640, 480, 6, "fixed"), daemon=True).start()
while True:
    time.sleep(5)
