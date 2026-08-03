"""SAFE real-robot perception test (NO arm motion): capture the fixed D435, run SAM3, deproject each
instance to a base-frame grasp point. Validates the real detection path before any grasp.

  python3 il/real_detect_test.py
"""
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "mycobot_mpc")))
import config as C
from geometry import transform_points

D435 = "043422070101"


def capture(serial=D435, w=848, h=480):
    import pyrealsense2 as rs
    pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
    prof = pipe.start(cfg); align = rs.align(rs.stream.color)
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    for _ in range(12):
        pipe.wait_for_frames(2000)
    f = align.process(pipe.wait_for_frames(2000))
    color = np.asanyarray(f.get_color_frame().get_data())                 # BGR
    depth = np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32) * scale
    intr = f.get_color_frame().profile.as_video_stream_profile().intrinsics
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.0]])
    pipe.stop()
    return color, depth, K


def sam3_labels(rgb, endpoint="tcp://127.0.0.1:5599", prompt="object", max_instances=10, timeout_ms=15000):
    import zmq
    h, w = rgb.shape[:2]
    s = zmq.Context.instance().socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0); s.setsockopt(zmq.RCVTIMEO, timeout_ms); s.setsockopt(zmq.SNDTIMEO, timeout_ms)
    s.connect(endpoint)
    hdr = {"h": h, "w": w, "encoding": "rgb8", "prompt": prompt, "max_instances": max_instances}
    s.send_multipart([json.dumps(hdr).encode(), np.ascontiguousarray(rgb).tobytes()])
    parts = s.recv_multipart()
    return np.frombuffer(parts[1], np.uint8).reshape(h, w)


def deproject_centroid(mask, depth, K, T, dmin=0.05, dmax=1.6):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    valid = mask & (depth > dmin) & (depth < dmax)
    vs, us = np.where(valid)
    if len(vs) < 60:
        return None, 0
    z = depth[vs, us]; x = (us - cx) / fx * z; y = (vs - cy) / fy * z
    pts = transform_points(T, np.stack([x, y, z], 1))
    return np.median(pts, 0), len(vs)                                    # robust centroid


def main():
    import cv2
    color, depth, K = capture()
    rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    T = np.array(json.load(open(os.path.join(C.OUT_DIR, "extrinsics_d435_static.json")))["T_base_cam435"], float)
    label = sam3_labels(rgb)
    ids = [int(i) for i in np.unique(label) if i > 0]
    print(f"SAM3: {len(ids)} instance(s); depth valid {int((depth>0.05).mean()*100)}% of frame")
    dets = []
    for i in ids:
        c, npx = deproject_centroid(label == i, depth, K, T)
        if c is not None:
            dets.append({"id": int(i), "xyz": [float(c[0]), float(c[1]), float(c[2])], "px": int(npx)})
            print(f"  obj {i}: base [{c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f}]  ({npx} px)")
    # save an overlay for inspection
    ov = color.copy()
    for i in ids:
        m = (label == i)
        ov[m] = (0.5 * ov[m] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    cv2.imwrite(os.path.join(C.OUT_DIR, "real_detect_overlay.png"), ov)
    print(f"reachable-looking (x 0.25-0.55, y +-0.2): "
          f"{sum(1 for d in dets if 0.25<d['xyz'][0]<0.55 and abs(d['xyz'][1])<0.22)}/{len(dets)}")
    print("overlay -> outputs/real_detect_overlay.png")


if __name__ == "__main__":
    main()
