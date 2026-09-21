#!/usr/bin/env python3
"""d405_grab :: system-python helper (pyrealsense2 lives there): grab an n-frame median of the aligned colour+depth
from a D405 and save npz {col, dep (m), fx, fy, cx, cy}. Used by rl/wrist_centre.py from the mjwarp env."""
import sys, numpy as np, pyrealsense2 as rs
serial, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 5
pipe, cfg = rs.pipeline(), rs.config(); cfg.enable_device(serial)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30); cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
prof = pipe.start(cfg); align = rs.align(rs.stream.color); ds = prof.get_device().first_depth_sensor().get_depth_scale()
intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
for _ in range(12): pipe.wait_for_frames()
deps = []
for _ in range(n):
    fs = align.process(pipe.wait_for_frames()); col = np.asanyarray(fs.get_color_frame().get_data()).copy()
    deps.append(np.asanyarray(fs.get_depth_frame().get_data()).astype(np.float32) * ds)
pipe.stop()
np.savez(out, col=col, dep=np.median(np.stack(deps), 0), fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy)
