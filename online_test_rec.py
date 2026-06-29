#!/usr/bin/env python3
"""Record /mycobot/drive_feedback (pos deg + vel deg/s) for --secs, save npz.
Used to verify the online welder produces smooth, velocity-continuous motion."""
import argparse, os, time
import numpy as np, rclpy
from sensor_msgs.msg import JointState

ap = argparse.ArgumentParser()
ap.add_argument("--secs", type=float, default=14.0)
ap.add_argument("--out", default="/tmp/sim_online/fb.npz")
a = ap.parse_args()

rclpy.init()
node = rclpy.create_node("fb_rec")
rec = []
node.create_subscription(JointState, "/mycobot/drive_feedback",
                         lambda m: rec.append((time.time(),
                                               [float(x) for x in m.position],
                                               [float(x) for x in m.velocity])), 100)
t0 = time.time()
while time.time() - t0 < a.secs:
    rclpy.spin_once(node, timeout_sec=0.02)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
if rec:
    t = np.array([r[0] for r in rec]) - rec[0][0]
    pos = np.array([r[1] for r in rec]); vel = np.array([r[2] for r in rec])
    np.savez(a.out, t=t, pos=pos, vel=vel)
    print(f"recorded {len(rec)} samples ({t[-1]:.1f}s) -> {a.out}")
else:
    print("no feedback recorded")
rclpy.shutdown()
