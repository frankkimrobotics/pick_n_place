#!/usr/bin/env python3
"""real_pnp_servo :: pick-and-place with wrist-D405 visual-servo grasp.

The fixed D435 only proposes a coarse box position; the grasp xy comes from
the wrist D405 looking straight DOWN from hover (close range + hand-eye
T_tcp_cam, spread 6 mm) — immune to the D435 extrinsic error and to the
±15 mm frame-to-frame jitter of the global detection.

    python3 real_pnp_servo.py --obj 0.312,-0.104,0.031 --place 0.42,0.02 --exec
"""
import argparse
import json
import os
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C                                          # noqa: E402
from latency_common import (CMD_LOG, StreamRecorder, close_sender, rpc,  # noqa: E402
                            plan_joint, plan_pose, qrad, read_state,
                            send_chunk, stream_at)
from joint_conventions import rad_to_linuxcnc_deg           # noqa: E402
from real_pnp_online import check_box, suction, suction_events, plot  # noqa: E402

TIP_BELOW_TCP = 0.019
_HE = os.path.join(HERE, "outputs", "handeye_corrected.json")
if not os.path.exists(_HE):
    _HE = os.path.join(HERE, "outputs", "handeye_20260703_004118.json")
T_TCP_CAM = np.array(json.load(open(_HE))["T_tcp_cam"], float)
D405_WRIST = "218622271300"


def fk_T(qd):
    r = rpc({"type": "fk", "q": qrad(qd)})
    p = np.array(r["pos"][0])
    x, y, z, w = r["quat"][0][1], r["quat"][0][2], r["quat"][0][3], r["quat"][0][0]
    # curobo quat is wxyz
    w, x, y, z = r["quat"][0]
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


_CAP_SRC = r'''
import sys, numpy as np, pyrealsense2 as rs
pipe = rs.pipeline(); cfg = rs.config()
cfg.enable_device("%s")
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
prof = pipe.start(cfg)
scale = prof.get_device().first_depth_sensor().get_depth_scale()
intr = prof.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
for _ in range(10):
    frames = pipe.wait_for_frames()
z = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * scale
pipe.stop()
np.savez(sys.argv[1], z=z, fx=intr.fx, fy=intr.fy, ppx=intr.ppx, ppy=intr.ppy)
''' % D405_WRIST


def wrist_cloud_base(qd):
    """Capture wrist D405 depth (system python + libusb librealsense),
    return points in base frame via FK @ hand-eye."""
    import subprocess
    import tempfile
    f = tempfile.mktemp(suffix=".npz")
    r = subprocess.run(["python3", "-c", _CAP_SRC, f],
                       env=dict(os.environ, PYTHONPATH=os.path.expanduser(
                           "~/librealsense/build/release")),
                       capture_output=True, text=True, timeout=120)
    if not os.path.exists(f):
        raise RuntimeError(f"wrist capture failed: {r.stderr[-300:]}")
    d = np.load(f)
    z = d["z"]
    h, w = z.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    ok = (z > 0.03) & (z < 0.5)
    x = (us - d["ppx"]) * z / d["fx"]
    y = (vs - d["ppy"]) * z / d["fy"]
    pc = np.stack([x[ok], y[ok], z[ok]], -1)
    T = fk_T(qd) @ T_TCP_CAM
    return (T[:3, :3] @ pc.T).T + T[:3, 3]


def box_top_center(pb, guess_xy, top_z, ball=0.06, slab=0.015):
    d = np.linalg.norm(pb[:, :2] - np.asarray(guess_xy), axis=1)
    sel = pb[(d < ball) & (np.abs(pb[:, 2] - top_z) < 0.025)]
    if len(sel) < 50:
        return None
    zt = np.percentile(sel[:, 2], 90)
    top = sel[np.abs(sel[:, 2] - zt) < slab]
    if len(top) < 30:
        return None
    return top[:, 0].mean(), top[:, 1].mean(), float(zt)


def leg(label, xyz, v):
    traj = plan_pose(qrad(read_state()[0]), list(xyz))
    assert traj is not None, f"{label} plan failed"
    check_box(traj)
    print(f"[{label}] -> {np.round(xyz,3)} @ {v} deg/s", flush=True)
    stream_at(traj, v)


def descend_press(x, y, z_target):
    """Small re-planned steps to the press depth; tolerate plan stops."""
    z_now = None
    z = None
    while True:
        qd, _ = read_state()
        cur = fk_T(qd)[:3, 3]
        z = cur[2] - 0.006 if z is None else z - 0.006
        if z < z_target:
            z = z_target
        tr = plan_pose(qrad(qd), [x, y, z])
        if tr is None:
            print(f"    plan stop at z={z:.3f}")
            return
        stream_at(np.asarray(tr), 4.0, settle=False)
        time.sleep(0.4)
        if z <= z_target:
            time.sleep(0.6)
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True, help="coarse x,y,top_z from D435")
    ap.add_argument("--place", required=True)
    ap.add_argument("--exec", action="store_true")
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/real_pnp_servo"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ox, oy, otop = map(float, a.obj.split(","))
    px, py = map(float, a.place.split(","))
    if not a.exec:
        print("[dry] would hover over", (ox, oy))
        return

    rec = StreamRecorder()
    rec.start()
    t0 = time.time()
    try:
        leg("hover", [ox, oy, otop + 0.10], 8.0)
        time.sleep(0.8)
        qd, _ = read_state()
        pb = wrist_cloud_base(qd)
        hit = box_top_center(pb, [ox, oy], otop)
        assert hit is not None, "wrist camera found no box top under hover"
        bx, by, btop = hit
        print(f"[servo] wrist-refined box: ({bx:.3f},{by:.3f}) top {btop:.3f} "
              f"(D435 was ({ox:.3f},{oy:.3f}) top {otop:.3f})")
        leg("center", [bx, by, btop + 0.05], 4.0)
        print("[press] descend to press", flush=True)
        descend_press(bx, by, max(btop - TIP_BELOW_TCP - 0.016, -0.005))
        suction(True)
        time.sleep(2.5)
        leg("lift", [bx, by, 0.20], 3.0)
        # grip check: wrist camera should NOT see a box top at table level below
        qd, _ = read_state()
        pb = wrist_cloud_base(qd)
        still = box_top_center(pb, [bx, by], btop)
        if still is not None and abs(still[2] - btop) < 0.03:
            print("[gripcheck] box still on table — seal FAILED")
            suction(False)
        else:
            print("[gripcheck] clear — transporting", flush=True)
            leg("transit", [px, py, 0.20], 5.0)
            print("[place] descend", flush=True)
            descend_press(px, py, btop - TIP_BELOW_TCP + 0.004)
            suction(False)
            time.sleep(1.2)
            leg("retreat", [px, py, 0.15], 3.0)
        tr = plan_joint(qrad(read_state()[0]), [float(v) for v in C.BASE_Q])
        if tr is not None:
            check_box(tr)
            stream_at(tr, 8.0)
    finally:
        try:
            suction(False)
        except Exception:
            pass
        time.sleep(0.5)
        rec.stop()
        close_sender()
        ts, qs, tqs, _ = rec.arrays()
        cmds = [(t, m) for t, m in CMD_LOG if "trajectory" in m]
        np.savez(os.path.join(a.out, "pnp_servo_log.npz"), ts=ts, qs=qs,
                 tqs=tqs, t0=t0, suction=np.array(suction_events),
                 cmd_json=np.array([json.dumps({"t": t, **m}) for t, m in cmds]))
        if len(ts) and cmds:
            plot(ts, qs, tqs, cmds, t0, a.out)


if __name__ == "__main__":
    main()
