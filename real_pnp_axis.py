#!/usr/bin/env python3
"""real_pnp_axis :: pick-and-place by image-space servoing to the cup's
CONTACT AXIS in the wrist D405 view.

The cup axis pixel was calibrated from direct observation (the physical
contact lands at ~(320,490) at tip depth). At hover, the cap's top-face
blob is detected in the wrist depth image; the arm is stepped laterally
until the blob center coincides with the axis pixel AT THE CAP'S DEPTH
(parallax-corrected along the axis), then descends straight down.
3D-transform biases largely cancel — the servo loop closes on pixels.

    python3 real_pnp_axis.py --obj 0.413,0.178,0.098 --place 0.44,-0.12 --exec
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C                                          # noqa: E402
from latency_common import (CMD_LOG, StreamRecorder, close_sender,  # noqa: E402
                            plan_joint, plan_pose, qrad, read_state,
                            stream_at)
from real_pnp_online import check_box, suction, suction_events, plot  # noqa: E402
from real_pnp_servo import T_TCP_CAM, _CAP_SRC, fk_T  # noqa: E402

TIP_BELOW_TCP = 0.019
TIP_PX = (320.0, 490.0)        # user-observed contact pixel at tip depth


def wrist_depth():
    f = tempfile.mktemp(suffix=".npz")
    r = subprocess.run(["python3", "-c", _CAP_SRC, f],
                       env=dict(os.environ, PYTHONPATH=os.path.expanduser(
                           "~/librealsense/build/release")),
                       capture_output=True, text=True, timeout=120)
    if not os.path.exists(f):
        raise RuntimeError(f"capture failed {r.stderr[-200:]}")
    return np.load(f)


def axis_px_at_depth(zc, d):
    """Pixel where the cup axis crosses camera depth zc (parallax along axis)."""
    R, t = T_TCP_CAM[:3, :3], T_TCP_CAM[:3, 3]
    p_tip_cam = R.T @ (np.array([0, 0, TIP_BELOW_TCP]) - t)
    d_cam = R.T @ np.array([0.0, 0.0, 1.0])
    # calibrate axis xy so that at tip depth it hits TIP_PX exactly
    p0 = p_tip_cam
    u0 = d["fx"] * p0[0] / p0[2] + d["ppx"]
    v0 = d["fy"] * p0[1] / p0[2] + d["ppy"]
    du, dv = TIP_PX[0] - u0, TIP_PX[1] - v0
    s = (zc - p_tip_cam[2]) / d_cam[2]
    p = p_tip_cam + s * d_cam
    return (d["fx"] * p[0] / p[2] + d["ppx"] + du,
            d["fy"] * p[1] / p[2] + d["ppy"] + dv)


CUP_ROI = (330, 480, 220, 460)     # r0,r1,c0,c1 — the cup's own pixels

def cap_blob(d, expected_zc, axis_px, min_pts=300, track_px=None):
    """Cap top-face: largest connected component of pixels at least 2 cm
    ABOVE the measured table plane (table depth = scene median), cup ROI
    excluded. No priors on expected depth or image location."""
    from scipy import ndimage
    z = d["z"].copy()
    h, w = z.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    zv = z[(z > 0.12) & (z < 0.5)]
    if len(zv) < 5000:
        return None
    table = np.median(zv)
    # cup/arm excluded by DEPTH (cup ~0.14 m), not by image region — the
    # target must be allowed to reach the contact-axis pixels
    m = (z > 0.17) & (z < table - 0.02)
    if m.sum() < min_pts:
        return None
    lab, n = ndimage.label(m)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    cands = []
    for k in range(1, n + 1):
        if sizes[k - 1] < min_pts:
            continue
        sel = lab == k
        cands.append((float(us[sel].mean()), float(vs[sel].mean()),
                      float(np.median(z[sel])), sizes[k - 1]))
    if not cands:
        return None
    if track_px is None:
        return max(cands, key=lambda c: c[3])[:3]   # largest on first frame
    return min(cands, key=lambda c: np.hypot(c[0] - track_px[0],
                                             c[1] - track_px[1]))[:3]


def cup_roi_depth(d):
    """Median depth inside the cup's own ROI (grip check: held object is NEAR)."""
    z = d["z"][CUP_ROI[0]:CUP_ROI[1], CUP_ROI[2]:CUP_ROI[3]]
    v = z[(z > 0.03) & (z < 0.6)]
    return float(np.median(v)) if len(v) > 200 else None


def leg(label, xyz, v):
    traj = plan_pose(qrad(read_state()[0]), list(xyz))
    assert traj is not None, f"{label} plan failed"
    check_box(traj)
    print(f"[{label}] -> {np.round(xyz,3)} @ {v} deg/s", flush=True)
    stream_at(traj, v)


from latency_common import rpc as _rpc


def numeric_J(qr):
    """Position jacobian (3x6) via central differences on the fk rpc."""
    J = np.zeros((3, 6))
    eps = np.radians(0.5)
    for j in range(6):
        qa = list(qr); qb = list(qr)
        qa[j] += eps; qb[j] -= eps
        pa = np.array(_rpc({"type": "fk", "q": qa})["pos"][0])
        pb = np.array(_rpc({"type": "fk", "q": qb})["pos"][0])
        J[:, j] = (pa - pb) / (2 * eps)
    return J


def local_move(dxyz, v_deg=3.0):
    """Planner-free minimal move: damped-least-squares joint step streamed
    as a short chunk. |dxyz| should be small (<= 3 cm)."""
    qd, _ = read_state()
    qr = np.array(qrad(qd))
    J = numeric_J(qr)
    dq = J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(3), np.asarray(dxyz))
    n = max(2, int(np.degrees(np.abs(dq).max()) / 0.5))
    traj = qr[None] + np.linspace(0, 1, n)[:, None] * dq[None]
    check_box(traj)
    stream_at(traj, v_deg)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--place", required=True)
    ap.add_argument("--exec", action="store_true")
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/real_pnp_axis"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ox, oy, otop = map(float, a.obj.split(","))
    px, py = map(float, a.place.split(","))
    if not a.exec:
        print("[dry]")
        return
    rec = StreamRecorder()
    rec.start()
    t0 = time.time()
    try:
        leg("hover", [ox, oy, otop + 0.08], 8.0)
        # --- image-space servo: put cap blob on the contact axis ---
        prev_err = 1e9
        converged = False
        GAINS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.4, 0.4, 0.4]
        for it in range(10):
            time.sleep(0.6)
            d = wrist_depth()
            qd, _ = read_state()
            cam_z = (fk_T(qd) @ T_TCP_CAM)[2, 3]
            expected_zc = cam_z - otop
            u_t0, v_t0 = axis_px_at_depth(expected_zc, d)
            blob = cap_blob(d, expected_zc, (u_t0, v_t0),
                            track_px=locals().get("pred_px"))
            assert blob is not None, "no cap blob in wrist view"
            u_c, v_c, zc = blob
            u_t, v_t = axis_px_at_depth(zc, d)
            err_px = np.hypot(u_c - u_t, v_c - v_t)
            prev_err = min(prev_err, err_px)
            # pixel error -> camera-frame lateral -> base frame
            delta_cam = np.array([(u_c - u_t) * zc / d["fx"],
                                  (v_c - v_t) * zc / d["fy"], 0.0])
            qd, _ = read_state()
            Rbc = (fk_T(qd) @ T_TCP_CAM)[:3, :3]
            delta_base = Rbc @ delta_cam
            print(f"[servo {it}] blob ({u_c:.0f},{v_c:.0f}) axis ({u_t:.0f},{v_t:.0f}) "
                  f"err {err_px:.0f}px -> base d=({delta_base[0]*1000:+.0f},"
                  f"{delta_base[1]*1000:+.0f}) mm", flush=True)
            if err_px < 8:
                converged = True
                break
            g = GAINS[it]
            # predict where THIS blob lands after the move (tracker prior)
            pred_px = (u_c + g * (u_t - u_c), v_c + g * (v_t - v_c))
            local_move([np.clip(g * delta_base[0], -0.035, 0.035),
                        np.clip(g * delta_base[1], -0.035, 0.035), 0.0])
        if not converged and prev_err > 12:
            print(f"[servo] NOT converged (best {prev_err:.0f}px) — aborting press")
            tr = plan_joint(qrad(read_state()[0]), [float(v) for v in C.BASE_Q])
            if tr is not None:
                check_box(tr)
                stream_at(tr, 8.0)
            return
        # cap height from wrist depth at converged position
        qd, _ = read_state()
        cam_z = (fk_T(qd) @ T_TCP_CAM)[2, 3]
        cap_top_base = cam_z - zc
        print(f"[servo] cap top z (wrist) = {cap_top_base:.3f}  (D435 said {otop:.3f})")
        # --- straight descend to press ---
        tcp = fk_T(qd)[:3, 3]
        press_z = cap_top_base + TIP_BELOW_TCP - 0.014   # tip 14mm below cap top
        print(f"[press] {tcp[2]:.3f} -> {press_z:.3f}", flush=True)
        z = tcp[2]
        while z > press_z + 1e-4:
            step = min(0.012, z - press_z)
            z -= step
            local_move([0.0, 0.0, -step], v_deg=3.0)
        time.sleep(0.5)
        suction(True)
        time.sleep(2.5)
        leg("lift", [tcp[0], tcp[1], 0.22], 3.0)
        time.sleep(0.5)
        d2 = wrist_depth()
        roi_z = cup_roi_depth(d2)
        print(f"[gripcheck] cup-ROI median depth {roi_z}")
        held = roi_z is not None and roi_z < 0.22   # held object sits close
        if not held:
            print("[gripcheck] nothing on the cup — seal FAILED")
            suction(False)
        else:
            print("[gripcheck] object on cup — transporting", flush=True)
            leg("transit", [px, py, 0.22], 5.0)
            qd, _ = read_state()
            drop_z = (0.22 - cap_top_base) + 0.006  # object bottom ~6mm above table
            print("[place] descend", flush=True)
            z = 0.22
            tgt = cap_top_base + TIP_BELOW_TCP + 0.004
            while z > tgt + 1e-4:
                step = min(0.012, z - tgt)
                z -= step
                local_move([0.0, 0.0, -step], v_deg=3.5)
            suction(False)
            time.sleep(1.2)
            leg("retreat", [px, py, 0.20], 3.0)
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
        np.savez(os.path.join(a.out, "pnp_axis_log.npz"), ts=ts, qs=qs,
                 tqs=tqs, t0=t0, suction=np.array(suction_events),
                 cmd_json=np.array([json.dumps({"t": t, **m}) for t, m in cmds]))
        if len(ts) and cmds:
            plot(ts, qs, tqs, cmds, t0, a.out)


if __name__ == "__main__":
    main()
