#!/usr/bin/env python3
"""calib_cup_sweep :: board-free D435 extrinsic correction using the robot
as the calibration target.

The arm sweeps the suction cup to N known FK poses (down-pointing, spread in
xyz); at each pose the D435 point cloud is captured and the cup TIP is
detected as the centroid of the lowest slab inside a search ball around the
predicted tip. The rigid correction T_corr (Kabsch) maps camera-derived tip
positions onto FK truth; the new extrinsic T_corr @ T_old overwrites
outputs/extrinsics_d435.json (old file backed up).

FK-vs-physical-tip: the calibrated just-touch depth (TOUCH_CALIB=0.019) means
the physical cup tip sits 19 mm below the URDF tcp frame; FK targets are
shifted accordingly before matching.

    python3 calib_cup_sweep.py --exec
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C                                          # noqa: E402
from latency_common import (close_sender, fk, plan_joint, plan_pose, qrad,  # noqa: E402
                            read_state, stream_at)

TIP_BELOW_TCP = 0.019
SCRATCH = os.environ.get("CAL_SCRATCH", "/tmp/cup_sweep")
POSES = [                     # xy spread + two heights for rotation observability
    (0.30, -0.12, 0.10), (0.30, 0.12, 0.10),
    (0.41, -0.12, 0.10), (0.41, 0.12, 0.10),
    (0.37, 0.00, 0.16), (0.30, 0.00, 0.13),
    (0.41, 0.00, 0.13), (0.37, -0.12, 0.16),
    (0.34, 0.06, 0.10), (0.40, -0.06, 0.12),
    (0.33, -0.06, 0.14), (0.37, 0.10, 0.12),
]


def capture(tag):
    out = os.path.join(SCRATCH, tag)
    r = subprocess.run(["python3", os.path.join(HERE, "d435_capture.py"),
                        "--out", out],
                       env=dict(os.environ, PYTHONPATH=os.path.expanduser(
                           "~/librealsense/build/release")),
                       capture_output=True, text=True, timeout=120)
    if not os.path.exists(os.path.join(out, "depth.npy")):
        raise RuntimeError(f"capture failed: {r.stdout[-200:]} {r.stderr[-200:]}")
    return out


def cloud_base(frame_dir, Tbc):
    z = np.load(os.path.join(frame_dir, "depth.npy")).astype(np.float32)
    ki = json.load(open(os.path.join(frame_dir, "K.json")))       # depth already m
    h, w = z.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    ok = (z > 0.15) & (z < 1.2)
    x = (us - ki["ppx"]) * z / ki["fx"]
    y = (vs - ki["ppy"]) * z / ki["fy"]
    pc = np.stack([x[ok], y[ok], z[ok]], -1)
    return (Tbc[:3, :3] @ pc.T).T + Tbc[:3, 3]


def find_tip(pb, predicted, ball=0.06, slab=0.02):
    # z-band tight around the prediction so the TABLE below the hovering
    # cup can never be the "lowest slab"
    d = np.linalg.norm(pb[:, :2] - predicted[:2], axis=1)
    sel = pb[(d < ball) & (pb[:, 2] > predicted[2] - 0.04)
             & (pb[:, 2] < predicted[2] + 0.05)]
    if len(sel) < 12:
        return None
    zmin = sel[:, 2].min()
    tip = sel[sel[:, 2] < zmin + slab]
    if len(tip) < 8:
        return None
    return tip.mean(0)


def kabsch(A, B):
    """T with B ~= R A + t."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = cb - R @ ca
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec", action="store_true")
    ap.add_argument("--translation-only", action="store_true")
    ap.add_argument("--write", action="store_true", help="write new extrinsic")
    a = ap.parse_args()
    os.makedirs(SCRATCH, exist_ok=True)
    ext_path = os.path.join(C.OUT_DIR, "extrinsics_d435.json")
    ext = json.load(open(ext_path))
    Tbc = np.array(ext["T_base_cam435"], float)

    pairs = []
    qd, _ = read_state()
    for i, (x, y, z) in enumerate(POSES):
        traj = None
        for _ in range(3):
            traj = plan_pose(qrad(qd), [x, y, z])
            if traj is not None:
                break
            time.sleep(0.5)
        if traj is None:
            print(f"[{i}] plan failed for {x,y,z}, skipping")
            continue
        if not a.exec:
            print(f"[{i}] plan ok -> {x,y,z} (dry)")
            continue
        stream_at(traj, 9.0)
        time.sleep(0.8)
        qd_here, _ = read_state()
        p_fk = np.array(fk(qd_here))
        p_tip_fk = p_fk - np.array([0, 0, TIP_BELOW_TCP])
        pbs = [cloud_base(capture(f"pose{i}_{k}"), Tbc) for k in range(3)]
        pb = np.concatenate(pbs)
        det = find_tip(pb, p_tip_fk)
        if det is None:
            print(f"[{i}] tip not found near {np.round(p_tip_fk,3)}")
            stream_at(np.asarray(traj)[::-1], 9.0)
            continue
        err = det - p_tip_fk
        print(f"[{i}] fk_tip {np.round(p_tip_fk,3)}  cam_tip {np.round(det,3)}"
              f"  err {np.round(err*1000,1)} mm")
        pairs.append((det, p_tip_fk))
        stream_at(np.asarray(traj)[::-1], 9.0)   # planner-free return to base

    if not a.exec:
        return
    close_sender()
    if len(pairs) < 4:
        sys.exit(f"only {len(pairs)} pairs — not enough")
    A = np.array([p[0] for p in pairs])
    B = np.array([p[1] for p in pairs])
    # median-based outlier rejection on the per-pair error vectors
    E = B - A
    med = np.median(E, axis=0)
    inl = np.linalg.norm(E - med, axis=1) < 0.015
    print(f"inliers: {inl.sum()}/{len(E)} (median err {np.round(med*1000,1)} mm)")
    if inl.sum() >= 3:
        A, B = A[inl], B[inl]
    if a.translation_only or inl.sum() < 5:
        T = np.eye(4)
        T[:3, 3] = (B - A).mean(0)
        print("[fit] translation-only")
    else:
        T = kabsch(A, B)
    res_before = np.linalg.norm(B - A, axis=1)
    Ah = (T[:3, :3] @ A.T).T + T[:3, 3]
    res_after = np.linalg.norm(B - Ah, axis=1)
    print(f"\nresidual before: mean {res_before.mean()*1000:.1f} mm  "
          f"max {res_before.max()*1000:.1f} mm")
    print(f"residual after : mean {res_after.mean()*1000:.1f} mm  "
          f"max {res_after.max()*1000:.1f} mm")
    print("correction translation (mm):", np.round(T[:3, 3] * 1000, 1))
    ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
    print(f"correction rotation: {ang:.2f} deg")
    if a.write:
        shutil.copy(ext_path, ext_path + ".bak")
        ext["T_base_cam435"] = (T @ Tbc).tolist()
        ext["calib_note"] = "cup-sweep correction " + time.strftime("%Y-%m-%d %H:%M")
        json.dump(ext, open(ext_path, "w"), indent=1)
        print(f"[write] new extrinsic -> {ext_path} (backup .bak)")


if __name__ == "__main__":
    main()
