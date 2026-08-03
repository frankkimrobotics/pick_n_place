"""convert_litdata :: pnp_dataset -> litdata streaming dataset for Lightning AI.

One sample per policy step of every SUCCESSFUL pick episode (meta.json
"picks"), sliding over the pick's 10 Hz frame range:

  obs (2 steps, t-1 and t; t-1 clamped at episode start):
    d405_jpg[2], d435_jpg[2]  -- RGB frames re-encoded as JPEG bytes (q=90)
    q[2,6] qd[2,6]            -- joint angles / velocities (rad, rad/s)
    eef_pos[2,3] eef_quat[2,4]-- tcp FK pose (wxyz), from eef.npz
    suction[2] rangefinder[2]
  action (next 1.6 s):
    ctrl_pts[16,6] -- control points of a clamped CUBIC B-spline over
                      s in [0, 1.5]: s maps to absolute time t+0.1+s, fit to
                      q[t+1..t+16] at s = 0,0.1,..,1.5 with a light P-spline
                      ridge (LAMBDA on CP 2nd differences, see below).
                      Evaluate the spline at any rate (e.g. 4 ms) for qref.
    suction_cmd[16] -- suction state at t+1..t+16 (uint8)
  ids: scene, pick, object, t  (frame index within the scene stream)

Knots (fixed, also in dataset_spec.json):
  [0]*4 + [1.5*j/13 for j in 1..12] + [1.5]*4

Run in the sam3 env (torch + litdata + cv2):
    ~/miniconda3/envs/sam3/bin/python convert_litdata.py \
        --out ~/pnp_litdata --workers 8
Scenes missing eef.npz or meta "picks" are skipped (run prep_eef.py /
postprocess_picks.py first); skips are printed, not silent.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
from litdata import optimize

DS = os.path.expanduser("~/pnp_dataset")
HORIZON = 16          # future steps in the action chunk
N_CTRL = 16           # B-spline control points
DEGREE = 3
DT = 0.1              # 10 Hz
# spline parameter s spans the 16 action sites: s = 0 is the FIRST action
# (absolute time t + 0.1 s), s = T_SPAN the last (t + 1.6 s)
T_SPAN = (HORIZON - 1) * DT

KNOTS = np.array([0.0] * (DEGREE + 1)
                 + [T_SPAN * j / (N_CTRL - DEGREE) for j in range(1, N_CTRL - DEGREE)]
                 + [T_SPAN] * (DEGREE + 1))


def bspline_basis(s):
    """Cox-de Boor: basis row (N_CTRL,) for clamped cubic at parameter s."""
    s = min(max(s, 0.0), T_SPAN - 1e-9)
    n_knots = len(KNOTS)
    b = np.zeros((n_knots - 1,))
    for i in range(n_knots - 1):
        b[i] = 1.0 if KNOTS[i] <= s < KNOTS[i + 1] else 0.0
    for p in range(1, DEGREE + 1):
        nb = np.zeros(n_knots - 1 - p)
        for i in range(len(nb)):
            left = right = 0.0
            if KNOTS[i + p] > KNOTS[i]:
                left = (s - KNOTS[i]) / (KNOTS[i + p] - KNOTS[i]) * b[i]
            if KNOTS[i + p + 1] > KNOTS[i + 1]:
                right = (KNOTS[i + p + 1] - s) / (KNOTS[i + p + 1] - KNOTS[i + 1]) * b[i + 1]
            nb[i] = left + right
        b = nb
    return b


# fit design: the 16 future samples sit on the clamped domain [0, 1.5]
# (well-conditioned collocation, cond ~19). Exact interpolation rings badly
# between samples at impact/latch events (executed q has +-250 deg/s velocity
# spikes), so a small P-spline ridge on control-point second differences
# smooths those: p99 fit error 0.68 deg (max ~6 deg only at impact frames),
# dense-evaluation overshoot p99 0.31 deg.
LAMBDA = 1e-3
FIT_S = np.array([DT * j for j in range(HORIZON)])
BASIS = np.stack([bspline_basis(s) for s in FIT_S])          # (16, 16)
_D2 = np.zeros((N_CTRL - 2, N_CTRL))
for _i in range(N_CTRL - 2):
    _D2[_i, _i:_i + 3] = [1.0, -2.0, 1.0]
FIT_MAT = np.linalg.solve(BASIS.T @ BASIS + LAMBDA * _D2.T @ _D2, BASIS.T)


def read_video_jpegs(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        assert ok
        out.append(buf.tobytes())
    cap.release()
    return out


def scene_samples(sdir):
    meta = json.load(open(os.path.join(sdir, "meta.json")))
    scene = meta["scene"]
    t = np.load(os.path.join(sdir, "traj.npz"))
    e = np.load(os.path.join(sdir, "eef.npz"))
    q, qd = t["q"].astype(np.float32), t["qd"].astype(np.float32)
    rf = t["rangefinder"].astype(np.float32)
    suc = t["suction"].astype(np.uint8)
    epos, equat = e["pos"].astype(np.float32), e["quat"].astype(np.float32)
    j405 = read_video_jpegs(os.path.join(sdir, "d405_rgb.mp4"))
    j435 = read_video_jpegs(os.path.join(sdir, "d435_rgb.mp4"))
    n = min(len(q), len(j405), len(j435))
    for pi, pk in enumerate(meta["picks"]):
        if not pk["success"] or pk["frame_latch"] < 0:
            continue
        f0, f1 = pk["frame_start"], min(pk["frame_end"], n)
        for tt in range(f0, f1 - HORIZON):
            tp = max(f0, tt - 1)
            fut = q[tt + 1: tt + 1 + HORIZON]               # (16, 6)
            ctrl = (FIT_MAT @ fut).astype(np.float32)       # (16, 6)
            yield dict(
                scene=scene, pick=pi, object=pk["object"], t=tt,
                d405_jpg=[j405[tp], j405[tt]],
                d435_jpg=[j435[tp], j435[tt]],
                q=q[[tp, tt]], qd=qd[[tp, tt]],
                eef_pos=epos[[tp, tt]], eef_quat=equat[[tp, tt]],
                suction=suc[[tp, tt]], rangefinder=rf[[tp, tt]],
                ctrl_pts=ctrl,
                suction_cmd=suc[tt + 1: tt + 1 + HORIZON].copy(),
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_litdata"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N scenes only")
    a = ap.parse_args()

    scenes = []
    for sdir in sorted(glob.glob(os.path.join(DS, "scene_*"))):
        mp = os.path.join(sdir, "meta.json")
        if not os.path.exists(mp) or not os.path.exists(os.path.join(sdir, "eef.npz")):
            print(f"[skip] {os.path.basename(sdir)} (missing meta/eef)")
            continue
        if "picks" not in json.load(open(mp)):
            print(f"[skip] {os.path.basename(sdir)} (no picks)")
            continue
        scenes.append(sdir)
    if a.limit:
        scenes = scenes[:a.limit]
    print(f"[convert] {len(scenes)} scenes -> {a.out}")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "dataset_spec.json"), "w") as f:
        json.dump(dict(
            obs_steps=2, horizon=HORIZON, dt=DT, chunk_seconds=HORIZON * DT,
            t_span=T_SPAN,
            n_ctrl=N_CTRL, degree=DEGREE, knots=KNOTS.tolist(),
            fit_s=FIT_S.tolist(), fit_lambda=LAMBDA,
            fit="ridge on ctrl-pt 2nd differences: "
                "C = (B'B + lambda*D2'D2)^-1 B' q_future",
            action="q [rad]; evaluate clamped cubic B-spline(ctrl_pts, knots) "
                   "on s in [0,1.5], where s=0 is absolute time t+0.1 s "
                   "(the first action) -- qref(t+0.1+s); suction_cmd is "
                   "per-0.1s state at t+0.1..t+1.6",
            images="d405_jpg/d435_jpg: [t-1, t] JPEG bytes, 432x240 BGR-encoded",
            quat="wxyz", n_scenes=len(scenes)), f, indent=1)

    optimize(fn=scene_samples, inputs=scenes, output_dir=a.out,
             num_workers=a.workers, chunk_bytes="128MB", mode="overwrite")
    print("[convert] done")


if __name__ == "__main__":
    main()
