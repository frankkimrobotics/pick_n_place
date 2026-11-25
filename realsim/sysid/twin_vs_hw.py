#!/usr/bin/env python3
"""Replay the 2026-11-14 hardware tracking runs (old and tuned stream gains) through the twin: same reference,
same gains. Reports hardware vs twin tracking error and plots both.   (mj4 env)"""
import glob, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
from look import load
import mujoco
from realsim.validate import settle, XML, Q_START
from realsim.twin import Pro630Twin
from realsim.config import make_cfg
D = os.path.expanduser("~/pnp_real_runs/sysid")
RUNS = [("sine_012345_fast_old", 1, "1 Hz 3 deg sine, deployed gains"), ("sine_012345_fast_new", 1, "1 Hz 3 deg sine, tuned gains"),
        ("sine_1_base", 1, "0.5 Hz 4 deg sine, deployed gains"), ("sine_1_u_h", 1, "0.5 Hz 4 deg sine, tuned gains"),
        ("check_j1_stepold", 1, "3 deg at 30 deg/s, deployed gains"), ("check_j1_stepnew", 1, "3 deg at 30 deg/s, tuned gains")]

def lag_rms(t, q, tr, r):
    best = min(((np.sqrt(((q - np.interp(t - L, tr, r)) ** 2).mean()), L) for L in np.arange(-0.08, 0.1, 0.002)))
    return 1e3 * best[1], float(np.sqrt(((q - np.interp(t, tr, r)) ** 2).mean())), float(np.abs(q - np.interp(t, tr, r)).max())

out, TR = {}, {}
for c, (tag, j, title) in enumerate(RUNS):
    f = sorted(glob.glob(f"{D}/*{tag}_pi.csv"))[-1]; Lg = load(f); ref = np.load(f[:-7] + "_ref.npy"); meta = json.load(open(f[:-7] + "_meta.json"))
    g = meta["gains"]
    over = dict(k0=g.get("k0", 10.0), k1=g.get("k1", 0.3), vff=g.get("vff", 1.0), lead=g.get("lead", 0.0),
                lead_p=g.get("lead_p"), vfric=np.radians(g.get("vfric", 0.0)), vmax=np.radians(g.get("vmax", 40.0)),
                vgain=1.025 * g.get("vel_scale", 17.0) / 17.0, tx_spike_p=0.0)
    over.update({k: (np.radians(v) if k == "amax" else v) for k, v in json.loads(os.environ.get("TW_OVER", "{}")).items()})
    m_ = mujoco.MjModel.from_xml_path(XML); d_ = mujoco.MjData(m_); d_.qpos[:6] = Q_START; mujoco.mj_forward(m_, d_)
    tw = Pro630Twin(m_, d_, make_cfg("real", **over))
    settle(tw)
    q0 = tw.d.qpos[:6].copy()
    pts = q0[None] + np.radians(ref - ref[0])
    t_a = tw.t + 0.4
    tw.send_spline(pts, t_a, dt=0.01)
    tw.log = []
    tw.advance(0.4 + len(ref) * 0.01 + 0.3)
    ts = np.array([r["t"] for r in tw.log]) - t_a; qs = np.degrees(np.array([r["q"][j] for r in tw.log]) - q0[j])
    tr = np.arange(len(ref)) * 0.01; r = ref[:, j] - ref[0, j]
    th = Lg["t"] - meta["t_anchor"]; qh = Lg["q"][:, j] - ref[0, j]
    kh = (th > 0) & (th < tr[-1]); ks = (ts > 0) & (ts < tr[-1])
    H = lag_rms(th[kh], qh[kh], tr, r); S = lag_rms(ts[ks], qs[ks], tr, r)
    diff = float(np.sqrt(((qs[ks] - np.interp(ts[ks], th, qh)) ** 2).mean()))
    out[tag] = dict(hw=dict(lag_ms=round(H[0]), rms=round(H[1], 3), max=round(H[2], 3)), twin=dict(lag_ms=round(S[0]), rms=round(S[1], 3), max=round(S[2], 3)), twin_vs_hw_rms=round(diff, 3))
    print(tag, out[tag], flush=True)
    TR.update({f"{c}_tr": tr, f"{c}_r": r, f"{c}_th": th[kh], f"{c}_qh": qh[kh], f"{c}_ts": ts[ks], f"{c}_qs": qs[ks]})
np.savez(f"{D}/twin_vs_hw_traces.npz", **TR)
json.dump(out, open(f"{D}/twin_vs_hw.json", "w"), indent=1); print("saved")
