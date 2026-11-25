#!/usr/bin/env python3
"""realsim.validate :: check the twin against the hardware measurements.

  1. step   : joint-1 10 deg step, robot_hal point-move law K0=6   (hw 2026-09-17, accel 4x)
  2. sine   : J2 4 deg / 0.35 Hz streamed through the spline path   (hw 2026-11-07: rms 0.28 deg)
  3. replay : the recorded real ZED-policy runs (~/pnp_real_runs/zed_2026*/steps.jsonl):
              feed the logged 10 Hz q_cmd stream at its logged timing through the twin and
              compare the simulated posfb with the logged q_meas, free-running per segment.
Each test runs at level "real" and "ideal" (the old harness assumption) side by side.

    conda activate mj4 && python realsim/validate.py [--runs 'zed_20261108_*'] [--out DIR]
"""
import argparse
import glob
import json
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from realsim.config import make_cfg  # noqa: E402
from realsim.twin import Pro630Twin  # noqa: E402

XML = os.path.expanduser("~/pnp_dataset_zed/scene_090/scene.xml")
Q_START = np.array([0.0, -0.34907, 1.39626, 0.17453, -1.5708, 0.0])
HW = dict(step=dict(onset_ms="36-52", rise_s="0.24-0.27", overshoot_pct="0.0-0.1", settle_s="0.42-0.48"),
          sine=dict(rms_deg=0.28))


def new_twin(level, q0=Q_START):
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    d.qpos[:6] = q0
    mujoco.mj_forward(m, d)
    tw = Pro630Twin(m, d, make_cfg(level))
    return tw


def settle(tw, s=0.5):
    tw.point_target = tw.d.qpos[:6].copy()
    tw.advance(s)
    tw.point_target = None


def test_step(level, j=0, amp=10.0):
    tw = new_twin(level)
    settle(tw)
    q0 = tw.d.qpos[j]
    tgt = tw.d.qpos[:6].copy()
    tgt[j] += np.radians(amp)
    tw.point_target = tgt
    t0, tr = tw.t, []
    for _ in range(150):
        tw.tick()
        tr.append((tw.t - t0, np.degrees(tw.d.qpos[j] - q0)))
    t, y = np.array(tr).T
    onset = t[np.argmax(np.abs(y) > 0.02)]
    rise = t[np.argmax(y > 0.9 * amp)] - t[np.argmax(y > 0.1 * amp)]
    over = max(0.0, (y.max() - amp) / amp * 100)
    outside = np.where(np.abs(y - amp) > 0.5)[0]
    settle_t = t[outside[-1] + 1] if len(outside) and outside[-1] + 1 < len(t) else np.nan
    return dict(onset_ms=round(1000 * onset, 1), rise_s=round(rise, 3), overshoot_pct=round(over, 2),
                settle_s=round(float(settle_t), 3))


class Streamer:
    """Desktop side of PiLink._send_spline_segment: last 4 targets + 1 extrapolated point,
    anchor = t_send - 2 dt snapped to a fixed grid; `hold` resets the history (flat)."""

    def __init__(self, tw, dt=0.1):
        self.tw, self.dt, self.h, self.t0 = tw, dt, [], None

    def send(self, q_from, q_to, t_k, hold=False, t_anchor_k=None):
        if hold:
            self.h = [np.asarray(q_to, float)] * 4
        else:
            if not self.h:
                self.h.append(np.asarray(q_from, float))
            self.h.append(np.asarray(q_to, float))
            self.h = self.h[-4:]
        h = list(self.h)
        while len(h) < 4:
            h.insert(0, h[0])
        ext = h[-1] + (h[-1] - h[-2])
        ta = (t_k if t_anchor_k is None else t_anchor_k) - 2 * self.dt
        if self.t0 is None:
            self.t0 = ta
        ta = self.t0 + round((ta - self.t0) / self.dt) * self.dt
        self.tw.send_spline(np.stack(h + [ext]), ta, self.dt)


def test_sine(level, j=1, amp=4.0, f=0.35, dur=6.0):
    tw = new_twin(level)
    settle(tw)
    st = Streamer(tw)
    qc = tw.d.qpos[:6].copy()
    base = qc.copy()
    errs = []
    for k in range(int(dur / 0.1)):
        t_k = tw.t
        q_to = base.copy()
        q_to[j] += np.radians(amp) * np.sin(2 * np.pi * f * (k + 1) * 0.1)
        st.send(qc, q_to, t_k)
        qc = q_to
        for _ in range(10):
            tw.tick()
            # anchored schedule: the reference value at time t (the target sent at k is due at t_k + dt)
            q_sched = base[j] + np.radians(amp) * np.sin(2 * np.pi * f * (tw.t - t_k + k * 0.1))
            if k > 10:
                errs.append(np.degrees(tw.d.qpos[j] - q_sched))
    e = np.array(errs)
    # tracking delay = lag that best aligns the response with the schedule
    return dict(rms_deg=round(float(np.sqrt(np.mean(e ** 2))), 3), max_deg=round(float(np.abs(e).max()), 3))


def load_run(path):
    rows = [json.loads(l) for l in open(path)]
    return rows


def test_replay(level, run_glob, z_min=0.06):
    files = sorted(glob.glob(os.path.expanduser(f"~/pnp_real_runs/{run_glob}/steps.jsonl")))
    per_run, all_err, all_lead_sim, all_lead_real = [], [], [], []
    for fp in files:
        rows = load_run(fp)
        if len(rows) < 20:
            continue
        t = np.array([r["t"] for r in rows])
        args = json.load(open(os.path.join(os.path.dirname(fp), "args.json")))
        ddt = 0.1 * float(args.get("slow", 1.0))          # decision period AND spline knot spacing
        kk = np.array([r["k"] for r in rows], float)
        t_grid = (t - kk * ddt).min() - 0.005 + kk * ddt  # loop start t0 + k*dt (rows log the END of decision k)
        qm = np.array([r["q_meas"] for r in rows])
        qc = np.array([r["q_cmd"] for r in rows])
        z = np.array([r["tip"][2] for r in rows])
        hold = [r.get("hold") is not None for r in rows]
        # contiguous POLICY-phase stretches: scripted legs (splice align / recover / place) move the
        # arm without logging q_cmd and leave time gaps -> break there and restart from q_meas
        plan = [r.get("plan") is not None for r in rows]
        jump = np.r_[0.0, np.degrees(np.abs(np.diff(qc, axis=0))).max(1)]
        bad = (np.r_[np.diff(t) > 0.25, False] | ~np.array(plan) | (jump > 4.0)
               | np.array([isinstance(r.get("hold"), str) and "splice" in r["hold"] for r in rows]))
        brk = np.where(bad)[0]
        segs, s0 = [], 0
        for b in list(brk) + [len(rows) - 1]:
            if b - s0 >= 15:
                segs.append((s0, b))
            s0 = b + 1
        errs = []
        for a, b in segs:
            if np.degrees(np.abs(qm[a:b + 1] - qm[a]).max()) < 1.0:
                continue
            tw = new_twin(level, q0=qm[a])
            st = Streamer(tw, dt=ddt)
            q_prev = qm[a].copy()
            # real timing (policy/real_rollout_zed.py): decision k starts at t0 + k*dt (dt = 0.1*slow),
            # reads q_meas there, sends its segment when processing ends (= logged t) anchored at t_k
            for k in range(a, b):
                g_k = t_grid[k] - t_grid[a]
                t_send = max(t[k] - t_grid[a], g_k)
                tw.advance(max(0.0, round((t_send - tw.t) / 0.01) * 0.01))
                st.send(q_prev, qc[k], tw.t, hold=hold[k], t_anchor_k=g_k)
                q_prev = qc[k]
                t_read = max(t_grid[k + 1] - t_grid[a], tw.t)
                tw.advance(max(0.0, round((t_read - tw.t) / 0.01) * 0.01))
                q_sim, _, _ = tw.joint_feedback()
                if z[k + 1] > z_min:                       # no contact on the real arm
                    e = np.degrees(q_sim - qm[k + 1])
                    errs.append(e)
                    all_lead_sim.append(np.degrees(np.abs(qc[k] - q_sim)).max())
                    all_lead_real.append(np.degrees(np.abs(qc[k] - qm[k + 1])).max())
        if errs:
            E = np.abs(np.array(errs))
            all_err.append(E)
            per_run.append(dict(run=os.path.basename(os.path.dirname(fp)), n=len(E),
                                mean_deg=round(float(E[:, :4].mean()), 3), p95_deg=round(float(np.percentile(E[:, :4], 95)), 3)))
    E = np.concatenate(all_err)
    ls, lr = np.array(all_lead_sim), np.array(all_lead_real)
    return dict(n_pairs=len(E), runs=len(per_run),
                mean_abs_deg_j1to4=round(float(E[:, :4].mean()), 3),
                p95_abs_deg_j1to4=round(float(np.percentile(E[:, :4], 95)), 3),
                per_joint_mean=np.round(E.mean(0), 3).tolist(),
                lag_cmd_minus_meas_p50=dict(sim=round(float(np.median(ls)), 2), real=round(float(np.median(lr)), 2)),
                lag_cmd_minus_meas_p95=dict(sim=round(float(np.percentile(ls, 95)), 2),
                                            real=round(float(np.percentile(lr, 95)), 2)),
                per_run=per_run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="zed_2026110*")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "outputs", "realsim_validation.json"))
    a = ap.parse_args()
    res = {"hardware": HW}
    for lvl in ("real", "ideal"):
        r = dict(step=test_step(lvl), sine=test_sine(lvl), replay=test_replay(lvl, a.runs))
        res[lvl] = r
        print(f"== {lvl}")
        print("  step  ", r["step"], " hw", HW["step"])
        print("  sine  ", r["sine"], " hw", HW["sine"])
        rp = {k: v for k, v in r["replay"].items() if k != "per_run"}
        print("  replay", rp)
    json.dump(res, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
