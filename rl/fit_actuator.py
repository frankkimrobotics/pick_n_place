#!/usr/bin/env python3
"""fit_actuator :: Isaac-Lab-style actuator model for the MuJoCo twin, fitted to real traces.

Model (per joint, mirroring isaaclab.actuators.DCMotor + ActuatorBaseCfg):
    tau = clip( kp*(q* - q) + kd*(qd* - qd),  +-tau_cap(qd) )      PD with DC saturation
    tau_cap(qd) = tau_sat * max(0, 1 - |qd| / v_rated)             back-EMF torque rolloff
    joint armature (reflected rotor inertia), viscous damping, Coulomb frictionloss
    command DELAY d (DelayedPDActuator): references reach the servo d seconds late

Data: real runs' steps.jsonl (10 Hz: q_cmd reference vs q_meas, URDF rad).
Fit:  teacher-forced one-step prediction (reset to measured state each decision,
      simulate 0.1 s, compare next measured) -> Nelder-Mead over
      {log kp, log kd, armature, viscous damping, per-joint frictionloss, delay}.
Validation: free-run rollout on a held-out run + the measured step-battery stats.

Output: fitted params JSON -> outputs/actuator_fit.json; apply with --write to patch
        the shared sim helpers' gain table (opt-in via PNP_ACT_FIT=1 at generator/env load).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import mujoco  # noqa: E402

XML = os.path.join(HERE, "scenes", "box_med.xml")
NJ = 6
CTRL_DT = 0.004


def load_run(run_dir):
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "steps.jsonl"))]
    q_m = np.array([r["q_meas"] for r in rows], float)
    q_c = np.array([r["q_cmd"] for r in rows], float)
    t = np.array([r["t"] for r in rows], float)
    # keep contiguous 10 Hz stretches (drop scripted-leg gaps)
    keep = np.where(np.diff(t) < 0.25)[0]
    return q_m, q_c, keep


class IsaacStyleSim:
    def __init__(self, params):
        self.m = mujoco.MjModel.from_xml_path(XML)
        self.d = mujoco.MjData(self.m)
        self.set_params(params)

    def set_params(self, p):
        self.kp = np.exp(p["log_kp"])
        self.kd = np.exp(p["log_kd"])
        self.tau_sat = np.asarray(p["tau_sat"], float)
        self.v_rated = float(p["v_rated"])
        self.delay = float(p["delay"])
        self.m.dof_armature[:NJ] = float(p["armature"])
        self.m.dof_damping[:NJ] = np.asarray(p["damping"], float)
        self.m.dof_frictionloss[:NJ] = np.asarray(p["frictionloss"], float)

    def step_decision(self, q0, qd0, q_ref_prev, q_ref, horizon=0.1):
        """simulate one 10 Hz decision: linear reference q_ref_prev->q_ref with command delay"""
        d = self.d
        d.qpos[:NJ] = q0
        d.qvel[:NJ] = qd0
        mujoco.mj_forward(self.m, d)
        n = int(round(horizon / CTRL_DT))
        for k in range(n):
            tt = k * CTRL_DT - self.delay
            a = np.clip(tt / horizon, 0.0, 1.0)
            qr = q_ref_prev + a * (q_ref - q_ref_prev)
            qdr = (q_ref - q_ref_prev) / horizon if 0 <= tt <= horizon else np.zeros(NJ)
            tau = self.kp * (qr - d.qpos[:NJ]) + self.kd * (qdr - d.qvel[:NJ])
            cap = self.tau_sat * np.clip(1.0 - np.abs(d.qvel[:NJ]) / self.v_rated, 0.0, 1.0)
            d.ctrl[:NJ] = np.clip(tau, -cap, cap)
            mujoco.mj_step(self.m, d)
        return d.qpos[:NJ].copy(), d.qvel[:NJ].copy()


def unpack(x):
    return dict(log_kp=x[0], log_kd=x[1], armature=abs(x[2]), delay=np.clip(x[3], 0.0, 0.15),
                damping=np.abs(x[4:10]), frictionloss=np.abs(x[10:16]),
                tau_sat=np.full(NJ, 60.0), v_rated=1.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs for fitting")
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--iters", type=int, default=250)
    a = ap.parse_args()

    data = []
    for rd in a.runs:
        q_m, q_c, keep = load_run(os.path.expanduser(rd))
        qd = np.gradient(q_m, 0.1, axis=0)
        for i in keep[:-1]:
            if i + 1 in keep or True:
                data.append((q_m[i], qd[i], q_c[max(i - 1, 0)], q_c[i], q_m[i + 1]))
    print(f"fit pairs: {len(data)}")
    sub = data[:: max(1, len(data) // 400)]          # ~400 pairs for speed

    sim_holder = {}

    def loss(x):
        p = unpack(x)
        key = "sim"
        if key not in sim_holder:
            sim_holder[key] = IsaacStyleSim(p)
        else:
            sim_holder[key].set_params(p)
        s = sim_holder[key]
        errs = []
        for q0, qd0, qrp, qr, q1 in sub:
            qs, _ = s.step_decision(np.array(q0), np.array(qd0), np.array(qrp), np.array(qr))
            errs.append(np.degrees(np.abs(qs - q1)))
        return float(np.mean(errs))

    # init: computed-torque-ish gains, Sept kv prior, today's 61 ms delay
    x0 = np.concatenate([[np.log(80.0), np.log(8.0), 0.15, 0.061],
                         np.full(6, 0.5), np.full(6, 0.3)])
    from scipy.optimize import minimize
    print("initial loss: %.4f deg" % loss(x0))
    res = minimize(loss, x0, method="Nelder-Mead",
                   options=dict(maxiter=a.iters, xatol=1e-3, fatol=1e-4, adaptive=True))
    p = unpack(res.x)
    print("fitted loss: %.4f deg (one-step mean abs)" % res.fun)
    print("kp %.1f kd %.2f armature %.3f delay %.0f ms" % (
        np.exp(res.x[0]), np.exp(res.x[1]), p["armature"], 1000 * p["delay"]))
    print("damping   :", np.round(p["damping"], 3).tolist())
    print("friction  :", np.round(p["frictionloss"], 3).tolist())

    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()}
    out["one_step_err_deg"] = res.fun
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "outputs", "actuator_fit.json"), "w"), indent=1)
    print("saved outputs/actuator_fit.json")

    if a.holdout:
        q_m, q_c, keep = load_run(os.path.expanduser(a.holdout))
        s = IsaacStyleSim(p)
        i0 = keep[10]
        q, qd = q_m[i0].copy(), np.zeros(NJ)
        errs = []
        for i in range(i0, min(i0 + 80, len(q_c) - 1)):
            q, qd = s.step_decision(q, qd, q_c[max(i - 1, 0)], q_c[i])
            errs.append(np.degrees(np.abs(q - q_m[i + 1])))
        e = np.array(errs)
        print(f"holdout free-run 8 s: mean {e.mean():.2f} deg, p95 {np.percentile(e,95):.2f} deg")


if __name__ == "__main__":
    main()
