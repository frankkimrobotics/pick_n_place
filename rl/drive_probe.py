#!/usr/bin/env python3
"""drive_probe :: replay the 2026-09-17 hardware experiments on the sim drive model.

Hardware references (joint 0, mycobot_mpc/README.md + logs/tuner_runs):
  * waypoint 10 deg step, robot_hal law K0=6 (vel cmd = -6*(q-q*)), vmax 50, accel 4x:
      onset(0.02 deg) 36-52 ms, rise 10-90 % 0.27 s, overshoot 0.0-0.1 %, settle 0.43 s,
      peak 45-50 deg/s
  * streamed sine 12 deg 0.5 Hz, law K0=20 K1=0.3 vff=1 (S17/S19):
      rms 0.10-0.12 deg, max 0.22-0.23 deg, lag ~0-5 ms
Run:  ~/miniconda3/envs/mjwarp/bin/python rl/drive_probe.py [--drive ideal]
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import warp as wp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import env_warp as E  # noqa: E402


def metrics(t, q, q0, tg):
    step = tg - q0
    norm = (q - q0) / step
    on = next((tt for tt, qq in zip(t, q) if abs(qq - q0) > math.radians(0.02)), None)
    r10 = next((tt for tt, n in zip(t, norm) if n >= 0.1), None)
    r90 = next((tt for tt, n in zip(t, norm) if n >= 0.9), None)
    over = max(0.0, (norm.max() - 1.0) * 100)
    settle = max([tt for tt, qq in zip(t, q) if abs(qq - tg) > math.radians(0.5)] or [0.0])
    v = np.gradient(q, t)
    return dict(onset_ms=None if on is None else on * 1000, rise_s=None if r10 is None or r90 is None else r90 - r10,
                overshoot_pct=over, settle_s=settle, peak_vel_deg_s=math.degrees(np.abs(v).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="real", choices=["real", "ideal"])
    ap.add_argument("--scene", default=os.path.join(HERE, "scenes", "box_med_ped.xml"))
    a = ap.parse_args()
    wp.init()
    env = E.PickEnv(nworld=1, xml=a.scene, mode="attach", dr=False, drive=a.drive)
    env.auto_reset = False
    j = 0
    # ---------------- 10 deg waypoint step (robot_hal waypoint law) ----------------
    env.reset(torch.ones(1, dtype=torch.bool, device=env.device))
    env.drive_law = "waypoint"
    q0 = float(env.qpos[0, j])
    tg = q0 + math.radians(10.0)
    t, q = [], []
    # sub-sample the physics at 10 ms to mimic the encoder stream: run decisions with a
    # zero action but a pre-set target, logging q_target-tracking at command-tick rate
    env.q_target[0, j] = tg
    env.q_target_prev[0, j] = tg          # step (no ramp): waypoint mode
    log = []
    for k in range(15):                   # 15 decisions = 1.5 s, logged every 10 ms command tick
        if env.drive == "real":
            env._step_real_drive(log=log)
        else:
            for c in range(env.n_cmd):
                for _ in range(env.substeps // env.n_cmd):
                    tau = (env.kp * (env.q_target - env.qpos[:, :6]) - env.kd * env.qvel[:, :6]).clamp(-100, 100)
                    env.ctrl[:, :6] = tau
                    E.mjw.step(env.m, env.d)
                log.append(env.qpos[:, :6].clone())
    t = [(i + 1) * E.DRIVE["cmd_dt"] for i in range(len(log))]
    q = [float(x[0, j]) for x in log]
    t, q = np.array(t), np.array(q)
    m = metrics(t, q, q0, tg)
    print(f"[{a.drive}] 10 deg waypoint step, joint {j}: onset {m['onset_ms']} ms | rise {m['rise_s']} s | "
          f"overshoot {m['overshoot_pct']:.1f} % | settle {m['settle_s']:.2f} s | peak {m['peak_vel_deg_s']:.0f} deg/s")
    print("   hardware:                onset 36-52 ms | rise 0.27 s | overshoot 0.0-0.1 % | settle 0.43 s | peak 45-50 deg/s")
    print("   drive-vs-joint tracking |q_drive - q| max: %.3f deg" % math.degrees(float((env.q_drive - env.qpos[:, :6]).abs().max())))

    # ---------------- streamed 12 deg 0.5 Hz sine (deployed streaming law) ----------------
    env.reset(torch.ones(1, dtype=torch.bool, device=env.device))
    env.drive_law = "stream"
    q0 = float(env.qpos[0, j])
    A, f, T = math.radians(12.0), 0.5, 4.0
    ref = lambda tau: q0 + A * (1 - math.cos(2 * math.pi * f * min(max(tau, 0), T))) / 2
    t, q, r = [], [], []
    n_dec = int(T * E.CTRL_HZ) + 5
    for k in range(n_dec):
        # decision k: the policy/planner streams the reference for the next 100 ms
        target = ref((k + 1) / E.CTRL_HZ)
        dq = torch.zeros(1, 7, device=env.device)
        dq[0, j] = (target - float(env.q_target[0, j])) / env.dq_max
        # bypass the action clamp for the probe (12 deg sine peaks at 19 deg/s > 2 deg/tick)
        env.q_target_prev = env.q_target.clone()
        env.q_target[0, j] = target
        env._step_real_drive() if env.drive == "real" else None
        if env.drive == "ideal":
            for _ in range(env.substeps):
                tau = (env.kp * (env.q_target - env.qpos[:, :6]) - env.kd * env.qvel[:, :6]).clamp(-100, 100)
                env.ctrl[:, :6] = tau
                E.mjw.step(env.m, env.d)
        tt = (k + 1) / E.CTRL_HZ
        t.append(tt); q.append(float(env.qpos[0, j])); r.append(ref(tt))
    e = np.degrees(np.array(q) - np.array(r))
    print(f"[{a.drive}] streamed sine 12 deg 0.5 Hz (sampled at decisions): rms {np.sqrt((e**2).mean()):.3f} deg | max {np.abs(e).max():.3f} deg")
    print("   hardware (S17/S19):           rms 0.10-0.12 deg | max 0.22-0.23 deg")


if __name__ == "__main__":
    main()
