#!/usr/bin/env python3
"""seal_probe :: is a seal REACHABLE under a given drive model? (physics probe, no learning)

attach-mode starts (cup hovering 2-4 cm above the object). A scripted controller
descends along the joint-space direction hover -> grasp at a chosen per-decision
rate, commands suction once the cup is within `--on_dist` of the grasp point,
then lifts. Reports latch rate, break rate, cup speed at first contact, and the
seal-gate factors, for --drive real and ideal.

  $PY rl/seal_probe.py --drive real --rate 1.0     # 1.0 deg/decision descent
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


def run(drive, rate_deg, on_dist, nworld, steps, dq_max, verbose=True):
    env = E.PickEnv(nworld=nworld, xml=os.path.join(HERE, "scenes", "box_med_ped.xml"),
                    mode="attach", dr=False, drive=drive, dq_max_deg=dq_max)
    env.auto_reset = False
    env.reset(torch.ones(nworld, dtype=torch.bool, device=env.device))
    # joint-space descent direction from IK: hover pose -> same xy at grasp height
    demo = {"__file__": os.path.join(os.path.dirname(HERE), "mjwarp_pick_demo.py")}
    exec(open(demo["__file__"]).read().split("if __name__")[0], demo)
    import mujoco
    dik = mujoco.MjData(env.mjm)
    gp = env._grasp_point()[0].cpu().numpy()
    q_now = env.qpos[0, :6].cpu().numpy()
    q_g, err = demo["ik"](env.mjm, dik, "tcp", [float(gp[0]), float(gp[1]), float(gp[2])], demo["R_DOWN"], q_now)
    d = (q_g - q_now)
    d = d / max(np.abs(d).max(), 1e-6)                       # largest joint moves `rate_deg` per decision
    step = torch.tensor(d * math.radians(rate_deg) / env.dq_max, device=env.device, dtype=torch.float32)
    step = step.clamp(-1, 1)
    latched = torch.zeros(nworld, dtype=torch.bool, device=env.device)
    broke_any = torch.zeros(nworld, dtype=torch.bool, device=env.device)
    v_contact = torch.full((nworld,), float("nan"), device=env.device)
    lifted = torch.zeros(nworld, dtype=torch.bool, device=env.device)
    t_latch = torch.full((nworld,), -1, dtype=torch.long, device=env.device)
    tcp_prev = env._tcp()[0].clone()
    for k in range(steps):
        tcp, R = env._tcp()
        dist = torch.norm(tcp - env._grasp_point(), dim=-1)
        a = torch.zeros(nworld, 7, device=env.device)
        desc = ~env.sealed
        a[desc, :6] = step
        a[env.sealed, :6] = -step * 0.5                       # lift once sealed
        a[:, 6] = torch.where(dist < on_dist, 1.0, -1.0)
        a[env.sealed, 6] = 1.0
        # cup speed at the moment of first contact with the object (proxy: within 1.5 cm)
        spd = torch.norm(tcp - tcp_prev, dim=-1) * E.CTRL_HZ
        first_contact = torch.isnan(v_contact) & (dist < 0.015)
        v_contact[first_contact] = spd[first_contact]
        tcp_prev = tcp.clone()
        obs, r, done, info = env.step(a)
        new = env.sealed & ~latched
        t_latch[new] = k
        latched |= env.sealed
        broke_any |= latched & ~env.sealed
        op = env._obj_pos()
        lifted |= env.sealed & ((op[:, 2] - float(env.half[2])) > 0.02)
    res = dict(drive=drive, rate_deg=rate_deg, latch=100 * latched.float().mean().item(),
               lift=100 * lifted.float().mean().item(), broke=100 * broke_any.float().mean().item(),
               v_contact_med=torch.nanmedian(v_contact).item(), t_latch_med=float(t_latch[t_latch >= 0].float().median()) if (t_latch >= 0).any() else -1)
    if verbose:
        print(f"[{drive}] rate {rate_deg:.1f} deg/dec: latched {res['latch']:.1f} %  lifted>2cm {res['lift']:.1f} %  "
              f"broke {res['broke']:.1f} %  cup speed at contact {res['v_contact_med']:.3f} m/s (gate {E.SEAL_VEL})  "
              f"median latch step {res['t_latch_med']:.0f}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="real", choices=["real", "ideal", "both"])
    ap.add_argument("--rate", type=float, default=None, help="deg per decision on the largest joint (default: sweep)")
    ap.add_argument("--on_dist", type=float, default=0.02)
    ap.add_argument("--nworld", type=int, default=256)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--dq_max", type=float, default=2.0)
    a = ap.parse_args()
    wp.init()
    drives = ["real", "ideal"] if a.drive == "both" else [a.drive]
    rates = [a.rate] if a.rate is not None else [0.5, 1.0, 2.0]
    for dv in drives:
        for rt in rates:
            run(dv, rt, a.on_dist, a.nworld, a.steps, a.dq_max)
