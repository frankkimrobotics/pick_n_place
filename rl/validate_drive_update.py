#!/usr/bin/env python3
"""validate_drive_update :: A/B the twin's drive model against the real robot (2026-09-20).

The deployed controller changed on 2026-09-20 (rl/real_policy_ctrl.STREAM_GAINS,
mycobot_mpc/robot_hal.run_stream_loop + spline_ref.py):

    old twin (pre-0920)   K0 = 20, linear per-decision reference ramp,     no reference lead
    new twin (deployed)   K0 = 10, uniform cubic B-spline reference,       lead = 45 ms,
                          and the streamed reference is bounded to +-3 deg of the MEASURED joint
                          (real_policy_ctrl.LEAD_MAX_DEG)

This script replays the DEPLOYED policy (rl/weights/resid1_real_best.pt, the fused residual)
deterministically -- dist.loc, i.e. tanh of the mean, not samples -- with the deployment's
observation noise (0.005) and no other domain randomisation, and prints for each twin variant:
success, seal, the per-episode peak joint speed distribution and the decision index of the
first seal.  Reference numbers from the robot (docs/, FINDINGS items 18-21):
peak |qd| 22-34 deg/s (median ~27), first contact ~3.3 s (decision ~33).

    CUDA_VISIBLE_DEVICES=1 $PY rl/validate_drive_update.py --nworld 256
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

# old / new twin variants: (DRIVE["k0"], drive_ref, lead_clip)
VARIANTS = {
    "old  (k0 20, linear, no lead clip)": (20.0, "linear", False),
    "new  (k0 10, spline, lead clip 3deg)": (10.0, "spline", True),
    "new- (k0 10, spline, NO lead clip)": (10.0, "spline", False),
    "mid  (k0 10, linear, no lead clip)": (10.0, "linear", False),
}


def run(env, pol, nworld, ep_len, dev, seed, obs_noise, dq_max_deg):
    env.rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env.auto_reset = False
    env.dq_max = float(np.radians(dq_max_deg))
    env.reset(torch.ones(nworld, dtype=torch.bool, device=dev))
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    info = {}
    for _ in range(ep_len):
        o = env.observe()
        if obs_noise:                      # the deployment adds the training obs noise (FINDINGS 16)
            o = o + torch.randn(o.shape, device=dev, generator=gen) * obs_noise
        with torch.no_grad():
            a = pol(o)                     # deterministic: tanh(mu), no sampling
        _, _, _, info = env.step(a)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "weights", "resid1_real_best.pt"))
    ap.add_argument("--nworld", type=int, default=256)
    ap.add_argument("--ep_len", type=int, default=150)
    ap.add_argument("--dq_max", type=float, default=2.0)
    ap.add_argument("--obs_noise", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene", default=os.path.join(HERE, "scenes", "box_med.xml"))
    ap.add_argument("--arch", default="paper")
    ap.add_argument("--only", default=None, help="comma-separated substrings of the variant names to run")
    a = ap.parse_args()

    import warp as wp
    wp.init()
    import env_warp as E
    from env_paper import PaperPickEnv
    from ppo import build_frozen_policy, describe_policy

    dev = "cuda:0"          # CUDA_VISIBLE_DEVICES selects the physical GPU
    # no DR: the drive/seal randomisation is off, the observation noise is added explicitly
    env = PaperPickEnv(nworld=a.nworld, device=dev, xml=a.scene, dr=False, drive="real",
                       ep_len=a.ep_len, grasp_shaping=True, obs_ee=True, reach_target="grasp",
                       lift_dense=True, w_reach=0.5, w_track_c=4, w_track_f=8, seed=a.seed)
    obs_dim = env.observe().shape[-1]
    pol = build_frozen_policy(a.ckpt, obs_dim, arch=a.arch, device=dev)
    print(f"[val] policy {a.ckpt} -> {describe_policy(pol, a.ckpt)}  obs_dim {obs_dim} "
          f"nworld {a.nworld} ep_len {a.ep_len} dq_max {a.dq_max} obs_noise {a.obs_noise}", flush=True)
    print("[val] real robot reference: peak |qd| 22-34 deg/s (median ~27), first contact ~3.3 s "
          "(decision ~33)", flush=True)

    k0_orig = E.DRIVE["k0"]
    try:
        for name, (k0, ref, clip) in VARIANTS.items():
            if a.only and not any(t.strip() in name for t in a.only.split(",")):
                continue
            E.DRIVE["k0"] = k0
            env.drive_ref = ref
            env.lead_clip = clip
            info = run(env, pol, a.nworld, a.ep_len, dev, a.seed, a.obs_noise, a.dq_max)
            pq = np.degrees(info["peak_qd"].cpu().numpy())
            ts = info["t_seal"].cpu().numpy()
            tg = info["t_goal"].cpu().numpy()
            sealed = ts < a.ep_len
            reached = tg < a.ep_len
            print(f"[val] {name:<38} success {float(info['placed'].float().mean()):6.2%}  "
                  f"seal {float(info['ever_sealed'].float().mean()):6.2%}  "
                  f"peak|qd| med {np.median(pq):5.1f}  p90 {np.percentile(pq, 90):5.1f}  "
                  f"max {pq.max():5.1f} deg/s  |  first seal med {np.median(ts[sealed]) if sealed.any() else float('nan'):5.1f} dec "
                  f"({(np.median(ts[sealed]) / 10 if sealed.any() else float('nan')):.2f} s)  "
                  f"p90 {np.percentile(ts[sealed], 90) if sealed.any() else float('nan'):5.1f}  |  "
                  f"t_goal med {np.median(tg[reached]) if reached.any() else float('nan'):5.1f} dec", flush=True)
    finally:
        E.DRIVE["k0"] = k0_orig


if __name__ == "__main__":
    main()
