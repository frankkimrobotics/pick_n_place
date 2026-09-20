#!/usr/bin/env python3
"""eval_residual :: deterministic evaluation of a residual-PPO checkpoint.

Loads a checkpoint written by ppo.py --residual_base (it carries `residual_base`,
`residual_bound` and, since 2026-09-20, `base_scale` / `dq_max_deg`), rebuilds the paper env
exactly as the training/eval convention does, and reports deterministic success, seal rate and
the SPEED metrics (decisions to the first seal / to the goal, per-episode peak joint speed) for

    base alone       a = base(o)                                        (dq_max = --base_dq_max)
    base + residual  a = clamp(s*base(o_b) + bound*tanh(res(o)), -1, 1) (dq_max = --dq_max)

`base` is built recursively, so a residual stacked on a residual is evaluated as the one fused
policy the robot actually runs.  Both rollouts use the same env seed (paired comparison).

    $PY rl/eval_residual.py ~/pnp_rl/resid3_fast/best.pt --nworld 1024 --dq_max 3 --base_dq_max 2
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def rollout(env, act_fn, nworld, ep_len, device, seed=0, dq_max_deg=None):
    """One deterministic pass; paired across policies via the same env seed."""
    env.rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env.auto_reset = False
    if dq_max_deg is not None:
        env.dq_max = float(np.radians(dq_max_deg))
    env.reset(torch.ones(nworld, dtype=torch.bool, device=device))
    info = {}
    for _ in range(ep_len):
        with torch.no_grad():
            a = act_fn(env.observe())
        _, _, _, info = env.step(a)
    g = lambda k: info[k].float().cpu().numpy()            # noqa: E731
    pq = np.degrees(g("peak_qd"))
    return dict(success=float(g("placed").mean()), seal=float(g("ever_sealed").mean()),
                t_seal=float(g("t_seal").mean()), t_goal=float(g("t_goal").mean()),
                qd_med=float(np.median(pq)), qd_p90=float(np.percentile(pq, 90)), qd_max=float(pq.max()),
                t_seal_ok=float(g("t_seal")[g("t_seal") < ep_len].mean()) if (g("t_seal") < ep_len).any() else float("nan"),
                t_goal_ok=float(g("t_goal")[g("t_goal") < ep_len].mean()) if (g("t_goal") < ep_len).any() else float("nan"))


def fmt(name, r, hz=10.0):
    return (f"[eval] {name:<28} success {r['success']:.2%}  seal {r['seal']:.2%}  "
            f"t_seal {r['t_seal']:.1f} dec ({r['t_seal'] / hz:.2f} s, sealed-only {r['t_seal_ok'] / hz:.2f} s)  "
            f"t_goal {r['t_goal']:.1f} dec ({r['t_goal'] / hz:.2f} s, reached-only {r['t_goal_ok'] / hz:.2f} s)  "
            f"peak|qd| med/p90/max {r['qd_med']:.1f}/{r['qd_p90']:.1f}/{r['qd_max']:.1f} deg/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="+", help="one or more residual checkpoints (the frozen base is evaluated once)")
    ap.add_argument("--base", default=None, help="override the checkpoint's residual_base path")
    ap.add_argument("--bound", type=float, default=None, help="override the checkpoint's residual_bound")
    ap.add_argument("--base_scale", type=float, default=None, help="override the checkpoint's base_scale")
    ap.add_argument("--dq_max", type=float, default=None, help="dq_max (deg) for the residual policy (default: ckpt)")
    ap.add_argument("--base_dq_max", type=float, default=None, help="dq_max (deg) the BASE was trained with")
    ap.add_argument("--nworld", type=int, default=512)
    ap.add_argument("--ep_len", type=int, default=150)
    ap.add_argument("--scene", default=os.path.join(HERE, "scenes", "box_med.xml"))
    ap.add_argument("--drive", default="real", choices=["real", "ideal"])
    ap.add_argument("--drive_ref", default=None, choices=["spline", "linear"])
    ap.add_argument("--no_dr", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="paper")
    ap.add_argument("--base_only", action="store_true", help="evaluate only the frozen base")
    a = ap.parse_args()

    import warp as wp
    wp.init()
    from env_paper import PaperPickEnv
    from ppo import FusedResidual, AC, build_frozen_policy, describe_policy

    dev = "cuda:0"
    ck = torch.load(a.ckpt[0], map_location=dev, weights_only=False)
    base_path = a.base or ck.get("residual_base")
    bound = a.bound if a.bound is not None else ck.get("residual_bound")
    if not base_path:
        raise SystemExit(f"[eval] {a.ckpt[0]} has no residual_base key (not a residual checkpoint); pass --base")
    bound = 0.3 if bound is None else float(bound)
    base_scale = float(a.base_scale if a.base_scale is not None else ck.get("base_scale", 1.0))
    dq_max = a.dq_max if a.dq_max is not None else ck.get("dq_max_deg")
    base_dq = a.base_dq_max if a.base_dq_max is not None else (dq_max * base_scale if dq_max else None)

    kw = {} if a.drive_ref is None else dict(drive_ref=a.drive_ref)
    env = PaperPickEnv(nworld=a.nworld, device=dev, xml=a.scene, dr=not a.no_dr, drive=a.drive,
                       ep_len=a.ep_len, grasp_shaping=True, obs_ee=True, reach_target="grasp",
                       lift_dense=True, w_reach=0.5, w_track_c=4, w_track_f=8, seed=a.seed, **kw)
    obs_dim = env.observe().shape[-1]

    base = build_frozen_policy(base_path, obs_dim, arch=a.arch, device=dev)
    print(f"[eval] base {base_path} ({describe_policy(base, base_path)}) bound {bound} base_scale {base_scale} "
          f"dq_max {dq_max} base_dq_max {base_dq} obs_dim {obs_dim} nworld {a.nworld} ep_len {a.ep_len} "
          f"dr={not a.no_dr} drive={a.drive} ref={env.drive_ref} lead_clip={env.lead_clip}", flush=True)

    runs = [("base", base, base_dq)]
    if not a.base_only:
        for path in a.ckpt:
            ckp = torch.load(path, map_location=dev, weights_only=False)
            res = AC(obs_dim=obs_dim, arch=a.arch, critic_extra=(5 if ckp.get("critic_priv") else 0)).to(dev)
            res.load_state_dict(ckp["ac"])
            res.eval()
            bnd = a.bound if a.bound is not None else float(ckp.get("residual_bound", 0.3))
            bsc = float(a.base_scale if a.base_scale is not None else ckp.get("base_scale", 1.0))
            dqm = a.dq_max if a.dq_max is not None else ckp.get("dq_max_deg")
            runs.append((f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)} @{ckp.get('step')}",
                         FusedResidual(base, res.pi, bnd, bsc).to(dev).eval(), dqm))
    for name, fn, dqm in runs:
        r = rollout(env, fn, a.nworld, a.ep_len, dev, seed=a.seed, dq_max_deg=dqm)
        print(fmt(f"{name} (dq {dqm})", r), flush=True)


if __name__ == "__main__":
    main()
