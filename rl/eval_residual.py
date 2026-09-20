#!/usr/bin/env python3
"""eval_residual :: deterministic evaluation of a residual-PPO checkpoint.

Loads a checkpoint written by ppo.py --residual_base (it carries the keys
`residual_base` and `residual_bound`), rebuilds the paper env exactly as the
training/eval convention does, and reports deterministic success + seal rate for

    base alone       a = tanh(base.pi(o))
    base + residual  a = clamp(tanh(base.pi(o)) + bound * tanh(res.pi(o)), -1, 1)

    $PY rl/eval_residual.py ~/pnp_rl/resid1_real/best.pt --nworld 512
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def rollout(env, act_fn, nworld, ep_len, device, seed=0):
    """One deterministic pass; paired across policies via the same env seed."""
    env.rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env.auto_reset = False
    env.reset(torch.ones(nworld, dtype=torch.bool, device=device))
    info = {}
    for _ in range(ep_len):
        with torch.no_grad():
            a = act_fn(env.observe())
        _, _, _, info = env.step(a)
    return (float(info["placed"].float().mean()),
            float(info["ever_sealed"].float().mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--base", default=None, help="override the checkpoint's residual_base path")
    ap.add_argument("--bound", type=float, default=None, help="override the checkpoint's residual_bound")
    ap.add_argument("--nworld", type=int, default=512)
    ap.add_argument("--ep_len", type=int, default=150)
    ap.add_argument("--scene", default=os.path.join(HERE, "scenes", "box_med.xml"))
    ap.add_argument("--drive", default="real", choices=["real", "ideal"])
    ap.add_argument("--no_dr", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="paper")
    a = ap.parse_args()

    import warp as wp
    wp.init()
    from env_paper import PaperPickEnv
    from ppo import AC

    dev = "cuda:0"
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    base_path = a.base or ck.get("residual_base")
    bound = a.bound if a.bound is not None else ck.get("residual_bound")
    if not base_path:
        raise SystemExit(f"[eval] {a.ckpt} has no residual_base key (not a residual checkpoint); pass --base")
    bound = 0.3 if bound is None else float(bound)

    env = PaperPickEnv(nworld=a.nworld, device=dev, xml=a.scene, dr=not a.no_dr, drive=a.drive,
                       ep_len=a.ep_len, grasp_shaping=True, obs_ee=True, reach_target="grasp",
                       lift_dense=True, w_reach=0.5, w_track_c=4, w_track_f=8, seed=a.seed)
    obs_dim = env.observe().shape[-1]

    base = AC(obs_dim=obs_dim, arch=a.arch).to(dev)
    ckb = torch.load(base_path, map_location=dev, weights_only=False)
    base.load_state_dict(ckb["ac"] if "ac" in ckb else ckb)
    base.eval()
    res = AC(obs_dim=obs_dim, arch=a.arch,
             critic_extra=(5 if ck.get("critic_priv") else 0)).to(dev)
    res.load_state_dict(ck["ac"])
    res.eval()
    print(f"[eval] ckpt {a.ckpt} step {ck.get('step')} base {base_path} bound {bound} "
          f"obs_dim {obs_dim} nworld {a.nworld} ep_len {a.ep_len} dr={not a.no_dr} drive={a.drive}", flush=True)

    def f_base(o):
        return torch.tanh(base.pi(o))

    def f_res(o):
        return (torch.tanh(base.pi(o)) + bound * torch.tanh(res.pi(o))).clamp(-1.0, 1.0)

    for name, fn in (("base", f_base), ("base+residual", f_res)):
        succ, seal = rollout(env, fn, a.nworld, a.ep_len, dev, seed=a.seed)
        print(f"[eval] {name:<14} success {succ:.2%}  seal {seal:.2%}", flush=True)


if __name__ == "__main__":
    main()
