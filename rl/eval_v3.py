#!/usr/bin/env python3
"""eval_v3 :: paired deterministic evaluation on rl/env_v3.ObstacleEnv.

Every policy is replayed on the SAME episodes (the env rng and torch seed are reset before
each rollout), with DR on, and reported with the v3 failure modes:

    success | seal | obst_hit | wall_hit | off-table | t_seal | t_goal | peak |qd|

A checkpoint that carries `residual_base` is rebuilt as the fused residual stack the robot
would run (ppo.FusedResidual), exactly as rl/eval_residual.py does.  A checkpoint trained on
a SMALLER observation (e.g. the 47-D env_v2 student) is loaded into a 54-D actor with the
extra columns zero-padded (ppo.AC.load_state_dict), i.e. it simply ignores the obstacle block
-- which is what makes "v2 student on v3" a meaningful baseline.

    # milestone 1: what the obstacles and the enlarged workspace cost a v2-trained student
    $PY rl/eval_v3.py ~/pnp_rl/dagger_v2c/bc_iter16.pt --nworld 1024 --both
    # milestone 3: student vs residual
    $PY rl/eval_v3.py ~/pnp_rl/dagger_v3/best.pt ~/pnp_rl/resid_v3/best.pt --nworld 1024
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def load_policy(path, obs_dim, arch="paper", device="cuda:0", dq_default=None):
    """-> (callable, dq_max_deg, label)."""
    from ppo import AC, FusedResidual, build_frozen_policy, describe_policy
    ck = torch.load(path, map_location=device, weights_only=False)
    ac = AC(obs_dim=obs_dim, arch=arch, critic_extra=(5 if ck.get("critic_priv") else 0)).to(device)
    ac.load_state_dict(ck["ac"])
    ac.eval()
    dq = ck.get("dq_max_deg", dq_default)
    if ck.get("residual_base"):
        base = build_frozen_policy(ck["residual_base"], obs_dim, arch=arch, device=device)
        pol = FusedResidual(base, ac.pi, float(ck.get("residual_bound", 0.3)),
                            float(ck.get("base_scale", 1.0))).to(device).eval()
        lab = f"{describe_policy(pol, path)}"
    else:
        pol = lambda o: torch.tanh(ac.pi(o))                       # noqa: E731
        lab = f"bc/ppo obs{ck.get('obs_dim', obs_dim)} step {ck.get('step')}"
    trained_dim = int(ck["ac"]["pi.0.weight"].shape[1]) if "pi.0.weight" in ck["ac"] else obs_dim
    if trained_dim != obs_dim:
        lab += f"  (trained on obs {trained_dim}, zero-padded to {obs_dim})"
    return pol, dq, lab


def rollout(env, act_fn, ep_len, seed=0, dq_max_deg=None):
    env.rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env.auto_reset = False
    if dq_max_deg is not None:
        env.dq_max = float(np.radians(dq_max_deg))
    env.reset(torch.ones(env.nworld, dtype=torch.bool, device=env.device))
    info = {}
    for _ in range(ep_len):
        with torch.no_grad():
            a = act_fn(env.observe())
        _, _, _, info = env.step(a)
    g = lambda k: info[k].float().cpu().numpy() if k in info else np.zeros(env.nworld)  # noqa: E731
    pq = np.degrees(g("peak_qd"))
    ok = g("placed")
    nd = g("n_dist")
    out = dict(success=float(ok.mean()), seal=float(g("ever_sealed").mean()),
               obst_hit=float(g("obst_hit").mean()), wall_hit=float(g("wall_hit").mean()),
               off=float(g("off").mean()),
               t_seal=float(g("t_seal").mean()) / 10.0, t_goal=float(g("t_goal").mean()) / 10.0,
               qd_med=float(np.median(pq)), qd_p90=float(np.percentile(pq, 90)), qd_max=float(pq.max()),
               by_n={int(k): round(float(ok[nd == k].mean()), 4) for k in np.unique(nd) if (nd == k).sum()})
    return out


def fmt(name, r):
    return (f"[eval_v3] {name:<46} success {r['success']:7.2%}  seal {r['seal']:6.2%}  "
            f"obst_hit {r['obst_hit']:6.2%}  wall_hit {r['wall_hit']:5.2%}  off {r['off']:5.2%}  "
            f"t_seal {r['t_seal']:4.2f} s  t_goal {r['t_goal']:4.2f} s  "
            f"peak|qd| {r['qd_med']:.1f}/{r['qd_p90']:.1f}/{r['qd_max']:.1f} deg/s  "
            f"by n_dist {r['by_n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="+")
    ap.add_argument("--nworld", type=int, default=1024)
    ap.add_argument("--ep_len", type=int, default=150)
    ap.add_argument("--drive", default="real", choices=["real", "ideal"])
    ap.add_argument("--arch", default="paper")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dq_max", type=float, default=None, help="override the action clamp (deg)")
    ap.add_argument("--obstacles", type=int, default=1)
    ap.add_argument("--both", action="store_true", help="evaluate each policy with obstacles OFF and ON")
    ap.add_argument("--no_dr", action="store_true")
    ap.add_argument("--out", default=None, help="write the table as json")
    a = ap.parse_args()

    import warp as wp
    wp.init()
    from env_v3 import ObstacleEnv
    dev = "cuda:0"
    modes = [False, True] if a.both else [bool(a.obstacles)]
    rows = []
    for obst in modes:
        env = ObstacleEnv(nworld=a.nworld, device=dev, dr=not a.no_dr, drive=a.drive,
                          ep_len=a.ep_len, obstacles=obst, seed=a.seed,
                          grasp_shaping=True, obs_ee=True, reach_target="grasp", lift_dense=True,
                          w_reach=0.5, w_track_c=4, w_track_f=8)
        obs_dim = env.observe().shape[-1]
        print(f"[eval_v3] obstacles={obst} obs_dim {obs_dim} nworld {a.nworld} ep_len {a.ep_len} "
              f"dr={not a.no_dr} drive={a.drive}", flush=True)
        for path in a.ckpt:
            pol, dq, lab = load_policy(path, obs_dim, a.arch, dev, dq_default=a.dq_max)
            dq = a.dq_max if a.dq_max is not None else dq
            r = rollout(env, pol, a.ep_len, seed=a.seed, dq_max_deg=dq)
            tag = f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"
            name = f"{tag} obst={int(obst)} dq={dq}"
            print(fmt(name, r), flush=True)
            print(f"           ^ {lab}", flush=True)
            rows.append(dict(ckpt=os.path.abspath(path), obstacles=obst, dq_max=dq, **r))
        del env
        torch.cuda.empty_cache()
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"[eval_v3] wrote {a.out}")


if __name__ == "__main__":
    main()
