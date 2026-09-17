#!/usr/bin/env python3
"""bc_bootstrap :: scripted-teacher behaviour cloning for the ATTACH stage under the
measured drive (2026-09-17). Random exploration never discovers the sustained press the
real drive needs (FINDINGS "Why the measured drive stalls"), so we clone a scripted
descend-press-lift controller into the PPO actor and let PPO refine from there.

Teacher (per world): IK direction from the hover pose to the grasp point (CPU MuJoCo IK
as in env_warp's hover grid), joint targets rate-limited to `--rate` deg/decision,
suction commanded within `--on_dist` of the grasp point, lift back along the same
direction once sealed. Gaussian action noise `--noise` for state coverage. Only
successful episodes (seal + 2 cm lift) are kept.

  $PY rl/bc_bootstrap.py --out ~/pnp_rl/bc_attach_real/bc_init.pt
  $PY rl/ppo.py --mode attach --dr --drive real --init ~/pnp_rl/bc_attach_real/bc_init.pt ...
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="real", choices=["real", "ideal"])
    ap.add_argument("--nworld", type=int, default=512)
    ap.add_argument("--batches", type=int, default=6, help="episode batches to collect")
    ap.add_argument("--rate", type=float, default=1.0, help="deg/decision on the largest joint")
    ap.add_argument("--on_dist", type=float, default=0.02)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/bc_attach_real/bc_init.pt"))
    ap.add_argument("--scene", default=os.path.join(HERE, "scenes", "box_med_ped.xml"))
    a = ap.parse_args()
    wp.init()
    import mujoco
    env = E.PickEnv(nworld=a.nworld, xml=a.scene, mode="attach", dr=True, drive=a.drive)
    env.auto_reset = False
    demo = {"__file__": os.path.join(os.path.dirname(HERE), "mjwarp_pick_demo.py")}
    exec(open(demo["__file__"]).read().split("if __name__")[0], demo)
    dik = mujoco.MjData(env.mjm)
    N, dev = a.nworld, env.device
    EP = E.EP_LEN_ATTACH
    X, Y = [], []
    n_ok = 0
    for b in range(a.batches):
        env.reset(torch.ones(N, dtype=torch.bool, device=dev))
        # per-world IK: hover pose -> grasp point (same xy, grasp height)
        gp = env._grasp_point().cpu().numpy()
        q0 = env.qpos[:, :6].cpu().numpy()
        q_goal = np.zeros_like(q0)
        for w in range(N):
            qg, err = demo["ik"](env.mjm, dik, "tcp", [float(gp[w, 0]), float(gp[w, 1]), float(gp[w, 2]) - 0.004],
                                 demo["R_DOWN"], q0[w])
            q_goal[w] = qg if err < 0.01 else q0[w]
        q_goal = torch.tensor(q_goal, device=dev, dtype=torch.float32)
        q_hover = env.qpos[:, :6].clone()
        obs_b, act_b = [], []
        rate = math.radians(a.rate)
        for k in range(EP):
            obs = env.observe()
            tcp, R = env._tcp()
            dist = torch.norm(tcp - env._grasp_point(), dim=-1)
            tgt = torch.where(env.sealed[:, None], q_hover, q_goal)
            dq = tgt - env.q_target                                   # remaining commanded motion
            scale = (rate / dq.abs().max(-1, keepdim=True).values.clamp(min=1e-6)).clamp(max=1.0)
            act6 = (dq * scale / env.dq_max).clamp(-1, 1)
            act6 = act6 + torch.randn_like(act6) * a.noise
            suc = torch.where((dist < a.on_dist) | env.sealed, 1.0, -1.0)[:, None]
            act = torch.cat([act6.clamp(-1, 1), suc], -1)
            obs_b.append(obs.clone()); act_b.append(act.clone())
            _, r, done, info = env.step(act)
        ok = info["placed"] | (env.sealed & ((env._obj_pos()[:, 2] - float(env.half[2])) > 0.02))
        n_ok += int(ok.sum())
        O = torch.stack(obs_b, 1)[ok]; A = torch.stack(act_b, 1)[ok]       # (n_ok, EP, dim)
        X.append(O.reshape(-1, O.shape[-1])); Y.append(A.reshape(-1, 7))
        print(f"[bc] batch {b}: success {100 * ok.float().mean():.1f} %  kept {int(ok.sum())} episodes", flush=True)
    X = torch.cat(X); Y = torch.cat(Y)
    print(f"[bc] dataset {X.shape[0]} steps from {n_ok} successful episodes (obs {X.shape[1]})", flush=True)
    from ppo import AC
    ac = AC(obs_dim=X.shape[1]).to(dev)
    opt = torch.optim.Adam(ac.pi.parameters(), lr=1e-3)
    idx = torch.arange(X.shape[0], device=dev)
    for ep in range(a.epochs):
        perm = idx[torch.randperm(idx.numel(), device=dev)]
        tot = 0.0
        for i in range(0, perm.numel(), 4096):
            j = perm[i:i + 4096]
            loss = ((ac.pi(X[j]) - Y[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * j.numel()
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"[bc] epoch {ep} mse {tot / perm.numel():.4f}", flush=True)
    with torch.no_grad():
        ac.log_std.fill_(-1.0)
    # replay the cloned policy (deterministic) to measure its own success
    env.reset(torch.ones(N, dtype=torch.bool, device=dev))
    for k in range(EP):
        with torch.no_grad():
            act = ac.pi(env.observe()).clamp(-1, 1)
        _, r, done, info = env.step(act)
    ok = info["placed"] | (env.sealed & ((env._obj_pos()[:, 2] - float(env.half[2])) > 0.02))
    print(f"[bc] cloned policy success (deterministic, {a.drive} drive): {100 * ok.float().mean():.1f} %", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"ac": ac.state_dict(), "step": 0}, a.out)
    print(f"[bc] saved {a.out}")


if __name__ == "__main__":
    main()
