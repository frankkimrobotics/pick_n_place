#!/usr/bin/env python3
"""bc_paper :: scripted-teacher behaviour cloning for the paper-style env (env_paper.PaperPickEnv):
reach from home -> descend -> press -> seal -> lift -> carry the object to the 3-D goal -> hold.
Same idea as bc_bootstrap.py for the attach stage: the sustained press a suction cup needs is not
discoverable by PPO exploration (paper_*, paper2_*, paper3_* runs, 2026-09-17), so clone a
scripted controller first and let PPO refine.

Teacher (per world, CPU MuJoCo IK once per phase, joint-space rate limit `--rate` deg/decision):
  1 hover : IK to grasp point + 4 cm (cup down)      until within 1.5 cm
  2 press : IK to grasp point - 4 mm, suction ON      until sealed
  3 lift  : back to the hover joint pose              until lifted > h_min
  4 carry : IK to tcp target = goal + (tcp - obj) offset at seal   then hold
Only episodes that end lifted and within --succ_tol of the goal are kept.

  $PY rl/bc_paper.py --drive real --out ~/pnp_rl/bc_paper_real/bc_init.pt
  $PY rl/ppo.py --env paper --arch paper --drive real --init ~/pnp_rl/bc_paper_real/bc_init.pt ...
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
from env_paper import PaperPickEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="real", choices=["real", "ideal"])
    ap.add_argument("--nworld", type=int, default=512)
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--rate", type=float, default=1.5, help="deg/decision on the largest joint")
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/bc_paper_real/bc_init.pt"))
    ap.add_argument("--scene", default=os.path.join(HERE, "scenes", "box_med.xml"))
    a = ap.parse_args()
    wp.init()
    import mujoco
    env = PaperPickEnv(nworld=a.nworld, xml=a.scene, dr=True, drive=a.drive)
    env.auto_reset = False
    demo = {"__file__": os.path.join(os.path.dirname(HERE), "mjwarp_pick_demo.py")}
    exec(open(demo["__file__"]).read().split("if __name__")[0], demo)
    dik = mujoco.MjData(env.mjm)
    N, dev, EP = a.nworld, env.device, env.paper["ep_len"]
    rate = math.radians(a.rate)

    def ik_batch(targets, seeds):
        out = np.zeros_like(seeds)
        for w in range(N):
            q, err = demo["ik"](env.mjm, dik, "tcp", [float(x) for x in targets[w]], demo["R_DOWN"], seeds[w])
            out[w] = q if err < 0.01 else seeds[w]
        return torch.tensor(out, device=dev, dtype=torch.float32)

    def act_to(tgt):
        dq = tgt - env.q_target
        sc = (rate / dq.abs().max(-1, keepdim=True).values.clamp(min=1e-6)).clamp(max=1.0)
        return (dq * sc / env.dq_max).clamp(-1, 1)

    X, Y, n_ok = [], [], 0
    for b in range(a.batches):
        env.reset(torch.ones(N, dtype=torch.bool, device=dev))
        gp = env._grasp_point().cpu().numpy()
        q0 = env.qpos[:, :6].cpu().numpy()
        q_hover = ik_batch(gp + np.array([0, 0, 0.04]), q0)
        q_grasp = ik_batch(gp + np.array([0, 0, -0.004]), q_hover.cpu().numpy())
        q_carry = None
        phase = torch.zeros(N, dtype=torch.long, device=dev)          # 0 hover 1 press 2 lift 3 carry
        obs_b, act_b = [], []
        for k in range(EP):
            obs = env.observe()
            tcp, R = env._tcp()
            gpt = env._grasp_point()
            dist = torch.norm(tcp - gpt, dim=-1)
            lifted = (env._obj_pos()[:, 2] - float(env.half[2])) > env.paper["h_min"]
            phase = torch.where((phase == 0) & (dist < 0.015), torch.ones_like(phase), phase)
            phase = torch.where((phase == 1) & env.sealed, torch.full_like(phase, 2), phase)
            newly_lifted = (phase == 2) & lifted
            if newly_lifted.any() and q_carry is None:
                # carry target for ALL worlds: keep the seal offset (tcp - obj) and aim the object at the goal
                off = (tcp - env._obj_pos()).cpu().numpy()
                q_carry = ik_batch(env.goal.cpu().numpy() + off, env.qpos[:, :6].cpu().numpy())
            phase = torch.where(newly_lifted & (q_carry is not None), torch.full_like(phase, 3), phase)
            tgt = torch.where((phase == 0)[:, None], q_hover,
                  torch.where((phase == 1)[:, None], q_grasp,
                  torch.where((phase == 2)[:, None], q_hover,
                              q_carry if q_carry is not None else q_hover)))
            act6 = act_to(tgt) + torch.randn(N, 6, device=dev) * a.noise
            suc = torch.where((phase >= 1) | (dist < 0.02), 1.0, -1.0)[:, None]
            act = torch.cat([act6.clamp(-1, 1), suc], -1)
            obs_b.append(obs.clone()); act_b.append(act.clone())
            _, r, done, info = env.step(act)
        ok = info["placed"]
        n_ok += int(ok.sum())
        O = torch.stack(obs_b, 1)[ok]; A = torch.stack(act_b, 1)[ok]
        X.append(O.reshape(-1, O.shape[-1])); Y.append(A.reshape(-1, 7))
        print(f"[bc] batch {b}: success {100 * ok.float().mean():.1f} % (sealed {100 * env.ever_sealed.float().mean():.0f} %, phase3 {100 * (phase == 3).float().mean():.0f} %) kept {int(ok.sum())}", flush=True)
    X = torch.cat(X); Y = torch.cat(Y)
    print(f"[bc] dataset {X.shape[0]} steps from {n_ok} successful episodes (obs {X.shape[1]})", flush=True)
    if X.shape[0] == 0:
        print("[bc] no successful episodes -- nothing to clone"); return
    from ppo import AC
    ac = AC(obs_dim=X.shape[1], arch="paper").to(dev)
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
    env.reset(torch.ones(N, dtype=torch.bool, device=dev))
    for k in range(EP):
        with torch.no_grad():
            act = ac.pi(env.observe()).clamp(-1, 1)
        _, r, done, info = env.step(act)
    print(f"[bc] cloned policy success (deterministic, {a.drive} drive): {100 * info['placed'].float().mean():.1f} %  sealed {100 * env.ever_sealed.float().mean():.0f} %", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"ac": ac.state_dict(), "step": 0}, a.out)
    print(f"[bc] saved {a.out}")


if __name__ == "__main__":
    main()
