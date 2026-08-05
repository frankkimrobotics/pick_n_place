"""ppo :: batched PPO on PickEnv — on-policy comparator to sac.py.

Same obs/action space, same logging schema (plot with plot_training.py).
    python rl/ppo.py --nworld 4096 --steps 12000000 --mode pnp --dr
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import warp as wp                                          # noqa: E402
from sac import mlp, OBS_DIM, ACT_DIM                      # noqa: E402


class AC(nn.Module):
    def __init__(self):
        super().__init__()
        self.pi = mlp(OBS_DIM, ACT_DIM, ln=False)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -0.5))
        self.v = mlp(OBS_DIM, 1)

    def dist(self, obs):
        mu = self.pi(obs)
        return torch.distributions.Normal(mu, self.log_std.exp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworld", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=12_000_000)
    ap.add_argument("--rollout", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=16384)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.003)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mode", default="pnp", choices=["full", "attach", "pnp"])
    ap.add_argument("--dr", action="store_true")
    ap.add_argument("--init", default=None)
    ap.add_argument("--target_max", type=float, default=0.30)
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/ppo1"))
    ap.add_argument("--scene", default=os.path.join(HERE, "scenes", "box_med.xml"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda:0"
    torch.manual_seed(0)
    wp.init()
    from env_warp import PickEnv
    env = PickEnv(nworld=a.nworld, device=dev, xml=a.scene, mode=a.mode, dr=a.dr, target_max=a.target_max)
    ac = AC().to(dev)
    if a.init:
        ck = torch.load(a.init, map_location=dev, weights_only=False)
        try:
            ac.pi.load_state_dict({k[4:]: v for k, v in ck["actor"].items()
                                   if k.startswith("net.")}, strict=False)
            print("[ppo] warm-started pi (partial)", flush=True)
        except Exception as e:
            print("[ppo] warm-start skipped:", e, flush=True)
    opt = torch.optim.Adam(ac.parameters(), lr=a.lr)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)

    N, T = a.nworld, a.rollout
    obs_b = torch.zeros(T, N, OBS_DIM, device=dev)
    act_b = torch.zeros(T, N, ACT_DIM, device=dev)
    logp_b = torch.zeros(T, N, device=dev)
    rew_b = torch.zeros(T, N, device=dev)
    done_b = torch.zeros(T, N, device=dev)
    val_b = torch.zeros(T + 1, N, device=dev)

    obs = env.observe()
    step, n_up, t0 = 0, 0, time.time()
    ep = dict(n=0, ret=0.0, len=0.0, placed=0, sealed=0,
              comp=np.zeros(len(env.RKEYS)))
    while step < a.steps:
        with torch.no_grad():
            for t in range(T):
                dist = ac.dist(obs)
                raw = dist.sample()
                actn = torch.tanh(raw)
                obs_b[t] = obs
                act_b[t] = raw
                logp_b[t] = dist.log_prob(raw).sum(-1)
                val_b[t] = ac.v(obs).squeeze(-1)
                obs, r, done, info = env.step(actn)
                rew_b[t] = r
                done_b[t] = done.float()
                if done.any():
                    di = torch.nonzero(done).squeeze(-1)
                    ep["n"] += di.numel()
                    ep["ret"] += float(info["ep_comp"][di].sum(-1).sum())
                    ep["len"] += float(info["ep_len"][di].float().sum())
                    ep["placed"] += int(info["placed"][di].sum())
                    ep["sealed"] += int(info["ever_sealed"][di].sum())
                    ep["comp"] += info["ep_comp"][di].sum(0).cpu().numpy()
            val_b[T] = ac.v(obs).squeeze(-1)
            adv = torch.zeros(T, N, device=dev)
            gae = torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                delta = rew_b[t] + a.gamma * (1 - done_b[t]) * val_b[t + 1] - val_b[t]
                gae = delta + a.gamma * a.lam * (1 - done_b[t]) * gae
                adv[t] = gae
            ret = adv + val_b[:T]
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        step += N * T

        fo = obs_b.reshape(-1, OBS_DIM)
        fa = act_b.reshape(-1, ACT_DIM)
        fl = logp_b.reshape(-1)
        fadv = adv.reshape(-1)
        fret = ret.reshape(-1)
        idx_all = torch.randperm(fo.shape[0], device=dev)
        pl = vl = el = 0.0
        nb = 0
        for _ in range(a.epochs):
            for k in range(0, fo.shape[0], a.minibatch):
                mb = idx_all[k:k + a.minibatch]
                dist = ac.dist(fo[mb])
                lp = dist.log_prob(fa[mb]).sum(-1)
                ratio = (lp - fl[mb]).exp()
                l1 = ratio * fadv[mb]
                l2 = ratio.clamp(1 - a.clip, 1 + a.clip) * fadv[mb]
                lpi = -torch.min(l1, l2).mean()
                lv = 0.5 * (ac.v(fo[mb]).squeeze(-1) - fret[mb]).pow(2).mean()
                lent = -dist.entropy().sum(-1).mean()
                loss = lpi + 0.5 * lv + a.ent * lent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
                opt.step()
                pl += float(lpi); vl += float(lv); el += float(-lent); nb += 1
        n_up += 1
        if n_up % 5 == 0:
            n_ep = max(1, ep["n"])
            rec = dict(step=step, updates=n_up, critic=vl / nb, actor=pl / nb,
                       alpha=0.0, entropy=el / nb, q_mean=float(val_b.mean()),
                       ep_ret=ep["ret"] / n_ep, ep_len=ep["len"] / n_ep,
                       success=ep["placed"] / n_ep, seal_rate=ep["sealed"] / n_ep,
                       sps=step / (time.time() - t0),
                       comp={k: ep["comp"][i] / n_ep
                             for i, k in enumerate(env.RKEYS)})
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"[ppo] {step:>10,} | succ {rec['success']:.2%} "
                  f"seal {rec['seal_rate']:.2%} ret {rec['ep_ret']:.2f} "
                  f"sps {rec['sps']:,.0f}", flush=True)
            ep = dict(n=0, ret=0.0, len=0.0, placed=0, sealed=0,
                      comp=np.zeros(len(env.RKEYS)))
            torch.save(dict(ac=ac.state_dict(), step=step),
                       os.path.join(a.out, "ac.pt"))
    torch.save(dict(ac=ac.state_dict(), step=step),
               os.path.join(a.out, "final.pt"))
    print("[ppo] done", flush=True)


if __name__ == "__main__":
    main()
