"""sac :: Soft Actor-Critic on the batched PickEnv (single GPU).

Logs (jsonl, every LOG_EVERY updates + every finished-episode batch):
  losses: critic, actor, alpha, entropy, q-values
  episodes: return, length, success/seal rates, and the FULL per-component
  reward breakdown (approach/align/press/seal/lift/transport/place/...)
Plot with rl/plot_training.py.

Run (mjwarp env):  python rl/sac.py --nworld 2048 --steps 3_000_000
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import warp as wp                                          # noqa: E402

OBS_DIM = 34
ACT_DIM = 7
LOG_EVERY = 200


def mlp(inp, out, hidden=512, layers=3, ln=True):
    seq = []
    d = inp
    for _ in range(layers):
        seq += [nn.Linear(d, hidden)]
        if ln:
            seq += [nn.LayerNorm(hidden)]
        seq += [nn.SiLU()]
        d = hidden
    seq += [nn.Linear(d, out)]
    return nn.Sequential(*seq)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(OBS_DIM, 2 * ACT_DIM, ln=False)

    def forward(self, obs):
        mu, log_std = self.net(obs).chunk(2, -1)
        log_std = log_std.clamp(-5, 2)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self(obs)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        pre = mu + std * eps
        a = torch.tanh(pre)
        logp = (-0.5 * (eps**2 + 2 * log_std + np.log(2 * np.pi))).sum(-1)
        logp = logp - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, logp


class Replay:
    def __init__(self, cap, device):
        self.cap = cap
        self.device = device
        self.obs = torch.zeros(cap, OBS_DIM, device=device)
        self.act = torch.zeros(cap, ACT_DIM, device=device)
        self.rew = torch.zeros(cap, device=device)
        self.nobs = torch.zeros(cap, OBS_DIM, device=device)
        self.done = torch.zeros(cap, device=device)
        self.ptr = 0
        self.full = False

    def add(self, o, a, r, no, d):
        n = o.shape[0]
        idx = (self.ptr + torch.arange(n, device=self.device)) % self.cap
        self.obs[idx] = o
        self.act[idx] = a
        self.rew[idx] = r
        self.nobs[idx] = no
        self.done[idx] = d.float()
        self.ptr = int((self.ptr + n) % self.cap)
        if self.ptr < n:
            self.full = True

    def sample(self, bs):
        hi = self.cap if self.full else self.ptr
        idx = torch.randint(0, hi, (bs,), device=self.device)
        return (self.obs[idx], self.act[idx], self.rew[idx],
                self.nobs[idx], self.done[idx])

    def __len__(self):
        return self.cap if self.full else self.ptr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworld", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--utd", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=20_000)
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/run1"))
    ap.add_argument("--scene", default=os.path.join(HERE, "scenes", "box_med.xml"))
    ap.add_argument("--mode", default="full", choices=["full", "attach", "pnp"])
    ap.add_argument("--init", default=None, help="warm-start ckpt (actor[+critics])")
    ap.add_argument("--dr", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = a.device
    torch.manual_seed(0)

    wp.init()
    from env_warp import PickEnv
    env = PickEnv(nworld=a.nworld, device=dev, xml=a.scene, mode=a.mode, dr=a.dr)
    print(f"[sac] env up: {a.nworld} worlds", flush=True)

    actor = Actor().to(dev)
    q1, q2 = mlp(OBS_DIM + ACT_DIM, 1).to(dev), mlp(OBS_DIM + ACT_DIM, 1).to(dev)
    q1t, q2t = mlp(OBS_DIM + ACT_DIM, 1).to(dev), mlp(OBS_DIM + ACT_DIM, 1).to(dev)
    q1t.load_state_dict(q1.state_dict())
    q2t.load_state_dict(q2.state_dict())
    log_alpha = torch.tensor(np.log(0.2), device=dev, requires_grad=True)
    tgt_ent = -float(ACT_DIM)
    opt_a = torch.optim.Adam(actor.parameters(), lr=a.lr)
    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=a.lr)
    opt_al = torch.optim.Adam([log_alpha], lr=a.lr)
    if a.init:
        ck = torch.load(a.init, map_location=dev, weights_only=False)
        actor.load_state_dict(ck["actor"])
        if "q1" in ck:
            q1.load_state_dict(ck["q1"]); q2.load_state_dict(ck["q2"])
            q1t.load_state_dict(ck["q1"]); q2t.load_state_dict(ck["q2"])
        print(f"[sac] warm-started from {a.init}", flush=True)
    buf = Replay(2_000_000, dev)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)

    obs = env.observe()
    step = 0
    n_updates = 0
    t0 = time.time()
    ep_stats = dict(n=0, ret=0.0, len=0.0, placed=0, sealed=0,
                    comp=np.zeros(len(env.RKEYS)))
    while step < a.steps:
        with torch.no_grad():
            if step < a.warmup:
                act = torch.rand(a.nworld, ACT_DIM, device=dev) * 2 - 1
            else:
                act, _ = actor.sample(obs)
        nobs, rew, done, info = env.step(act)
        buf.add(obs, act, rew, nobs, done)
        obs = nobs
        step += a.nworld

        if done.any():
            di = torch.nonzero(done).squeeze(-1)
            ep_stats["n"] += di.numel()
            ep_stats["ret"] += float(info["ep_comp"][di].sum(-1).sum())
            ep_stats["len"] += float(info["ep_len"][di].float().sum())
            ep_stats["placed"] += int(info["placed"][di].sum())
            ep_stats["sealed"] += int(info["ever_sealed"][di].sum())
            ep_stats["comp"] += info["ep_comp"][di].sum(0).cpu().numpy()

        if len(buf) >= a.warmup:
            closs = aloss = alos = ent = qm = 0.0
            for _ in range(a.utd):
                o, ac, r, no, dn = buf.sample(a.batch)
                with torch.no_grad():
                    na, nlogp = actor.sample(no)
                    tq = torch.min(q1t(torch.cat([no, na], -1)),
                                   q2t(torch.cat([no, na], -1))).squeeze(-1)
                    y = r + a.gamma * (1 - dn) * (tq - log_alpha.exp() * nlogp)
                qa1 = q1(torch.cat([o, ac], -1)).squeeze(-1)
                qa2 = q2(torch.cat([o, ac], -1)).squeeze(-1)
                lq = F.mse_loss(qa1, y) + F.mse_loss(qa2, y)
                opt_q.zero_grad(); lq.backward(); opt_q.step()
                pa, plogp = actor.sample(o)
                qpi = torch.min(q1(torch.cat([o, pa], -1)),
                                q2(torch.cat([o, pa], -1))).squeeze(-1)
                la = (log_alpha.exp().detach() * plogp - qpi).mean()
                opt_a.zero_grad(); la.backward(); opt_a.step()
                lal = -(log_alpha.exp() * (plogp.detach() + tgt_ent)).mean()
                opt_al.zero_grad(); lal.backward(); opt_al.step()
                with torch.no_grad():
                    for pt, p in zip(list(q1t.parameters()) + list(q2t.parameters()),
                                     list(q1.parameters()) + list(q2.parameters())):
                        pt.mul_(1 - a.tau).add_(p, alpha=a.tau)
                closs += float(lq); aloss += float(la); alos += float(lal)
                ent += float(-plogp.mean()); qm += float(qpi.mean())
                n_updates += 1

            if n_updates % LOG_EVERY < a.utd:
                n_ep = max(1, ep_stats["n"])
                rec = dict(step=step, updates=n_updates,
                           critic=closs / a.utd, actor=aloss / a.utd,
                           alpha=float(log_alpha.exp()), entropy=ent / a.utd,
                           q_mean=qm / a.utd,
                           ep_ret=ep_stats["ret"] / n_ep,
                           ep_len=ep_stats["len"] / n_ep,
                           success=ep_stats["placed"] / n_ep,
                           seal_rate=ep_stats["sealed"] / n_ep,
                           sps=step / (time.time() - t0),
                           comp={k: ep_stats["comp"][i] / n_ep
                                 for i, k in enumerate(env.RKEYS)})
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(f"[sac] {step:>9,} | succ {rec['success']:.2%} "
                      f"seal {rec['seal_rate']:.2%} ret {rec['ep_ret']:.2f} "
                      f"critic {rec['critic']:.3f} sps {rec['sps']:,.0f}",
                      flush=True)
                ep_stats = dict(n=0, ret=0.0, len=0.0, placed=0, sealed=0,
                                comp=np.zeros(len(env.RKEYS)))
                torch.save(dict(actor=actor.state_dict(), step=step),
                           os.path.join(a.out, "actor.pt"))
    torch.save(dict(actor=actor.state_dict(), q1=q1.state_dict(),
                    q2=q2.state_dict(), step=step),
               os.path.join(a.out, "final.pt"))
    print("[sac] done", flush=True)


if __name__ == "__main__":
    main()
