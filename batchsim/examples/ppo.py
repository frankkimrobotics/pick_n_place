"""Minimal PPO on PickLiftEnv: a smoke test that batched RL runs end to end through the latency-aware twin.
  python -m batchsim.examples.ppo --nworld 1024 --iters 40 --device cuda:0        (mj4 env)"""
import argparse, json, os, sys, time
import torch, torch.nn as nn
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from batchsim.env import PickLiftEnv


class AC(nn.Module):
    def __init__(self, od, ad, h=256):
        super().__init__()
        mk = lambda o: nn.Sequential(nn.Linear(od, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, o))
        self.pi, self.v = mk(ad), mk(1)
        self.logstd = nn.Parameter(torch.full((ad,), -0.7))

    def dist(self, o):
        return torch.distributions.Normal(self.pi(o), self.logstd.exp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="~/pnp_dataset_mega/scene_980/scene.xml")
    ap.add_argument("--nworld", type=int, default=1024); ap.add_argument("--iters", type=int, default=40); ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--dr", type=int, default=1); ap.add_argument("--level", default="real")
    ap.add_argument("--out", default="~/pnp_rollouts_realsim/batch/ppo_smoke.json")
    a = ap.parse_args()
    env = PickLiftEnv(a.xml, a.nworld, a.device, dr=bool(a.dr), level=a.level, infer_mean=0.02, infer_sd=0.004, infer_min=0.01)   # an MLP policy is fast
    ac = AC(env.obs_dim, env.act_dim).to(a.device); opt = torch.optim.Adam(ac.parameters(), 3e-4)
    obs = env.reset(); log = []; t0 = time.time()
    for it in range(a.iters):
        O, A, LP, R, D, V = [], [], [], [], [], []
        succ = done_n = bad = 0; held = 0.0
        for _ in range(a.T):
            with torch.no_grad():
                dist = ac.dist(obs); act = dist.sample(); lp = dist.log_prob(act).sum(1); v = ac.v(obs).squeeze(1)
            nobs, r, d, info = env.step(act)
            O.append(obs); A.append(act); LP.append(lp); R.append(r); D.append(d.float()); V.append(v)
            succ += int(info["success"].sum()); done_n += int(d.sum()); held += float(info["held"].float().mean()); bad += int(info["bad"].sum())
            obs = nobs
        with torch.no_grad():
            adv, last, nv = [], 0, ac.v(obs).squeeze(1)
            for t in reversed(range(a.T)):
                delta = R[t] + 0.99 * nv * (1 - D[t]) - V[t]; last = delta + 0.99 * 0.95 * (1 - D[t]) * last; adv.insert(0, last); nv = V[t]
            adv = torch.stack(adv); ret = adv + torch.stack(V)
        O, A, LP, adv, ret = torch.cat(O), torch.cat(A), torch.cat(LP), adv.flatten(), ret.flatten()
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        for _ in range(4):
            for idx in torch.randperm(len(O), device=a.device).split(8192):
                dist = ac.dist(O[idx]); ratio = (dist.log_prob(A[idx]).sum(1) - LP[idx]).exp()
                loss = -torch.min(ratio * adv[idx], ratio.clamp(0.8, 1.2) * adv[idx]).mean() + 0.5 * (ac.v(O[idx]).squeeze(1) - ret[idx]).pow(2).mean() - 0.001 * dist.entropy().sum(1).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(ac.parameters(), 1.0); opt.step()
        row = dict(it=it, rew_per_step=float(torch.stack(R).mean()), held_frac=held / a.T, success=succ, episodes=done_n, bad_worlds=bad,
                   env_steps_per_s=(it + 1) * a.T * a.nworld / (time.time() - t0), cmd_latency_ms=float(1000 * info["latency"].mean()))
        log.append(row); print(json.dumps(row), flush=True)
    json.dump(log, open(os.path.expanduser(a.out), "w"), indent=1)


if __name__ == "__main__":
    main()
