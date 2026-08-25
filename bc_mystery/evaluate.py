"""evaluate :: batched closed-loop evaluation harness (BC-mystery style).

Pluggable policy interface, chunked execution, optional history stacking —
the knobs Mysteries 1/2/4 are measured with. Success = ppo4-era criterion
(at rest, released, <=3.5 cm, lifted) + the full ladder for context, plus
an offline proxy (action MSE vs the spline expert on identical states) so
val-loss-vs-success divergence is measurable per eval.

Policies:
    --policy expert            spline expert itself (upper anchor)
    --policy random|zero       lower anchors
    --policy ckpt:PATH         torch module: act(obs[N,obs_dim]) -> [N,K,7]

    python bc_mystery/evaluate.py --policy expert --episodes 512
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "rl"))
sys.path.insert(0, HERE)
import warp as wp                                          # noqa: E402
from collect import SplineExpert, TermTracker, T_EP        # noqa: E402


class RandomPolicy:
    def __init__(self, K):
        self.K = K

    def act(self, obs):
        return torch.rand(obs.shape[0], self.K, 7, device=obs.device) * 2 - 1


class ZeroPolicy:
    def __init__(self, K):
        self.K = K

    def act(self, obs):
        return torch.zeros(obs.shape[0], self.K, 7, device=obs.device)


class HistoryWrap:
    """Stack the last H observations as policy input (Mystery 2 knob)."""
    def __init__(self, policy, H):
        self.policy = policy
        self.H = H
        self.buf = None

    def reset(self, N, obs_dim, dev):
        self.buf = torch.zeros(N, self.H, obs_dim, device=dev)

    def act(self, obs):
        self.buf = torch.roll(self.buf, -1, dims=1)
        self.buf[:, -1] = obs
        return self.policy.act(self.buf.reshape(obs.shape[0], -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="expert")
    ap.add_argument("--episodes", type=int, default=512)
    ap.add_argument("--nworld", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=25,
                    help="actions produced per policy call")
    ap.add_argument("--exec_len", type=int, default=25,
                    help="actions executed before re-querying (<= chunk)")
    ap.add_argument("--history", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scene", default=os.path.join(
        os.path.dirname(HERE), "rl", "scenes", "box_med_ped.xml"))
    a = ap.parse_args()
    wp.init()
    from env_warp import PickEnv
    rng = np.random.default_rng(a.seed)
    env = PickEnv(nworld=a.nworld, mode="pnp", dr=False, xml=a.scene,
                  lift_req=0.30)
    env.auto_reset = False
    dev = env.device

    expert = SplineExpert(env, rng)        # also the offline-MSE reference
    if a.policy == "expert":
        policy = None                      # expert acts directly per tick
    elif a.policy == "random":
        policy = RandomPolicy(a.chunk)
    elif a.policy == "zero":
        policy = ZeroPolicy(a.chunk)
    elif a.policy.startswith("ckpt:"):
        policy = torch.load(a.policy[5:], map_location=dev,
                            weights_only=False)
        policy.eval()
    else:
        raise SystemExit(f"unknown policy {a.policy}")
    if a.history > 0 and policy is not None:
        policy = HistoryWrap(policy, a.history)

    n_rounds = (a.episodes + a.nworld - 1) // a.nworld
    met_all, mse_all = [], []
    for rd in range(n_rounds):
        env.reset(torch.ones(a.nworld, dtype=torch.bool, device=dev))
        expert.plan()
        if isinstance(policy, HistoryWrap):
            policy.reset(a.nworld, env.observe().shape[1], dev)
        obs = env.observe()
        trk = TermTracker(a.nworld, dev)
        chunk, ci = None, 0
        for t in range(T_EP):
            ref = expert.act(t)            # expert action on the SAME state
            if policy is None:
                act = ref
            else:
                if chunk is None or ci >= a.exec_len:
                    with torch.no_grad():
                        chunk = policy.act(obs).clamp(-1, 1)
                    ci = 0
                act = chunk[:, min(ci, chunk.shape[1] - 1)]
                ci += 1
                mse_all.append(float(((act - ref) ** 2).mean()))
            obs, r, done, info = env.step(act)
            trk.update(done, info)
        met_all.append(trk.table().cpu().numpy())
    M = np.concatenate(met_all)[:a.episodes]
    # cols: seal setd v1 v2 v3 final_d max_lift final_spd
    print(f"policy={a.policy} chunk={a.chunk} exec={a.exec_len} "
          f"hist={a.history} | eps {len(M)}")
    print(f"  seal {M[:,0].mean():.1%}  setd {M[:,1].mean():.1%}  "
          f"V1 {M[:,2].mean():.1%}  V2 {M[:,3].mean():.1%}  "
          f"V3 {M[:,4].mean():.1%}")
    print(f"  place err p50 {np.percentile(M[:,5],50)*100:.1f} cm  "
          f"p90 {np.percentile(M[:,5],90)*100:.1f} cm | "
          f"lift p50 {np.percentile(M[:,6],50)*100:.1f} cm")
    if mse_all:
        print(f"  offline action-MSE vs expert: {np.mean(mse_all):.4f}")


if __name__ == "__main__":
    main()
