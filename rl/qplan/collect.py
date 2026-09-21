#!/usr/bin/env python3
"""qplan.collect :: roll a behaviour policy in the twin and store chunked transitions (M0).

Storage is by TRAJECTORY, not by transition: a 150-decision episode holds 150 overlapping
chunked transitions and storing them flat would repeat every observation H + 1 times.  One
episode is 151 x 40 + 150 x 7 + 150 values = 14.6 kB in fp16; a 4096-world batch is 60 MB and
carries 614 400 chunked transitions.  `critic.QBuffer` slices the chunks out on the GPU.

Per decision t the buffer therefore yields exactly what the plan asks for:
    obs_t (40) | executed chunk a_{t:t+H} (H x 7, end padded by repeating the last action)
    r_succ (H) | r_time (H) | done | obs_{t+H} | the executed chunk at t+H (bootstrap action)
plus the per-episode labels (placed, final distance, t_seal, t_goal, peak |qd|, env.privileged()
and the tracker-error draw).

Behaviour modes
    pi       the frozen policy alone
    explore  pi, but with probability --p_explore a world commits to ONE uniformly drawn
             candidate from proposals.propose() and executes it OPEN-LOOP for H decisions.
             This is the "arbitrary proposal" mode: it is what puts the planner's candidate
             manifold into the replay buffer (the critic is queried there, so it must be
             trained there).
    planner  the Q-planner itself (used by iterate.py for the online iterations).

    $PY rl/qplan/collect.py --nworld 4096 --batches 4 --mode explore --tag m0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (ACT_DIM, DATA_ROOT, H, OBS_DIM, R_DIST_W, R_SUCC_W, R_TIME_W,  # noqa: E402
                    load_pi, make_env, seed_env)
import proposals as PR                                                             # noqa: E402


# --------------------------------------------------------------------------- behaviour policies
class PiActor:
    name = "pi"

    def __init__(self, pol):
        self.pol = pol

    def reset(self, n, device):
        pass

    def __call__(self, obs, t):
        return self.pol(obs)


class ExploreActor:
    """pi + open-loop proposal chunks (the arbitrary-proposal collection mode)."""
    name = "explore"

    def __init__(self, pol, p=0.25, n_cand=32, gen=None, sigma_shared=0.15, sigma_step=0.05):
        self.pol, self.p, self.n_cand, self.gen = pol, float(p), int(n_cand), gen
        self.kw = dict(sigma_shared=sigma_shared, sigma_step=sigma_step)

    def reset(self, n, device):
        self.buf = torch.zeros(n, H, ACT_DIM, device=device)
        self.left = torch.zeros(n, dtype=torch.long, device=device)

    def __call__(self, obs, t):
        a_pi = self.pol(obs)
        n = obs.shape[0]
        u = torch.rand(n, device=obs.device, generator=self.gen) if self.gen is not None \
            else torch.rand(n, device=obs.device)
        start = (self.left == 0) & (u < self.p)
        if start.any():
            cand, _ = PR.sample_one(a_pi, n_cand=self.n_cand, gen=self.gen, **self.kw)
            self.buf = torch.where(start[:, None, None], cand, self.buf)
            self.left = torch.where(start, torch.full_like(self.left, H), self.left)
        use = self.left > 0
        a = torch.where(use[:, None], self.buf[:, 0], a_pi)
        self.buf = torch.roll(self.buf, -1, dims=1)
        self.left = (self.left - 1).clamp(min=0)
        return a


class ScaleActor:
    """pi with a BLANKET joint-action scale that is re-drawn per episode / per segment.

    Added after the M1 iteration-1 diagnosis (`diag.py`): trained on `pi` + `explore` data alone
    the TIME head just re-encodes the SUCCESS head -- every deviation in that buffer is a
    deviation that FAILS (explore episodes place 18 % of the time), so "chunk deviates" and
    "episode is slow" are the same feature and Q_time ranks pi's own chunk fastest.  A blanket
    scale s in [lo, hi] held for a whole episode or segment produces episodes that mostly still
    SUCCEED but reach the goal at measurably different times, which is the contrast the time head
    needs -- and s * pi's chunk is exactly the planner's structured candidate family.

    Two scales per episode (before / after the first seal) plus, on half the worlds, a re-draw
    every --seg decisions, so the critic also sees that the right speed is phase dependent.
    """
    name = "scale"

    def __init__(self, pol, env, lo=0.6, hi=1.6, seg=10, p_seg=0.5, p=0.08, n_cand=32,
                 gen=None, sigma_shared=0.10, sigma_step=0.04):
        self.pol, self.env, self.lo, self.hi = pol, env, float(lo), float(hi)
        self.seg, self.p_seg, self.p, self.n_cand, self.gen = int(seg), float(p_seg), float(p), int(n_cand), gen
        self.kw = dict(sigma_shared=sigma_shared, sigma_step=sigma_step)

    def _u(self, n, device):
        u = torch.rand(n, device=device, generator=self.gen) if self.gen is not None \
            else torch.rand(n, device=device)
        return self.lo + (self.hi - self.lo) * u

    def reset(self, n, device):
        self.dev = device
        self.s_pre, self.s_post = self._u(n, device), self._u(n, device)
        r = torch.rand(n, device=device, generator=self.gen) if self.gen is not None \
            else torch.rand(n, device=device)
        self.segmented = r < self.p_seg
        self.buf = torch.zeros(n, H, ACT_DIM, device=device)
        self.left = torch.zeros(n, dtype=torch.long, device=device)

    def __call__(self, obs, t):
        n = obs.shape[0]
        if t > 0 and t % self.seg == 0:
            ns = self._u(n, self.dev)
            self.s_pre = torch.where(self.segmented, ns, self.s_pre)
            self.s_post = torch.where(self.segmented, ns, self.s_post)
        a = self.pol(obs)
        s = torch.where(self.env.ever_sealed, self.s_post, self.s_pre)[:, None]
        a = torch.cat([(s * a[:, :6]).clamp(-1, 1), a[:, 6:]], -1)
        u = torch.rand(n, device=obs.device, generator=self.gen) if self.gen is not None \
            else torch.rand(n, device=obs.device)
        start = (self.left == 0) & (u < self.p)
        if start.any():
            cand, _ = PR.sample_one(a, n_cand=self.n_cand, gen=self.gen, **self.kw)
            self.buf = torch.where(start[:, None, None], cand, self.buf)
            self.left = torch.where(start, torch.full_like(self.left, H), self.left)
        use = self.left > 0
        out = torch.where(use[:, None], self.buf[:, 0], a)
        self.buf = torch.roll(self.buf, -1, dims=1)
        self.left = (self.left - 1).clamp(min=0)
        return out


# --------------------------------------------------------------------------- rollout -> shard
@torch.no_grad()
def rollout_traj(env, actor, ep_len, device, seed, time_ref="at_goal"):
    """One paired, deterministic batch of full episodes.  Returns CPU tensors."""
    N = env.nworld
    seed_env(env, seed, device)
    actor.reset(N, device)
    obs = env.observe()
    OBS = torch.zeros(ep_len + 1, N, OBS_DIM, dtype=torch.float16, device=device)
    ACT = torch.zeros(ep_len, N, ACT_DIM, dtype=torch.float16, device=device)
    ATG = torch.zeros(ep_len, N, dtype=torch.bool, device=device)
    info = {}
    for t in range(ep_len):
        OBS[t] = obs.half()
        a = actor(obs, t).clamp(-1, 1)
        ACT[t] = a.half()
        obs, _r, _d, info = env.step(a)
        # what stops the time head's clock.  "placed" = the scripted place-down finished and the
        # object is resting within tolerance (the robot's real end of cycle); "at_goal" = the
        # object is merely held at the goal (the pre-place-phase definition, kept for comparison).
        ATG[t] = info["placed_now"] if (time_ref == "placed" and "placed_now" in info) \
            else info["at_goal"]
    OBS[ep_len] = obs.half()
    out = dict(
        obs=OBS.transpose(0, 1).contiguous().cpu(),        # (E, T+1, 40)
        act=ACT.transpose(0, 1).contiguous().cpu(),        # (E, T, 7)
        at_goal=ATG.transpose(0, 1).contiguous().cpu(),    # (E, T)
        placed=info["placed"].float().cpu(),
        final_d=info["final_d"].float().cpu(),
        t_seal=info["t_seal"].float().cpu(),
        t_goal=info["t_goal"].float().cpu(),
        ever_sealed=info["ever_sealed"].float().cpu(),
        peak_qd=info["peak_qd"].float().cpu(),
        priv=env.privileged().float().cpu(),
        obj_err=env.obj_err.float().cpu(),
    )
    for k in ("t_placed", "place_err"):
        if k in info:
            out[k] = info[k].float().cpu()
    out["r_term"] = R_SUCC_W * out["placed"] - R_DIST_W * (out["final_d"] * 100.0)
    return out


def shard_meta(out, ep_len, extra=None):
    m = dict(episodes=int(out["obs"].shape[0]), ep_len=int(ep_len), H=H,
             transitions=int(out["obs"].shape[0] * ep_len),
             success=float(out["placed"].mean()), seal=float(out["ever_sealed"].mean()),
             t_seal=float(out["t_seal"].mean()), t_goal=float(out["t_goal"].mean()),
             final_d_med=float(out["final_d"].median()),
             t_placed=float(out["t_placed"].mean()) if "t_placed" in out else None,
             place_err_med=float(out["place_err"].median()) if "place_err" in out else None,
             r_term=float(out["r_term"].mean()),
             r_time_w=R_TIME_W, r_succ_w=R_SUCC_W, r_dist_w=R_DIST_W)
    if extra:
        m.update(extra)
    return m


def save_shard(out, path, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = dict(out)
    out["meta"] = meta
    torch.save(out, path)
    return os.path.getsize(path)


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworld", type=int, default=4096)
    ap.add_argument("--batches", type=int, default=4, help="episode batches (each = nworld episodes)")
    ap.add_argument("--ep_len", type=int, default=150)
    ap.add_argument("--mode", default="explore", choices=["pi", "explore", "scale"])
    ap.add_argument("--scale_lo", type=float, default=0.6)
    ap.add_argument("--scale_hi", type=float, default=1.6)
    ap.add_argument("--seg", type=int, default=10)
    ap.add_argument("--p_explore", type=float, default=0.25)
    ap.add_argument("--n_cand", type=int, default=32)
    ap.add_argument("--dq_max", type=float, default=3.0)
    ap.add_argument("--obj_err", type=int, default=0, help="1 = tracker-error DR on (see README)")
    ap.add_argument("--place_phase", type=int, default=0,
                    help="1 = scripted place-down + release + settle (env_paper place_phase)")
    ap.add_argument("--time_ref", default="auto", choices=["auto", "at_goal", "placed"],
                    help="what stops the time head's clock (auto: placed when place_phase else at_goal)")
    ap.add_argument("--no_dr", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="m0")
    ap.add_argument("--out", default=os.path.join(DATA_ROOT, "data"))
    a = ap.parse_args()

    import warp as wp
    wp.init()
    dev = "cuda:0"
    torch.manual_seed(a.seed)
    env = make_env(a.nworld, device=dev, seed=a.seed, dr=not a.no_dr, obj_err=bool(a.obj_err),
                   dq_max_deg=a.dq_max, ep_len=a.ep_len, place_phase=bool(a.place_phase))
    time_ref = a.time_ref if a.time_ref != "auto" else ("placed" if a.place_phase else "at_goal")
    pol = load_pi(dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed + 777)
    if a.mode == "pi":
        actor = PiActor(pol)
    elif a.mode == "explore":
        actor = ExploreActor(pol, p=a.p_explore, n_cand=a.n_cand, gen=gen)
    else:
        actor = ScaleActor(pol, env, lo=a.scale_lo, hi=a.scale_hi, seg=a.seg,
                           p=a.p_explore, n_cand=a.n_cand, gen=gen)

    print(f"[collect] mode={a.mode} nworld={a.nworld} batches={a.batches} ep_len={a.ep_len} "
          f"dq_max={a.dq_max} dr={not a.no_dr} obj_err={bool(a.obj_err)} out={a.out}", flush=True)
    tot_tr, tot_t = 0, 0.0
    for b in range(a.batches):
        t0 = time.time()
        out = rollout_traj(env, actor, a.ep_len, dev, seed=a.seed + 1000 * b, time_ref=time_ref)
        dt = time.time() - t0
        meta = shard_meta(out, a.ep_len, dict(mode=a.mode, seed=a.seed + 1000 * b, batch=b,
                                              p_explore=a.p_explore if a.mode != "pi" else 0.0,
                                              obj_err=bool(a.obj_err), dq_max=a.dq_max,
                                              place_phase=bool(a.place_phase), time_ref=time_ref,
                                              seconds=round(dt, 1)))
        path = os.path.join(a.out, f"{a.tag}_{a.mode}_b{b:02d}.pt")
        mb = save_shard(out, path, meta) / 1e6
        tot_tr += meta["transitions"]
        tot_t += dt
        print(f"[collect] {os.path.basename(path)} {meta['transitions']:,} transitions in {dt:.1f}s "
              f"({meta['transitions'] / dt:,.0f}/s, {mb:.0f} MB) | success {meta['success']:.2%} "
              f"seal {meta['seal']:.2%} t_goal {meta['t_goal']:.1f} dec", flush=True)
    print(f"[collect] TOTAL {tot_tr:,} transitions in {tot_t / 60:.1f} min "
          f"= {tot_tr / tot_t:,.0f} transitions/s ({200000 / (tot_tr / tot_t):.0f} s per 200k)", flush=True)
    with open(os.path.join(a.out, f"{a.tag}_{a.mode}_summary.json"), "w") as f:
        json.dump(dict(transitions=tot_tr, seconds=tot_t, per_s=tot_tr / tot_t, args=vars(a)), f, indent=1)


if __name__ == "__main__":
    main()
