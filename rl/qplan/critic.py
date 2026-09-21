#!/usr/bin/env python3
"""qplan.critic :: Q_phi(obs, chunk) with HL-Gauss categorical regression (M1).

    Q_phi : R^40 x R^35 -> (Q_succ, Q_time)
    trunk  MLP [512, 512, 256] (ELU, the repo's `paper` arch) on the concatenated, normalised
           (obs, chunk); two HL-Gauss heads of 51 bins each.
    HL-Gauss (Farebrother et al. 2024): the scalar target y is projected onto the bin support
           with a Gaussian kernel of sigma = 0.75 bin widths (truncated-Gaussian CDF differences,
           renormalised), the loss is cross-entropy, and Q = sum_b v_b softmax_b.
    target y = sum_{i<H} gamma^i r_{t+i} + (1 - done) gamma^H Q_bar(o_{t+H}, c_{t+H})
           with an EMA target network (tau 0.005).  c_{t+H} is the chunk the BEHAVIOUR policy
           executed at o_{t+H} (`--target_chunk exec`, the default: an exact SARSA backup -- on
           the M0 data that policy IS pi, and on the online iterations it is the planner, which
           turns the loop into policy iteration) or repeat(pi(o_{t+H}), H) (`--target_chunk pi`).

    $PY rl/qplan/critic.py --data ~/pnp_rl/qplan/data --steps 20000 --out ~/pnp_rl/qplan/q0
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (ACT_DIM, CHUNK_DIM, DATA_ROOT, GAMMA, H, HLG_SIGMA_BINS, N_BINS,  # noqa: E402
                    OBS_DIM, R_TIME_W, V_RANGE, V_RANGE_PLACE, V_SOFT, speed_excess)

# The head set is taken from the checkpoint's `v_range` (QChunk.head_names), so a two-head critic
# trained before the speed head existed still loads and still runs the planner.
HEADS = ("succ", "time", "speed")


# --------------------------------------------------------------------------- HL-Gauss
class HLGauss(nn.Module):
    """Categorical value head over `n_bins` uniform bins of [vmin, vmax]."""

    def __init__(self, vmin, vmax, n_bins=N_BINS, sigma_bins=HLG_SIGMA_BINS):
        super().__init__()
        edges = torch.linspace(vmin, vmax, n_bins + 1)
        self.register_buffer("edges", edges)
        self.register_buffer("centers", 0.5 * (edges[:-1] + edges[1:]))
        self.sigma = float(sigma_bins * (vmax - vmin) / n_bins)
        self.vmin, self.vmax, self.n_bins = float(vmin), float(vmax), int(n_bins)

    def project(self, y):
        """(B,) scalar targets -> (B, n_bins) probabilities."""
        y = y.clamp(self.vmin, self.vmax).unsqueeze(-1)
        cdf = torch.special.ndtr((self.edges - y) / self.sigma)          # (B, n_bins+1)
        p = cdf[:, 1:] - cdf[:, :-1]
        return p / p.sum(-1, keepdim=True).clamp_min(1e-8)

    def q(self, logits):
        return (torch.softmax(logits, -1) * self.centers).sum(-1)

    def loss(self, logits, y):
        return -(self.project(y) * torch.log_softmax(logits, -1)).sum(-1)


def mlp(inp, out, hidden=(512, 512, 256)):
    seq, d = [], inp
    for h in hidden:
        seq += [nn.Linear(d, h), nn.ELU()]
        d = h
    seq += [nn.Linear(d, out)]
    return nn.Sequential(*seq)


class QChunk(nn.Module):
    """Shared trunk + one HL-Gauss head per value.

    `sep_speed=True` gives the SPEED head its own trunk.  Measured (README "Speed head"): with a
    shared trunk the third head's gradients cost the ranking quality of the other two -- the same
    planner that scored 99.0 % on the two-head critic scores 97.7 % on the three-head one -- while
    a separate trunk leaves succ/time exactly as they were and prices speed just as well.
    """

    def __init__(self, obs_dim=OBS_DIM, chunk_dim=CHUNK_DIM, hidden=(512, 512, 256),
                 n_bins=N_BINS, v_range=None, sep_speed=False):
        super().__init__()
        v_range = V_RANGE if v_range is None else v_range
        self.v_range = {k: tuple(v) for k, v in v_range.items()}
        # ORDER matters (the trunk's output slice is positional): always succ, time, speed.
        self.head_names = tuple(k for k in HEADS if k in self.v_range)
        self.obs_dim, self.chunk_dim, self.n_bins = obs_dim, chunk_dim, n_bins
        self.sep_speed = bool(sep_speed) and "speed" in self.head_names
        self.main_heads = tuple(k for k in self.head_names if not (self.sep_speed and k == "speed"))
        self.trunk = mlp(obs_dim + chunk_dim, n_bins * len(self.main_heads), hidden=hidden)
        if self.sep_speed:
            self.trunk_v = mlp(obs_dim + chunk_dim, n_bins, hidden=hidden)
        self.heads = nn.ModuleDict({k: HLGauss(*self.v_range[k], n_bins=n_bins)
                                    for k in self.head_names})
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    def set_norm(self, mean, std):
        self.obs_mean.copy_(mean)
        self.obs_std.copy_(std.clamp_min(1e-3))

    def logits(self, obs, chunk):
        x = torch.cat([(obs - self.obs_mean) / self.obs_std, chunk], -1)
        z = self.trunk(x)
        out = {k: z[..., i * self.n_bins:(i + 1) * self.n_bins]
               for i, k in enumerate(self.main_heads)}
        if self.sep_speed:
            out["speed"] = self.trunk_v(x)
        return out

    def forward(self, obs, chunk):
        lg = self.logits(obs, chunk)
        return {k: self.heads[k].q(lg[k]) for k in self.head_names}

    @torch.no_grad()
    def score(self, obs, chunks):
        """obs (N, 40), chunks (N, C, H, 7) -> {head: (N, C)} in one batched call."""
        N, C = chunks.shape[0], chunks.shape[1]
        o = obs[:, None, :].expand(N, C, obs.shape[-1]).reshape(N * C, -1)
        c = chunks.reshape(N * C, -1)
        q = self.forward(o, c)
        return {k: v.view(N, C) for k, v in q.items()}


# --------------------------------------------------------------------------- replay buffer
class QBuffer:
    """All episodes ever collected, on the GPU in fp16; chunks are sliced per batch."""

    def __init__(self, device="cuda:0", gamma=GAMMA, hor=H):
        self.device, self.gamma, self.H = device, gamma, hor
        self.obs = self.act = self.atg = self.qex = None
        self.rterm = self.placed = self.group = None
        self.srcs = []
        self.qex_derived = 0

    def add_file(self, path, group=0):
        d = torch.load(path, map_location="cpu", weights_only=False)
        self.add(d, os.path.basename(path), group=group)

    def add(self, d, name="mem", group=0):
        """group 0 = the FIXED offline pool (pi / exploration / scale / object-error episodes),
        group 1 = episodes the online loop generated.  `batch(..., fixed_frac)` then guarantees a
        minimum share of group 0 in every gradient batch, so the buffer filling up with the
        planner's own increasingly-similar rollouts cannot silently take the contrast out of the
        time head (M2 drift, see README)."""
        dev = self.device
        obs = d["obs"].to(dev)
        act = d["act"].to(dev)
        atg = d["at_goal"].to(dev)
        if "qd_ex" in d:
            qex = d["qd_ex"].to(dev)
        else:
            # shard collected before the speed head existed: recover the per-decision excess from
            # the stored observation (qd occupies obs[6:12], and obs_{t+1} holds the velocity the
            # decision-t action produced).  Costs the 0.005 rad/s observation noise; the collector
            # stores the clean value from now on.
            qex = speed_excess(obs[:, 1:, 6:12].float(), V_SOFT).half()
            self.qex_derived += int(obs.shape[0])
        rt = d["r_term"].to(dev).float()
        pl = d["placed"].to(dev).float()
        gp = torch.full((obs.shape[0],), int(group), dtype=torch.int8, device=dev)
        if self.obs is None:
            self.obs, self.act, self.atg, self.qex, self.rterm, self.placed, self.group = \
                obs, act, atg, qex, rt, pl, gp
        else:
            assert obs.shape[1] == self.obs.shape[1], "ep_len mismatch between shards"
            self.obs = torch.cat([self.obs, obs])
            self.act = torch.cat([self.act, act])
            self.atg = torch.cat([self.atg, atg])
            self.qex = torch.cat([self.qex, qex])
            self.rterm = torch.cat([self.rterm, rt])
            self.placed = torch.cat([self.placed, pl])
            self.group = torch.cat([self.group, gp])
        self.srcs.append((name, int(obs.shape[0]), int(group)))

    # -- shape helpers -----------------------------------------------------
    @property
    def E(self):
        return 0 if self.obs is None else int(self.obs.shape[0])

    @property
    def T(self):
        return 0 if self.act is None else int(self.act.shape[1])

    @property
    def n_trans(self):
        return self.E * self.T

    def split(self, holdout=0.05, seed=0):
        """Deterministic STRIDED holdout: episode i is held out iff i % k == 0.  Index-based so
        that appending new episodes (iterate.py) never moves an episode from val to train --
        a reshuffle would leak every previously held-out episode into training."""
        k = max(2, int(round(1.0 / max(1e-6, holdout))))
        ar = torch.arange(self.E, device=self.device)
        self.idx_val = ar[ar % k == 0]
        self.idx_tr = ar[ar % k != 0]
        g = self.group[self.idx_tr]
        self.idx_tr_fixed = self.idx_tr[g == 0]
        self.idx_tr_online = self.idx_tr[g == 1]
        return self.idx_tr.numel(), self.idx_val.numel()

    def obs_stats(self, n=200_000):
        e = torch.randint(0, self.E, (n,), device=self.device)
        t = torch.randint(0, self.T, (n,), device=self.device)
        o = self.obs[e, t].float()
        return o.mean(0), o.std(0)

    # -- batch -------------------------------------------------------------
    def batch(self, B, idx=None, gen=None, e=None, t=None, fixed_frac=None):
        dev, T, Hh = self.device, self.T, self.H
        if e is None:
            if fixed_frac is not None and idx is None and self.idx_tr_online.numel() > 0:
                nf = max(1, int(round(fixed_frac * B)))
                a_ = self.idx_tr_fixed[torch.randint(0, self.idx_tr_fixed.numel(), (nf,),
                                                     device=dev, generator=gen)]
                b_ = self.idx_tr_online[torch.randint(0, self.idx_tr_online.numel(), (B - nf,),
                                                      device=dev, generator=gen)]
                e = torch.cat([a_, b_])
            else:
                pool = self.idx_tr if idx is None else idx
                sel = torch.randint(0, pool.numel(), (B,), device=dev, generator=gen)
                e = pool[sel]
            t = torch.randint(0, T, (B,), device=dev, generator=gen)
        ar = torch.arange(Hh, device=dev)
        ti = t[:, None] + ar                                  # (B, H) decision indices
        valid = (ti <= T - 1).float()
        tic = ti.clamp(max=T - 1)
        eb = e[:, None].expand(-1, Hh)
        obs = self.obs[e, t].float()
        chunk = self.act[eb, tic].float().reshape(B, -1)
        # rewards
        r_time = -R_TIME_W * (~self.atg[eb, tic]).float() * valid
        r_succ = self.rterm[e][:, None] * (ti == (T - 1)).float()
        r_speed = -self.qex[eb, tic].float() * valid
        disc = self.gamma ** ar
        y = {"succ": (r_succ * disc).sum(-1), "time": (r_time * disc).sum(-1),
             "speed": (r_speed * disc).sum(-1)}
        # bootstrap
        nt = (t + Hh).clamp(max=T)
        nobs = self.obs[e, nt].float()
        nti = (t[:, None] + Hh + ar).clamp(max=T - 1)
        nchunk = self.act[eb, nti].float().reshape(B, -1)
        done = (t + Hh >= T).float()
        return dict(obs=obs, chunk=chunk, y=y, nobs=nobs, nchunk=nchunk, done=done, e=e, t=t)

    def mc_return(self, e, t, chunk=20000):
        """Exact discounted return from decision t of episode e (for calibration)."""
        T, dev = self.T, self.device
        ar = torch.arange(T, device=dev)
        outs, outs_v = [], []
        for i in range(0, e.numel(), chunk):
            ee, tt = e[i:i + chunk], t[i:i + chunk]
            k = ar[None, :] - tt[:, None]                     # (b, T)
            m = (k >= 0).float()
            disc = self.gamma ** k.clamp(min=0).float() * m
            outs.append((-R_TIME_W * (~self.atg[ee]).float() * disc).sum(-1))
            outs_v.append((-self.qex[ee].float() * disc).sum(-1))
        g_time, g_speed = torch.cat(outs), torch.cat(outs_v)
        g_succ = self.rterm[e] * (self.gamma ** (T - 1 - t).float())
        return {"succ": g_succ, "time": g_time, "speed": g_speed}


# --------------------------------------------------------------------------- training
def ema_(tgt, src, tau):
    with torch.no_grad():
        for a, b in zip(tgt.parameters(), src.parameters()):
            a.mul_(1 - tau).add_(b, alpha=tau)
        for a, b in zip(tgt.buffers(), src.buffers()):
            a.copy_(b)


def train_q(buf, q=None, steps=20000, batch=4096, lr=3e-4, tau=0.005, gamma=GAMMA,
            device="cuda:0", target_chunk="exec", pol=None, seed=0, log_every=1000,
            log=print, holdout=0.05, fixed_frac=None, v_range=None, sep_speed=False):
    if q is None:
        q = QChunk(v_range=v_range, sep_speed=sep_speed).to(device)
        m, s = buf.obs_stats()
        q.set_norm(m, s)
    qt = QChunk(v_range=q.v_range, sep_speed=q.sep_speed).to(device)
    qt.load_state_dict(q.state_dict())
    for p in qt.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    buf.split(holdout=holdout, seed=seed)
    gH = gamma ** H
    hist = []
    t0 = time.time()
    for it in range(1, steps + 1):
        b = buf.batch(batch, gen=gen, fixed_frac=fixed_frac)
        with torch.no_grad():
            if target_chunk == "pi":
                a = pol(b["nobs"])
                nc = a[:, None, :].expand(-1, H, -1).reshape(batch, -1)
            else:
                nc = b["nchunk"]
            qn = qt(b["nobs"], nc)
            y = {k: b["y"][k] + (1 - b["done"]) * gH * qn[k] for k in q.head_names}
        lg = q.logits(b["obs"], b["chunk"])
        losses = {k: q.heads[k].loss(lg[k], y[k]).mean() for k in q.head_names}
        loss = sum(losses.values())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
        opt.step()
        ema_(qt, q, tau)
        if it % log_every == 0 or it == 1:
            with torch.no_grad():
                qpred = {k: q.heads[k].q(lg[k]) for k in q.head_names}
                td = {k: float((qpred[k] - y[k]).abs().mean()) for k in q.head_names}
            row = dict(step=it, loss=float(loss), gnorm=float(gn), sec=round(time.time() - t0, 1))
            for k in q.head_names:
                row[f"l_{k}"] = float(losses[k])
                row[f"td_{k}"] = td[k]
                row[f"q_{k}"] = float(qpred[k].mean())
            hist.append(row)
            log(f"[q] {it:6d}/{steps} loss {row['loss']:.4f} ("
                + " ".join(f"{k} {row[f'l_{k}']:.4f}" for k in q.head_names) + ") |TD| "
                + " ".join(f"{k} {td[k]:.4f}" for k in q.head_names) + " Q "
                + " ".join(f"{k} {row[f'q_{k}']:+.3f}" for k in q.head_names)
                + f" [{row['sec']:.0f}s]")
    return q, hist


# --------------------------------------------------------------------------- calibration
@torch.no_grad()
def calibrate(q, buf, n=50_000, device="cuda:0", bins=10, seed=0):
    """Predicted Q vs the EXACT Monte-Carlo return on held-out episodes."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 5)
    pool = buf.idx_val
    sel = torch.randint(0, pool.numel(), (n,), device=device, generator=gen)
    e = pool[sel]
    t = torch.randint(0, buf.T, (n,), device=device, generator=gen)
    b = buf.batch(n, e=e, t=t)
    pred = q(b["obs"], b["chunk"])
    mc = buf.mc_return(e, t)
    out = {}
    for k in q.head_names:
        p, g = pred[k].float(), mc[k].float()
        order = torch.argsort(p)
        ps, gs = p[order], g[order]
        chunks = torch.chunk(torch.arange(n, device=device), bins)
        rel = [(float(ps[c].mean()), float(gs[c].mean()), int(c.numel())) for c in chunks]
        out[k] = dict(pred_mean=float(p.mean()), real_mean=float(g.mean()),
                      bias=float((p - g).mean()), mae=float((p - g).abs().mean()),
                      rmse=float(((p - g) ** 2).mean().sqrt()),
                      corr=float(torch.corrcoef(torch.stack([p, g]))[0, 1]),
                      ece=float(np.mean([abs(a - bb) * c for a, bb, c in rel]) / (n / bins)),
                      reliability=rel)
    # success-probability calibration at the FIRST decision (the interpretable number)
    t0 = torch.zeros(pool.numel(), dtype=torch.long, device=device)
    b0 = buf.batch(pool.numel(), e=pool, t=t0)
    p0 = q(b0["obs"], b0["chunk"])["succ"] / (GAMMA ** (buf.T - 1))
    out["succ"]["t0_pred_success"] = float(p0.mean())
    out["succ"]["t0_real_success"] = float(buf.placed[pool].mean())
    return out


def fmt_cal(c):
    s = []
    for k in [k for k in HEADS if k in c]:
        d = c[k]
        s.append(f"[cal] {k:<5} pred {d['pred_mean']:+.4f} real {d['real_mean']:+.4f} "
                 f"bias {d['bias']:+.4f} mae {d['mae']:.4f} rmse {d['rmse']:.4f} corr {d['corr']:.3f}")
    d = c["succ"]
    s.append(f"[cal] success@t0 predicted {d['t0_pred_success']:.2%} vs realised "
             f"{d['t0_real_success']:.2%} (held-out episodes)")
    return "\n".join(s)


def load_buffer(paths, device="cuda:0", log=print, group=0, buf=None):
    buf = QBuffer(device=device) if buf is None else buf
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.pt"))) if os.path.isdir(p) else [p]
    for f in files:
        buf.add_file(f, group=group)
        log(f"[buf] + {os.path.basename(f)} -> {buf.E} episodes, {buf.n_trans:,} transitions")
    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=[os.path.join(DATA_ROOT, "data")])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--target_chunk", default="exec", choices=["exec", "pi"])
    ap.add_argument("--holdout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sep_speed", action="store_true",
                    help="give the speed head its own trunk (keeps succ/time exactly as the "
                         "two-head critic learned them)")
    ap.add_argument("--place", action="store_true",
                    help="use the wider time support of the ep_len 180 place-phase episodes")
    ap.add_argument("--out", default=os.path.join(DATA_ROOT, "q0"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda:0"
    torch.manual_seed(a.seed)
    buf = load_buffer(a.data, device=dev)
    pol = None
    if a.target_chunk == "pi":
        import warp as wp
        wp.init()
        from common import load_pi
        pol = load_pi(dev)
    q, hist = train_q(buf, steps=a.steps, batch=a.batch, lr=a.lr, tau=a.tau, gamma=a.gamma,
                      device=dev, target_chunk=a.target_chunk, pol=pol, seed=a.seed,
                      holdout=a.holdout, v_range=(V_RANGE_PLACE if a.place else V_RANGE),
                      sep_speed=a.sep_speed)
    cal = calibrate(q, buf, device=dev, seed=a.seed)
    print(fmt_cal(cal), flush=True)
    torch.save(dict(q=q.state_dict(), args=vars(a), hist=hist, cal=cal, v_range=q.v_range,
                    sep_speed=q.sep_speed, sources=buf.srcs, episodes=buf.E),
               os.path.join(a.out, "q.pt"))
    with open(os.path.join(a.out, "train.json"), "w") as f:
        json.dump(dict(args=vars(a), hist=hist, cal=cal, episodes=buf.E,
                       transitions=buf.n_trans), f, indent=1)
    print(f"[q] saved {a.out}/q.pt", flush=True)


if __name__ == "__main__":
    main()
