#!/usr/bin/env python3
"""qplan.planner :: score N chunk proposals with Q and execute the softmax-Q weighted chunk (M1).

At every decision (10 Hz):
  1. c_pi = repeat(pi(o_t), H)                       -- the cheap chunk approximation (common.py)
  2. N candidates around it                          -- proposals.propose()
  3. Q_succ, Q_time for all N in ONE batched call    -- critic.QChunk.score()
  4. keep only candidates within 10 % of the best Q_succ
        thr = max - 0.1*|max|   (the plan's "0.9 * max"; written this way so it still means
        "within 10 % of the best" when the best value is negative)
  5. w = softmax(Q_time / lambda) over the survivors, chunk = sum_i w_i c_i
  6. execute the FIRST action of that chunk and re-plan next decision (receding horizon, so the
     deployed 10 Hz loop is unchanged).  --open_loop executes all H actions instead (ablation).

Ablations (`--eval`): pi alone | weighted | best-of-N | N = 8/64 | lambda = 0.02/0.2 | open loop.

    $PY rl/qplan/planner.py --eval --q ~/pnp_rl/qplan/q0/q.pt --nworld 1024
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
from common import (ACT_DIM, DATA_ROOT, H, OBS_DIM, load_pi, make_env, seed_env)  # noqa: E402
from critic import QChunk                                                         # noqa: E402
import proposals as PR                                                            # noqa: E402

NEG = -1e9


class QPlanner:
    """Stateful 10 Hz actor: __call__(obs, t) -> action (N, 7)."""

    def __init__(self, pol, q, n_cand=32, lam=0.05, succ_frac=0.9, sigma_shared=0.15,
                 sigma_step=0.05, rule="weighted", open_loop=False, gen=None, vel_margin=None,
                 vel_mode="cap", w_speed=0.0, speed_margin=None, ema=None, noise=None,
                 fine=False, hash_tab=None):
        """w_speed / speed_margin / ema: the SPEED head (2026-09-21), three separate levers.

        w_speed    score the softmax on Q_time + w_speed * Q_speed instead of Q_time alone.  This
                   is the shape of fix the README's M1 gate asks for: the planner pays for joint
                   speed in the same currency it pays for time, so it can still pick a faster
                   chunk where the drive can absorb it and must not where it cannot.
        speed_margin  a SOFT gate instead (score on Q_time alone): drop candidates whose Q_speed
                   is more than `margin` below the pi chunk's own Q_speed.  Unlike the hard
                   VELOCITY mask (`vel_mode="mask"`, which biases the survivors slow because
                   clamped candidates equal pi exactly where pi saturates), this gate is on the
                   critic's PREDICTED excess, so it drops a fast candidate only where Q believes
                   the drive will actually overshoot.
        ema        after mixing, execute alpha * a_mix + (1 - alpha) * a_prev on the JOINT
                   channels (the suction logit is left alone -- its sign is the command and a
                   blend would latch a decision late).  a_prev is read from the observation
                   (columns 18:24), so the rule is stateless and exports as part of the graph.
        noise      fixed gaussian-family table (proposals.noise_table) -> deterministic planner.

        vel_margin / vel_mode: cap on the commanded joint speed relative to pi's own.

        The commanded delta IS the commanded joint speed (|a| x dq_max x CTRL_HZ = |a| x 30 deg/s),
        so `v_pi = max_{h,j} |c_pi[h,j]|` is pi's own peak for this chunk and the planner is not
        allowed past `vel_margin * v_pi`.

        vel_mode="mask"  drop every candidate above the cap BEFORE the softmax.  Measured: this
            biases the surviving set SLOW -- candidates are clamped to [-1,1], so where pi
            saturates the fast ladder entries are identical to pi and survive, while where pi is
            slow (exactly where a speed-up is safe) they are the ones removed.  The mixture then
            sits below pi and t_goal got WORSE than pi (8.39 s vs 7.77 s, M2 v2 iteration 1).
        vel_mode="cap" (default)  score and mix the candidates unchanged, then rescale the joint
            channels of the EXECUTED chunk so its peak obeys the cap.  Same guarantee on the
            commanded peak, no bias in the selection.
        """
        self.pol, self.q = pol, q
        self.n_cand, self.lam, self.succ_frac = int(n_cand), float(lam), float(succ_frac)
        self.kw = dict(sigma_shared=sigma_shared, sigma_step=sigma_step)
        self.rule, self.open_loop, self.gen = rule, bool(open_loop), gen
        self.vel_margin = None if vel_margin is None else float(vel_margin)
        self.vel_mode = str(vel_mode)
        self.w_speed = float(w_speed)
        self.speed_margin = None if speed_margin is None else float(speed_margin)
        self.ema = None if ema is None else float(ema)
        self.noise = noise
        self.hash_tab = hash_tab
        self.kw["fine"] = bool(fine)
        self.has_speed = "speed" in getattr(q, "head_names", ("succ", "time"))
        if (self.w_speed or self.speed_margin is not None) and not self.has_speed:
            raise ValueError("this critic has no speed head (retrain with qplan/critic.py)")
        vm = "" if self.vel_margin is None else f",vel{vel_mode}<={self.vel_margin}"
        sp = (",fine" if fine else "") + (",hash" if hash_tab is not None else "") + \
             (f",wsp={self.w_speed}" if self.w_speed else "") + \
             (f",spgate={self.speed_margin}" if self.speed_margin is not None else "") + \
             (f",ema={self.ema}" if self.ema is not None else "")
        self.name = f"qplan[{rule},N={n_cand},lam={lam}{vm}{sp}{',open' if open_loop else ''}]"
        self.diag = {}

    def reset(self, n, device):
        self.buf = torch.zeros(n, H, ACT_DIM, device=device)
        self.left = torch.zeros(n, dtype=torch.long, device=device)
        self.diag = dict(n_alive=0.0, w_pi=0.0, w_max=0.0, not_pi=0.0, dq=0.0,
                         vel_drop=0.0, forced_pi=0.0, sp_drop=0.0, calls=0)

    # ------------------------------------------------------------------ plan
    @torch.no_grad()
    def plan(self, obs):
        a_pi = self.pol(obs)
        noise = self.noise
        if self.hash_tab is not None:                 # deterministic, observation-hashed noise
            z = (obs - self.q.obs_mean) / self.q.obs_std
            noise = PR.hash_noise(z, self.hash_tab, self.kw["sigma_shared"], self.kw["sigma_step"])
        cand = PR.propose(a_pi, n_cand=self.n_cand, gen=self.gen, noise=noise,
                          **self.kw)                                           # (N, C, H, 7)
        if self.rule == "pi":
            return cand[:, 0], None
        qs = self.q.score(obs, cand)
        s, tq = qs["succ"], qs["time"]
        vq = qs["speed"] if self.has_speed else None
        smax = s.max(dim=1, keepdim=True).values
        thr = smax - (1.0 - self.succ_frac) * smax.abs()
        alive = s >= thr - 1e-6
        if self.speed_margin is not None:
            # soft speed gate, referenced to the PI CHUNK (candidate 0), not to the best
            # candidate: the planner is allowed to be as rough as pi is, never rougher by more
            # than `margin` of predicted discounted excess.
            sp_ok = vq >= vq[:, :1] - self.speed_margin
            self.diag["sp_drop"] += float((~sp_ok).float().mean())
            alive = alive & sp_ok
        vdrop = 0.0
        if self.vel_margin is not None and self.vel_mode == "mask":
            vpi = cand[:, 0, :, :6].abs().amax(dim=(1, 2))               # (N,)
            vc = cand[:, :, :, :6].abs().amax(dim=(2, 3))                # (N, C)
            vel_ok = vc <= self.vel_margin * vpi[:, None] + 1e-6
            vdrop = float((~vel_ok).float().mean())
            alive = alive & vel_ok
        # never leave a world with nothing to execute: fall back to pi's own chunk
        none_alive = ~alive.any(dim=1)
        if none_alive.any():
            alive = alive.clone()
            alive[none_alive, 0] = True
        self.diag["forced_pi"] += float(none_alive.float().mean())
        self.diag["vel_drop"] += vdrop
        sc = tq if not self.w_speed else tq + self.w_speed * vq
        if self.rule == "best":
            lg = torch.where(alive, sc, torch.full_like(sc, NEG))
            w = torch.zeros_like(sc)
            w.scatter_(1, lg.argmax(1, keepdim=True), 1.0)
        else:
            lg = torch.where(alive, sc / self.lam, torch.full_like(sc, NEG))
            w = torch.softmax(lg, dim=1)
        chunk = (w[:, :, None, None] * cand).sum(1)
        if self.vel_margin is not None and self.vel_mode == "cap":
            vpi = cand[:, 0, :, :6].abs().amax(dim=(1, 2))
            vch = chunk[:, :, :6].abs().amax(dim=(1, 2))
            sc = (self.vel_margin * vpi / vch.clamp_min(1e-6)).clamp(max=1.0)
            self.diag["vel_drop"] += float((sc < 1.0 - 1e-6).float().mean())
            chunk = torch.cat([chunk[..., :6] * sc[:, None, None], chunk[..., 6:]], -1)
        d = self.diag
        d["n_alive"] += float(alive.float().sum(1).mean())
        d["w_pi"] += float(w[:, 0].mean())
        d["w_max"] += float(w.max(1).values.mean())
        d["not_pi"] += float((w.argmax(1) != 0).float().mean())
        d["dq"] += float((chunk[:, 0, :6] - cand[:, 0, 0, :6]).abs().mean())
        d["calls"] += 1
        return chunk, w

    # ------------------------------------------------------------------ act
    def _ema(self, a, obs):
        """alpha * a + (1 - alpha) * a_prev on the joint channels; a_prev = obs[:, 18:24]."""
        if self.ema is None:
            return a
        ap = obs[:, 18:24]
        j = self.ema * a[:, :6] + (1.0 - self.ema) * ap
        return torch.cat([j.clamp(-1, 1), a[:, 6:]], -1)

    def __call__(self, obs, t=0):
        if not self.open_loop:
            chunk, _ = self.plan(obs)
            return self._ema(chunk[:, 0], obs)
        need = self.left == 0
        if need.any():
            chunk, _ = self.plan(obs)
            self.buf = torch.where(need[:, None, None], chunk, self.buf)
            self.left = torch.where(need, torch.full_like(self.left, H), self.left)
        a = self.buf[:, 0]
        self.buf = torch.roll(self.buf, -1, dims=1)
        self.left = self.left - 1
        return a

    def diag_mean(self):
        c = max(1, self.diag.get("calls", 1))
        return {k: (v / c if k != "calls" else v) for k, v in self.diag.items()}


class PiActor:
    """pi alone, optionally with a BLANKET joint-action scale (the "just move 1.3x faster"
    control the plan calls out: it costs 8 pp of success in the twin, which is the point of
    letting the critic pick fast chunks only where they are safe)."""

    def __init__(self, pol, scale=1.0):
        self.pol, self.scale = pol, float(scale)
        self.name = "pi" if scale == 1.0 else f"pi x{scale}"

    def reset(self, n, device):
        pass

    def __call__(self, obs, t=0):
        a = self.pol(obs)
        if self.scale != 1.0:
            a = torch.cat([(self.scale * a[..., :6]).clamp(-1, 1), a[..., 6:]], -1)
        return a


# --------------------------------------------------------------------------- paired eval
@torch.no_grad()
def rollout(env, actor, ep_len, device, seed=0):
    """eval_residual.rollout's metric set (+ the final-distance median), paired by `seed`."""
    seed_env(env, seed, device)
    actor.reset(env.nworld, device)
    info = {}
    for t in range(ep_len):
        a = actor(env.observe(), t)
        _, _, _, info = env.step(a.clamp(-1, 1))
    g = lambda k: info[k].float().cpu().numpy()                       # noqa: E731
    pq = np.degrees(g("peak_qd"))
    ts, tg, fd = g("t_seal"), g("t_goal"), g("final_d")
    extra = {}
    if "t_placed" in info:
        tp, pe = g("t_placed"), g("place_err")
        extra = dict(t_placed=float(tp.mean()), place_err_med=float(np.median(pe)),
                     t_placed_ok=float(tp[tp < ep_len].mean()) if (tp < ep_len).any() else float("nan"))
    return dict(**extra, success=float(g("placed").mean()), seal=float(g("ever_sealed").mean()),
                t_seal=float(ts.mean()), t_goal=float(tg.mean()),
                qd_med=float(np.median(pq)), qd_p90=float(np.percentile(pq, 90)),
                qd_max=float(pq.max()), final_d_med=float(np.median(fd)),
                final_d_mean=float(fd.mean()),
                t_seal_ok=float(ts[ts < ep_len].mean()) if (ts < ep_len).any() else float("nan"),
                t_goal_ok=float(tg[tg < ep_len].mean()) if (tg < ep_len).any() else float("nan"))


def fmt(name, r, hz=10.0):
    pl = ""
    if "t_placed" in r:
        pl = (f"t_placed {r['t_placed'] / hz:.2f} s (placed-only {r['t_placed_ok'] / hz:.2f} s)  "
              f"place_err med {r['place_err_med'] * 100:.1f} cm  ")
    return (f"[eval] {name:<34} success {r['success']:.2%}  seal {r['seal']:.2%}  " + pl +
            f"t_seal {r['t_seal']:.1f} dec ({r['t_seal'] / hz:.2f} s, sealed-only {r['t_seal_ok'] / hz:.2f} s)  "
            f"t_goal {r['t_goal']:.1f} dec ({r['t_goal'] / hz:.2f} s, reached-only {r['t_goal_ok'] / hz:.2f} s)  "
            f"final_d med {r['final_d_med'] * 100:.1f} cm  "
            f"peak|qd| med/p90/max {r['qd_med']:.1f}/{r['qd_p90']:.1f}/{r['qd_max']:.1f} deg/s")


def load_q(path, device="cuda:0"):
    ck = torch.load(path, map_location=device, weights_only=False)
    sep = ck.get("sep_speed", any(k.startswith("trunk_v.") for k in ck["q"]))
    # the head SET comes from the checkpoint, never from the current default: critics saved
    # before the speed head existed have two heads (and some have no `v_range` at all, so read
    # the supports straight off the stored bin edges).
    vr = ck.get("v_range") or {k.split(".")[1]: (float(v[0]), float(v[-1]))
                               for k, v in ck["q"].items() if k.endswith(".edges")}
    q = QChunk(v_range=vr, sep_speed=sep).to(device)
    q.load_state_dict(ck["q"])
    q.eval()
    for p in q.parameters():
        p.requires_grad_(False)
    return q, ck


SWEEP = [(32, 0.05, 0.95), (32, 0.05, 0.97), (32, 0.02, 0.95), (32, 0.20, 0.97),
         (16, 0.05, 0.90), (32, 0.05, 0.80)]


def build_variants(pol, q, dev, seed=0, which="full", vel_margin=None, a=None):
    def gen():
        g = torch.Generator(device=dev)
        g.manual_seed(seed + 31337)
        return g
    V = [("pi alone", PiActor(pol))]
    if which == "none":
        return V
    # ---- speed-head groups (2026-09-21) ---------------------------------
    if which in ("speed", "spgate", "smooth", "final", "combo"):
        n, lam, sf = a.n_cand, a.lam, a.succ_frac
        nz = PR.noise_table(n, gen=gen(), device=dev) if a.fixed_noise else None
        ht = None
        if a.hash_tab:                       # the EXACT tables baked into an exported graph
            ht = {k: (v.to(dev) if torch.is_tensor(v) else v)
                  for k, v in torch.load(a.hash_tab, map_location=dev, weights_only=False).items()}
        elif a.hash:
            ht = PR.hash_tables(n, gen=gen(), device=dev, gain=a.hash_gain)
        mk = lambda **kw: QPlanner(pol, q, n, lam, succ_frac=sf, gen=gen(), noise=nz,  # noqa: E731
                                   fine=bool(a.fine), hash_tab=ht,
                                   sigma_shared=a.sigma_shared,
                                   sigma_step=a.sigma_step, **kw)
        if which == "speed":
            V.append((f"planner N={n} lam={lam} (no speed term)", mk()))
            for w in a.w_speed:
                V.append((f"planner w_speed={w}", mk(w_speed=w)))
        elif which == "combo":
            # cross the two speed levers with the EMA: w_speed = 0 means "no speed term",
            # sp_margin < 0 means "no speed gate"
            for w in a.w_speed:
                for m in a.sp_margin:
                    V.append((f"w_speed={w} gate={m} ema={a.ema}",
                              mk(w_speed=w, speed_margin=(None if m < 0 else m),
                                 ema=(a.ema or None))))
        elif which == "spgate":
            V.append((f"planner N={n} lam={lam} (no speed term)", mk()))
            for m in a.sp_margin:
                V.append((f"planner Q_time, speed gate margin={m}", mk(speed_margin=m)))
        elif which == "smooth":
            kw = {}
            if a.w_speed_pick:
                kw["w_speed"] = a.w_speed_pick
            if a.sp_margin_pick is not None:
                kw["speed_margin"] = a.sp_margin_pick
            V.append(("(a) weighted mixture", mk(**kw)))
            V.append(("(b) best candidate after the gates", mk(rule="best", **kw)))
            V.append((f"(c) weighted + EMA alpha={a.ema}", mk(ema=a.ema, **kw)))
        else:                                    # final: the chosen setting, both rules
            kw = {}
            if a.w_speed_pick:
                kw["w_speed"] = a.w_speed_pick
            if a.sp_margin_pick is not None:
                kw["speed_margin"] = a.sp_margin_pick
            if a.ema:
                kw["ema"] = a.ema
            V.append(("qplan_v1 (chosen)", mk(**kw)))
        return V
    if which == "best":
        V.append(("planner N=16 lam=0.1", QPlanner(pol, q, 16, 0.1, gen=gen(), vel_margin=vel_margin)))
        V.append(("best-of-N N=16", QPlanner(pol, q, 16, 0.1, rule="best", gen=gen(),
                                             vel_margin=vel_margin)))
        V.append(("planner N=16 lam=0.1 vel<=1.2", QPlanner(pol, q, 16, 0.1, gen=gen(),
                                                            vel_margin=1.2)))
        return V
    if which == "velcap":
        for vmg, vmd in [(1.05, "cap"), (1.15, "cap"), (1.3, "cap"), (None, None)]:
            V.append((f"planner N=16 lam=0.1 vel={vmd}<={vmg}",
                      QPlanner(pol, q, 16, 0.1, gen=gen(), vel_margin=vmg,
                               vel_mode=vmd or "cap")))
        return V
    if which == "pareto":
        for n, lam, sf in [(16, 0.10, 0.90), (16, 0.20, 0.90), (16, 0.40, 0.90), (16, 1.00, 0.90)]:
            V.append((f"planner N={n} lam={lam} succ_frac={sf}",
                      QPlanner(pol, q, n, lam, succ_frac=sf, gen=gen(), vel_margin=vel_margin)))
        return V
    if which == "confirm":
        for n, lam, sf in [(32, 0.05, 0.90), (16, 0.05, 0.90), (32, 0.05, 0.80), (16, 0.05, 0.80)]:
            V.append((f"planner N={n} lam={lam} succ_frac={sf}",
                      QPlanner(pol, q, n, lam, succ_frac=sf, gen=gen(), vel_margin=vel_margin)))
        return V
    if which == "sweep":
        for n, lam, sf in SWEEP:
            V.append((f"planner N={n} lam={lam} succ_frac={sf}",
                      QPlanner(pol, q, n, lam, succ_frac=sf, gen=gen(), vel_margin=vel_margin)))
        return V
    V += [("pi blanket x1.3 (control)", PiActor(pol, 1.3))]
    V += [("planner weighted N=32 lam=0.05", QPlanner(pol, q, 32, 0.05, gen=gen(), vel_margin=vel_margin)),
          ("best-of-N N=32", QPlanner(pol, q, 32, 0.05, rule="best", gen=gen(), vel_margin=vel_margin))]
    if which == "full":
        V += [("planner N=8 lam=0.05", QPlanner(pol, q, 8, 0.05, gen=gen(), vel_margin=vel_margin)),
              ("planner N=64 lam=0.05", QPlanner(pol, q, 64, 0.05, gen=gen(), vel_margin=vel_margin)),
              ("planner N=32 lam=0.02", QPlanner(pol, q, 32, 0.02, gen=gen(), vel_margin=vel_margin)),
              ("planner N=32 lam=0.2", QPlanner(pol, q, 32, 0.20, gen=gen(), vel_margin=vel_margin)),
              ("planner N=32 lam=0.01", QPlanner(pol, q, 32, 0.01, gen=gen(), vel_margin=vel_margin)),
              ("planner N=32 open-loop", QPlanner(pol, q, 32, 0.05, open_loop=True, gen=gen(), vel_margin=vel_margin))]
    return V


def gate_m1(base, plan):
    d_succ = plan["success"] - base["success"]
    d_goal = (base["t_goal"] - plan["t_goal"]) / 10.0
    qd_ok = plan["qd_p90"] <= base["qd_p90"] + 2.0
    return dict(d_success_pp=100 * d_succ, d_t_goal_s=d_goal, qd_p90=plan["qd_p90"],
                qd_budget=base["qd_p90"] + 2.0,
                passed=bool(d_succ >= 0.01 and d_goal >= 0.3 and qd_ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--q", default=os.path.join(DATA_ROOT, "q0", "q.pt"))
    ap.add_argument("--nworld", type=int, default=1024)
    ap.add_argument("--ep_len", type=int, default=150)
    ap.add_argument("--dq_max", type=float, default=3.0)
    ap.add_argument("--obj_err", type=int, default=0)
    ap.add_argument("--place_phase", type=int, default=0)
    ap.add_argument("--no_dr", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vel_margin", type=float, default=None,
                    help="hard velocity mask: drop chunks whose peak commanded joint delta "
                         "exceeds this multiple of pi's own (e.g. 1.2)")
    ap.add_argument("--which", default="full",
                    choices=["full", "core", "none", "sweep", "confirm", "pareto", "velcap",
                             "best", "speed", "spgate", "smooth", "final", "combo"])
    ap.add_argument("--n_cand", type=int, default=16)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--succ_frac", type=float, default=0.9)
    ap.add_argument("--w_speed", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--sp_margin", type=float, nargs="+", default=[0.05, 0.15, 0.4])
    ap.add_argument("--w_speed_pick", type=float, default=0.0)
    ap.add_argument("--sp_margin_pick", type=float, default=None)
    ap.add_argument("--ema", type=float, default=0.7)
    ap.add_argument("--hash", type=int, default=0,
                    help="1 = gaussian family from sin(W.obs) instead of an RNG (deterministic, "
                         "and what the exported graph computes)")
    ap.add_argument("--hash_gain", type=float, default=5.0)
    ap.add_argument("--hash_tab", default=None,
                    help="path to <stem>_hash.pt written by qplan/export.py (evaluate exactly "
                         "the planner that was exported)")
    ap.add_argument("--sigma_shared", type=float, default=0.15)
    ap.add_argument("--sigma_step", type=float, default=0.05)
    ap.add_argument("--fine", type=int, default=0,
                    help="1 = replace the gaussian family with extra ladder scales (a fully "
                         "DETERMINISTIC proposal set -- what the exported graph builds)")
    ap.add_argument("--fixed_noise", type=int, default=0,
                    help="1 = one FIXED gaussian-noise table for the whole run (what the exported "
                         "TensorRT graph bakes in)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import warp as wp
    wp.init()
    dev = "cuda:0"
    torch.manual_seed(a.seed)
    env = make_env(a.nworld, device=dev, seed=a.seed, dr=not a.no_dr, obj_err=bool(a.obj_err),
                   dq_max_deg=a.dq_max, ep_len=a.ep_len, place_phase=bool(a.place_phase))
    pol = load_pi(dev)
    q, ck = load_q(a.q, dev)
    print(f"[plan] q={a.q} trained on {ck.get('episodes')} episodes; nworld {a.nworld} "
          f"ep_len {a.ep_len} dq_max {a.dq_max} dr={not a.no_dr} obj_err={bool(a.obj_err)}", flush=True)
    rows = {}
    for name, actor in build_variants(pol, q, dev, seed=a.seed, which=a.which,
                                      vel_margin=a.vel_margin, a=a):
        t0 = time.time()
        r = rollout(env, actor, a.ep_len, dev, seed=a.seed)
        r["seconds"] = round(time.time() - t0, 1)
        if isinstance(actor, QPlanner):
            r["diag"] = actor.diag_mean()
        rows[name] = r
        print(fmt(name, r) + f"  [{r['seconds']:.0f}s]", flush=True)
        if isinstance(actor, QPlanner):
            d = r["diag"]
            print(f"       survivors {d['n_alive']:.1f}/{actor.n_cand}  w(pi) {d['w_pi']:.3f}  "
                  f"w_max {d['w_max']:.3f}  argmax!=pi {d['not_pi']:.1%}  "
                  f"|dchunk| {d['dq']:.4f}  vel_drop {d['vel_drop']:.1%}  "
                  f"sp_drop {d.get('sp_drop', 0.0):.1%}  "
                  f"forced_pi {d['forced_pi']:.1%}", flush=True)
    base = rows.get("pi alone")
    if base is not None and a.which in ("speed", "spgate", "smooth", "final", "combo", "pareto"):
        print(f"\n[front] {'variant':<42} {'success':>8} {'t_goal':>8} {'qd p90':>7} "
              f"{'d_succ':>7} {'d_time':>7} {'gate':>5}")
        for nm, r in rows.items():
            if nm.startswith("_"):
                continue
            g = gate_m1(base, r)
            print(f"[front] {nm:<42} {r['success']:>7.2%} {r['t_goal'] / 10:>7.2f}s "
                  f"{r['qd_p90']:>7.1f} {g['d_success_pp']:>+6.2f} {g['d_t_goal_s']:>+6.2f} "
                  f"{'PASS' if g['passed'] else 'fail':>5}", flush=True)
            rows[nm]["gate"] = g
    if "planner weighted N=32 lam=0.05" in rows:
        g = gate_m1(rows["pi alone"], rows["planner weighted N=32 lam=0.05"])
        print(f"[gate M1] d_success {g['d_success_pp']:+.2f} pp (need >= +1.0)  "
              f"d_t_goal {g['d_t_goal_s']:+.2f} s (need >= +0.30)  "
              f"peak qd p90 {g['qd_p90']:.1f} <= {g['qd_budget']:.1f}  -> "
              f"{'PASS' if g['passed'] else 'FAIL'}", flush=True)
        rows["_gate"] = g
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(dict(args=vars(a), rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
