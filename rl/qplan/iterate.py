#!/usr/bin/env python3
"""qplan.iterate :: the twin self-improvement loop (M2).

K iterations of
    deploy   the planner in --nworld DR worlds for --ep_len decisions
    append   ALL episodes (successes AND failures) to the replay buffer
    retrain  Q for --q_steps steps on the WHOLE buffer with the EMA target (pi never moves)
    eval     paired deterministic 1024-episode comparison pi vs planner
    log      ~/pnp_rl/qplan/iterN/{deploy.pt, iter.json}
and a small-multiples plot of success / t_goal / peak |qd| vs iteration.

    $PY rl/qplan/iterate.py --iters 10 --q ~/pnp_rl/qplan/q0/q.pt --data ~/pnp_rl/qplan/data
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
from common import DATA_ROOT, GAMMA, load_pi, make_env                      # noqa: E402
import collect as CO                                                        # noqa: E402
import critic as CR                                                         # noqa: E402
import planner as PL                                                        # noqa: E402

# Gates.  The ABSOLUTE peak-speed gate of 36 deg/s is the Pro 630's firmware following-error
# ceiling, but the twin over-reads peaks by ~25 % versus the robot (FINDINGS 22: pi alone reads
# 43.6 deg/s p90 here and 22-34 deg/s on the arm), so the twin-side criterion is RELATIVE:
# the planner may not exceed pi's own p90 by more than 2 deg/s.  Both are reported.
GATE = dict(success=0.99, t_goal_s=6.5, qd_p90=36.0, qd_rel=2.0)


class MixedDeploy:
    """Deploy the planner on (1 - frac) of the worlds and a blanket-scaled pi on the rest.

    Without this the loop is self-referential on the SPEED axis: the planner's own episodes all
    run at the speed the planner already chose, so each retrain sees less and less contrast
    between "same trajectory, faster" and "same trajectory, slower" and Q_time decays back to a
    copy of Q_succ -- exactly the failure `diag.py` found before the `scale` batches existed.
    `--explore_frac` keeps that axis alive, the way DAgger keeps the teacher in the loop.
    """

    def __init__(self, planner, pol, env, frac=0.25, lo=0.6, hi=1.6, gen=None):
        self.p, self.pol, self.env = planner, pol, env
        self.frac, self.lo, self.hi, self.gen = float(frac), float(lo), float(hi), gen

    def _u(self, n, dev):
        u = torch.rand(n, device=dev, generator=self.gen) if self.gen is not None \
            else torch.rand(n, device=dev)
        return self.lo + (self.hi - self.lo) * u

    def reset(self, n, device):
        self.p.reset(n, device)
        r = torch.rand(n, device=device, generator=self.gen) if self.gen is not None \
            else torch.rand(n, device=device)
        self.mask = r < self.frac
        self.s_pre, self.s_post = self._u(n, device), self._u(n, device)

    def __call__(self, obs, t):
        a = self.p(obs, t)
        b = self.pol(obs)
        s = torch.where(self.env.ever_sealed, self.s_post, self.s_pre)[:, None]
        b = torch.cat([(s * b[:, :6]).clamp(-1, 1), b[:, 6:]], -1)
        return torch.where(self.mask[:, None], b, a)

    def diag_mean(self):
        return self.p.diag_mean()


def gate_m2(r, base=None):
    budget = None if base is None else base["qd_p90"] + GATE["qd_rel"]
    ok = dict(success=r["success"] >= GATE["success"],
              t_goal=r["t_goal"] / 10.0 <= GATE["t_goal_s"],
              qd_abs=r["qd_p90"] <= GATE["qd_p90"],
              qd_rel=(budget is not None and r["qd_p90"] <= budget))
    return dict(**ok, qd_budget=budget, qd_p90=r["qd_p90"],
                passed=bool(ok["success"] and ok["t_goal"] and ok["qd_rel"]),
                passed_abs=bool(ok["success"] and ok["t_goal"] and ok["qd_abs"]))


def score_iter(r, base):
    """Ranking used to keep the best critic: success first (the M2 gate's headline), time as the
    tie-break, and a hard veto on exceeding pi's p90 + 2 deg/s."""
    if r["qd_p90"] > base["qd_p90"] + GATE["qd_rel"] + 1e-9:
        return -1e9 + r["success"]
    return r["success"] * 100.0 - 0.02 * (r["t_goal"] / 10.0)


# --------------------------------------------------------------------------- plot
def plot_curve(rows, out_png, base=None):
    """Small multiples: three measures of different scale never share one y-axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e2e1dc"
    SER = ["#2a78d6", "#eb6834", "#1baf7a"]
    it = [r["iter"] for r in rows]
    panels = [
        ("Success rate", [100 * r["eval"]["planner"]["success"] for r in rows], "%",
         100 * GATE["success"], None if base is None else 100 * base["success"], "{:.1f}%"),
        ("Time to goal", [r["eval"]["planner"]["t_goal"] / 10 for r in rows], "s",
         GATE["t_goal_s"], None if base is None else base["t_goal"] / 10, "{:.2f} s"),
        ("Peak |qd| p90", [r["eval"]["planner"]["qd_p90"] for r in rows], "deg/s",
         (GATE["qd_p90"] if base is None else base["qd_p90"] + GATE["qd_rel"]),
         None if base is None else base["qd_p90"], "{:.1f}"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.2), sharex=True,
                             gridspec_kw=dict(hspace=0.32))
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (title, ys, unit, gate, bval, f), c in zip(axes, panels, SER):
        ax.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=9, length=0)
        ax.plot(it, ys, color=c, lw=2, marker="o", ms=5, zorder=3,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5)
        ax.axhline(gate, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.annotate(f"gate {f.format(gate)}", (it[0], gate), color=INK2, fontsize=8.5,
                    xytext=(4, 3), textcoords="offset points", va="bottom")
        if bval is not None:
            ax.axhline(bval, color="#a8a79f", lw=1.2, zorder=1)
            ax.annotate(f"pi {f.format(bval)}", (it[0], bval), color="#7a7973", fontsize=8.5,
                        xytext=(2, -11), textcoords="offset points")
        ax.annotate(f.format(ys[-1]), (it[-1], ys[-1]), color=c, fontsize=9.5, weight="bold",
                    xytext=(6, 0), textcoords="offset points", va="center")
        ax.set_title(f"{title}  ({unit})", color=INK, fontsize=11, loc="left", pad=8)
        lo = min(ys + [gate] + ([bval] if bval is not None else []))
        hi = max(ys + [gate] + ([bval] if bval is not None else []))
        pad = max(1e-6, 0.12 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(min(it) - 0.3, max(it) + 0.9)
    axes[-1].set_xlabel("self-improvement iteration", color=INK2, fontsize=9.5)
    axes[-1].set_xticks(it)
    fig.suptitle("Q-Planning twin iterations - planner vs the frozen policy",
                 color=INK, fontsize=13, x=0.055, ha="left", y=0.975)
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--nworld", type=int, default=4096)
    ap.add_argument("--eval_nworld", type=int, default=1024)
    ap.add_argument("--ep_len", type=int, default=150)
    ap.add_argument("--q", default=os.path.join(DATA_ROOT, "q0", "q.pt"))
    ap.add_argument("--data", nargs="+", default=[os.path.join(DATA_ROOT, "data")])
    ap.add_argument("--q_steps", type=int, default=2500)
    ap.add_argument("--fixed_frac", type=float, default=0.3,
                    help="minimum share of every gradient batch drawn from the FIXED offline pool")
    ap.add_argument("--fixed_data", nargs="*", default=[],
                    help="extra offline shards for the fixed pool (e.g. the object-error data)")
    ap.add_argument("--fresh", action="store_true",
                    help="retrain Q FROM SCRATCH for --q_steps every iteration instead of warm-"
                         "starting.  steps_curve.py shows planning quality is non-monotone in the "
                         "number of TD steps (peak ~20k on this buffer, 95 %% at 60k), so a warm "
                         "start accumulates steps past the optimum and the loop decays.")
    ap.add_argument("--patience", type=int, default=2,
                    help="stop after this many consecutive declines of the paired eval score")
    ap.add_argument("--vel_margin", type=float, default=1.05,
                    help="cap on the executed chunk's peak commanded joint speed, as a multiple "
                         "of pi's own peak for that chunk; 0 = off")
    ap.add_argument("--vel_mode", default="cap", choices=["cap", "mask"])
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--target_chunk", default="exec", choices=["exec", "pi"])
    ap.add_argument("--n_cand", type=int, default=32)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--dq_max", type=float, default=3.0)
    ap.add_argument("--obj_err", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=DATA_ROOT)
    ap.add_argument("--start_iter", type=int, default=1)
    ap.add_argument("--explore_frac", type=float, default=0.0,
                    help="fraction of deployment worlds that run blanket-scaled pi instead of the planner")
    a = ap.parse_args()

    import warp as wp
    wp.init()
    dev = "cuda:0"
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    env = make_env(a.nworld, device=dev, seed=a.seed, dr=True, obj_err=bool(a.obj_err),
                   dq_max_deg=a.dq_max, ep_len=a.ep_len)
    eenv = make_env(a.eval_nworld, device=dev, seed=a.seed, dr=True, obj_err=bool(a.obj_err),
                    dq_max_deg=a.dq_max, ep_len=a.ep_len)
    pol = load_pi(dev)
    q, _ck = PL.load_q(a.q, dev)
    for p in q.parameters():
        p.requires_grad_(True)
    buf = CR.load_buffer(a.data, device=dev, group=0)
    if a.fixed_data:
        buf = CR.load_buffer(a.fixed_data, device=dev, group=0, buf=buf)
    print(f"[iter] fixed pool: {buf.E} episodes / {buf.n_trans:,} transitions | "
          f"q_steps {a.q_steps} fresh={a.fresh} fixed_frac {a.fixed_frac} "
          f"vel {a.vel_mode} {a.vel_margin} "
          f"patience {a.patience}", flush=True)
    vm = None if not a.vel_margin else float(a.vel_margin)

    def mk_planner(seed):
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        return PL.QPlanner(pol, q, a.n_cand, a.lam, gen=g, vel_margin=vm, vel_mode=a.vel_mode)

    base = PL.rollout(eenv, PL.PiActor(pol), a.ep_len, dev, seed=a.seed)
    print(PL.fmt("pi alone (reference)", base), flush=True)
    rows = []
    best = dict(score=-1e18, iter=0)
    declines = 0
    hist_path = os.path.join(a.out, "iterations.json")
    if os.path.exists(hist_path) and a.start_iter > 1:
        rows = json.load(open(hist_path))["rows"]
    for k in range(a.start_iter, a.iters + 1):
        d = os.path.join(a.out, f"iter{k}")
        os.makedirs(d, exist_ok=True)
        t0 = time.time()
        # ---- deploy -----------------------------------------------------
        actor = mk_planner(a.seed + 90000 + k)
        if a.explore_frac > 0:
            eg = torch.Generator(device=dev)
            eg.manual_seed(a.seed + 70000 + k)
            actor = MixedDeploy(actor, pol, env, frac=a.explore_frac, gen=eg)
        out = CO.rollout_traj(env, actor, a.ep_len, dev, seed=a.seed + 50000 + k)
        meta = CO.shard_meta(out, a.ep_len, dict(mode="planner", iter=k, n_cand=a.n_cand,
                                                 lam=a.lam, obj_err=bool(a.obj_err),
                                                 explore_frac=a.explore_frac,
                                                 diag=actor.diag_mean()))
        CO.save_shard(out, os.path.join(d, "deploy.pt"), meta)
        t_dep = time.time() - t0
        print(f"[iter {k}] deploy {meta['transitions']:,} transitions in {t_dep:.0f}s | "
              f"success {meta['success']:.2%} seal {meta['seal']:.2%} "
              f"t_goal {meta['t_goal']:.1f} dec", flush=True)
        # ---- append + retrain -------------------------------------------
        buf.add(out, f"iter{k}", group=1)
        t1 = time.time()
        q, qhist = CR.train_q(buf, q=(None if a.fresh else q), steps=a.q_steps,
                              batch=a.batch, lr=a.lr, tau=a.tau,
                              device=dev, target_chunk=a.target_chunk, pol=pol,
                              seed=a.seed + k, log_every=max(1, a.q_steps // 4),
                              fixed_frac=a.fixed_frac)
        cal = CR.calibrate(q, buf, device=dev, seed=a.seed + k)
        print(CR.fmt_cal(cal), flush=True)
        t_tr = time.time() - t1
        # ---- paired eval -------------------------------------------------
        t2 = time.time()
        ev_actor = mk_planner(a.seed + 31337)
        ev = PL.rollout(eenv, ev_actor, a.ep_len, dev, seed=a.seed)
        print(PL.fmt(f"iter {k} planner", ev), flush=True)
        g = gate_m2(ev, base)
        sc = score_iter(ev, base)
        print(f"[gate M2] iter {k}: success {ev['success']:.2%} (>=99%) {'OK' if g['success'] else 'no'} | "
              f"t_goal {ev['t_goal'] / 10:.2f}s (<=6.5) {'OK' if g['t_goal'] else 'no'} | "
              f"qd p90 {ev['qd_p90']:.1f} (rel <={g['qd_budget']:.1f} {'OK' if g['qd_rel'] else 'no'};"
              f" abs <=36 {'OK' if g['qd_abs'] else 'no'}) -> "
              f"{'PASS' if g['passed'] else 'FAIL'} (rel gate)", flush=True)
        row = dict(iter=k, score=sc, deploy=meta, eval=dict(planner=ev, pi=base), gate=g,
                   diag=ev_actor.diag_mean(), cal=cal, q_hist=qhist[-1] if qhist else None,
                   buffer=dict(episodes=buf.E, transitions=buf.n_trans),
                   seconds=dict(deploy=round(t_dep), train=round(t_tr),
                                eval=round(time.time() - t2)))
        rows.append(row)
        with open(os.path.join(d, "iter.json"), "w") as f:
            json.dump(row, f, indent=1)
        torch.save(dict(q=q.state_dict(), iter=k, args=vars(a)), os.path.join(d, "q.pt"))
        if sc > best["score"]:
            best = dict(score=sc, iter=k, eval=ev, gate=g)
            torch.save(dict(q=q.state_dict(), iter=k, args=vars(a), eval=ev),
                       os.path.join(a.out, "q_best.pt"))
            declines = 0
            print(f"[iter {k}] new best (score {sc:.3f}) -> {a.out}/q_best.pt", flush=True)
        else:
            declines += 1
            print(f"[iter {k}] decline {declines}/{a.patience} (score {sc:.3f} "
                  f"vs best {best['score']:.3f} @ iter {best['iter']})", flush=True)
        with open(hist_path, "w") as f:
            json.dump(dict(args=vars(a), base=base, rows=rows, best=best), f, indent=1)
        png = plot_curve(rows, os.path.join(a.out, "iterations.png"), base=base)
        print(f"[iter {k}] done in {time.time() - t0:.0f}s -> {d} | plot {png}", flush=True)
        if declines >= a.patience:
            print(f"[iter] early stop: {declines} consecutive declines; best = iter "
                  f"{best['iter']}", flush=True)
            break
    b = best.get("eval")
    if b:
        print(PL.fmt(f"BEST iter {best['iter']}", b), flush=True)
        print(f"[gate M2] BEST iter {best['iter']}: vs pi  success "
              f"{100 * (b['success'] - base['success']):+.2f} pp  t_goal "
              f"{(base['t_goal'] - b['t_goal']) / 10:+.2f} s  qd p90 {b['qd_p90']:.1f} vs budget "
              f"{base['qd_p90'] + GATE['qd_rel']:.1f}", flush=True)
    print("[iter] finished", flush=True)


if __name__ == "__main__":
    main()
