#!/usr/bin/env python3
"""qplan.diag :: why does the planner not move t_goal?

Rolls pi and, at every decision, scores the full candidate set, then reports PER FAMILY
(nominal / gaussian / scale 0.7,1.2,1.4 / suction flip / safety) the mean Q_succ, mean Q_time,
the survival rate under the Q_succ filter and the planner weight -- split by episode phase
(before the seal / after the seal), because the two phases want opposite things.

It also scores a pure SCALE LADDER (s * pi's chunk for s in --ladder) so the two questions
    "does Q_time believe a faster chunk finishes earlier?"  (monotone in s?)
    "does Q_succ let it through?"                            (survival vs s)
are answered separately.

    $PY rl/qplan/diag.py --q ~/pnp_rl/qplan/q0/q.pt --nworld 512
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ACT_DIM, DATA_ROOT, H, load_pi, make_env, seed_env   # noqa: E402
from planner import load_q                                              # noqa: E402
import proposals as PR                                                  # noqa: E402


def families(n_cand=32, ladder=True):
    """Index -> family label, mirroring proposals.propose()."""
    n_g, n_s, n_safe = PR.family_sizes(n_cand)
    lab = ["nominal"] + ["gauss"] * (n_g - 1)
    if ladder:
        lab += [f"lad{sc}" for sc in PR.LADDER]
        n_noisy = max(1, (n_s - len(PR.LADDER)) // 3)
        for sc in (0.7, 1.2, 1.4):
            lab += [f"gx{sc}"] * n_noisy
    else:
        for sc in (0.7, 1.2, 1.4):
            lab += [f"x{sc}"] * max(1, n_s // 3)
    lab = lab[:n_g + n_s]
    for k in range(max(0, n_g + n_s - 2), n_g + n_s):
        lab[k] = "suck_flip"
    lab += ["safety"] * n_safe
    return lab[:n_cand]


def ladder_chunks(a_pi, scales):
    c = a_pi[:, None, None, :].expand(-1, len(scales), H, ACT_DIM).clone()
    for i, s in enumerate(scales):
        c[:, i, :, :6] = c[:, i, :, :6] * s
    return c.clamp(-1, 1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default=os.path.join(DATA_ROOT, "q0", "q.pt"))
    ap.add_argument("--nworld", type=int, default=512)
    ap.add_argument("--ep_len", type=int, default=150)
    ap.add_argument("--dq_max", type=float, default=3.0)
    ap.add_argument("--succ_frac", type=float, default=0.9)
    ap.add_argument("--ladder", type=float, nargs="+",
                    default=[0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.7])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(DATA_ROOT, "diag.json"))
    a = ap.parse_args()
    import warp as wp
    wp.init()
    dev = "cuda:0"
    env = make_env(a.nworld, device=dev, seed=a.seed, dr=True, obj_err=False,
                   dq_max_deg=a.dq_max, ep_len=a.ep_len)
    pol = load_pi(dev)
    q, _ = load_q(a.q, dev)
    g = torch.Generator(device=dev)
    g.manual_seed(a.seed + 31337)
    lab = families(32)
    fams = sorted(set(lab))
    fidx = {f: torch.tensor([i for i, l in enumerate(lab) if l == f], device=dev) for f in fams}

    acc = {ph: {f: dict(qs=0.0, qt=0.0, alive=0.0, w=0.0, n=0) for f in fams}
           for ph in ("pre", "post")}
    lad = {ph: dict(qs=np.zeros(len(a.ladder)), qt=np.zeros(len(a.ladder)),
                    alive=np.zeros(len(a.ladder)), n=0) for ph in ("pre", "post")}
    seed_env(env, a.seed, dev)
    for t in range(a.ep_len):
        obs = env.observe()
        a_pi = pol(obs)
        cand = PR.propose(a_pi, n_cand=32, gen=g)
        qq = q.score(obs, cand)
        s, tq = qq["succ"], qq["time"]
        smax = s.max(1, keepdim=True).values
        alive = s >= smax - (1 - a.succ_frac) * smax.abs() - 1e-6
        w = torch.softmax(torch.where(alive, tq / 0.05, torch.full_like(tq, -1e9)), 1)
        post = env.ever_sealed
        lc = ladder_chunks(a_pi, a.ladder)
        lq = q.score(obs, lc)
        ls, lt = lq["succ"], lq["time"]
        lalive = ls >= ls.max(1, keepdim=True).values - (1 - a.succ_frac) * ls.max(1, keepdim=True).values.abs() - 1e-6
        for ph, m in (("pre", ~post), ("post", post)):
            if not m.any():
                continue
            for f in fams:
                i = fidx[f]
                d = acc[ph][f]
                d["qs"] += float(s[m][:, i].mean()) * int(m.sum())
                d["qt"] += float(tq[m][:, i].mean()) * int(m.sum())
                d["alive"] += float(alive[m][:, i].float().mean()) * int(m.sum())
                d["w"] += float(w[m][:, i].sum(1).mean()) * int(m.sum())
                d["n"] += int(m.sum())
            L = lad[ph]
            L["qs"] += ls[m].mean(0).cpu().numpy() * int(m.sum())
            L["qt"] += lt[m].mean(0).cpu().numpy() * int(m.sum())
            L["alive"] += lalive[m].float().mean(0).cpu().numpy() * int(m.sum())
            L["n"] += int(m.sum())
        env.step(a_pi)

    out = {}
    for ph in ("pre", "post"):
        print(f"\n=== {ph}-seal ===")
        print(f"{'family':<10} {'Q_succ':>8} {'Q_time':>8} {'survive':>8} {'weight':>8}")
        out[ph] = {}
        for f in fams:
            d = acc[ph][f]
            n = max(1, d["n"])
            r = dict(q_succ=d["qs"] / n, q_time=d["qt"] / n, survive=d["alive"] / n, weight=d["w"] / n)
            out[ph][f] = r
            print(f"{f:<10} {r['q_succ']:>8.4f} {r['q_time']:>8.4f} {r['survive']:>8.1%} {r['weight']:>8.1%}")
        L = lad[ph]
        n = max(1, L["n"])
        print(f"  scale ladder: {'s':>6} {'Q_succ':>9} {'Q_time':>9} {'survive':>8}")
        out[ph]["ladder"] = []
        for i, s_ in enumerate(a.ladder):
            row = dict(scale=s_, q_succ=float(L["qs"][i] / n), q_time=float(L["qt"][i] / n),
                       survive=float(L["alive"][i] / n))
            out[ph]["ladder"].append(row)
            print(f"                {s_:>6.2f} {row['q_succ']:>9.4f} {row['q_time']:>9.4f} {row['survive']:>8.1%}")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n[diag] -> {a.out}")


if __name__ == "__main__":
    main()
