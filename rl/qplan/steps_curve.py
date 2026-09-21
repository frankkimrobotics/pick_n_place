#!/usr/bin/env python3
"""qplan.steps_curve :: how many TD steps should Q get?

The M2 loop decayed monotonically while the critic's CALIBRATION kept improving (TD |bias| ->
0, corr 0.851 -> 0.884 over ten iterations).  That is only possible if TD convergence and
planning quality pull in opposite directions, so this script trains the SAME critic on the SAME
fixed offline buffer for a range of step counts and evaluates the planner with each one.

    $PY rl/qplan/steps_curve.py --data ~/pnp_rl/qplan/data --steps 2500 5000 10000 20000 40000
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
from common import DATA_ROOT, V_RANGE, V_RANGE_PLACE, load_pi, make_env   # noqa: E402
import critic as CR                                            # noqa: E402
import planner as PL                                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=[os.path.join(DATA_ROOT, "data")])
    ap.add_argument("--steps", type=int, nargs="+", default=[2500, 5000, 10000, 20000, 40000])
    ap.add_argument("--nworld", type=int, default=1024)
    ap.add_argument("--ep_len", type=int, default=150)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_cand", type=int, default=16)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--vel_margin", type=float, default=0.0)
    ap.add_argument("--place_phase", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(DATA_ROOT, "steps_curve.json"))
    a = ap.parse_args()
    import warp as wp
    wp.init()
    dev = "cuda:0"
    torch.manual_seed(a.seed)
    buf = CR.load_buffer(a.data, device=dev)
    env = make_env(a.nworld, device=dev, seed=a.seed, dr=True, obj_err=False, ep_len=a.ep_len,
                   place_phase=bool(a.place_phase))
    pol = load_pi(dev)
    vm = None if not a.vel_margin else a.vel_margin
    base = PL.rollout(env, PL.PiActor(pol), a.ep_len, dev, seed=a.seed)
    print(PL.fmt("pi alone", base), flush=True)
    rows = []
    for st in sorted(a.steps):
        t0 = time.time()
        q, hist = CR.train_q(buf, steps=st, batch=a.batch, lr=a.lr, device=dev, seed=a.seed,
                             log_every=max(1, st),
                             v_range=(V_RANGE_PLACE if a.place_phase else V_RANGE))
        cal = CR.calibrate(q, buf, device=dev, seed=a.seed)
        q.eval()
        for p in q.parameters():
            p.requires_grad_(False)
        g = torch.Generator(device=dev)
        g.manual_seed(a.seed + 31337)
        act = PL.QPlanner(pol, q, a.n_cand, a.lam, gen=g, vel_margin=vm)
        r = PL.rollout(env, act, a.ep_len, dev, seed=a.seed)
        row = dict(steps=st, eval=r, cal={k: {kk: v[kk] for kk in
                                              ("bias", "mae", "corr", "pred_mean", "real_mean")}
                                          for k, v in cal.items()},
                   diag=act.diag_mean(), td=hist[-1], seconds=round(time.time() - t0))
        rows.append(row)
        print(PL.fmt(f"planner @ {st} TD steps", r), flush=True)
        print(f"       cal succ bias {cal['succ']['bias']:+.4f} corr {cal['succ']['corr']:.3f} | "
              f"time bias {cal['time']['bias']:+.4f} corr {cal['time']['corr']:.3f} | "
              f"w(pi) {row['diag']['w_pi']:.3f} survivors {row['diag']['n_alive']:.1f} "
              f"[{row['seconds']}s]", flush=True)
        with open(a.out, "w") as f:
            json.dump(dict(args=vars(a), base=base, rows=rows), f, indent=1)
    print(f"[steps] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
