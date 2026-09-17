#!/usr/bin/env python3
"""plot_ab :: overlay training curves of several PPO runs (log.jsonl) — e.g. the
measured-drive vs ideal-drive A/B.  python3 rl/plot_ab.py ~/pnp_rl/rd_attach_real ~/pnp_rl/rd_attach_ideal --out /tmp/ab.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--out", default="/tmp/ab.png")
ap.add_argument("--comps", default="seal,lift,approach,sat,smooth,act")
a = ap.parse_args()
comps = a.comps.split(",")
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
axes = axes.ravel()
for r in a.runs:
    L = [json.loads(l) for l in open(os.path.join(r, "log.jsonl"))]
    st = [x["step"] / 1e6 for x in L]
    name = os.path.basename(r.rstrip("/"))
    axes[0].plot(st, [100 * x["success"] for x in L], label=name)
    axes[1].plot(st, [100 * x["seal_rate"] for x in L], label=name)
    axes[2].plot(st, [x["ep_ret"] for x in L], label=name)
    axes[3].plot(st, [x["ep_len"] for x in L], label=name)
    for c in comps:
        if c in L[-1]["comp"]:
            axes[4].plot(st, [x["comp"].get(c, 0) for x in L], label=f"{name}:{c}", lw=1)
    axes[5].plot(st, [x["sps"] for x in L], label=name)
for ax, t in zip(axes, ["success %", "seal rate %", "episode return", "episode length", "reward components /ep", "env steps/s"]):
    ax.set_title(t, fontsize=10); ax.grid(alpha=.3); ax.set_xlabel("M steps")
    ax.legend(fontsize=7)
fig.suptitle("PPO attach-from-scratch: measured drive vs ideal drive (2026-09-17)")
fig.tight_layout(); fig.savefig(a.out, dpi=110)
for r in a.runs:
    L = [json.loads(l) for l in open(os.path.join(r, "log.jsonl"))]
    x = L[-1]
    print(f"{os.path.basename(r.rstrip('/')):22s} step {x['step']/1e6:.2f}M  succ {100*x['success']:.1f}%  seal {100*x['seal_rate']:.1f}%  ret {x['ep_ret']:.2f}  len {x['ep_len']:.1f}  sps {x['sps']:.0f}  "
          + " ".join(f"{c}={x['comp'].get(c, 0):.2f}" for c in comps if c in x["comp"]))
print("saved", a.out)
