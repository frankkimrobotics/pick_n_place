#!/usr/bin/env python3
"""plot_dagger :: per-iteration metrics of bc_curobo.py (metrics.json) — student success, seal
rate and return vs DAgger iteration, dataset size on a twin axis.
  python3 rl/plot_dagger.py ~/pnp_rl/dagger_ideal ~/pnp_rl/dagger_real --out /tmp/dagger.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--out", default="/tmp/dagger.png")
a = ap.parse_args()
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for r in a.runs:
    f = os.path.join(r, "metrics.json")
    if not os.path.exists(f):
        continue
    M = json.load(open(f))
    it = [m["iter"] for m in M]
    name = os.path.basename(r.rstrip("/"))
    axes[0].plot(it, [100 * m["student_success"] for m in M], "o-", label=f"{name} student")
    axes[0].axhline(100 * M[-1]["teacher_success"], ls="--", lw=1, color=axes[0].lines[-1].get_color(), label=f"{name} teacher")
    axes[1].plot(it, [100 * m["student_seal"] for m in M], "o-", label=name)
    axes[2].plot(it, [m["student_return"] for m in M], "o-", label=name)
for ax, t in zip(axes, ["deterministic student success %", "student seal rate %", "student episode return"]):
    ax.set_title(t, fontsize=10); ax.set_xlabel("DAgger iteration (0 = one-shot BC)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.suptitle("DAgger from the scripted/cuRobo teacher (paper env)")
fig.tight_layout(); fig.savefig(a.out, dpi=110)
for r in a.runs:
    f = os.path.join(r, "metrics.json")
    if os.path.exists(f):
        for m in json.load(open(f)):
            print(f"{os.path.basename(r.rstrip('/')):14s} iter {m['iter']}  dataset {m['dataset']:7d}  student success {100*m['student_success']:.1f}%  seal {100*m['student_seal']:.0f}%  return {m['student_return']:.2f}  (teacher {100*m['teacher_success']:.0f}%)")
print("saved", a.out)
