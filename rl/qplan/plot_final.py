#!/usr/bin/env python3
"""qplan.plot_final :: the M2 figure -- success / t_goal / peak |qd| vs iteration.

Three measures of different scale never share one y-axis, so this is small multiples with one
panel each.  Iteration 0 is the OFFLINE critic (no online data yet) and is the point every loop
has to beat.  Both loops are drawn: the plan's literal recipe (raw union of the buffer, 10 k
warm-start TD steps per iteration) and the corrected one (>=35 % fixed-pool batches, 2.5 k steps
at lr 1e-4, velocity cap, best-checkpoint + early stop).

    $PY rl/qplan/plot_final.py
"""
import json
import os
import sys

import numpy as np

DATA = os.path.expanduser("~/pnp_rl/qplan")
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e2e1dc", "#fcfcfb"
SER = {"v1": "#2a78d6", "v3": "#eb6834", "v4": "#1baf7a"}
GATE = dict(success=99.0, t_goal=6.5, qd_rel=2.0)


def load(name, it0):
    p = os.path.join(DATA, name, "iterations.json") if name else os.path.join(DATA, "iterations.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    xs = [0] + [r["iter"] for r in d["rows"]]
    ev = [it0] + [r["eval"]["planner"] for r in d["rows"]]
    return dict(base=d["base"], x=xs, ev=ev)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Iteration 0 = the OFFLINE critic (20 k TD steps on the M0 buffer) with the deployed
    # planner, seed 0 -- the point every loop has to beat.  v3 runs a velocity-capped planner,
    # so its iteration 0 is the nearest measured cap (1.3 x pi) rather than the uncapped point.
    sc = json.load(open(os.path.join(DATA, "steps_curve.json")))
    it0 = [r["eval"] for r in sc["rows"] if r["steps"] == 20000][0]
    it0_cap = json.load(open(os.path.join(DATA, "m1_velcap.json")))["rows"]["planner N=16 lam=0.1 vel=cap<=1.3"]
    runs = [("v1", "plan recipe: raw buffer, 10k warm TD steps / iteration", load(None, it0)),
            ("v3", "corrected: 35 % fixed batch, 2.5k steps @1e-4, vel cap 1.2", load("v3", it0_cap)),
            ("v4", "from scratch: 20k TD steps / iteration on the grown buffer", load("v5", it0))]
    runs = [r for r in runs if r[2]]
    base = runs[0][2]["base"]

    panels = [("Success rate", "%", lambda e: 100 * e["success"], GATE["success"],
               100 * base["success"], "{:.1f}%"),
              ("Time to goal", "s", lambda e: e["t_goal"] / 10, GATE["t_goal"],
               base["t_goal"] / 10, "{:.2f} s"),
              ("Peak |qd| p90", "deg/s", lambda e: e["qd_p90"],
               base["qd_p90"] + GATE["qd_rel"], base["qd_p90"], "{:.1f}")]

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 9.0), sharex=True, gridspec_kw=dict(hspace=0.34))
    fig.patch.set_facecolor(SURF)
    for ax, (title, unit, fn, gate, bval, f) in zip(axes, panels):
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=9, length=0)
        allv = [gate, bval]
        for i, (key, _lab, d) in enumerate(runs):
            ys = [fn(e) for e in d["ev"]]
            allv += ys
            ax.plot(d["x"], ys, color=SER[key], lw=2, marker="o", ms=4.5, zorder=3,
                    markeredgecolor=SURF, markeredgewidth=1.2)
            # the end labels of two loops that stop at the same iteration would overlap
            dy = 0 if len({dd["x"][-1] for _k, _l, dd in runs}) == len(runs) else 7 * (i - 1)
            ax.annotate(f.format(ys[-1]), (d["x"][-1], ys[-1]), color=SER[key], fontsize=8.5,
                        weight="bold", xytext=(5, dy), textcoords="offset points", va="center")
        ax.axhline(gate, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.annotate(f"gate {f.format(gate)}", (0, gate), color=INK2, fontsize=8.5,
                    xytext=(2, 3), textcoords="offset points", va="bottom")
        ax.axhline(bval, color="#a8a79f", lw=1.2, zorder=1)
        xmax = max(max(d["x"]) for _k, _l, d in runs)
        ax.annotate(f"pi alone {f.format(bval)}", (xmax, bval), color="#7a7973", fontsize=8.5,
                    xytext=(0, 4), textcoords="offset points", ha="right", va="bottom")
        lo, hi = min(allv), max(allv)
        pad = max(1e-6, 0.16 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(-0.35, max(max(d["x"]) for _k, _l, d in runs) + 1.0)
        ax.set_title(f"{title}  ({unit})", color=INK, fontsize=11, loc="left", pad=8)
    axes[-1].set_xlabel("self-improvement iteration  (0 = offline critic, before any online data)",
                        color=INK2, fontsize=9.5)
    axes[-1].set_xticks(list(range(0, max(max(d["x"]) for _k, _l, d in runs) + 1)))
    handles = [plt.Line2D([], [], color=SER[k], lw=2, marker="o", ms=4.5, label=lab)
               for k, lab, _d in runs]
    axes[0].legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.13), frameon=False,
                   fontsize=8.8, labelcolor=INK2, ncol=1, handlelength=1.6)
    fig.suptitle("Q-Planning twin iterations - planner vs the frozen policy",
                 color=INK, fontsize=13, x=0.055, ha="left", y=1.045)
    out = os.path.join(DATA, "iterations.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURF)
    print(f"[plot] -> {out}")


if __name__ == "__main__":
    main()
