"""plot_training :: SAC curves — losses, success/seal, and the full
per-component reward breakdown over training. Usage:
    python rl/plot_training.py ~/pnp_rl/run1 [out.png]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK2, MUTED, SURF = "#52514e", "#c3c2b7", "#fcfcfb"

run = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else f"{run}/training_curves.png"
recs = [json.loads(l) for l in open(f"{run}/log.jsonl")]
S = [r["step"] / 1e6 for r in recs]

fig, ax = plt.subplots(2, 3, figsize=(15, 8), dpi=130, facecolor=SURF)
for a in ax.ravel():
    a.set_facecolor(SURF)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.grid(axis="y", color=MUTED, lw=0.3, alpha=0.5)
    a.set_xlabel("env steps (M)", fontsize=9)

ax[0, 0].plot(S, [r["success"] for r in recs], color=BLUE, lw=1.6, label="place success")
ax[0, 0].plot(S, [r["seal_rate"] for r in recs], color=ORANGE, lw=1.4, label="seal rate")
ax[0, 0].set_title("task success", fontsize=10); ax[0, 0].legend(fontsize=8, frameon=False)
ax[0, 0].set_ylim(-0.02, 1.02)

ax[0, 1].plot(S, [r["ep_ret"] for r in recs], color=BLUE, lw=1.4)
ax[0, 1].set_title("episode return", fontsize=10)

ax[0, 2].plot(S, [r["ep_len"] for r in recs], color=BLUE, lw=1.4)
ax[0, 2].set_title("episode length (steps)", fontsize=10)

ax[1, 0].plot(S, [r["critic"] for r in recs], color=BLUE, lw=1.2, label="critic")
ax[1, 0].plot(S, [r["actor"] for r in recs], color=ORANGE, lw=1.2, label="actor")
ax[1, 0].set_title("losses", fontsize=10); ax[1, 0].legend(fontsize=8, frameon=False)

ax[1, 1].plot(S, [r["entropy"] for r in recs], color=BLUE, lw=1.2, label="policy entropy")
ax[1, 1].plot(S, [r["alpha"] for r in recs], color=ORANGE, lw=1.2, label="alpha")
ax[1, 1].plot(S, [r["q_mean"] for r in recs], color=AQUA, lw=1.2, label="Q mean")
ax[1, 1].set_title("SAC internals", fontsize=10); ax[1, 1].legend(fontsize=8, frameon=False)

keys = ["approach", "seal", "lift", "transport", "place", "drop"]
cols = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, INK2]
for k, c in zip(keys, cols):
    ax[1, 2].plot(S, [r["comp"][k] for r in recs], color=c, lw=1.2, label=k)
ax[1, 2].set_title("reward components / episode", fontsize=10)
ax[1, 2].legend(fontsize=7, frameon=False, ncol=2)

fig.suptitle(f"SAC pick-and-place — {run}", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out, facecolor=SURF)
print("saved", out)
