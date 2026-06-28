#!/usr/bin/env python3
"""plot_touch :: joint position + velocity profiles and a time-aligned RGBD filmstrip for one
recorded welded-touch episode (servo_touch.py --rec-dir ... writes obj00/).

The joints are ~50 Hz and the camera ~30 Hz; to put the frames on the SAME time axis as the
joint plots we sample a uniform time grid and ZERO-ORDER-HOLD (duplicate the most recent frame
over each interval), so every grid slot has an image even where the camera was slower.
"""
import os, sys, json
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})

HERE = os.path.dirname(os.path.abspath(__file__))
d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "outputs", "touch_rec", "obj00")
K = int(sys.argv[2]) if len(sys.argv) > 2 else 16          # number of filmstrip slots

J = np.load(os.path.join(d, "joints.npz")); t, pos, vel = J["t"], J["pos"], J["vel"]
meta = json.load(open(os.path.join(d, "meta.json"))); marks = meta.get("marks", {})
frames = [json.loads(l) for l in open(os.path.join(d, "frames.jsonl"))]
ft = np.array([f["t"] for f in frames]); fgap = [f["gap"] for f in frames]
t0, t1 = float(min(t.min(), ft.min())), float(max(t.max(), ft.max()))

# ZOH frame indices on a uniform grid (duplicates the last frame across the interval)
grid = np.linspace(t0, t1, K)
idxs = [max(0, int(np.searchsorted(ft, g, side="right")) - 1) for g in grid]

fig = plt.figure(figsize=(16, 11))
gs = GridSpec(4, 1, height_ratios=[3, 3, 2.1, 2.1], hspace=0.38)
ax_p = fig.add_subplot(gs[0]); ax_v = fig.add_subplot(gs[1], sharex=ax_p)
ax_rgb = fig.add_subplot(gs[2], sharex=ax_p); ax_d = fig.add_subplot(gs[3], sharex=ax_p)
cols = plt.cm.tab10(np.arange(6)); mc = {"approach": "tab:green", "blend": "tab:orange", "contact": "tab:red"}

# shade the three phases across every panel
bounds = [marks.get("approach", t0), marks.get("blend"), marks.get("contact"), t1]
spanc = ["#e8f5e9", "#e3f2fd", "#ffebee"]; spanl = ["approach (fast)", "welded descent (slow)", "contact + settle"]
for ax in (ax_p, ax_v, ax_rgb, ax_d):
    for a, b, c in zip(bounds[:-1], bounds[1:], spanc):
        if a is not None and b is not None:
            ax.axvspan(a, b, color=c, alpha=0.6, zorder=0)

for j in range(6):
    ax_p.plot(t, pos[:, j], color=cols[j], lw=1.4, label=f"J{j+1}")
    ax_v.plot(t, vel[:, j], color=cols[j], lw=1.4, label=f"J{j+1}")
ax_p.set_ylabel("joint position (deg)"); ax_v.set_ylabel("joint velocity (deg/s)")
ax_p.legend(ncol=6, fontsize=8, loc="best"); ax_v.axhline(0, color="gray", lw=0.6)

# contact gap (mm) on a twin axis of the velocity panel = the contact signal
ax_g = ax_v.twinx()
gmm = [np.nan if (g is None or g * 1000 > 60) else g * 1000 for g in fgap]   # clip far-field
ax_g.plot(ft, gmm, color="k", lw=1.4, alpha=0.55, label="contact gap")
ax_g.set_ylabel("contact gap (mm)", color="k"); ax_g.set_ylim(0, 60)
ax_g.axhline(5, color="tab:red", ls=":", lw=1, alpha=0.6)   # gap-contact threshold

for ax in (ax_p, ax_v):
    ax.grid(alpha=0.3)
    for k, tv in marks.items():
        ax.axvline(tv, color=mc.get(k, "k"), ls="--", lw=1.3, alpha=0.85)
for k, tv in marks.items():
    ax_p.text(tv, ax_p.get_ylim()[1], f" {k}", color=mc.get(k, "k"), fontsize=10,
              rotation=90, va="top", ha="left")
for a, b, lab in zip(bounds[:-1], bounds[1:], spanl):
    if a is not None and b is not None:
        ax_p.text((a + b) / 2, ax_p.get_ylim()[0], lab, fontsize=9, color="dimgray",
                  ha="center", va="bottom", style="italic")

# time-aligned RGBD filmstrips (ZOH / duplicated frames)
w = (t1 - t0) / K * 0.94
for g, ix in zip(grid, idxs):
    c = cv2.cvtColor(cv2.imread(os.path.join(d, "frames", f"{ix:04d}_c.jpg")), cv2.COLOR_BGR2RGB)
    dp = cv2.cvtColor(cv2.imread(os.path.join(d, "frames", f"{ix:04d}_d.jpg")), cv2.COLOR_BGR2RGB)
    ax_rgb.imshow(c, extent=[g - w / 2, g + w / 2, 0, 1], aspect="auto")
    ax_d.imshow(dp, extent=[g - w / 2, g + w / 2, 0, 1], aspect="auto")
    ax_rgb.text(g, 1.03, f"{ft[ix]:.1f}s", fontsize=7, ha="center", va="bottom")
for ax, lab in ((ax_rgb, "RGB"), (ax_d, "depth")):
    ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_ylabel(lab)
    for k, tv in marks.items():
        ax.axvline(tv, color=mc.get(k, "k"), ls="--", lw=1.3, alpha=0.85)
ax_d.set_xlabel("time since approach (s)"); ax_p.set_xlim(t0, t1)
fig.suptitle(f"Welded touch (pregrasp+contact) — surf={meta.get('surf'):.3f}  "
             f"target={meta.get('target'):.3f}  contact_z={meta.get('contact_z')}  "
             f"| {meta.get('n_joint')} joint @~50Hz, {meta.get('n_frame')} frames @~30Hz", fontsize=11)
out = os.path.join(d, "touch_profile.png"); fig.savefig(out, dpi=110, bbox_inches="tight")
print("wrote", out)
