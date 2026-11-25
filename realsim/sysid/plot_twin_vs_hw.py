"""plot realsim/sysid/twin_vs_hw.py traces (system python: matplotlib is not in mj4)"""
import os, sys, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
D = os.path.expanduser("~/pnp_real_runs/sysid"); T = np.load(f"{D}/twin_vs_hw_traces.npz"); R = json.load(open(f"{D}/twin_vs_hw.json"))
titles = ["1 Hz 3 deg sine, deployed gains", "1 Hz 3 deg sine, tuned gains", "0.5 Hz 4 deg sine, deployed gains", "0.5 Hz 4 deg sine, tuned gains", "3 deg at 30 deg/s, deployed gains", "3 deg at 30 deg/s, tuned gains"]
fig, ax = plt.subplots(2, 6, figsize=(25, 7))
for c, (tag, title) in enumerate(zip(R, titles)):
    tr, r, th, qh, ts, qs = (T[f"{c}_{k}"] for k in ("tr", "r", "th", "qh", "ts", "qs"))
    xl = (1.5, 3.5) if "sine" in tag else (0.4, 1.3)
    a = ax[0, c]; a.plot(tr, r, "k", lw=1.2, label="reference"); a.plot(th, qh, "C3", lw=1, label="real arm"); a.plot(ts, qs, "C0", lw=1, ls="--", label="twin")
    a.set_xlim(*xl); a.set_title(title, fontsize=10); a.grid(alpha=.3); a.legend(fontsize=7); a.set_ylabel("J2 deg")
    b = ax[1, c]; b.plot(th, qh - np.interp(th, tr, r), "C3", lw=1, label=f"real rms {R[tag]['hw']['rms']:.3f}"); b.plot(ts, qs - np.interp(ts, tr, r), "C0", lw=1, ls="--", label=f"twin rms {R[tag]['twin']['rms']:.3f}")
    b.set_xlim(*xl); b.grid(alpha=.3); b.legend(fontsize=7); b.set_ylabel("tracking error (deg)"); b.set_xlabel("s")
fig.tight_layout(); fig.savefig(f"{D}/plot_twin_vs_hw.png", dpi=70); print("saved")
