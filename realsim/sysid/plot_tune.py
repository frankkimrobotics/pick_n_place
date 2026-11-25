import glob, sys, os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__)); from look import load
D = os.path.expanduser("~/pnp_real_runs/sysid")
def get(tag, j):
    f = sorted(glob.glob(f"{D}/*{tag}_pi.csv"))[-1]; L = load(f); ref = np.load(f[:-7] + "_ref.npy"); m = json.load(open(f[:-7] + "_meta.json"))
    tr = m["t_anchor"] + np.arange(len(ref)) * 0.01; t = L["t"]; k = (t > tr[0]) & (t < tr[-1])
    return t[k] - tr[0], L["q"][k, j] - ref[0, j], np.interp(t[k], tr, ref[:, j]) - ref[0, j]
fig, ax = plt.subplots(2, 3, figsize=(17, 8))
for col, (old, new, j, title, xl) in enumerate([("sine_012345_fast_old", "sine_012345_fast_new", 1, "J2, 1 Hz 3 deg sine (all joints moving)", (1.5, 4.0)),
                                               ("check_j1_stepold", "check_j1_stepnew", 1, "J2, 3 deg at 30 deg/s then stop", (0.4, 1.3)),
                                               ("check_j4_stepold", "check_j4_stepnew", 4, "J5, 3 deg at 30 deg/s then stop", (0.4, 1.3))]):
    t, q, r = get(old, j); t2, q2, r2 = get(new, j)
    a = ax[0, col]; a.plot(t, r, "k", lw=1.2, label="reference"); a.plot(t, q, "C3", lw=1, label="before (deployed gains)"); a.plot(t2, q2, "C0", lw=1, label="after (tuned)")
    a.set_xlim(*xl); a.set_title(title); a.set_ylabel("deg"); a.grid(alpha=.3); a.legend(fontsize=8)
    b = ax[1, col]; b.plot(t, q - r, "C3", lw=1, label=f"before: rms {np.sqrt(((q-r)**2).mean()):.3f} deg"); b.plot(t2, q2 - r2, "C0", lw=1, label=f"after: rms {np.sqrt(((q2-r2)**2).mean()):.3f} deg")
    b.set_xlim(*xl); b.set_ylabel("tracking error (deg)"); b.set_xlabel("s"); b.grid(alpha=.3); b.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{D}/plot_tuning.png", dpi=80); print("saved")
