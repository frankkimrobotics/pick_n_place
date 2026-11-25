"""plots of the bare (open-loop velocity command) runs: one figure for pulses, one for the slow ramps."""
import glob, sys, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__)); from look import load
D = os.path.expanduser("~/pnp_real_runs/sysid")
for kind in ("bare", "ramp"):
    fig, ax = plt.subplots(6, 2, figsize=(15, 17), sharex="col" if kind == "ramp" else False)
    for j in range(6):
        fs = sorted(glob.glob(f"{D}/*{kind}_j{j}_pi.csv"))
        if not fs: continue
        L = load(fs[-1]); t = L["t"] - L["t"][0]
        tt = np.arange(0, t[-1], 0.01); q = np.interp(tt, t, L["q"][:, j]); v = np.gradient(q, 0.01)
        a = ax[j, 0]; a.plot(t, L["cv"][:, j], "k", lw=1, label="velocity command"); a.plot(tt, v, "C0", lw=1, label="measured (d posfb/dt)")
        a.set_ylabel(f"J{j+1}  deg/s"); a.grid(alpha=.3)
        b = ax[j, 1]; b.plot(t, L["tq"][:, j], "C3", lw=.8); b.set_ylabel("torqfb"); b.grid(alpha=.3)
        b2 = b.twinx(); b2.plot(t, L["q"][:, j] - L["q"][0, j], "C2", lw=.8); b2.set_ylabel("position - start (deg)", color="C2")
    ax[0, 0].legend(loc="upper left", fontsize=8); ax[0, 0].set_title("open-loop velocity command vs response"); ax[0, 1].set_title("drive torque feedback (red) and displacement (green)")
    ax[5, 0].set_xlabel("s"); ax[5, 1].set_xlabel("s")
    fig.tight_layout(); fig.savefig(f"{D}/plot_{kind}.png", dpi=80); print("saved", f"{D}/plot_{kind}.png")
