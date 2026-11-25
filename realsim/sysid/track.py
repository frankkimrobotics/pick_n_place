"""tracking quality of a closed-loop run: lag (cross-correlation), rms/max error vs the TRUE reference at the Pi's clock."""
import glob, sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from look import load
D = os.path.expanduser("~/pnp_real_runs/sysid")
def score(tagglob):
    f = sorted(glob.glob(f"{D}/*{tagglob}_pi.csv"))[-1]; base = f[:-7]
    L = load(f); ref = np.load(base + "_ref.npy"); meta = json.load(open(base + "_meta.json"))
    tr = meta["t_anchor"] + np.arange(len(ref)) * 0.01
    res = {}
    for j in range(6):
        if np.ptp(ref[:, j]) < 0.5: continue
        t = L["t"]; q = L["q"][:, j]
        m = (t > tr[0]) & (t < tr[-1])
        best = None
        for lag in np.arange(-0.08, 0.12, 0.002):
            e = q[m] - np.interp(t[m] - lag, tr, ref[:, j])
            r = np.sqrt((e ** 2).mean())
            if best is None or r < best[1]: best = (lag, r)
        e0 = q[m] - np.interp(t[m], tr, ref[:, j])
        res[f"j{j+1}"] = dict(lag_ms=round(1e3 * best[0]), rms=round(float(np.sqrt((e0 ** 2).mean())), 3), max=round(float(np.abs(e0).max()), 3), rms_at_lag=round(float(best[1]), 3))
    return res
if __name__ == "__main__":
    for g in sys.argv[1:]: print(g, score(g))
