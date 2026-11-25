"""fit the bare-drive runs: per joint  v = K * delay(T) * 1/(tau s + 1) * accel-limit  of the velocity command,
breakaway deadband from the slow ramps, friction/gravity from the drive torque feedback."""
import glob, sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from look import load
D = os.path.expanduser("~/pnp_real_runs/sysid")
DT = 0.002
def sim(u, K, Td, tau, amax):
    n = int(round(Td / DT)); ud = np.concatenate([np.full(n, u[0]), u[:len(u) - n]]) * K
    v = np.zeros(len(u)); x = ud[0]
    for i in range(len(u)):
        dv = (ud[i] - x) / tau * DT if tau > 1e-4 else ud[i] - x
        x += np.clip(dv, -amax * DT, amax * DT); v[i] = x
    return v
out = {}
for j in range(6):
    L = load(sorted(glob.glob(f"{D}/*bare_j{j}_pi.csv"))[-1]); t = L["t"] - L["t"][0]
    tt = np.arange(0, t[-1], DT)
    idx = np.clip(np.searchsorted(t, tt, side="right") - 1, 0, len(t) - 1)
    u = L["cv"][idx, j]                                   # zero-order hold of the written command
    q = np.interp(tt, t, L["q"][:, j]) - L["q"][0, j]
    best = None
    for Td in np.arange(0.0, 0.081, 0.004):
        for tau in (0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06):
            for amax in (300, 500, 800, 1200, 1e9):
                v0 = sim(u, 1.0, Td, tau, amax); p0 = np.cumsum(v0) * DT
                K = float(p0 @ q / (p0 @ p0))             # displacement-matching gain... drift-free pulses
                e = np.sqrt(((K * p0 - q) ** 2).mean())
                if best is None or e < best[0]: best = (e, K, Td, tau, amax)
    e, K, Td, tau, amax = best
    # steady-state gain per plateau + torque
    v = np.gradient(q, DT); tq = np.interp(tt, t, L["tq"][:, j])
    edges = np.where(np.abs(np.diff(u)) > 0.3)[0]; edges = edges[np.diff(np.r_[-999, edges]) > 20]
    rows = []
    for a, b in zip(edges, np.r_[edges[1:], len(tt) - 1]):
        ss = slice(a + int(0.5 * (b - a)), b - 10); c = u[ss].mean()
        rows.append((c, v[ss].mean(), tq[ss].mean()))
    rows = np.array(rows); mv = np.abs(rows[:, 0]) > 0.5
    gain = float(np.median(rows[mv, 1] / rows[mv, 0]))
    pos, neg = rows[rows[:, 0] > 0.5], rows[rows[:, 0] < -0.5]
    # torque = g + c*sign(v) + b*v
    A = np.c_[np.ones(mv.sum()), np.sign(rows[mv, 0]), rows[mv, 0]]; g, c, bv = np.linalg.lstsq(A, rows[mv, 2], rcond=None)[0]
    # ramp: breakaway command (first |v|>0.15 deg/s sustained) for each start from rest
    R = load(sorted(glob.glob(f"{D}/*ramp_j{j}_pi.csv"))[-1]); tr = R["t"] - R["t"][0]; t2 = np.arange(0, tr[-1], 0.01)
    ur = np.interp(t2, tr, R["cv"][:, j]); qr = np.interp(t2, tr, R["q"][:, j]); vr = np.gradient(qr, 0.01)
    vr = np.convolve(vr, np.ones(5) / 5, "same"); brk = []
    starts = np.where((np.abs(ur[:-1]) < 1e-6) & (np.abs(ur[1:]) > 1e-6))[0]
    for s0 in starts:
        k = np.where(np.abs(vr[s0:s0 + 300]) > 0.1)[0]
        if len(k): brk.append(abs(float(ur[s0 + k[0]])))
    out[f"j{j+1}"] = dict(pos_rms_deg=round(e, 4), K=round(K, 4), gain_ss=round(gain, 4), dead_ms=round(1e3 * Td), tau_ms=round(1e3 * tau),
                          amax=amax if amax < 1e8 else None, breakaway_cmd_deg_s=[round(x, 2) for x in brk],
                          torq_gravity=round(float(g), 4), torq_coulomb=round(float(c), 4), torq_visc_per_deg_s=round(float(bv), 5))
    print(f"J{j+1}", out[f"j{j+1}"], flush=True)
json.dump(out, open(f"{D}/fit_bare.json", "w"), indent=1)
