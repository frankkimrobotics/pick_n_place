#!/usr/bin/env python3
"""3x3 montage of a grid-touch session, arranged by physical (x,y): J2 shoulder cmd-vs-actual,
phase-shaded, contact marked. Usage: python3 montage_grid.py <session.npz> <out.png> [title]"""
import sys, json
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def build_cmd(cmd_log, ts, j):
    chunks = []
    for tsend, m in cmd_log:
        if "trajectory" in m: chunks.append(("traj", float(m.get("t_anchor", tsend)), float(m["traj_dt"]), np.array(m["trajectory"], float)))
        elif m.get("hold"): chunks.append(("hold", float(tsend), None, None))
    chunks.sort(key=lambda c: c[1]); out = np.full(len(ts), np.nan); last = None
    for k, t in enumerate(ts):
        active = None
        for c in chunks:
            if c[1] <= t: active = c
            else: break
        if active is None: continue
        if active[0] == "hold":
            if last is not None: out[k] = last
        else:
            _, anc, dt, wps = active; idx = (t - anc) / dt
            cmd = wps[0] if idx <= 0 else wps[-1] if idx >= len(wps)-1 else wps[int(idx)]*(1-(idx-int(idx)))+wps[int(idx)+1]*(idx-int(idx))
            out[k] = cmd[j]; last = cmd[j]
    return out

def spans(ep):
    t0 = ep["t0"]; jt = ep["junc_t"]; tc = (ep["t_contact"]-t0) if ep.get("t_contact") else ep["dur"]
    tlr = (ep["t_liftret"]-t0) if ep.get("t_liftret") else tc
    s = [("approach",0,jt,"#d4e6fb"),("descent",jt,tc,"#d6f0d2")]
    if ep.get("t_contact"): s.append(("hold",tc,tlr,"#fbe0cd"))
    s.append(("lift/return",tlr,ep["t_end"]-t0,"#e8e8e8")); return s

sess, out = sys.argv[1], sys.argv[2]; title = sys.argv[3] if len(sys.argv) > 3 else "grid touch"
d = np.load(sess, allow_pickle=True); T=d["T"]; Q=d["Q"]; cmd_log=json.loads(str(d["cmd_log"])); eps=json.loads(str(d["episodes"]))
live = [e for e in eps if not e.get("skipped") and e.get("t0") is not None]
xs = sorted(set(round(e["xy"][0],2) for e in live)); ys = sorted(set(round(e["xy"][1],2) for e in live), reverse=True)
fig, axs = plt.subplots(len(ys), len(xs), figsize=(4.2*len(xs), 3.0*len(ys)), squeeze=False)
J = 1  # J2 shoulder (moves most in descent)
for e in live:
    r = ys.index(round(e["xy"][1],2)); c = xs.index(round(e["xy"][0],2)); ax = axs[r][c]
    t0 = e["t0"]; m = (T>=t0-0.3)&(T<=e["t_end"]+0.3); tt = T[m]-t0
    for _,a,b,col in spans(e):
        if b>a: ax.axvspan(a,b,color=col,lw=0)
    ax.plot(tt, build_cmd(cmd_log, T[m], J), "--", color="#b30000", lw=1.2)
    ax.plot(tt, Q[m][:,J], "-", color="#003a99", lw=1.2)
    if e.get("t_contact"): ax.axvline(e["t_contact"]-t0, color="crimson", lw=1.0)
    gr = e.get("gap_rest_mm"); grs = f"{gr:+.1f}mm" if gr is not None else "-"
    ax.set_title(f"[{e['xy'][0]:.1f},{e['xy'][1]:+.1f}]  seat {grs}", fontsize=9)
    ax.grid(alpha=0.2); ax.tick_params(labelsize=7)
    if c==0: ax.set_ylabel("J2 (deg)", fontsize=8)
    if r==len(ys)-1: ax.set_xlabel("t (s)", fontsize=8)
leg = [Patch(facecolor=c,label=l) for l,_,_,c in spans(live[0])] + [plt.Line2D([],[],color="#b30000",ls="--",label="cmd"), plt.Line2D([],[],color="#003a99",label="actual")]
fig.legend(handles=leg, loc="lower center", ncol=6, fontsize=8, frameon=False, bbox_to_anchor=(0.5,0.0))
fig.suptitle(title, fontsize=13); fig.tight_layout(rect=[0,0.04,1,0.98]); fig.savefig(out, dpi=115); print("saved", out)
