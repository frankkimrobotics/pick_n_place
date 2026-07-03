#!/usr/bin/env python3
"""Task C: characterize cuRobo joint rotation from base across the workspace (PLAN-ONLY, no motion).
Plans base->down-pose for a grid of table targets, records per-joint deviation from BASE_Q,
flags excessive (>90 / >180 deg) rotation. Saves plots to outputs/taskC_planner/."""
import os, sys, socket, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C
from geometry import R_from_two_axes, R_to_quat_wxyz
OUT = os.path.join(C.OUT_DIR, "taskC_planner"); os.makedirs(OUT, exist_ok=True)
DOWN = list(map(float, R_to_quat_wxyz(R_from_two_axes(np.array([0, 0, -1.0])))))
BASE_DEG = np.degrees(np.array(C.BASE_Q))

def rpc(d):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=40); s.sendall((json.dumps(d) + "\n").encode()); b = b""
    while not b.endswith(b"\n"): b += s.recv(65536)
    s.close(); return json.loads(b)

def analyze(tag, seed):
    xs = np.linspace(0.28, 0.48, 6); ys = np.linspace(-0.22, 0.22, 7); z = 0.10
    rows = []
    for x in xs:
        for y in ys:
            r = rpc({"type": "plan_pose", "start_q": list(map(float, seed)),
                     "goal_pose": [float(x), float(y), z] + DOWN, "max_attempts": 16})
            if not r.get("success"):
                rows.append((x, y, None, None)); continue
            qf_deg = np.degrees(np.array(r["trajectory"][-1]))
            dev = qf_deg - BASE_DEG                         # per-joint deviation from base
            rows.append((x, y, dev, float(np.max(np.abs(dev)))))
    return rows

def plot(rows, tag):
    ok = [(x, y, dev, m) for (x, y, dev, m) in rows if m is not None]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    # (1) xy map colored by max joint deviation
    xs = [r[0] for r in ok]; ys = [r[1] for r in ok]; ms = [r[3] for r in ok]
    sc = ax[0].scatter(xs, ys, c=ms, cmap="RdYlGn_r", s=260, vmin=0, vmax=200, edgecolor="k")
    for x, y, _, m in ok: ax[0].annotate(f"{m:.0f}", (x, y), ha="center", va="center", fontsize=7)
    fig.colorbar(sc, ax=ax[0], label="max |joint - base| (deg)")
    ax[0].set_title(f"[{tag}] max joint rotation from base"); ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)")
    ax[0].axhline(0, color="gray", lw=.5)
    # (2) histogram of max deviation
    ax[1].hist(ms, bins=20, color="steelblue", edgecolor="k")
    ax[1].axvline(90, color="orange", ls="--", label="90° ok-limit"); ax[1].axvline(180, color="red", ls="--", label="180°")
    ax[1].set_title("distribution of max joint rotation"); ax[1].set_xlabel("max |joint-base| (deg)"); ax[1].legend()
    # (3) per-joint worst-case deviation
    devs = np.array([r[2] for r in ok])                    # N x 6
    worst = np.max(np.abs(devs), axis=0); med = np.median(np.abs(devs), axis=0)
    j = np.arange(1, 7); ax[2].bar(j - 0.2, worst, 0.4, label="worst", color="indianred")
    ax[2].bar(j + 0.2, med, 0.4, label="median", color="lightsteelblue")
    ax[2].axhline(90, color="orange", ls="--"); ax[2].axhline(180, color="red", ls="--")
    ax[2].set_title("per-joint rotation from base"); ax[2].set_xlabel("joint"); ax[2].set_ylabel("|dev| (deg)"); ax[2].legend()
    fig.suptitle(f"Task C — cuRobo joint rotation from base ({tag}); {sum(1 for _,_,_,m in ok if m>180)}/{len(ok)} poses >180deg",
                 fontsize=13)
    fig.tight_layout()
    p = os.path.join(OUT, f"rotation_{tag}.png"); fig.savefig(p, dpi=110); print("saved", p)
    n180 = sum(1 for _, _, _, m in ok if m > 180); n90 = sum(1 for _, _, _, m in ok if m > 90)
    print(f"  [{tag}] {len(ok)}/{len(rows)} planned; >90deg: {n90}, >180deg: {n180}; worst per-joint: {np.round(worst,0)}")
    return ok

if __name__ == "__main__":
    print("== baseline: seed = BASE_Q ==")
    rows = analyze("baseline", C.BASE_Q)
    plot(rows, "baseline")
