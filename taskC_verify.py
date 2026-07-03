#!/usr/bin/env python3
"""Task C fix + verification: the suction cup is rotationally symmetric and the tcp lies on the
J6 axis, so holding J6 at its base value should NOT move the tip or change the approach axis.
Verify that over the workspace (FK tip error must be ~0), and plot before/after max rotation.
Saves to outputs/taskC_planner/."""
import os, sys, socket, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C
from geometry import R_from_two_axes, R_to_quat_wxyz, quat_wxyz_to_R
OUT = os.path.join(C.OUT_DIR, "taskC_planner"); os.makedirs(OUT, exist_ok=True)
DOWN = list(map(float, R_to_quat_wxyz(R_from_two_axes(np.array([0, 0, -1.0])))))
BASE = np.array(C.BASE_Q); BASE_DEG = np.degrees(BASE)

def rpc(d):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=40); s.sendall((json.dumps(d) + "\n").encode()); b = b""
    while not b.endswith(b"\n"): b += s.recv(65536)
    s.close(); return json.loads(b)
def fk(qrad):
    r = rpc({"type": "fk", "q": list(map(float, qrad))}); return np.array(r["pos"][0]), np.array(r["quat"][0])

xs = np.linspace(0.28, 0.48, 6); ys = np.linspace(-0.22, 0.22, 7)
tip_err = []; ax_err = []; dev_before = []; dev_after = []
for x in xs:
    for y in ys:
        r = rpc({"type": "plan_pose", "start_q": list(map(float, BASE)),
                 "goal_pose": [float(x), float(y), 0.10] + DOWN, "max_attempts": 16})
        if not r.get("success"): continue
        qf = np.array(r["trajectory"][-1])
        qflat = qf.copy(); qflat[5] = BASE[5]                     # hold J6 at base
        p0, q0 = fk(qf); p1, q1 = fk(qflat)
        tip_err.append(np.linalg.norm(p0 - p1) * 1000)           # mm
        za = quat_wxyz_to_R(q0)[:, 2]; zb = quat_wxyz_to_R(q1)[:, 2]  # approach axis
        ax_err.append(np.degrees(np.arccos(np.clip(za @ zb, -1, 1))))
        dev_before.append(np.max(np.abs(np.degrees(qf) - BASE_DEG)))
        dev_after.append(np.max(np.abs(np.degrees(qflat) - BASE_DEG)))
tip_err = np.array(tip_err); ax_err = np.array(ax_err)
db = np.array(dev_before); da = np.array(dev_after)

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].hist(tip_err, bins=20, color="seagreen", edgecolor="k")
ax[0].set_title(f"tip position shift from J6-flatten\nmax={tip_err.max():.3f} mm (≈0 → SAFE)")
ax[0].set_xlabel("tip shift (mm)")
ax[1].hist(ax_err, bins=20, color="teal", edgecolor="k")
ax[1].set_title(f"approach-axis change\nmax={ax_err.max():.4f}° (≈0 → SAFE)"); ax[1].set_xlabel("approach change (deg)")
ax[2].hist(db, bins=15, alpha=.6, label=f"before (free J6): max {db.max():.0f}°", color="indianred", edgecolor="k")
ax[2].hist(da, bins=15, alpha=.6, label=f"after (J6=base): max {da.max():.0f}°", color="steelblue", edgecolor="k")
ax[2].axvline(90, color="orange", ls="--"); ax[2].axvline(180, color="red", ls="--")
ax[2].set_title("max joint rotation from base"); ax[2].set_xlabel("max |joint-base| (deg)"); ax[2].legend()
fig.suptitle("Task C fix: hold J6 at base (cup is symmetric) — tip pose preserved, excessive rotation removed", fontsize=13)
fig.tight_layout(); p = os.path.join(OUT, "j6_flatten_verify.png"); fig.savefig(p, dpi=110)
print("saved", p)
print(f"tip shift  max={tip_err.max():.4f} mm   approach change max={ax_err.max():.5f} deg")
print(f"max rotation from base:  before(free J6)={db.max():.0f}°   after(J6=base)={da.max():.0f}°")
print(f">90° poses:  before={int((db>90).sum())}/{len(db)}   after={int((da>90).sum())}/{len(da)}")
