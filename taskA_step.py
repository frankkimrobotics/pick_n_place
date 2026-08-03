#!/usr/bin/env python3
"""Task A: per-joint STEP RESPONSE characterization (baseline for gain tuning).
For each joint, command a 10deg and 20deg step from base, densely record :9999 feedback,
measure rise-time (10-90%) + overshoot + settling, and plot commanded-vs-actual. Every stepped
config is FK-checked to stay in a safe box BEFORE moving. Saves to outputs/taskA_gains/.

  source /opt/ros/humble/setup.bash
  python3 taskA_step.py
"""
import os, sys, socket, json, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C
from joint_conventions import linuxcnc_deg_to_rad
OUT = os.path.join(C.OUT_DIR, "taskA_gains"); os.makedirs(OUT, exist_ok=True)
PI = "10.0.0.27"
BASE_DEG = np.array([0.0, -110.0, 80.0, -80.0, -90.0, 0.0])       # LinuxCNC base pose (deg)
SAFE = dict(x=(0.05, 0.52), y=(-0.32, 0.32), z=(0.05, 0.66))     # tip must stay in this box

def rpc(d):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=30); s.sendall((json.dumps(d) + "\n").encode()); b = b""
    while not b.endswith(b"\n"): b += s.recv(65536)
    s.close(); return json.loads(b)
def send_chunk(m):
    k = socket.create_connection((PI, 9994), timeout=3); k.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    k.sendall((json.dumps(m) + "\n").encode()); k.close()
def fk_tip(deg):
    r = rpc({"type": "fk", "q": [float(v) for v in linuxcnc_deg_to_rad(deg)]}); return np.array(r["pos"][0])
def safe(deg):
    p = fk_tip(deg)
    return (SAFE["x"][0] < p[0] < SAFE["x"][1] and SAFE["y"][0] < p[1] < SAFE["y"][1] and SAFE["z"][0] < p[2] < SAFE["z"][1])

def goto(deg, dur=2.5, settle=True):
    send_chunk({"trajectory": [list(map(float, deg)), list(map(float, deg))], "traj_dt": 1.0, "t_anchor": time.time() + 0.05})
    if settle: time.sleep(dur)

def step_and_record(j, target, dur):
    """Send a step to `target`, record joint j at ~feedback rate; return (t[], y_deg[])."""
    s = socket.create_connection((PI, 9999), timeout=3); buf = b""; samp = []
    t0 = time.time()
    send_chunk({"trajectory": [list(map(float, target)), list(map(float, target))], "traj_dt": 1.0, "t_anchor": t0 + 0.03})
    while time.time() - t0 < dur:
        buf += s.recv(4096)
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            try: samp.append((time.time() - t0, json.loads(line)["joints_deg"][j]))
            except Exception: pass
    s.close(); return np.array([a for a, _ in samp]), np.array([b for _, b in samp])

def metrics(t, y, y0, yT):
    """rise 10-90%, overshoot %, settle 2%, ramp velocity, peak % — robust to incomplete steps."""
    S = yT - y0
    if abs(S) < 1e-6 or len(y) < 5: return None
    yn = (y - y0) / S
    peak = float(np.max(yn)) * 100.0
    over = max(0.0, (np.max(yn) - 1.0) * 100.0)
    rise = float("nan"); vel = float("nan")
    idx10 = np.where(yn >= 0.1)[0]; idx90 = np.where(yn >= 0.9)[0]; idx70 = np.where(yn >= 0.7)[0]
    if len(idx10) and len(idx90): rise = t[idx90[0]] - t[idx10[0]]
    if len(idx10) and len(idx70) and idx70[0] > idx10[0]:
        i0, i1 = idx10[0], idx70[0]; vel = abs((y[i1] - y[i0]) / (t[i1] - t[i0]))
    settled = np.where(np.abs(yn - 1.0) > 0.02)[0]
    settle = float(t[settled[-1]]) if len(settled) else 0.0
    return dict(rise=rise, over=over, settle=settle, peak=peak, vel=vel)

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="baseline"); TAG = ap.parse_args().tag
    goto(BASE_DEG, dur=6.0)                                            # settle to base from wherever the restart left it
    steps = [10.0, 20.0]; DUR = 7.0; RET = 5.5
    results = {}
    fig, axes = plt.subplots(2, 3, figsize=(17, 9)); axes = axes.ravel()
    for j in range(6):
        ax = axes[j]; results[j] = {}
        for stp in steps:
            d = +1.0 if safe(BASE_DEG + np.eye(6)[j] * stp) else (-1.0 if safe(BASE_DEG - np.eye(6)[j] * stp) else 0.0)
            if d == 0.0:
                print(f"J{j+1} {stp:.0f}deg: NO SAFE DIRECTION, skipped"); continue
            target = BASE_DEG.copy(); target[j] += stp * d
            t, y = step_and_record(j, target, DUR)
            goto(BASE_DEG, dur=RET)                                    # return to base
            m = metrics(t, y, BASE_DEG[j], target[j])
            results[j][stp] = m
            if m is None: continue
            yn = (y - BASE_DEG[j]) / (target[j] - BASE_DEG[j])
            ax.plot(t, yn, label=f"{stp:.0f}°  rise {m['rise']:.2f}s  v {m['vel']:.0f}°/s  OS {m['over']:.0f}%  peak {m['peak']:.0f}%")
            print(f"J{j+1} {stp:.0f}°(dir{int(d):+d}): rise={m['rise']:.2f}s vel={m['vel']:.1f}°/s overshoot={m['over']:.0f}% "
                  f"peak={m['peak']:.0f}% settle={m['settle']:.2f}s")
        ax.axhline(1.0, color="gray", lw=.7); ax.axhline(0.9, color="orange", lw=.5, ls=":"); ax.axhline(0.1, color="orange", lw=.5, ls=":")
        ax.set_title(f"Joint {j+1}"); ax.set_xlabel("t since step (s)"); ax.set_ylabel("normalized response"); ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_ylim(-0.1, 1.3)
    fig.suptitle(f"Task A — per-joint step response ({TAG}): normalized actual vs step command", fontsize=14)
    fig.tight_layout()
    p = os.path.join(OUT, f"step_response_{TAG}.png"); fig.savefig(p, dpi=110); print("saved", p)
    # summary table
    with open(os.path.join(OUT, f"step_metrics_{TAG}.txt"), "w") as f:
        f.write("joint  step  rise_s  vel_deg_s  overshoot_%  peak_%  settle_s\n")
        for j in range(6):
            for stp, m in results[j].items():
                if m: f.write(f"J{j+1}  {stp:.0f}  {m['rise']:.2f}  {m['vel']:.1f}  {m['over']:.0f}  {m['peak']:.0f}  {m['settle']:.2f}\n")
    goto(BASE_DEG, dur=2.5)
    print("done; arm at base")

if __name__ == "__main__":
    main()
