#!/usr/bin/env python3
"""Task A (operational probe): FF-driven fast-move tracking. Plans a J1 move, retimes it to a
range of target peak velocities, streams each, and records :9999 to measure actual-vs-commanded
velocity + acceleration + tracking lag. This is the test that reflects real pick-and-place motion
(feedforward engaged) and CAN reveal accel/velocity ceilings (unlike a static step).
Saves to outputs/taskA_gains/.  Usage: python3 taskA_fast.py --tag accel_backup"""
import os, sys, socket, json, time, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C
from joint_conventions import linuxcnc_deg_to_rad, rad_to_linuxcnc_deg
OUT = os.path.join(C.OUT_DIR, "taskA_gains"); os.makedirs(OUT, exist_ok=True)
PI = "10.0.0.27"
BASE = np.array(C.BASE_Q)                     # rad
BASE_DEG = np.array([0.0, -110.0, 80.0, -80.0, -90.0, 0.0])
SAFE = dict(x=(0.05, 0.55), y=(-0.34, 0.34), z=(0.05, 0.66))

def rpc(d):
    s = socket.create_connection(("127.0.0.1", 9997), timeout=30); s.sendall((json.dumps(d) + "\n").encode()); b = b""
    while not b.endswith(b"\n"): b += s.recv(65536)
    s.close(); return json.loads(b)
def send_chunk(m):
    k = socket.create_connection((PI, 9994), timeout=3); k.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    k.sendall((json.dumps(m) + "\n").encode()); k.close()
def fk_tip(rad):
    return np.array(rpc({"type": "fk", "q": list(map(float, rad))})["pos"][0])
def goto(deg, dur):
    send_chunk({"trajectory": [list(map(float, deg)), list(map(float, deg))], "traj_dt": 1.0, "t_anchor": time.time() + 0.05}); time.sleep(dur)

def stream_record(traj_deg, dt, jidx, rec_dur):
    """Stream a joint trajectory (deg waypoints) at traj_dt=dt; record joint jidx from :9999."""
    s = socket.create_connection((PI, 9999), timeout=3); s.settimeout(0.5); buf = b""; samp = []
    t0 = time.time(); anchor = t0 + 0.12
    send_chunk({"trajectory": [list(map(float, w)) for w in traj_deg], "traj_dt": dt, "t_anchor": anchor})
    while time.time() - t0 < rec_dur:
        try: d = s.recv(4096)
        except socket.timeout: continue
        if not d: break
        buf += d
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            try: samp.append((time.time() - t0, json.loads(line)["joints_deg"][jidx]))
            except Exception: pass
    s.close()
    return np.array([a for a, _ in samp]), np.array([b for _, b in samp]), anchor - t0

def vel(t, y, rate=50.0):
    """Resample to a uniform grid (kills uneven-dt derivative spikes), smooth, differentiate."""
    if len(t) < 12: return t, np.zeros_like(t)
    order = np.argsort(t); t = t[order]; y = y[order]
    keep = np.concatenate([[True], np.diff(t) > 1e-4])            # drop duplicate timestamps
    t, y = t[keep], y[keep]
    tu = np.arange(t[0], t[-1], 1.0 / rate)
    yu = np.interp(tu, t, y)
    if len(yu) >= 7: yu = np.convolve(yu, np.ones(7) / 7, mode="same")
    return tu, np.gradient(yu, tu)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="accel_backup"); TAG = ap.parse_args().tag
    goto(BASE_DEG, 6.0)
    goal = BASE.copy(); goal[0] += np.radians(35.0)                 # J1 +35deg move
    if not (SAFE["x"][0] < fk_tip(goal)[0] < SAFE["x"][1] and SAFE["y"][0] < fk_tip(goal)[1] < SAFE["y"][1]):
        print("target unsafe"); return
    r = rpc({"type": "plan_joint", "start_q": list(map(float, BASE)), "goal_q": list(map(float, goal)), "max_attempts": 16})
    if not r.get("success"): print("plan failed"); return
    traj = np.array(r["trajectory"]); dt0 = r["dt"]
    step_deg = np.degrees(np.max(np.abs(np.diff(traj[:, 0]))))       # per-waypoint J1 step (deg)
    MOVE = 35.0; y0 = BASE_DEG[0]
    targets = [15.0, 30.0, 45.0]
    fig, ax = plt.subplots(1, len(targets), figsize=(6 * len(targets), 5.2), sharey=True)
    summ = []
    def crossing(t, yn, thr):
        i = np.where(yn >= thr)[0]; return float(t[i[0]]) if len(i) else float("nan")
    for k, vt in enumerate(targets):
        dt = step_deg / vt                                          # retime so commanded peak vel = vt
        T = dt * (len(traj) - 1)
        traj_deg = [list(map(float, rad_to_linuxcnc_deg(w))) for w in traj]
        t, y, off = stream_record(traj_deg, dt, 0, T + 2.5)
        goto(BASE_DEG, T + 3.0)
        yn = (y - y0) / MOVE                                        # position-based (robust, no derivative)
        t10 = crossing(t, yn, 0.1); t90 = crossing(t, yn, 0.9)
        dead = t10 - off                                           # command->first-motion dead time
        mv = (0.8 * MOVE / (t90 - t10)) if (t90 == t90 and t10 == t10 and t90 > t10) else float("nan")  # 10-90 velocity
        peak_pos = float(np.nanmax(yn)) * 100
        summ.append((vt, T, dead, t90 - off if t90 == t90 else float("nan"), mv, peak_pos))
        print(f"target {vt:.0f}deg/s (cmd dur {T:.2f}s): dead={dead:.2f}s  t90={t90-off:.2f}s  move-vel={mv:.1f}deg/s  reached={peak_pos:.0f}%")
        cmd_j1 = np.array([rad_to_linuxcnc_deg(w)[0] for w in traj]); tc = np.arange(len(traj)) * dt + off
        a = ax[k]
        a.plot(tc, cmd_j1, "b--", label=f"commanded (dur {T:.1f}s)")
        a.plot(t, y, "r-", label=f"actual (t90 {t90-off:.1f}s, v {mv:.0f}deg/s)")
        a.axhline(y0 + 0.9 * MOVE, color="orange", ls=":", lw=.7)
        a.set_title(f"target {vt:.0f} deg/s"); a.set_xlabel("t (s)"); a.grid(alpha=.3); a.legend(fontsize=8)
    ax[0].set_ylabel("J1 position (deg)")
    fig.suptitle(f"Task A FF-move probe ({TAG}): J1 +35deg, commanded vs actual POSITION (dead-time + tracking)", fontsize=12)
    fig.tight_layout(); p = os.path.join(OUT, f"fast_move_{TAG}.png"); fig.savefig(p, dpi=110); print("saved", p)
    with open(os.path.join(OUT, f"fast_move_{TAG}.txt"), "w") as f:
        f.write("target_deg_s  cmd_dur_s  dead_s  t90_s  move_vel_deg_s  reached_%\n")
        for vt, T, dd, t9, mv, pp in summ: f.write(f"{vt:.0f}  {T:.2f}  {dd:.2f}  {t9:.2f}  {mv:.1f}  {pp:.0f}\n")
    goto(BASE_DEG, 3.0); print("done; arm at base")

if __name__ == "__main__":
    main()
