#!/usr/bin/env python3
"""Phase-0 system ID for the streaming controller's reaction latency.

Commands a bounded position RAMP on one joint from rest (commanded velocity = a step 0->v->0) and
logs actual joint angle at max rate, to separate:
  * PURE DEAD-TIME L  -- delay from commanded-velocity onset to the first actual motion
                         (transport + drive pipeline; beatable only by prediction / HW), and
  * ACTUATOR LAG tau  -- first-order rise of actual velocity toward the commanded velocity
                         (beatable by feedforward / higher bandwidth).
--hold-test additionally ramps at steady velocity then sends a mid-motion HOLD; L_hold is the
reaction latency of the real HOLD path, and L_hold - L_rest ~= the online_servo command-buffer
depth (the Phase-2 target).

Safe by construction: bounded amplitude, constant velocity <= --vel-deg (default 20, cap 55),
J1 base-yaw by default (free space). NOT a fast probe.

  source /opt/ros/humble/setup.bash
  python3 step_id.py --joint 0 --amps 3,5,8 --vel-deg 20 --hold-test
  python3 step_id.py --replot outputs/step_id/session.npz     # remake plots only
"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C
from latency_common import read_state, send_chunk, StreamRecorder, CMD_LOG
OUT = os.path.join(C.OUT_DIR, "step_id"); os.makedirs(OUT, exist_ok=True)
JN = ["J1", "J2", "J3", "J4", "J5", "J6"]
ANCHOR_LEAD = 0.12                                                      # the deliberate t_anchor lead

def ramp_chunk(q0_deg, joint, delta, vel, dt=0.02):
    """Waypoints (linuxcnc deg) ramping `joint` by `delta` deg at constant `vel` deg/s."""
    dur = abs(delta) / max(vel, 1e-6); n = max(int(round(dur / dt)), 2)
    wps = [list(q0_deg) for _ in range(n + 1)]
    for i in range(n + 1): wps[i][joint] = q0_deg[joint] + delta * (i / n)
    return wps, dt

def send_ramp(wps, dt):
    t_send = time.time(); t_anchor = t_send + ANCHOR_LEAD
    send_chunk({"trajectory": [list(map(float, w)) for w in wps], "traj_dt": dt, "t_anchor": t_anchor})
    return t_send, t_anchor

def smooth_vel(T, x, win=5):
    v = np.gradient(x, T)
    v = np.clip(v, -80.0, 80.0)                                        # reject feedback glitches (drives cap ~60 deg/s)
    if len(v) >= win: v = np.convolve(v, np.ones(win) / win, mode="same")
    return v

def onset_after(T, sig, thr, t_after):
    for i in range(len(T)):
        if T[i] >= t_after and abs(sig[i]) > thr: return T[i], i
    return None, None

def fit_L_tau(T, pos, t_anchor, vcmd, q0j):
    """L (s) = actual-motion onset minus t_anchor; tau (s) = first-order velocity-rise constant.
    Onset is detected on POSITION (|pos-q0| > 0.15 deg) -- robust at the ~46 Hz feedback rate."""
    t_on = None
    for i in range(len(T)):
        if T[i] >= t_anchor and abs(pos[i] - q0j) > 0.15: t_on = T[i]; break
    if t_on is None: return None
    L = t_on - t_anchor; vel = smooth_vel(T, pos)
    m = (T >= t_on) & (T <= t_on + 0.8); tt = T[m] - t_on; vv = np.abs(vel[m])
    tau = None; r2 = None
    try:
        from scipy.optimize import curve_fit
        f = lambda t, tau: abs(vcmd) * (1.0 - np.exp(-t / max(tau, 1e-3)))
        p, _ = curve_fit(f, tt, vv, p0=[0.05], maxfev=8000, bounds=(1e-3, 1.0)); tau = float(p[0])
        resid = vv - f(tt, tau); ss = np.sum((vv - vv.mean()) ** 2)
        r2 = float(1 - np.sum(resid ** 2) / ss) if ss > 0 else None
    except Exception:
        tgt = abs(vcmd)
        i10 = np.argmax(vv >= 0.1 * tgt); i90 = np.argmax(vv >= 0.9 * tgt)
        if i90 > i10: tau = (tt[i90] - tt[i10]) / 2.2                  # 10-90% ~ 2.2 tau
    return {"L": float(L), "tau": tau, "r2": r2, "t_on": float(t_on), "vpeak": float(np.max(np.abs(vel[m]))) if m.any() else None}

def analyze(sess):
    d = np.load(sess, allow_pickle=True)
    T = d["T"]; Q = d["Q"]; meta = json.loads(str(d["meta"])); joint = meta["joint"]; vel = meta["vel"]
    print(f"\n==== step_id: {JN[joint]}  vel={vel} deg/s ====")
    print("  amp   L_rest(ms)   tau(ms)   R2    vpeak(deg/s)")
    Ls = []; taus = []
    for st in meta["steps"]:
        r = fit_L_tau(T, Q[:, joint], st["t_anchor"], vel, st["q0"][joint])
        if r is None: print(f"  {st['amp']:+.0f}     no-onset"); continue
        tau_ms = f"{r['tau']*1000:.0f}" if r["tau"] else "  -"
        r2s = f"{r['r2']:.2f}" if r["r2"] is not None else " - "
        print(f"  {st['amp']:+.0f}     {r['L']*1000:7.0f}    {tau_ms:>6}   {r2s}   {r['vpeak']:.1f}")
        Ls.append(r["L"]);
        if r["tau"]: taus.append(r["tau"])
        st["fit"] = r
    if Ls: print(f"  ---> L_rest mean {np.mean(Ls)*1000:.0f} ms (sigma {np.std(Ls)*1000:.0f})", end="")
    if taus: print(f" ; tau mean {np.mean(taus)*1000:.0f} ms")
    else: print()
    holds = meta.get("holds") or ([meta["hold"]] if meta.get("hold") else [])
    Lr = np.mean(Ls) if Ls else float("nan")
    Lhs = []; stops = []
    vel_s = smooth_vel(T, Q[:, joint])
    for h in holds:
        th = h["t_hold"]; m = (T >= th - 0.25) & (T < th)
        v_pre = float(np.median(np.abs(vel_s[m]))) if m.any() else vel
        t_dec, _ = onset_after(T, [-1 if abs(v) < 0.5 * v_pre else 0 for v in vel_s], 0.5, th)
        t_stop, _ = onset_after(T, [-1 if abs(v) < 0.5 else 0 for v in vel_s], 0.5, th)
        Lh = (t_dec - th) if t_dec else None; h["L_hold"] = Lh; h["v_pre"] = v_pre
        if Lh is not None:
            Lhs.append(Lh); stops.append(t_stop - th if t_stop else np.nan)
            print(f"  HOLD: v_pre {v_pre:.1f} deg/s ; L_hold {Lh*1000:.0f} ms ; stop {(t_stop-th)*1000:.0f} ms")
    if Lhs:
        med = float(np.median(Lhs)); good = [x for x in Lhs if 0.4 * med <= x <= 2.5 * med]
        print(f"  ---> L_hold median {med*1000:.0f} ms over {len(Lhs)} reps ({len(Lhs)-len(good)} glitch-excluded) ; "
              f"stop median {np.nanmedian(stops)*1000:.0f} ms ; **buffer L_hold-L_rest ~= {(med-Lr)*1000:.0f} ms**")
    return T, Q, meta, joint, vel

def plot(T, Q, meta, joint, vel, png):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (axp, axv) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for st in meta["steps"]:
        ta = st["t_anchor"]; m = (T >= ta - 0.2) & (T <= ta + abs(st["amp"]) / vel + 0.8)
        tt = T[m] - ta; pos = Q[m, joint] - st["q0"][joint]
        axp.plot(tt, pos, lw=1.3, label=f"{st['amp']:+.0f}°")
        axv.plot(tt, smooth_vel(T, Q[:, joint])[m], lw=1.1)
        f = st.get("fit")
        if f: axp.axvline(f["t_on"] - ta, color="gray", ls=":", lw=0.8)
    axp.axvline(0, color="crimson", lw=1.2, label="cmd onset (t_anchor)")
    axp.set_ylabel(f"{JN[joint]} Δ (deg)"); axp.grid(alpha=0.25); axp.legend(fontsize=8, loc="lower right")
    axp.set_title(f"step_id {JN[joint]} — dotted = actual-motion onset (gap to red = L_rest)")
    axv.axvline(0, color="crimson", lw=1.2); axv.axhline(vel, color="k", ls=":", lw=0.8, label=f"cmd vel {vel}")
    axv.set_ylabel("velocity (deg/s)"); axv.set_xlabel("time since t_anchor (s)"); axv.grid(alpha=0.25); axv.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(png, dpi=110); plt.close(fig); print(f"  plot -> {png}")
    holds = [h for h in (meta.get("holds") or ([meta["hold"]] if meta.get("hold") else [])) if h.get("L_hold") is not None]
    if holds:
        vel_s = smooth_vel(T, Q[:, joint])
        fig, ax = plt.subplots(figsize=(11, 4))
        for k, h in enumerate(holds):
            th = h["t_hold"]; m = (T >= th - 0.6) & (T <= th + 1.2)
            ax.plot(T[m] - th, vel_s[m], lw=1.3, label=f"rep{k+1} L_hold {h['L_hold']*1000:.0f} ms")
            ax.axvline(h["L_hold"], color="green", lw=0.8, ls="--")
        ax.axvline(0, color="crimson", lw=1.4, label="HOLD sent")
        ax.set_xlabel("time since HOLD (s)"); ax.set_ylabel("velocity (deg/s)"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
        ax.set_title("HOLD reaction (green dashed = decel onset) — L_hold - L_rest = buffer drain"); fig.tight_layout()
        hpng = png.replace(".png", "_hold.png"); fig.savefig(hpng, dpi=110); plt.close(fig); print(f"  plot -> {hpng}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", type=int, default=0, help="0..5 (default J1 base-yaw, safest)")
    ap.add_argument("--amps", default="5,10,15", help="step amplitudes (deg), comma-sep")
    ap.add_argument("--vel-deg", type=float, default=20.0, help="ramp velocity (deg/s, cap 55)")
    ap.add_argument("--hold-delta", type=float, default=25.0, help="ramp size for the HOLD test (deg)")
    ap.add_argument("--hold-test", action="store_true")
    ap.add_argument("--hold-reps", type=int, default=3, help="repeat the HOLD test N times for a mean")
    ap.add_argument("--replot", default="")
    args = ap.parse_args()
    if args.replot:
        T, Q, meta, joint, vel = analyze(args.replot); plot(T, Q, meta, joint, vel, os.path.join(OUT, "step_id.png")); return
    vel = min(args.vel_deg, 55.0); joint = args.joint
    amps = [float(a) for a in args.amps.split(",")]
    print(f"[step_id] {JN[joint]}  amps={amps}  vel={vel} deg/s  hold={args.hold_test}")

    rec = StreamRecorder(); rec.start(); time.sleep(0.8)               # persistent :9999 stream; baseline
    steps = []
    for amp in amps:
        q0 = read_state()[0]
        wps, dt = ramp_chunk(q0, joint, amp, vel)
        t_send, t_anchor = send_ramp(wps, dt); print(f"   step {amp:+.0f}°  (t_anchor set)")
        time.sleep(ANCHOR_LEAD + abs(amp) / vel + 1.0)                 # let it finish + settle
        steps.append({"amp": amp, "q0": list(q0), "t_send": t_send, "t_anchor": t_anchor})
        back, dt = ramp_chunk(read_state()[0], joint, -amp, vel); send_ramp(back, dt)  # return to q0 (unmeasured)
        time.sleep(ANCHOR_LEAD + abs(amp) / vel + 1.0)
    holds = []
    if args.hold_test:
        for rep in range(args.hold_reps):
            q0 = read_state()[0]; wps, dt = ramp_chunk(q0, joint, args.hold_delta, vel)
            t_send, t_anchor = send_ramp(wps, dt)
            time.sleep(ANCHOR_LEAD + 0.7)                              # into steady velocity, still moving
            t_hold = time.time(); send_chunk({"hold": True}); print(f"   HOLD rep {rep+1}/{args.hold_reps} sent mid-motion")
            time.sleep(1.4)
            holds.append({"amp": args.hold_delta, "q0": list(q0), "t_send": t_send, "t_anchor": t_anchor, "t_hold": t_hold})
            cur = read_state()[0]; back, dt = ramp_chunk(cur, joint, q0[joint] - cur[joint], vel)  # ease back to q0
            send_ramp(back, dt); time.sleep(ANCHOR_LEAD + abs(args.hold_delta) / vel + 1.0)
    rec.stop(); time.sleep(0.3)
    T, Q, TQ, _ = rec.arrays()
    meta = {"joint": joint, "vel": vel, "steps": steps, "holds": holds, "anchor_lead": ANCHOR_LEAD}
    sess = os.path.join(OUT, "session.npz")
    np.savez(sess, T=T, Q=Q, TQ=TQ, meta=json.dumps(meta), cmd_log=json.dumps(CMD_LOG))
    print(f"[saved] {sess}  ({len(T)} samples @ ~{len(T)/max(T[-1]-T[0],1e-6):.0f} Hz)")
    T, Q, meta, joint, vel = analyze(sess); plot(T, Q, meta, joint, vel, os.path.join(OUT, "step_id.png"))

if __name__ == "__main__":
    main()
