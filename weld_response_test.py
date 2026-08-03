#!/usr/bin/env python3
"""Streaming-response touch test on a GRID of table points.

Per point: WELDED approach->vertical-descent (one motion, non-zero junction, no dead-stop) while a
CONCURRENT D405 cup-region monitor watches the contact gap (the RGBD stream wired into contact
detection). On contact the main thread streams an updated HOLD chunk; we log how fast the robot
stops (drive dead-time) and the overshoot. Then a WELDED lift->return-to-base (one retimed chunk,
no stop) at the approach speed, so the NEXT point is probed fresh from the base pose.

Grid: NxN points centred on --center with --spacing between adjacent points.
Default = 3x3 about [0.4,0.0], 0.2 m apart -> x in {0.2,0.4,0.6}, y in {-0.2,0,0.2}.

A background Recorder logs (t, joints, torque, contact-gap) the whole run; every streamed chunk is
logged too, so we can reconstruct the COMMANDED joint reference and plot cmd-vs-actual + contact
per episode with phase-coloured backgrounds. Raw session saved to npz -> replot without the robot.

  source /opt/ros/humble/setup.bash
  python3 weld_response_test.py                       # 3x3 grid touch + plots
  python3 weld_response_test.py --replot outputs/weld_grid/session.npz   # just remake plots
"""
import argparse, json, os, socket, sys, time, threading
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C
import cup_region as CR
from joint_conventions import rad_to_linuxcnc_deg
from latency_common import (rpc, send_chunk, read_state, qrad, fk, plan_pose as plan,
                            stream_at, to_base, Recorder, DOWN, CMD_LOG)
SER = "218622271300"; W, H = 640, 480
CUP_REGION = os.path.join(C.OUT_DIR, "cup_region_fixed.npz")
OUT = os.path.join(C.OUT_DIR, "weld_grid"); os.makedirs(OUT, exist_ok=True)

# ---- CONCURRENT D405 cup-region contact monitor (the RGBD stream wired into contact) ----
class CupMonitor(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.lock = threading.Lock(); self.stop_evt = threading.Event()
        self._gap = None; self._t = 0.0; self.cup_d = None
        self._hist = __import__("collections").deque(maxlen=8)          # (t, gap) for dgap/dt
    def run(self):
        import pyrealsense2 as rs
        pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(SER)
        cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30); cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
        prof = pipe.start(cfg); scale = prof.get_device().first_depth_sensor().get_depth_scale(); align = rs.align(rs.stream.color)
        cup_m, above_m = CR.load(CUP_REGION); cds = []
        while not self.stop_evt.is_set():
            try:
                f = align.process(pipe.wait_for_frames(1000))
                depth = np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32) * scale
                cd, sd = CR.read_depths(depth, cup_m, above_m)
                if cd is not None: cds.append(cd); cds = cds[-30:]
                cref = float(np.median(cds)) if cds else None
                with self.lock:
                    self.cup_d = cref
                    self._gap = (sd - cref) if (sd is not None and cref is not None) else None
                    self._t = time.time()
                    if self._gap is not None: self._hist.append((self._t, self._gap))
            except Exception: pass
        pipe.stop()
    def gap(self):
        with self.lock: return self._gap, self._t
    SLOPE_MAX = 0.045    # m/s: physical descent ceiling (~4 deg/s tip rate); steeper = depth glitch

    def gap_and_rate(self):
        """Return (gap_robust, dgap/dt in m/s, quality). Uses a THEIL-SEN slope (median of pairwise
        slopes) -- immune to single-sample depth glitches that wreck least-squares -- plus a physical
        rate clamp. Negative slope = approaching. Non-physical/untrusted -> slope 0 (caller falls back
        to the raw g<=0 threshold). gap is the median of the last 3 samples (robust to a spike)."""
        with self.lock:
            pts = list(self._hist)
        if len(pts) < 4: return (pts[-1][1] if pts else None), 0.0, 0.0
        ts = np.array([p[0] for p in pts]); gs = np.array([p[1] for p in pts])
        g = float(np.median(gs[-3:]))                                  # robust current gap
        sl = [(gs[j] - gs[i]) / (ts[j] - ts[i]) for i in range(len(pts)) for j in range(i + 1, len(pts)) if ts[j] - ts[i] > 1e-3]
        if not sl: return g, 0.0, 0.0
        slope = float(np.median(sl))                                   # Theil-Sen: robust to outliers
        if abs(slope) > self.SLOPE_MAX: return g, 0.0, 0.0             # glitch -> untrusted, raw fallback
        icpt = float(np.median(gs - slope * ts)); ss = float(np.sum((gs - gs.mean()) ** 2))
        r2 = float(1 - np.sum((gs - (slope * ts + icpt)) ** 2) / ss) if ss > 1e-12 else 0.0
        return g, slope, r2

def weld_retime(path, junc, v_app, v_des, ramp_deg, dt):
    """Resample the concatenated approach+descent path at uniform dt with a velocity profile that
    CRUISES at v_app, decels to v_des over `ramp_deg` before the junction, then holds v_des ->
    smooth non-zero-velocity junction (no dead-stop). v in deg/s of the fastest joint."""
    seg = np.degrees(np.max(np.abs(np.diff(path, axis=0)), axis=1)); s = np.concatenate([[0], np.cumsum(seg)])
    s_j = s[junc]; total = s[-1]
    def vel(si):
        if si < s_j - ramp_deg: return v_app
        if si < s_j: return v_app + (v_des - v_app) * (si - (s_j - ramp_deg)) / max(ramp_deg, 1e-6)
        return v_des
    def at(si):
        i = np.searchsorted(s, si) - 1; i = min(max(i, 0), len(s) - 2)
        a = (si - s[i]) / max(s[i + 1] - s[i], 1e-9)
        return path[i] * (1 - a) + path[i + 1] * a
    out = [path[0]]; si = 0.0; junc_i = None
    while si < total - 1e-6:
        si = min(si + vel(si) * dt, total); out.append(at(si))
        if junc_i is None and si >= s_j: junc_i = len(out) - 1
    return np.array(out), (junc_i if junc_i is not None else len(out) - 1)

# ---------------------------------------------------------------------------
# Reconstruct the COMMANDED joint reference (linuxcnc deg) at query times from the chunk log.
def build_cmd(cmd_log, ts):
    chunks = []
    for tsend, m in cmd_log:
        if "trajectory" in m: chunks.append(("traj", float(m.get("t_anchor", tsend)), float(m["traj_dt"]), np.array(m["trajectory"], float)))
        elif m.get("hold"):   chunks.append(("hold", float(tsend), None, None))
    chunks.sort(key=lambda c: c[1])
    out = np.full((len(ts), 6), np.nan); last = None
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
            if idx <= 0: cmd = wps[0]
            elif idx >= len(wps) - 1: cmd = wps[-1]
            else:
                i0 = int(idx); a = idx - i0; cmd = wps[i0] * (1 - a) + wps[i0 + 1] * a
            out[k] = cmd; last = cmd
    return out

def phase_spans(ep):
    """(label, t_rel_start, t_rel_end, color) covering approach/descent/hold/lift-return."""
    t0 = ep["t0"]; jt = ep["junc_t"]
    tc = (ep["t_contact"] - t0) if ep.get("t_contact") else ep["dur"]     # descent -> contact, else weld end
    tlr = (ep["t_liftret"] - t0) if ep.get("t_liftret") else tc
    sp = [("approach", 0.0, jt, "#d4e6fb"), ("descent", jt, tc, "#d6f0d2")]
    if ep.get("t_contact"): sp.append(("hold", tc, tlr, "#fbe0cd"))
    sp.append(("lift/return", tlr, ep["t_end"] - t0, "#e8e8e8"))
    return sp

def plot_episode(ep, T, Q, TQ, GAP, cmd_log, path_png, gap_thresh=0.0):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    t0 = ep["t0"]; m = (T >= t0 - 0.3) & (T <= ep["t_end"] + 0.3)
    if m.sum() < 3: print(f"   (no telemetry for {path_png})"); return
    tt = T[m] - t0; q = Q[m]; tq = TQ[m]; gp = GAP[m]
    cmd = build_cmd(cmd_log, T[m])                                        # linuxcnc deg, same convention as q
    spans = phase_spans(ep)
    fig, axs = plt.subplots(7, 1, figsize=(11, 14), sharex=True)
    for ax in axs:
        for _, a, b, col in spans:
            if b > a: ax.axvspan(a, b, color=col, lw=0)
    for j in range(6):
        ax = axs[j]
        ax.plot(tt, cmd[:, j], "--", color="#b30000", lw=1.4, label="cmd")
        ax.plot(tt, q[:, j], "-", color="#003a99", lw=1.4, label="actual")
        ax.set_ylabel(f"J{j+1} (deg)"); ax.grid(alpha=0.25)
        if j == 0: ax.legend(loc="upper right", fontsize=8)
    axg = axs[6]
    axg.plot(tt, gp * 1000.0, "-", color="#5b2d90", lw=1.5, label="contact gap (above_d-cup_d)")
    axg.axhline(gap_thresh * 1000.0, color="k", ls=":", lw=1, label=f"threshold {gap_thresh*1000:.0f} mm")
    if ep.get("t_contact"):
        tc = ep["t_contact"] - t0
        for ax in axs: ax.axvline(tc, color="crimson", lw=1.2, ls="-")
        axg.annotate("CONTACT -> HOLD", (tc, axg.get_ylim()[1]), fontsize=8, color="crimson", ha="left", va="top")
    axg.set_ylabel("gap (mm)"); axg.set_xlabel("time since approach start (s)"); axg.grid(alpha=0.25); axg.legend(loc="upper right", fontsize=8)
    leg = [Patch(facecolor=c, label=l) for l, _, _, c in spans]
    lat = f"{ep['latency']*1000:.0f} ms" if ep.get("latency") else "n/a"
    ov = f"{ep['overshoot']:+.1f} mm" if ep.get("overshoot") is not None else "n/a"
    fig.suptitle(f"touch [{ep['xy'][0]:.2f}, {ep['xy'][1]:+.2f}]   latency {lat}   overshoot {ov}", fontsize=13)
    fig.legend(handles=leg, loc="lower center", ncol=len(leg), fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=[0, 0.03, 1, 0.98]); fig.savefig(path_png, dpi=110); plt.close(fig)
    print(f"   plot -> {path_png}")

def make_plots(sessfile, gap_thresh=0.0):
    d = np.load(sessfile, allow_pickle=True)
    T = d["T"]; Q = d["Q"]; TQ = d["TQ"]; GAP = d["GAP"]
    cmd_log = json.loads(str(d["cmd_log"])); episodes = json.loads(str(d["episodes"]))
    for ep in episodes:
        if ep.get("skipped"): continue
        png = os.path.join(OUT, f"ep{ep['idx']:02d}_x{ep['xy'][0]:.2f}_y{ep['xy'][1]:+.2f}.png".replace("+", "p").replace("-", "m"))
        plot_episode(ep, T, Q, TQ, GAP, cmd_log, png, gap_thresh)

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", default="0.4,0.0", help="grid centre x,y (m)")
    ap.add_argument("--spacing", type=float, default=0.2, help="distance between adjacent grid points (m)")
    ap.add_argument("--grid-n", type=int, default=3, help="grid is N x N")
    ap.add_argument("--v-app", type=float, default=45.0, help="approach cruise (deg/s, fastest joint)")
    ap.add_argument("--v-des", type=float, default=4.0, help="descent speed at contact (deg/s)")
    ap.add_argument("--ramp-deg", type=float, default=12.0, help="decel window before junction (deg)")
    ap.add_argument("--weld-dt", type=float, default=0.05)
    ap.add_argument("--pre-z", type=float, default=0.05, help="pregrasp FK z (above table)")
    ap.add_argument("--desc-z", type=float, default=-0.02, help="descent endpoint FK z (contact stops it first)")
    ap.add_argument("--contact-gap", type=float, default=0.0)
    ap.add_argument("--torque-abort", type=float, default=0.14)
    ap.add_argument("--lead-time", type=float, default=0.196, help="predictive stop: fire HOLD when "
                    "gap+dgap/dt*lead <= contact-gap (lead = measured L_hold ~0.196 s)")
    ap.add_argument("--slope-min", type=float, default=-0.001, help="require dgap/dt < this (m/s, "
                    "i.e. actually approaching) to trust the prediction")
    ap.add_argument("--no-predict", dest="predict", action="store_false", help="disable prediction (raw g<=0 A/B baseline)")
    ap.set_defaults(predict=True)
    ap.add_argument("--creep", action="store_true", help="confirm-creep: stop early (short), then step "
                    "down slowly and seat the cup by gap-response collapse -> light controlled touch")
    ap.add_argument("--creep-approach", type=float, default=0.005, help="fire the fast HOLD at gap<=this (m, +ve=land short)")
    ap.add_argument("--creep-step", type=float, default=0.001, help="creep step down (m)")
    ap.add_argument("--v-creep", type=float, default=1.5, help="creep speed (deg/s)")
    ap.add_argument("--creep-seat", type=float, default=0.0004, help="seated when a step drops gap < this (m)")
    ap.add_argument("--creep-max", type=int, default=8, help="max creep steps")
    ap.add_argument("--replot", default="", help="skip the robot; regenerate plots from this session npz")
    args = ap.parse_args()
    if args.replot:
        make_plots(args.replot, args.contact_gap); return

    cx, cy = [float(v) for v in args.center.split(",")]
    g = args.grid_n; off = (g - 1) / 2.0
    xs = [round(cx + (i - off) * args.spacing, 3) for i in range(g)]
    ys = [round(cy + (j - off) * args.spacing, 3) for j in range(g)]
    pts = [[x, y] for x in xs for y in ys]
    print(f"[grid] {g}x{g} about ({cx},{cy}) spacing {args.spacing} -> {len(pts)} points  x={xs} y={ys}")

    mon = CupMonitor(); mon.start()
    print("[monitor] warming up D405..."); time.sleep(3.0)
    rec = Recorder(gap_fn=lambda: mon.gap()[0], min_dt=0.03); rec.start()
    to_base(args.v_app)                                                # start every probe from base
    episodes = []
    for k, oxy in enumerate(pts):
        print(f"\n=== point {k+1}/{len(pts)}: [{oxy[0]:.3f},{oxy[1]:+.3f}] ===")
        ep = {"idx": k, "xy": oxy, "skipped": False}
        q0 = read_state()[0]
        pa = plan(qrad(q0), [oxy[0], oxy[1], args.pre_z])
        if pa is None: print("   approach plan failed (unreachable) -> skip"); ep["skipped"] = True; episodes.append(ep); continue
        pd = plan(list(pa[-1]), [oxy[0], oxy[1], args.desc_z])
        if pd is None: print("   descent plan failed -> skip"); ep["skipped"] = True; episodes.append(ep); continue
        path = np.vstack([pa, pd[1:]]); junc = len(pa) - 1
        path[:, 5] = float(C.BASE_Q[5])                                 # hold J6 at base
        weld, junc_i = weld_retime(path, junc, args.v_app, args.v_des, args.ramp_deg, args.weld_dt)
        td = [list(map(float, rad_to_linuxcnc_deg(w))) for w in weld]
        # STREAM the whole welded approach->descent as one chunk (no stop at pregrasp)
        t0 = time.time(); send_chunk({"trajectory": td, "traj_dt": args.weld_dt, "t_anchor": t0 + 0.12})
        dur = args.weld_dt * (len(weld) - 1); junc_t = junc_i * args.weld_dt
        ep.update({"t0": t0, "junc_t": junc_t, "dur": dur})
        # concurrently watch the cup-region gap; on contact stream a HOLD (the "updated chunk").
        # PREDICTIVE: fire one dead-time early using dgap/dt so the arm coasts to gap=0 (no overshoot).
        t_contact = None; z_contact = None; fire_mode = None
        while time.time() - t0 < dur + 3.0:
            gap, slope, r2 = mon.gap_and_rate(); _, tq = read_state()
            armed = (time.time() - t0) > junc_t + 0.5                   # only arm in the descent phase
            fire = False; mode = None
            thr = args.creep_approach if args.creep else args.contact_gap   # creep: stop early (short)
            use_predict = args.predict and not args.creep
            if armed and t_contact is None and gap is not None:
                proj = gap + slope * args.lead_time
                if use_predict and slope < args.slope_min and r2 > 0.5 and proj <= thr:
                    fire, mode = True, "predict"
                elif gap <= thr:
                    fire, mode = True, ("creep-appr" if args.creep else "raw")
            if fire:
                t_contact = time.time(); z_contact = fk(read_state()[0])[2]; fire_mode = mode
                send_chunk({"hold": True})
                print(f"   [CONTACT/{mode}] gap {gap:+.3f} slope {slope*1000:+.1f}mm/s -> HOLD; FK z={z_contact:.3f}")
                break
            if abs(tq[2]) > args.torque_abort: send_chunk({"hold": True}); print("   torque abort"); break
            time.sleep(0.01)
        if t_contact is None: send_chunk({"hold": True}); print("   no contact (weld finished) -> hold")
        # measure response: watch tip z settle after the HOLD
        zs = []; t_stop = None
        for _ in range(120):
            z = fk(read_state()[0])[2]; zs.append((time.time(), z)); time.sleep(0.02)
            if len(zs) >= 6 and abs(zs[-1][1] - zs[-6][1]) < 0.0003: t_stop = zs[-1][0]; break
        z_rest = zs[-1][1] if zs else None
        lat = (t_stop - t_contact) if (t_stop and t_contact) else None
        over = (z_contact - z_rest) * 1000 if (z_contact is not None and z_rest is not None) else None
        time.sleep(0.25); gap_rest, _, _ = mon.gap_and_rate()            # honest press: <0 past surface, >0 short
        if gap_rest is not None and abs(gap_rest) > 0.05: gap_rest = None  # reject depth glitch (e.g. +293mm)
        gr_mm = gap_rest * 1000 if gap_rest is not None else None
        ep.update({"t_contact": t_contact, "z_contact": z_contact, "z_rest": z_rest, "latency": lat,
                   "overshoot": over, "fire_mode": fire_mode, "gap_rest_mm": gr_mm})
        if lat is not None: print(f"   RESPONSE[{fire_mode}]: latency={lat*1000:.0f} ms  coast={over:+.1f} mm  "
                                  f"gap_rest={gr_mm:+.1f} mm (0=landed, <0=pressed past)" if gr_mm is not None else "")
        # CONFIRM-CREEP: from the short stop, step down slowly and seat the cup by gap-response collapse
        # (settled gap is clean, sigma~0.4mm; a step that barely drops the gap = the cup has seated).
        if args.creep and t_contact is not None:
            prev_g = gap_rest; cn = 0; below = 0; it = 0
            while cn < args.creep_max and it < 24:
                it += 1; g, _, _ = mon.gap_and_rate()
                if g is None: break
                if g <= args.contact_gap:                              # confirm by 2 consecutive reads
                    below += 1
                    if below >= 2: print(f"   creep: gap {g*1000:+.1f}mm <= target (confirmed)"); break
                    time.sleep(0.2); continue                          # re-read without descending
                below = 0
                if prev_g is not None and cn > 0 and abs(prev_g - g) < args.creep_seat:
                    print(f"   creep: seated (|dgap| {abs(prev_g-g)*1000:.2f}mm < {args.creep_seat*1000:.1f})"); break
                curq = read_state()[0]; cz = fk(curq)[2]
                tr = plan(qrad(curq), [oxy[0], oxy[1], cz - args.creep_step])
                if tr is None: print("   creep: plan failed"); break
                prev_g = g; stream_at(tr, args.v_creep); cn += 1
                _, tq = read_state()
                if abs(tq[2]) > args.torque_abort: send_chunk({"hold": True}); print("   creep: torque abort"); break
            send_chunk({"hold": True}); time.sleep(0.35)
            gf, _, _ = mon.gap_and_rate()
            if gf is not None and abs(gf) > 0.05: gf = None
            gr_mm = gf * 1000 if gf is not None else None
            ep["gap_rest_mm"] = gr_mm; ep["creep_n"] = cn
            print(f"   CREEP {cn} steps -> seat gap_rest={gr_mm:+.1f} mm" if gr_mm is not None else f"   CREEP {cn} steps (gap glitch)")
        # WELDED lift -> return-to-base: concatenate into ONE retimed chunk (no stop at the lift pose),
        # at the approach speed -> next point is probed fresh from base
        ep["t_liftret"] = time.time()
        lift = plan(qrad(read_state()[0]), [oxy[0], oxy[1], 0.12])
        if lift is not None:
            rj = rpc({"type": "plan_joint", "start_q": list(map(float, lift[-1])), "goal_q": [float(v) for v in C.BASE_Q], "max_attempts": 12})
            if rj.get("success"): stream_at(np.vstack([lift, np.array(rj["trajectory"])[1:]]), args.v_app)   # one welded chunk
            else: stream_at(lift, args.v_app); to_base(args.v_app)
        else: to_base(args.v_app)
        ep["t_end"] = time.time(); episodes.append(ep)

    rec.stop_evt.set(); mon.stop_evt.set(); time.sleep(0.5)
    # ---- save raw session (arrays + logs) BEFORE plotting so data always survives ----
    T = np.array(rec.ts); Q = np.array(rec.qs); TQ = np.array(rec.tqs); GAP = np.array(rec.gaps)
    sess = os.path.join(OUT, "session.npz")
    np.savez(sess, T=T, Q=Q, TQ=TQ, GAP=GAP, cmd_log=json.dumps(CMD_LOG), episodes=json.dumps(episodes))
    print(f"\n[saved] {sess}  ({len(T)} telemetry samples, {len(CMD_LOG)} chunks)")

    mode = "PREDICT" if args.predict else "RAW threshold"
    print(f"\n============ GRID TOUCH: WELDED-REACTIVE RESPONSE ({mode}, lead={args.lead_time*1000:.0f}ms) ============")
    print("  pt  xy                 fire     latency   gap_rest   note")
    lats = []; grs = []
    for ep in episodes:
        if ep.get("skipped"): print(f"  {ep['idx']+1:2d}  [{ep['xy'][0]:.3f},{ep['xy'][1]:+.3f}]     -         -         -      UNREACHABLE"); continue
        ls = f"{ep['latency']*1000:.0f}ms" if ep.get('latency') else " none"
        gr = f"{ep['gap_rest_mm']:+.1f}mm" if ep.get('gap_rest_mm') is not None else " -"
        fm = ep.get("fire_mode") or "-"
        note = "" if ep.get("t_contact") else "no-contact"
        print(f"  {ep['idx']+1:2d}  [{ep['xy'][0]:.3f},{ep['xy'][1]:+.3f}]   {fm:>7}   {ls:>7}   {gr:>8}   {note}")
        if ep.get('latency'): lats.append(ep['latency'])
        if ep.get('gap_rest_mm') is not None: grs.append(ep['gap_rest_mm'])
    if lats: print(f"  latency  : mean {np.mean(lats)*1000:.0f} ms  std {np.std(lats)*1000:.0f} ms")
    if grs: print(f"  gap_rest : mean {np.mean(grs):+.1f} mm  std {np.std(grs):.1f} mm   (0=landed on surface; goal |gap_rest|->0)")
    print("done; arm at base. generating plots...")
    make_plots(sess, args.contact_gap)

if __name__ == "__main__":
    main()
