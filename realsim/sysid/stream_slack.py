#!/usr/bin/env python3
"""realsim.sysid.stream_slack :: L0 of the command-latency plan (docs/runlog_latency_plan.html).

Streams a slow free-space J1 sine through the EXECUTOR's own path (rl/real_policy_ctrl.PiLink,
ref_mode "spline": one 5-point B-spline segment per 100 ms decision, anchored at t_k - 2 dt), with an
emulated host-processing delay between the decision time t_k and the send (--proc_ms: 30 = today's
executor, 5 = after L1 "send before encode"). Records every send, every Pi ack (Pi t_recv), the 100 Hz
feedback and the Pi's per-tick CSV, then reports

  slack_k   = t_end_k - LEAD - t_recv_{k+1}      (Pi clock; t_end_k = t_anchor_k + 4 dt, the end of the
                                                welded reference before the next message lands)
  starved   = Pi ticks with t_tick + LEAD >= t_end of the newest welded message (Pi holds, v_ref = 0)
  transport = t_recv - (t_send - clock_offset)  (clock_offset = PiLink's min(rx - t_pi))
  ripple    = 10 Hz component of cmd_vel (J1) relative to its mean |cmd_vel|
  tracking  = rms |q - target| on the Pi (deg)

Safety: refuses unless the arm is within --start_tol of START_Q; moves J1 only, +-amp deg; no suction.

    PY=~/miniconda3/envs/ilpolicy/bin/python
    $PY realsim/sysid/stream_slack.py --proc_ms 30 --exec
    $PY realsim/sysid/stream_slack.py --analyse ~/pnp_real_runs/stream_slack/<run>
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "rl")]
import real_policy_ctrl as rpc  # noqa: E402
from real_policy_ctrl import PiLink, START_Q  # noqa: E402

GAINS = {k: getattr(rpc, n) for k, n in (("deployed", "STREAM_GAINS"), ("tuned", "STREAM_GAINS_TUNED"),
                                         ("fast", "STREAM_GAINS_FAST"), ("fast5", "STREAM_GAINS_FAST_5MS")) if hasattr(rpc, n)}


def drain(link, seen_fb, fb, acks_out, seen_ack):
    with link.lock:
        for s in link.samples:
            if s[0] > seen_fb[0]:
                fb.append((s[0], s[1], *s[2], *s[3]))
        if link.samples:
            seen_fb[0] = link.samples[-1][0]
    for t_host, m in list(link.acks):
        key = (m.get("seq"), m.get("t_recv"))
        if key not in seen_ack:
            seen_ack.add(key)
            acks_out.append({"t_host": t_host, **m})


def run(a):
    dt = 0.1
    rpc.STREAM_GAINS = dict(GAINS[a.stream])
    link = PiLink(a.pi, a.exec, ref_mode="spline")
    t_w = time.time()
    while not link.ok() and time.time() - t_w < 6:
        time.sleep(0.1)
    if not link.ok():
        raise SystemExit("no Pi link (:9999 feedback and :9998 commands)")
    time.sleep(1.0)
    q0, _, age = link.state()
    dev = np.degrees(np.abs(q0 - START_Q)).max()
    print(f"[slack] q0 {np.round(np.degrees(q0), 2)} deg, {dev:.1f} deg from START_Q, fb age {1e3 * age:.0f} ms, "
          f"clock_offset {1e3 * link.clock_offset:.2f} ms", flush=True)
    if dev > a.start_tol:
        raise SystemExit(f"arm is {dev:.1f} deg from START_Q (> {a.start_tol}); move it home first")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(os.path.expanduser(a.out), f"{stamp}_p{int(a.proc_ms)}")
    os.makedirs(out, exist_ok=True)
    fb, acks, cmds, seen_ack, seen_fb = [], [], [], set(), [0.0]
    amp, w = np.radians(a.amp), 2 * np.pi / a.period
    n = int(round(a.secs / dt))
    t0 = time.time() + 0.3
    q_prev = q0.copy()
    for k in range(n + 10):                               # 10 extra decisions: hold at q0
        t_k = t0 + k * dt
        while time.time() < t_k:
            time.sleep(0.0005)
        if a.proc_ms > 0:                                 # emulated host processing (image read/encode, IK, guards)
            t_busy = t_k + a.proc_ms * 1e-3
            while time.time() < t_busy:
                pass
        tt = (k + 1) * dt                                 # target is the reference at t_k + dt
        q_t = q0.copy()
        if k < n:
            q_t[0] = q0[0] + amp * np.sin(w * tt) * min(1.0, tt / 1.0) * min(1.0, max(0.0, (n * dt - tt)) / 1.0)
        t_send = time.time()
        link.send_segment(q_prev, q_t, t_k, dt=dt, seq=k, tag=f"slack:{k}", hold=False)
        cmds.append({"k": k, "t_k": t_k, "t_send": t_send, "clock_offset": link.clock_offset,
                     "t_anchor_pi": link.spline_t0 + round((t_k - 2 * dt - link.spline_t0) / dt) * dt - link.clock_offset,
                     "q1_deg": float(np.degrees(q_t[0]))})
        q_prev = q_t
        drain(link, seen_fb, fb, acks, seen_ack)
    t_end = time.time() + 1.5
    while time.time() < t_end:
        drain(link, seen_fb, fb, acks, seen_ack)
        time.sleep(0.02)
    link.close()
    meta = {"stream": a.stream, "gains": rpc.STREAM_GAINS, "proc_ms": a.proc_ms, "amp_deg": a.amp, "period": a.period,
            "secs": a.secs, "dt": dt, "stamp": stamp}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    json.dump(cmds, open(os.path.join(out, "cmds.json"), "w"))
    json.dump(acks, open(os.path.join(out, "acks.json"), "w"))
    np.save(os.path.join(out, "fb.npy"), np.array(fb))
    time.sleep(1.0)                                       # Pi writes its CSV when the stream ends
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", f"pi@{a.pi}", "ls -t ~/Desktop/mpc/logs/robot_hal_policy_*.csv | head -1"],
                       capture_output=True, text=True, timeout=15)
    name = r.stdout.strip()
    if name:
        subprocess.run(["scp", "-q", f"pi@{a.pi}:{name}", os.path.join(out, "pi.csv")], timeout=30)
    print(f"[slack] saved {out} (pi log {os.path.basename(name) or 'MISSING'})", flush=True)
    return out


def analyse(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    cmds = json.load(open(os.path.join(d, "cmds.json")))
    acks = [x for x in json.load(open(os.path.join(d, "acks.json"))) if str(x.get("tag", "")).startswith("slack:")]
    lead, dt = float(meta["gains"].get("lead", 0.04)), meta["dt"]
    by_k = {c["k"]: c for c in cmds}
    ak = sorted(((int(x["seq"]), float(x["t_recv"]), float(x["t_host"])) for x in acks if x.get("seq") is not None))
    ks = np.array([x[0] for x in ak]); trecv = np.array([x[1] for x in ak]); thost_ack = np.array([x[2] for x in ak])
    t_end_pi = np.array([by_k[k]["t_anchor_pi"] + 4 * dt for k in ks])
    tsend = np.array([by_k[k]["t_send"] for k in ks]); off = np.array([by_k[k]["clock_offset"] for k in ks])
    tk = np.array([by_k[k]["t_k"] for k in ks])
    slack = (t_end_pi[:-1] - lead - trecv[1:]) * 1e3
    transport = (trecv - (tsend - off)) * 1e3
    rtt = (thost_ack - tsend) * 1e3
    res = {"run": os.path.basename(d), "proc_ms": meta["proc_ms"], "n_cmds": len(cmds), "n_acks": int(len(ks)),
           "missing_acks": int(len(cmds) - len(ks)),
           "slack_ms": {"min": float(slack.min()), "p5": float(np.percentile(slack, 5)), "median": float(np.median(slack))},
           "send_after_tk_ms": {"median": float(np.median((tsend - tk) * 1e3)), "max": float(((tsend - tk) * 1e3).max())},
           "transport_ms": {"median": float(np.median(transport)), "p99": float(np.percentile(transport, 99)), "max": float(transport.max())},
           "send_to_ack_rtt_ms": {"median": float(np.median(rtt)), "p99": float(np.percentile(rtt, 99)), "max": float(rtt.max())}}
    p = os.path.join(d, "pi.csv")
    if os.path.exists(p):
        rows = list(csv.DictReader(open(p)))
        tt = np.array([float(r["t_epoch"]) for r in rows])
        q = np.array([[float(r[f"q{j}"]) for j in range(6)] for r in rows])
        tg = np.array([[float(r[f"target{j}"]) for j in range(6)] for r in rows])
        cv = np.array([float(r["cmd_vel0"]) for r in rows])
        idx = np.searchsorted(trecv, tt, side="right") - 1          # newest welded message at each tick
        valid = idx >= 0
        last_msg = len(trecv) - 1
        live = valid & (idx < last_msg)                              # exclude the end of the stream
        starved = live & (tt + lead >= t_end_pi[np.clip(idx, 0, None)])
        per = np.diff(tt) * 1e3
        seg = cv[live]
        spec = np.abs(np.fft.rfft(seg - seg.mean())) if len(seg) > 64 else None
        if spec is not None:
            f = np.fft.rfftfreq(len(seg), np.median(np.diff(tt)))
            i10 = np.argmin(np.abs(f - 10.0))
            res["ripple_10hz_rel"] = float(spec[max(0, i10 - 1):i10 + 2].max() / (np.abs(seg).mean() * len(seg) / 2 + 1e-9))
        res.update({"pi_ticks": int(len(tt)), "pi_tick_ms": {"median": float(np.median(per)), "max": float(per.max())},
                    "starved_ticks": int(starved.sum()),
                    "track_rms_deg_j1": float(np.sqrt(np.mean((q[live, 0] - tg[live, 0]) ** 2))),
                    "track_max_deg_j1": float(np.abs(q[live, 0] - tg[live, 0]).max())})
    else:
        res["pi_csv"] = "missing"
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(d, "result.json"), "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyse")
    ap.add_argument("--proc_ms", type=float, default=30.0)
    ap.add_argument("--stream", choices=list(GAINS), default="fast" if "fast" in GAINS else "tuned")
    ap.add_argument("--amp", type=float, default=4.0, help="J1 amplitude, deg")
    ap.add_argument("--period", type=float, default=4.0)
    ap.add_argument("--secs", type=float, default=30.0)
    ap.add_argument("--start_tol", type=float, default=8.0)
    ap.add_argument("--pi", default="192.168.50.2")
    ap.add_argument("--out", default="~/pnp_real_runs/stream_slack")
    ap.add_argument("--exec", action="store_true")
    a = ap.parse_args()
    if a.analyse:
        analyse(os.path.expanduser(a.analyse)); return
    d = run(a)
    if a.exec:
        analyse(d)


if __name__ == "__main__":
    main()
