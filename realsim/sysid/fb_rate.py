#!/usr/bin/env python3
"""fb_rate :: passive measurement of the Pi joint-feedback stream (:9999). Sends nothing.

Reports: Pi-side sample spacing, gaps, Pi->desktop transport, and how often the drive data
actually changes. With the arm still, positions do not change, so freshness is judged on the torque
vector (drive torque feedback is noisy, so a new drive report almost always changes it). The median
run length of identical torque vectors x the stream period = the real source update period.

    python3 realsim/sysid/fb_rate.py --secs 30 [--host 192.168.50.2] [--out fb.json]
"""
import argparse
import json
import socket
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.50.2")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--secs", type=float, default=30)
    ap.add_argument("--out")
    a = ap.parse_args()
    s = socket.create_connection((a.host, a.port), timeout=2)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    buf, tp, tr, q, tau = b"", [], [], [], []
    t0 = time.time()
    while time.time() - t0 < a.secs:
        d = s.recv(65536)
        if not d:
            break
        t_rx = time.time()
        buf += d
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            try:
                m = json.loads(line)
            except ValueError:
                continue
            tp.append(m["timestamp"]); tr.append(t_rx); q.append(m["joints_deg"]); tau.append(m.get("torque", [0] * 6))
    s.close()
    tp, tr, q, tau = map(np.asarray, (tp, tr, q, tau))
    dt = np.diff(tp) * 1e3
    lat = (tr - tp) * 1e3
    tch = np.any(np.diff(tau, axis=0) != 0, axis=1)
    qch = np.any(np.diff(q, axis=0) != 0, axis=1)
    runs, r = [], 1
    for c in tch:
        if c:
            runs.append(r); r = 1
        else:
            r += 1
    per = np.median(dt)
    out = {
        "n": int(len(tp)), "rate_hz": float((len(tp) - 1) / (tp[-1] - tp[0])),
        "pi_dt_ms": {k: float(f(dt)) for k, f in (("median", np.median), ("p95", lambda x: np.percentile(x, 95)),
                                                 ("p99", lambda x: np.percentile(x, 99)), ("max", np.max))},
        "gaps_gt_2x": int((dt > 2 * per).sum()),
        "transport_ms": {k: float(f(lat)) for k, f in (("median", np.median), ("p95", lambda x: np.percentile(x, 95)),
                                                      ("p99", lambda x: np.percentile(x, 99)), ("max", np.max))},
        "torque_changed_frac": float(tch.mean()), "q_changed_frac": float(qch.mean()),
        "torque_run_median": float(np.median(runs)) if runs else None,
        "source_update_ms_est": float(np.median(runs) * per) if runs else None,
        "q_std_deg": [float(x) for x in q.std(0)],
    }
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
