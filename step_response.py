#!/usr/bin/env python3
"""Measure the command-to-motion DEAD-TIME of the streaming controller, end-to-end.

Streams a small ramped step on ONE joint to online_servo (:9994) anchored at a known
wall-time, logs the :9999 motion feedback, and reports the lag from the commanded
reference onset to the actual motion onset. Run with the controller's --lead = 0 (pure
follower) so the number is the RAW dead-time. Compare before vs after the
elerob_online.hal 20ms->4ms fast-thread switch to see how much it recovers.

    python3 step_response.py            # +4 deg on joint 1, default
    python3 step_response.py --joint 2 --step-deg 5

Safe: small ramped move, capped speed; the welder holds the end position afterward.
"""
import argparse
import json
import os
import socket
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mycobot_mpc")))

PI = "10.0.0.27"


def read_once():
    s = socket.create_connection((PI, 9999), timeout=3); s.settimeout(2); b = b""
    try:
        while b"\n" not in b:
            b += s.recv(4096)
    except Exception:
        pass
    s.close()
    return list(json.loads(b.split(b"\n")[0])["joints_deg"])


def logger(stop, samples):
    """Continuously read :9999, timestamping each sample on arrival."""
    s = socket.create_connection((PI, 9999), timeout=3); s.settimeout(0.5)
    buf = b""
    while not stop.is_set():
        try:
            buf += s.recv(8192)
        except socket.timeout:
            continue
        except Exception:
            break
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            t = time.time()
            try:
                jd = json.loads(line)["joints_deg"]
                samples.append((t, [float(x) for x in jd]))
            except Exception:
                pass
    s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", type=int, default=1, help="joint index 0-5 to step")
    ap.add_argument("--step-deg", type=float, default=4.0, help="step size (deg)")
    ap.add_argument("--ramp", type=float, default=0.10, help="ramp duration of the step (s)")
    ap.add_argument("--settle", type=float, default=2.5, help="log this long after the step (s)")
    ap.add_argument("--onset-deg", type=float, default=0.3, help="motion-onset threshold (deg)")
    ap.add_argument("--out", default=None, help="optional CSV path for the raw trace")
    a = ap.parse_args()

    cur = read_once()
    print(f"current joints_deg = {[round(x,1) for x in cur]}; stepping joint {a.joint} "
          f"by {a.step_deg:+.1f} deg over {a.ramp*1000:.0f} ms")

    # build the ramp chunk (LinuxCNC deg, dt=0.01): current -> current+step, then hold
    dt = 0.01
    n = max(1, int(round(a.ramp / dt)))
    chunk = []
    for i in range(n + 1):
        w = list(cur); w[a.joint] += a.step_deg * (i / n); chunk.append(w)
    chunk += [list(chunk[-1])] * int(round(0.4 / dt))          # hold tail so welder clamps the goal
    vmax = a.step_deg / a.ramp
    print(f"  chunk: {len(chunk)} pts, peak {vmax:.0f} deg/s")

    stop = threading.Event(); samples = []
    th = threading.Thread(target=logger, args=(stop, samples), daemon=True); th.start()
    time.sleep(0.5)                                            # capture a pre-step baseline

    anchor = time.time() + 0.20                                # ramp STARTS here (motion onset reference)
    sock = socket.create_connection((PI, 9994), timeout=3)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.sendall((json.dumps({"trajectory": chunk, "traj_dt": dt, "t_anchor": anchor}) + "\n").encode())
    sock.close()
    print(f"  chunk sent; anchor (commanded onset) at t={anchor:.3f}")

    time.sleep((anchor - time.time()) + a.ramp + a.settle)
    stop.set(); th.join(timeout=1)

    if not samples:
        print("NO feedback samples captured (:9999)"); return
    ts = np.array([s[0] for s in samples])
    qj = np.array([s[1][a.joint] for s in samples])
    base = float(np.median(qj[ts < anchor])) if np.any(ts < anchor) else float(qj[0])
    moved = np.where((ts > anchor) & (np.abs(qj - base) > a.onset_deg))[0]
    if len(moved) == 0:
        print(f"  joint did not move > {a.onset_deg} deg (check controller/stream)"); return
    onset = ts[moved[0]]
    dead_time = onset - anchor
    # time to 50% of the step
    target50 = base + 0.5 * a.step_deg
    reach50 = np.where((ts > anchor) & (np.sign(a.step_deg) * (qj - target50) >= 0))[0]
    t50 = (ts[reach50[0]] - anchor) if len(reach50) else float("nan")
    final = float(np.median(qj[ts > ts[-1] - 0.3]))
    print("\n========== RESULT ==========")
    print(f"  baseline           : {base:.2f} deg")
    print(f"  DEAD-TIME (onset)  : {dead_time*1000:6.0f} ms   <-- set --lead to ~this")
    print(f"  time to 50% step   : {t50*1000:6.0f} ms")
    print(f"  final              : {final:.2f} deg  (target {base + a.step_deg:.2f})")
    print(f"  feedback rate      : {1.0/np.median(np.diff(ts)):.0f} Hz "
          f"({len(samples)} samples) -- bump --stream-rate for finer resolution")
    if a.out:
        np.savetxt(a.out, np.column_stack([ts - anchor, qj]), delimiter=",",
                   header="t_rel_to_anchor_s,joint_deg", comments="")
        print(f"  raw trace -> {a.out}")


if __name__ == "__main__":
    main()
