#!/usr/bin/env python3
"""realsim.sysid.pi_probe :: small, guarded identification runs on the real Pro 630 (robot_hal on the Pi).

Everything goes through robot_hal's streamed-reference path (:9998), whose law is
    u = vff*v_ref - K0*(q - q_ref) - K1*(qd - v_ref),   vel_cmd = vel_scale * u   -> drive velocity command
so   K0 = K1 = 0, vff = 1   is the BARE drive: the velocity command is the slope of the reference, no feedback.

Guards (desktop side, on the 100 Hz :9999 joint stream):
  * the probed joint may not leave  start +- --box  deg, the other joints start +- --box_other deg
  * any violation, or a stalled stream, sends a closed-loop hold at the START pose (ends the stream on the Pi)
The Pi's own 100 Hz log (q, velocity command, velfb, torqfb, t_epoch) is fetched afterwards.

  python3 realsim/sysid/pi_probe.py zero  --secs 3                       # all joints, zero velocity command
  python3 realsim/sysid/pi_probe.py check --joint 0                      # closed-loop +1 deg / back (deaf-drive check)
  python3 realsim/sysid/pi_probe.py bare  --joint 0 --vels 1 2 5         # open-loop velocity pulses
"""
import argparse
import base64
import json
import os
import socket
import threading
import time

import numpy as np

PI = "192.168.50.2"
VS = 17.0                                    # drive units per deg/s (2026-09-17)
CL_GAINS = dict(k0=10.0, k1=0.3, vff=1.0, vmax=20.0, vel_scale=VS, lead=0.045)      # deployed stream gains, slow clamp


class Pi:
    def __init__(self):
        self.cmd = socket.create_connection((PI, 9998), timeout=3)
        self.cmd.settimeout(0.2)
        self.buf = b""
        self.status = {}
        self.st = socket.create_connection((PI, 9999), timeout=3)
        self.q, self.tq, self.t_rx, self.rows = None, None, 0.0, []
        self.rec = False
        threading.Thread(target=self._stream, daemon=True).start()
        t0 = time.time()
        while self.q is None and time.time() - t0 < 3:
            time.sleep(0.01)
        if self.q is None:
            raise SystemExit("no joint stream on :9999")

    def _stream(self):
        f = self.st.makefile("r")
        for line in f:
            try:
                m = json.loads(line)
            except ValueError:
                continue
            self.q, self.tq, self.t_rx = np.array(m["joints_deg"]), np.array(m["torque"]), time.time()
            if self.rec:
                self.rows.append([m["timestamp"], time.time(), *m["joints_deg"], *m["torque"]])

    def poll(self):
        try:
            self.buf += self.cmd.recv(1 << 16)
        except socket.timeout:
            pass
        out = []
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            try:
                m = json.loads(line)
            except ValueError:
                continue
            out.append(m)
            if m.get("state") in ("idle", "moving", "streaming", "done"):
                self.status = m
        return out

    def send(self, obj):
        self.cmd.sendall((json.dumps(obj) + "\n").encode())

    def hold(self, q, why):
        print(f"[guard] {why} -> closed-loop hold at {np.round(q, 2).tolist()}", flush=True)
        self.send({"target_deg": [float(x) for x in q], "duration": 3.0, "controller": "mpc",
                   "gains": {"k0": 6.0, "k1": 0.0, "vmax": 10.0, "vel_scale": VS}, "vel_cmd_max": 10.0 * VS})

    def get_log(self, name, path):
        self.send({"get_log": name})
        t0 = time.time()
        self.cmd.settimeout(2.0)
        while time.time() - t0 < 20:
            for m in self.poll():
                if m.get("state") == "log":
                    open(path, "wb").write(base64.b64decode(m["log_content_base64"]))
                    self.cmd.settimeout(0.2)
                    return True
                if m.get("state") == "log_error":
                    return False
        return False


def run_stream(pi, ref, gains, vmax, box, tag, out):
    """ref [N,6] deg at 10 ms. Returns the fetched Pi log path (or None)."""
    q0 = pi.q.copy()
    if np.abs(ref[0] - q0).max() > 0.5:
        raise SystemExit(f"reference starts {np.abs(ref[0] - q0).max():.2f} deg from the arm")
    stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + tag
    pi.poll()
    pi.rows, pi.rec = [], True
    t_anchor = time.time() + 0.4
    for i in range(0, len(ref), 200):                       # 2 s pieces, all sent ahead of time
        pi.send({"chunk": ref[i:i + 201].tolist(), "traj_dt": 0.01, "t_anchor": t_anchor + i * 0.01, "seq": i // 200,
                 "gains": gains, "vel_cmd_max": float(vmax * VS * 1.05 + 5), "log_stamp": stamp})
    t_end = t_anchor + len(ref) * 0.01 + 3.0
    aborted, done = None, None
    while time.time() < t_end:
        for m in pi.poll():
            if m.get("state") == "done" and m.get("controller") == "stream" and stamp in (m.get("last_log_name") or ""):
                done = m
        if done:
            break
        dev = np.abs(pi.q - q0)
        if aborted is None:
            if (dev > box).any():
                aborted = f"joint {int(np.argmax(dev - box)) + 1} left its box ({dev.max():.2f} deg)"
            elif time.time() - pi.t_rx > 0.3:
                aborted = "joint stream stalled"
            if aborted:
                pi.hold(q0, aborted)
    pi.rec = False
    time.sleep(0.3)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, f"{stamp}_stream.npy"), np.array(pi.rows))
    np.save(os.path.join(out, f"{stamp}_ref.npy"), ref)
    json.dump(dict(tag=tag, gains=gains, vmax=vmax, t_anchor=t_anchor, q0=q0.tolist(), aborted=aborted, done=done),
              open(os.path.join(out, f"{stamp}_meta.json"), "w"), indent=1)
    name = (done or {}).get("last_log_name") or pi.status.get("last_log_name")
    path = None
    if name:
        path = os.path.join(out, f"{stamp}_pi.csv")
        if not pi.get_log(name, path):
            path = None
    print(f"[run] {tag}: {'ABORTED ' + aborted if aborted else 'ok'}; end dev {np.round(pi.q - q0, 2).tolist()}; "
          f"log {path}", flush=True)
    return path, aborted


def vel_profile(segs):
    """[(v deg/s, seconds), ...] -> velocity per 10 ms tick"""
    return np.concatenate([np.full(int(round(T / 0.01)), float(v)) for v, T in segs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["zero", "check", "bare", "ramp", "sine"])
    ap.add_argument("--joint", type=int, default=0, help="0-based")
    ap.add_argument("--secs", type=float, default=3.0)
    ap.add_argument("--vels", type=float, nargs="+", default=[1.0, 2.0, 5.0])
    ap.add_argument("--disp", type=float, default=4.0, help="max displacement of one pulse (deg)")
    ap.add_argument("--box", type=float, default=8.0)
    ap.add_argument("--box_other", type=float, default=2.0)
    ap.add_argument("--out", default="~/pnp_real_runs/sysid")
    ap.add_argument("--amp", type=float, default=4.0)
    ap.add_argument("--freq", type=float, default=0.5)
    ap.add_argument("--cycles", type=float, default=5.0)
    ap.add_argument("--joints", type=int, nargs="+", default=None, help="sine: joints moved together (0-based)")
    ap.add_argument("--gains", default="", help="sine: json overriding the deployed stream gains")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    pi = Pi()
    q0 = pi.q.copy()
    print(f"[pi] q = {np.round(q0, 3).tolist()}  torque = {np.round(pi.tq, 4).tolist()}  state {pi.status.get('state')}", flush=True)
    box = np.full(6, a.box_other)
    bare = dict(k0=0.0, k1=0.0, vff=1.0, vel_scale=VS, lead=0.0)
    if a.mode == "zero":
        v = np.zeros((int(a.secs * 100), 6))
        gains, vmax, tag = dict(bare, vmax=1.0), 1.0, "zero"
    elif a.mode == "check":
        V = a.vels[0] if a.tag else 2.0                     # --tag given: step test, --vels V, --disp D
        D = a.disp if a.tag else 1.0
        prof = vel_profile([(0, 0.5), (V, D / V), (0, 1.0), (-V, D / V), (0, 1.0)])
        v = np.zeros((len(prof), 6)); v[:, a.joint] = prof
        gains = dict(CL_GAINS, vmax=max(20.0, 1.5 * V))
        if a.gains:
            gains.update(json.loads(a.gains))
        vmax, tag = gains["vmax"], f"check_j{a.joint}" + (f"_{a.tag}" if a.tag else "")
        box[a.joint] = D + 2.0
    elif a.mode == "bare":
        segs = [(0, 1.0)]
        for s in a.vels:
            T = min(1.0, a.disp / s)
            segs += [(s, T), (0, 0.7), (-s, T), (0, 0.7)]
        prof = vel_profile(segs)
        v = np.zeros((len(prof), 6)); v[:, a.joint] = prof
        gains, vmax, tag = dict(bare, vmax=max(a.vels)), max(a.vels), f"bare_j{a.joint}"
        box[a.joint] = a.box
    elif a.mode == "sine":                                   # closed-loop tracking test: faded sine about the start pose
        n = int(a.cycles / a.freq * 100)
        tt = np.arange(n) * 0.01
        env = np.clip(np.minimum(tt, tt[-1] - tt) * a.freq, 0, 1)
        env = 0.5 - 0.5 * np.cos(np.pi * env)
        pos = a.amp * env * np.sin(2 * np.pi * a.freq * tt)
        js = a.joints if a.joints is not None else [a.joint]
        v = np.zeros((n + 50, 6))
        for jj in js:
            v[:n, jj] = np.gradient(pos, 0.01)
            box[jj] = a.amp + 3.0
        gains = dict(CL_GAINS, vmax=40.0)
        if a.gains:
            gains.update(json.loads(a.gains))
        vmax, tag = gains["vmax"], f"sine_{''.join(str(x) for x in js)}_{a.tag}"
    else:                                                    # slow velocity ramp through zero: stiction / deadband
        up = np.linspace(0, 1.5, 300)
        tri = np.concatenate([up, up[::-1]])
        prof = np.concatenate([np.zeros(50), tri, np.zeros(50), -tri, np.zeros(50), -tri, np.zeros(50), tri, np.zeros(50)])
        v = np.zeros((len(prof), 6)); v[:, a.joint] = prof
        gains, vmax, tag = dict(bare, vmax=2.0), 2.0, f"ramp_j{a.joint}"
        box[a.joint] = a.box
    ref = q0[None] + np.cumsum(v, 0) * 0.01
    ref = np.vstack([q0[None], ref])
    run_stream(pi, ref, gains, vmax, box, tag, out)


if __name__ == "__main__":
    main()
