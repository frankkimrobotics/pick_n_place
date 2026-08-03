#!/usr/bin/env python3
"""Shared low-level helpers for the MyCobot streaming-controller latency work.

Sockets to the planner RPC (:9997), the LinuxCNC command port (:9994) and feedback (:9999),
FK / plan wrappers, a velocity-retiming streamer, and a high-rate telemetry Recorder. Imported by
weld_response_test.py (benchmark) and step_id.py (Phase-0 system ID) so there is one copy.
"""
import json, socket, time, threading
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "mycobot_mpc")))
import config as C
from geometry import R_from_two_axes, R_to_quat_wxyz
from joint_conventions import linuxcnc_deg_to_rad, rad_to_linuxcnc_deg

PI = "10.0.0.27"; PLANNER = ("127.0.0.1", 9997); CMD_PORT = 9994; FB_PORT = 9999
DOWN = list(map(float, R_to_quat_wxyz(R_from_two_axes(np.array([0, 0, -1.0])))))
CMD_LOG = []                                        # [(t_send, msg_dict), ...] for cmd-vs-actual

def rpc(d):
    s = socket.create_connection(PLANNER, timeout=40); s.sendall((json.dumps(d) + "\n").encode()); b = b""
    while not b.endswith(b"\n"): b += s.recv(65536)
    s.close(); return json.loads(b)
class Sender:
    """Persistent, auto-reconnecting TCP sender to the LinuxCNC command port (:9994).

    online_servo._chunk_client reads newline-delimited messages in a loop on ONE connection, so a
    single held-open socket (TCP_NODELAY) beats connect-per-send: no per-message TCP handshake, no
    per-message server-thread spawn (which can reorder a chunk vs a following HOLD), and no per-send
    connect jitter. Survives a LinuxCNC restart by transparently reconnecting on the next send."""
    def __init__(self, addr=(PI, CMD_PORT)):
        self.addr = addr; self.sock = None; self.lock = threading.Lock()
    def _connect(self):
        self.sock = socket.create_connection(self.addr, timeout=3)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    def send(self, m):
        data = (json.dumps(m) + "\n").encode()
        with self.lock:
            for attempt in (1, 2):                   # one transparent reconnect+resend on failure
                try:
                    if self.sock is None: self._connect()
                    self.sock.sendall(data); return True
                except OSError:
                    try:
                        if self.sock: self.sock.close()
                    except OSError: pass
                    self.sock = None
                    if attempt == 2: raise
        return False
    def close(self):
        with self.lock:
            if self.sock:
                try: self.sock.close()
                except OSError: pass
                self.sock = None

_SENDER = Sender()
def send_chunk(m):
    CMD_LOG.append((time.time(), m))                # log every command
    _SENDER.send(m)
def close_sender(): _SENDER.close()
def read_state():
    s = socket.create_connection((PI, FB_PORT), timeout=3); b = b""
    while b"\n" not in b: b += s.recv(4096)
    s.close(); d = json.loads(b.split(b"\n")[0]); return list(d["joints_deg"]), list(d.get("torque", [0] * 6))
def qrad(qd): return [float(v) for v in linuxcnc_deg_to_rad(qd)]
def fk(qd): return np.array(rpc({"type": "fk", "q": qrad(qd)})["pos"][0])
def plan_pose(start_q, xyz, quat=None):
    r = rpc({"type": "plan_pose", "start_q": start_q, "goal_pose": list(xyz) + (list(quat) if quat else DOWN), "max_attempts": 16})
    return np.array(r["trajectory"]) if r.get("success") else None
def plan_joint(start_q, goal_q):
    r = rpc({"type": "plan_joint", "start_q": start_q, "goal_q": [float(v) for v in goal_q], "max_attempts": 12})
    return np.array(r["trajectory"]) if r.get("success") else None
def stream_at(traj, v_deg, hold_j6=True, settle=True):
    """Stream a joint (rad) trajectory retimed so the fastest joint moves at v_deg (deg/s)."""
    traj = np.array(traj, float)
    if hold_j6: traj[:, 5] = float(C.BASE_Q[5])
    seg = np.degrees(np.max(np.abs(np.diff(traj, axis=0)), axis=1)) if len(traj) > 1 else np.array([1.0])
    dt = max(float(np.max(seg)) / max(v_deg, 1e-6), 0.01)
    send_chunk({"trajectory": [list(map(float, rad_to_linuxcnc_deg(w))) for w in traj], "traj_dt": dt, "t_anchor": time.time() + 0.12})
    if settle: time.sleep(dt * (len(traj) - 1) + 1.5)
    return dt
def to_base(v_deg):
    t = plan_joint(qrad(read_state()[0]), C.BASE_Q)
    if t is not None: stream_at(t, v_deg)

class Recorder(threading.Thread):
    """High-rate telemetry: polls :9999 as fast as it can (or every min_dt s), logs (t, q[6], tq[6]).
    Optional gap_fn() pulls an external scalar (e.g. the contact gap) alongside each sample."""
    def __init__(self, gap_fn=None, min_dt=0.0):
        super().__init__(daemon=True); self.gap_fn = gap_fn; self.min_dt = min_dt
        self.stop_evt = threading.Event(); self.ts = []; self.qs = []; self.tqs = []; self.gaps = []
    def run(self):
        while not self.stop_evt.is_set():
            try:
                q, tq = read_state(); g = self.gap_fn() if self.gap_fn else np.nan
                self.ts.append(time.time()); self.qs.append(list(q)); self.tqs.append(list(tq))
                self.gaps.append(float(g) if g is not None else np.nan)
            except Exception: pass
            if self.min_dt: time.sleep(self.min_dt)
    def stop(self): self.stop_evt.set()
    def arrays(self): return np.array(self.ts), np.array(self.qs), np.array(self.tqs), np.array(self.gaps)

class StreamRecorder(threading.Thread):
    """Holds :9999 open and timestamps every pushed feedback line at its native rate (~46 Hz), with
    no per-sample connect. Receive-time stamps carry a ~constant offset that cancels in deltas
    (L, tau, L_hold). Preferred over Recorder for system ID."""
    def __init__(self, gap_fn=None):
        super().__init__(daemon=True); self.gap_fn = gap_fn; self.stop_evt = threading.Event()
        self.ts = []; self.qs = []; self.tqs = []; self.gaps = []
    def run(self):
        try:
            s = socket.create_connection((PI, FB_PORT), timeout=3); s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception: return
        buf = b""
        while not self.stop_evt.is_set():
            try:
                d = s.recv(4096)
                if not d: break
                buf += d
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip(): continue
                    dd = json.loads(line); t = time.time()
                    self.ts.append(t); self.qs.append(list(dd["joints_deg"])); self.tqs.append(list(dd.get("torque", [0] * 6)))
                    self.gaps.append(float(self.gap_fn()) if self.gap_fn else np.nan)
            except Exception: continue
        try: s.close()
        except Exception: pass
    def stop(self): self.stop_evt.set()
    def arrays(self): return np.array(self.ts), np.array(self.qs), np.array(self.tqs), np.array(self.gaps)
