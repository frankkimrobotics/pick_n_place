#!/usr/bin/env python3
"""real_policy_ctrl :: run the paper-env pick policy (TensorRT engine) as a 10 Hz controller of the
real Pro 630 through the Pi's robot_hal streaming interface (chunks on :9998, feedback on :9999).

    desktop (this script, 10 Hz)                       Pi (robot_hal, 100 Hz)          STM32 drives
    obs = [q, qd, p_obj, p_goal, a_prev,               chunk -> welded q_ref(t)        velocity mode
           tcp, cup_axis, grasp-tcp, q_target-q]  ->   u = vff*v_ref - K0(q-q_ref)
    a = policy(obs); q_target += a[:6]*dq_max                - K1(qd-v_ref)
    chunk [q_target_prev, q_target] over 100 ms  ->   vel_cmd = 17*u  -> CAN
    suction = a[6] > 0 (3-tick release hysteresis) -> halcmd setp pro600.digital_out00

The observation is built EXACTLY like env_paper.PaperPickEnv.observe() (measured-drive variant,
40-D): joint angles in URDF radians (joint_conventions.linuxcnc_deg_to_rad), finite-difference
joint velocity, object centre and goal in the robot base frame, the previous action, cup tip and
cup axis from MuJoCo FK of the training scene, grasp point (object top centre + cup radius) minus
tip, and the commanded-minus-measured lag term.

Nothing moves without --exec. Safety: per-decision delta <= dq_max (2 deg -> 20 deg/s, under the
36 deg/s firmware ceiling), elbow box |j2| <= 70 deg |j3| <= 145 deg, LinuxCNC soft limits, cup tip
keep-out (x >= 0.12 m: wall at x = -0.30, z >= object top - 3 cm, z <= 0.6, |y| <= 0.45),
150-decision timeout, feedback-stale abort (0.5 s), SIGINT -> stop streaming (Pi holds) + suction off.
Object pose is entered (--obj centre x y z, --half hx hy hz) or taken from the cup tip after jogging
the cup onto the object top (--obj_from_tcp). The z of the object top must lie in the training range
(sim: table top z = 0, object top 0.04; --force overrides the check).

  # 1. self-test in the simulator (no robot): obs layout + closed loop through the same code path
  $PY rl/real_policy_ctrl.py --selftest --episodes 16
  # 2. dry run against the live robot (prints what it would send)
  $PY rl/real_policy_ctrl.py --obj 0.38 0.0 0.02 --goal 0.30 -0.15 0.15
  # 3. execute
  $PY rl/real_policy_ctrl.py --obj 0.38 0.0 0.02 --goal 0.30 -0.15 0.15 --exec
"""
import argparse
import json
import math
import os
import signal
import socket
import sys
import threading
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "..", "mycobot_mpc")))
from joint_conventions import linuxcnc_deg_to_rad, rad_to_linuxcnc_deg, LINUXCNC_SOFT_LIMITS_DEG  # noqa: E402

CTRL_DT = 0.10
CUP_R = 0.008                       # env_warp.CUP_R
DQ_MAX_DEG = 2.0                    # training clamp (--dq_max)
ELBOW_DEG = (70.0, 145.0)           # |j2|, |j3| (URDF deg), same box as real_pnp_online.check_box
# fix 2 (2026-09-20): lead = drive dead-time so the Pi samples the reference where the arm will be when the
# command takes effect; with the lag compensated K0 can drop 20 -> 10 (A/B on the robot: velocity ripple rms
# j3/j4 1.53/1.45 -> 0.94/0.68 deg/s, same-tick position error 1.14/1.62 -> 0.71/0.98 deg). lead with K0=20
# made the approach aggressive enough to false-trigger the contact guard 4 cm above the object.
STREAM_GAINS = dict(k0=10.0, k1=0.3, vmax=50.0, vel_scale=17.0, vff=1.0, lead=0.045)
PERIOD_MS = 10
VEL_CMD_MAX = 850                   # Pi clamp on vel_cmd AFTER vel_scale (units = 17 per deg/s): 850 = 50 deg/s.
                                    # Run 1 (2026-09-20) used 100 = 5.9 deg/s -> the arm crawled and timed out.
START_Q = np.array([0.0, -0.349066, 1.396263, 0.174533, -1.570796, 0.0])   # config.START_Q (URDF rad)
OBJ_TOP_TRAIN = (0.02, 0.07)        # object-top z range seen in training (sim object top = 0.04)
OBS_NOISE = 0.005                   # training observation noise (env dr): the BC policy STALLS without it
                                    # (nominal sim, no noise 31 % / with 0.005 78 % / dr 85 %, 2026-09-19)
LEAD_MAX_DEG = 3.0                  # sent reference may lead the measured joint by this much: K0*3 deg saturates the
                                    # Pi law at vmax (full speed) while a blocked joint only ever sees a bounded ref
TAU_FIRM, TAU_HARD, W_J3 = 0.08, 0.13, 0.5   # contact_detector thresholds on the :9999 torque field
TAU_PRESS = 0.11                    # press force target for the seal (between firm 0.08 and hard 0.13)
TAU_ABORT = 0.25                    # pick mode: >= TAU_HARD holds position (vacuum builds), only this aborts
TAU_ABORT_ATTACHED = 0.40           # after the attach the arm lifts and carries: posture torque rises, only a real collision aborts
ATTACH_AFTER = 0.5                  # s of contact with suction on before the object is assumed attached (vacuum build-up).
                                    # 1.0 s in series 2; halved 2026-09-20 (speed-up pass) -- the press primitive now
                                    # reaches TAU_PRESS before the dwell starts, so the dwell is pure vacuum time.
SCRIPT_VMAX_DEG = 25.0              # default peak joint speed of the scripted (non-policy) legs, --script_vmax.
                                    # 25 deg/s peak = 13.3 deg/s average under the min-jerk profile, well under the
                                    # ~36 deg/s following-error ceiling (the last real series peaked at 22-34 deg/s).
SCRIPT_VMAX_SLOW = 12.0             # legs that carry the object or approach the table (post-attach lift, place-down)
AT_GOAL_DWELL = 0.5                 # s the object must stay within 3.5 cm of the goal before the place (1.0 in series 2)
PHASE_T = {}                        # phase boundary timestamps (time.time()), filled by run_episode / main


# ---------------------------------------------------------------- observation (mirrors PaperPickEnv.observe)
class ObsBuilder:
    def __init__(self, xml, half):
        import mujoco
        self.mujoco = mujoco
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.half = np.asarray(half, float)

    def fk(self, q):
        self.d.qpos[:6] = q
        self.mujoco.mj_kinematics(self.m, self.d)
        p = self.d.site_xpos[self.sid].copy()
        R = self.d.site_xmat[self.sid].reshape(3, 3).copy()
        return p, R

    def grasp_point(self, p_obj):
        g = np.asarray(p_obj, float).copy()
        g[2] += self.half[2] + CUP_R
        return g

    def build(self, q, qd, p_obj, p_goal, a_prev, q_target):
        tcp, R = self.fk(q)
        parts = [q, qd, p_obj, p_goal, a_prev, tcp, R[:, 2], self.grasp_point(p_obj) - tcp, q_target - q]
        return np.concatenate([np.asarray(x, np.float32).reshape(-1) for x in parts]), tcp, R


# ---------------------------------------------------------------- policy
def load_policy(stem, use_torch=False, obs_dim=40):
    if not use_torch and os.path.exists(stem + ".plan"):
        try:
            from export_trt import TrtPolicy
            pol = TrtPolicy(stem + ".plan")
            print(f"[policy] TensorRT engine {stem}.plan")
            return pol
        except Exception as e:  # noqa: BLE001
            print(f"[policy] TensorRT engine unusable ({e}); falling back to torch")
    import torch
    from ppo import AC
    ck = torch.load(stem + ".pt", map_location="cpu", weights_only=False)
    ac = AC(obs_dim=obs_dim, arch="paper", critic_extra=(5 if ck.get("critic_priv") else 0))
    ac.load_state_dict(ck["ac"])
    ac.eval()
    base_path, bound = ck.get("residual_base"), float(ck.get("residual_bound", 0.3))
    if base_path:                        # residual checkpoint: fuse base + bounded correction
        base = AC(obs_dim=obs_dim, arch="paper")
        ckb = torch.load(base_path, map_location="cpu", weights_only=False)
        base.load_state_dict(ckb["ac"] if "ac" in ckb else ckb)
        base.eval()
        print(f"[policy] torch {stem}.pt (residual on {base_path}, bound {bound})")

        def f(obs):
            with torch.no_grad():
                o = torch.as_tensor(obs, dtype=torch.float32)[None]
                return torch.clamp(torch.tanh(base.pi(o)) + bound * torch.tanh(ac.pi(o)), -1.0, 1.0)[0].numpy()
        return f
    print(f"[policy] torch {stem}.pt")

    def f(obs):
        with torch.no_grad():
            return torch.tanh(ac.pi(torch.as_tensor(obs, dtype=torch.float32)[None]))[0].numpy()
    return f


# ---------------------------------------------------------------- robot links
class PiLink:
    """Feedback stream (:9999, ~100 Hz joints_deg) + command socket (:9998)."""

    def __init__(self, host, exec_, ref_mode="linear"):
        self.host, self.exec = host, exec_
        self.ref_mode = ref_mode                  # "linear" (sampled chunk) | "spline" (B-spline control points)
        self.hist = deque(maxlen=8)               # targets already sent, for the spline control polygon
        self.spline_t0 = None                     # anchor grid: keeps the Pi welding onto one uniform knot base
        self.samples = deque(maxlen=400)          # (t_recv, t_robot, q_rad[6])
        self.lock = threading.Lock()
        self.clock_offset = 0.0
        self.cmd_sock = None
        self.status = {}
        self.acks = deque(maxlen=50)
        self.stop = False
        threading.Thread(target=self._stream_loop, daemon=True).start()
        threading.Thread(target=self._cmd_loop, daemon=True).start()

    def _stream_loop(self):
        offs = deque(maxlen=300)
        while not self.stop:
            try:
                s = socket.create_connection((self.host, 9999), timeout=5)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(3.0)
                buf = b""
                while not self.stop:
                    data = s.recv(65536)
                    if not data:
                        break
                    buf += data
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        try:
                            m = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        tr = time.time()
                        tj = float(m.get("timestamp", tr))
                        offs.append(tr - tj)
                        with self.lock:
                            self.samples.append((tr, tj, linuxcnc_deg_to_rad(m["joints_deg"][:6]), np.asarray(m.get("torque", [0.0] * 6)[:6], float)))
                            self.clock_offset = min(offs)
            except (OSError, socket.timeout):
                pass
            time.sleep(0.5)

    def _cmd_loop(self):
        while not self.stop:
            try:
                s = socket.create_connection((self.host, 9998), timeout=5)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(5.0)
                self.cmd_sock = s
                buf = b""
                while not self.stop:
                    data = s.recv(65536)
                    if not data:
                        break
                    buf += data
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        try:
                            m = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if str(m.get("state", "")).startswith("ack"):
                            self.acks.append((time.time(), m))
                        else:
                            self.status = m
            except (OSError, socket.timeout):
                pass
            self.cmd_sock = None
            time.sleep(0.5)

    def ok(self):
        with self.lock:
            fresh = bool(self.samples) and (time.time() - self.samples[-1][0]) < 0.5
        return fresh and self.cmd_sock is not None

    def state(self):
        """(q_rad, qd_rad/s, age_s): latest sample and a 50 ms finite-difference velocity."""
        with self.lock:
            if not self.samples:
                return None, None, 1e9
            tr, tj, q, _ = self.samples[-1]
            qd = np.zeros(6)
            for tr0, tj0, q0, _ in reversed(self.samples):
                if tj - tj0 >= 0.045:
                    qd = (q - q0) / (tj - tj0)
                    break
            return q.copy(), qd, time.time() - tr

    def torque(self):
        with self.lock:
            return self.samples[-1][3].copy() if self.samples else np.zeros(6)

    def torque_baseline(self, seconds=1.0):
        t_end = time.time() + seconds
        while time.time() < t_end:
            time.sleep(0.05)
        with self.lock:
            tq = np.array([s[3] for s in self.samples if s[0] > t_end - seconds])
        return np.median(tq, axis=0) if len(tq) else np.zeros(6)

    def _send(self, cmd):
        if not self.exec:
            return
        s = self.cmd_sock
        if s is None:
            raise RuntimeError("command socket not connected")
        s.sendall((json.dumps(cmd) + "\n").encode())

    extrapolate = True      # fix 1 (2026-09-20): keep reference ahead of the next chunk's arrival

    def send_segment(self, q_from, q_to, t_start, dt=CTRL_DT, seq=0, tag=None, hold=False):
        if hold:                       # true hold: flat reference (extrapolation / spline history would keep the previous motion going)
            if self.ref_mode == "spline":
                self.hist.clear()
                for _ in range(4):
                    self.hist.append(np.asarray(q_to, float))
                return self._send_spline_segment(q_to, q_to, t_start, dt, seq, tag)
            pts = [[round(float(x), 4) for x in rad_to_linuxcnc_deg(q_to)]] * 3
            self._send({"chunk": pts, "traj_dt": dt, "t_anchor": t_start - self.clock_offset, "seq": seq, "tag": tag or f"hold:{seq}",
                        "gains": STREAM_GAINS, "period_ms": PERIOD_MS, "vel_cmd_max": VEL_CMD_MAX, "log_stamp": "policy"})
            return
        """Linear reference from q_from to q_to over dt, anchored at desktop time t_start.
        With `extrapolate`, a third point continues the same velocity for one more dt: the next chunk
        (anchored at t_start + dt) arrives a few ms late and would otherwise leave the welder past the
        end of the reference (v_ref = 0, velocity command dips once per decision = the 10 Hz ripple).
        The welder drops points at t >= the next anchor, so the extrapolated point is only followed for
        those few ms -- or for at most dt if the desktop stalls, after which the Pi holds."""
        if self.ref_mode == "spline":
            return self._send_spline_segment(q_from, q_to, t_start, dt, seq, tag)
        pts = [[round(float(x), 4) for x in rad_to_linuxcnc_deg(q_from)], [round(float(x), 4) for x in rad_to_linuxcnc_deg(q_to)]]
        if self.extrapolate:
            q_ext = np.asarray(q_to, float) + (np.asarray(q_to, float) - np.asarray(q_from, float))
            pts.append([round(float(x), 4) for x in rad_to_linuxcnc_deg(q_ext)])
        self._send({"chunk": pts, "traj_dt": dt, "t_anchor": t_start - self.clock_offset, "seq": seq, "tag": tag or f"pol:{seq}",
                    "gains": STREAM_GAINS, "period_ms": PERIOD_MS, "vel_cmd_max": VEL_CMD_MAX, "log_stamp": "policy"})

    def _send_spline_segment(self, q_from, q_to, t_start, dt, seq, tag):
        """fix 3 (2026-09-19): C2 reference. Send the last 4 targets + 1 extrapolated point as the
        control polygon of a uniform cubic B-spline (Pi: spline_ref.SplineRef); the Pi re-welds the
        tail every decision, so the curve stays C2 across decision boundaries instead of kinking.

        Phase: the target sent at decision t is the reference value at t + dt (same as the linear
        segment q_from -> q_to over [t, t+dt]), so the newest target sits at knot t_start + dt and
        the anchor (first of the 5 points) is t_start - 2*dt. The extrapolated 5th point absorbs the
        one-knot lag of an approximating B-spline (see spline_ref docstring) and keeps the reference
        alive until the next chunk lands. Anchors are snapped to a fixed dt grid so the Pi can weld
        onto the same knot base (exact C2) instead of restarting the spline every decision."""
        if not self.hist:
            self.hist.append(np.asarray(q_from, float))
        self.hist.append(np.asarray(q_to, float))
        h = list(self.hist)[-4:]
        while len(h) < 4:                          # first decisions: pad with the current target pose
            h.insert(0, h[0])
        q_ext = h[-1] + (h[-1] - h[-2]) if len(h) >= 2 else h[-1]
        pts = [[round(float(x), 4) for x in rad_to_linuxcnc_deg(p)] for p in (h + [q_ext])]
        t_anchor = t_start - 2.0 * dt
        if self.spline_t0 is None:
            self.spline_t0 = t_anchor
        else:
            t_anchor = self.spline_t0 + round((t_anchor - self.spline_t0) / dt) * dt
        self._send({"spline": pts, "traj_dt": dt, "t_anchor": t_anchor - self.clock_offset, "seq": seq,
                    "tag": tag or f"pol:{seq}", "gains": STREAM_GAINS, "period_ms": PERIOD_MS,
                    "vel_cmd_max": VEL_CMD_MAX, "log_stamp": "policy"})

    def send_path(self, qs, dt, t_start):
        pts = [[round(float(x), 4) for x in rad_to_linuxcnc_deg(q)] for q in qs]
        if self.ref_mode == "spline":
            # keep one reference type per stream (a linear chunk arriving mid spline stream would be
            # welded as control points anyway); duplicating the first/last point makes the clamped
            # B-spline start and end exactly on q_from / q_to.
            pts = [pts[0]] + pts + [pts[-1]]
            self.hist.clear(); self.spline_t0 = None
            self._send({"spline": pts, "traj_dt": dt, "t_anchor": t_start - self.clock_offset, "seq": 0, "tag": "path",
                        "gains": STREAM_GAINS, "period_ms": PERIOD_MS, "vel_cmd_max": VEL_CMD_MAX, "log_stamp": "policy_path"})
            return
        self._send({"chunk": pts, "traj_dt": dt, "t_anchor": t_start - self.clock_offset, "seq": 0, "tag": "path",
                    "gains": STREAM_GAINS, "period_ms": PERIOD_MS, "vel_cmd_max": VEL_CMD_MAX, "log_stamp": "policy_path"})

    def suction(self, on):
        self._send({"suction": 1 if on else 0, "tag": "suction"})

    def close(self):
        self.stop = True


class SimLink:
    """Same interface backed by PaperPickEnv (1 world, measured drive, no DR) for --selftest."""

    def __init__(self, drive="real", nworld=1):
        import torch
        import warp as wp
        wp.init()
        from env_paper import PaperPickEnv
        self.torch = torch
        self.N = nworld
        self.env = PaperPickEnv(nworld=nworld, device="cuda:0", xml=os.path.join(HERE, "scenes", "box_med.xml"), dr=False, drive=drive, ep_len=150,
                                grasp_shaping=True, obs_ee=True, reach_target="grasp", lift_dense=True, w_reach=0.5, w_track_c=4, w_track_f=8)
        self.env.auto_reset = False
        self.exec = True
        self.clock_offset = 0.0
        self.pending = None
        self.info = {}
        self.reset()

    def reset(self):
        self.env.reset(self.torch.ones(self.N, dtype=self.torch.bool, device="cuda:0"))
        self.pending = None
        self.info = {}

    def ok(self):
        return True

    def state(self):
        return self.env.qpos[:, :6].cpu().numpy().astype(float), self.env.qvel[:, :6].cpu().numpy().astype(float), 0.0

    def obj(self):
        return self.env._obj_pos().cpu().numpy().astype(float)

    def goal(self):
        return self.env.goal.cpu().numpy().astype(float)

    def env_obs(self):
        return self.env.observe().cpu().numpy()

    def step(self, a7):
        """Advance all worlds by one decision with the controller's actions (N,7)."""
        _, r, done, info = self.env.step(self.torch.tensor(np.atleast_2d(a7), dtype=self.torch.float32, device="cuda:0"))
        self.info = {"placed": info["placed"].cpu().numpy()}
        return bool(done.all())

    def send_segment(self, *a, **k):
        pass

    def send_path(self, *a, **k):
        pass

    def suction(self, on):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------- live object tracking (rl/d435_track.py over UDP)
class ObjectTracker:
    """Latest D435 detection {cx, cy, top, n} from rl/d435_track.py (udp 127.0.0.1:9701)."""

    def __init__(self, port=9701):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)
        self.last = None
        self.t_last = 0.0

    bias = (0.0, 0.0)     # camera-to-robot bias of the estimate (m), measured with the cylinder centred under the cup tip

    def poll(self):
        while True:
            try:
                data = self.sock.recv(4096)
            except BlockingIOError:
                break
            try:
                m = json.loads(data)
            except json.JSONDecodeError:
                continue
            if m.get("cx") is not None:
                m["cx"] -= self.bias[0]; m["cy"] -= self.bias[1]
                for c in m.get("cands", []):
                    c["cx"] -= self.bias[0]; c["cy"] -= self.bias[1]
                self.last, self.t_last = m, time.time()
        return self.last if (time.time() - self.t_last) < 0.5 else None


# ---------------------------------------------------------------- safety
class Guard:
    def __init__(self, obs_builder, obj_top_z, force=False):
        self.ob = obs_builder
        self.z_floor = obj_top_z - 0.03
        self.force = force
        self.n_clip = 0

    def check_q(self, q):
        """Return a list of violated constraints for joint vector q (URDF rad)."""
        v = []
        d = np.degrees(q)
        if abs(d[1]) > ELBOW_DEG[0] + 1e-6:
            v.append(f"j2 {d[1]:.1f} > {ELBOW_DEG[0]}")
        if abs(d[2]) > ELBOW_DEG[1] + 1e-6:
            v.append(f"j3 {d[2]:.1f} > {ELBOW_DEG[1]}")
        lc = rad_to_linuxcnc_deg(q)
        for i, (lo, hi) in enumerate(LINUXCNC_SOFT_LIMITS_DEG):
            if not (lo + 2 <= lc[i] <= hi - 2):
                v.append(f"linuxcnc j{i + 1} {lc[i]:.1f} outside [{lo},{hi}]")
        tcp, _ = self.ob.fk(q)
        if tcp[0] < 0.12:
            v.append(f"tip x {tcp[0]:.3f} < 0.12 (wall keep-out)")
        if abs(tcp[1]) > 0.45:
            v.append(f"tip |y| {abs(tcp[1]):.3f} > 0.45")
        if tcp[2] < self.z_floor:
            v.append(f"tip z {tcp[2]:.3f} < floor {self.z_floor:.3f}")
        if tcp[2] > 0.60:
            v.append(f"tip z {tcp[2]:.3f} > 0.60")
        return v


TRACK_MAX_STEP = 0.05
LIFT_AFTER_ATTACH = 0.08            # scripted vertical lift right after the attach, before the policy carries
CARRY_ALPHA, CARRY_STEP = 0.5, 0.5   # after the attach: EMA on the action + step cap (policy dithers at 5 Hz in the hold phase)
TRAIN_GRASP_Z = 0.048               # sim training: object top 0.04 + cup radius; the policy descends to this absolute height


def scripted_move(link, guard, ob, q_to, label, v_peak_deg=None, min_T=0.4, verbose=True):
    """One scripted (non-policy) leg: min-jerk from the MEASURED q to q_to, then wait on feedback.

    Speed-up pass (2026-09-20): every scripted leg used to be a constant-velocity ramp at
    max_deg / 6 deg/s followed by a blind `time.sleep(T + 1.4)` pad. Two changes:

      * min-jerk time scaling s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5 (zero velocity AND zero
        acceleration at both ends, peak speed = 1.875 x average). The B-spline the Pi builds from
        the 10 Hz samples therefore has no velocity step at either end, so the whole leg can run at
        a peak of `v_peak_deg` (default --script_vmax 25 deg/s) instead of a 6 deg/s ramp, still
        far below the ~36 deg/s drive following-error fault.
      * the fixed 1.4 s pad is replaced by a feedback wait: poll the joint stream at 20 Hz until
        max|q - q_to| < 0.7 deg on 3 consecutive polls (hard timeout T + 2.0 s, warns with the
        residual). Nothing is waited on when nothing is being sent (dry run / SimLink).

    v_peak_deg=None uses the module default SCRIPT_VMAX_DEG (set from --script_vmax).
    Returns the measured q at the end, or None if the path violates the guard (same message and
    "stop here" semantics as before).
    """
    v_peak = float(SCRIPT_VMAX_DEG if v_peak_deg is None else v_peak_deg)
    q_from, _, _ = link.state()
    q_from = np.asarray(q_from, float).reshape(-1)[:6].copy()
    q_to = np.asarray(q_to, float).reshape(-1)[:6].copy()
    max_deg = float(np.degrees(np.abs(q_to - q_from)).max())
    T = max(float(min_T), max_deg / (v_peak / 1.875))          # v_peak = 1.875 * (max_deg / T)
    n = max(2, int(round(T / CTRL_DT)) + 1)
    T = CTRL_DT * (n - 1)                                      # snap to the 0.1 s sample grid
    qs = []
    for i in range(n):
        tau = i / (n - 1)
        s = tau * tau * tau * (10.0 - 15.0 * tau + 6.0 * tau * tau)
        qs.append(q_from + (q_to - q_from) * s)
    if any(guard.check_q(qq) for qq in qs):
        print(f"[guard] {label} path violates limits -> stopping here")
        return None
    executing = bool(getattr(link, "exec", False)) and hasattr(link, "samples")   # SimLink: no real motion
    if verbose:
        print(f"[ctrl] {label}: {T:.1f} s ({max_deg:.1f} deg, peak {v_peak:.0f} deg/s){'' if executing else '  [dry run]'}")
    t_start = time.time() + 0.2
    link.send_path(qs, 0.1, t_start)
    t_end = t_start + T
    while True:                                                # the path's own duration
        d = t_end - time.time()
        if d <= 0:
            break
        time.sleep(min(d, 0.05))
    if executing:
        good, err, t_dl = 0, None, t_end + 2.0                 # then wait on feedback (20 Hz poll)
        while time.time() < t_dl:
            q_m, _, age = link.state()
            if q_m is not None and age < 0.5:
                err = float(np.degrees(np.abs(np.asarray(q_m, float) - q_to)).max())
                good = good + 1 if err < 0.7 else 0
                if good >= 3:
                    break
            time.sleep(0.05)
        if good < 3:
            print(f"[warn] {label}: not settled after {T + 2.0:.1f} s (residual {err if err is None else round(err, 2)} deg) -- continuing")
    q_meas, _, _ = link.state()
    return np.asarray(q_meas, float).copy() if q_meas is not None else q_to.copy()


def retract_to_hover(link, ob, guard, q_now, tip_now, z_hover):
    """IK straight up to z_hover above the current tip xy, streamed as a min-jerk path; returns the hover joints."""
    import mujoco
    demo = {"__file__": os.path.join(ROOT, "mjwarp_pick_demo.py")}
    exec(open(demo["__file__"]).read().split("if __name__")[0], demo)
    q_up, err = demo["ik"](ob.m, mujoco.MjData(ob.m), "tcp", [float(tip_now[0]), float(tip_now[1]), float(z_hover)], demo["R_DOWN"], q_now)
    if err > 0.005:
        return np.asarray(q_now, float).copy()
    q_meas = scripted_move(link, guard, ob, np.array(q_up), "retract to hover")
    return np.asarray(q_now, float).copy() if q_meas is None else q_meas


# ---------------------------------------------------------------- controller loop
def run_episode(link, policy, ob, guard, p_obj, p_goal, n_steps, dq_max, log, obs_noise=OBS_NOISE, lead_max_deg=LEAD_MAX_DEG,
                tau_base=None, verbose=True, touch_only=False, tracker=None, track_freeze=0.03):
    """10 Hz policy loop on the real robot.
    q_virtual : the policy's integrated target (what the observation's lag term uses; it may wind up
                during a press exactly as in training)
    q_send    : the reference streamed to the Pi = measured q + clip(q_virtual - q, +-lead_max)
                (the sim drive bounds its following error the same way; on the robot it also keeps the
                velocity command small when the cup is blocked by the object)
    contact   : J2/J3 torque rise above the baseline (contact_detector thresholds); firm contact holds the
                reference at the measured position, hard contact aborts. Firm contact + suction for
                ATTACH_AFTER s => the object is assumed on the cup and p_obj follows the tip (the sim's
                kinematic attach; there is no object tracking here)."""
    rng = np.random.default_rng(0)
    q, qd, age = link.state()
    q_virtual = q.copy()
    q_send_prev = q.copy()
    a_prev = np.zeros(7, np.float32)
    sealed_cmd = False
    rel_count = 0
    lead_max = math.radians(lead_max_deg)
    attached, attach_off, t_contact_on = False, None, None
    p_obj = np.asarray(p_obj, float).copy()
    n_track = [0]
    n_retract = [0]
    n_hadapt = [0]
    tq_hist = []
    z_shift_frozen = [0.0]
    min_tip_near = [9.9]
    t_at_goal = None
    press_floor = [9.9]
    press_reached = [False]
    press_xy = [None]; press_z = [0.0]; t_press_ok = [None]; z_attach = [0.0]; lifted_once = [False]
    import mujoco
    _demo = {"__file__": os.path.join(ROOT, "mjwarp_pick_demo.py")}
    exec(open(_demo["__file__"]).read().split("if __name__")[0], _demo)
    ik_fn, R_DOWN, ik_data = _demo["ik"], _demo["R_DOWN"], mujoco.MjData(ob.m)
    n_blob0 = [None]
    if abs(2 * float(ob.half[2]) + CUP_R - TRAIN_GRASP_Z) > 0.005:
        print(f"[ctrl] object top {2 * float(ob.half[2]):.3f} m: observation z shifted by {TRAIN_GRASP_Z - (2 * float(ob.half[2]) + CUP_R):+.3f} m so the policy sees its trained grasp height")
    t0 = time.time()
    PHASE_T["ep0"] = t0
    rows = []
    reason = "timeout"
    for k in range(n_steps):
        t_dec = t0 + k * CTRL_DT
        d = t_dec - time.time()
        if d > 0:
            time.sleep(d)
        if not link.ok():
            reason = "feedback stale / command socket down"
            break
        q, qd, age = link.state()
        tcp_now, _ = ob.fk(q)
        if attached:
            p_obj = tcp_now + attach_off
        elif tracker is not None and (tcp_now[2] - ob.grasp_point(p_obj)[2]) > 0.25 and np.linalg.norm(ob.grasp_point(p_obj) - tcp_now) > track_freeze:
            det = tracker.poll()          # follow the object in xy only while the arm is high (its shadow/links merge into the blob below ~25 cm)
            if det is not None and det.get("cands"):
                # several objects on the table (s3_04, 2026-09-20): follow the candidate NEAREST to the current
                # estimate, not the tracker's largest blob (that switched the target to a bottle 8 cm away)
                near = min(det["cands"], key=lambda c: np.hypot(c["cx"] - p_obj[0], c["cy"] - p_obj[1]))
                det = dict(det, **near) if np.hypot(near["cx"] - p_obj[0], near["cy"] - p_obj[1]) < 0.30 else None
            if det is not None and det["n"] > 300 and (n_blob0[0] is None or 0.5 * n_blob0[0] < det["n"] < 1.8 * n_blob0[0]):
                if n_blob0[0] is None: n_blob0[0] = det["n"]
                new_xy = np.array([det["cx"], det["cy"]])
                step = new_xy - p_obj[:2]
                dist = np.linalg.norm(step)
                if dist < 0.30:
                    if dist > TRACK_MAX_STEP:                             # rate limit: <= 5 cm per decision (0.5 m/s)
                        step = step * (TRACK_MAX_STEP / dist)
                    p_obj[:2] = p_obj[:2] + step
                    n_track[0] += 1
        # object moved away while the cup was already low: the policy never saw that in training (run 2,
        # 2026-09-20: it pressed the table 30 cm from the object). Retract to a hover and let it re-approach.
        d_g = np.linalg.norm(ob.grasp_point(p_obj) - tcp_now)
        d_lat = np.hypot(*(ob.grasp_point(p_obj)[:2] - tcp_now[:2]))
        if tracker is not None and tcp_now[2] < 0.15 and d_lat > 0.12 and not attached:
            if verbose:
                print(f"[ctrl] step {k}: object moved away (lateral {100 * d_lat:.0f} cm at tip z {tcp_now[2]:.2f}) -> retract to hover and re-approach")
            q_hover = retract_to_hover(link, ob, guard, q, tcp_now, 0.15)
            q_virtual = q_hover.copy(); q_send_prev = q_hover.copy(); a_prev[:] = 0.0
            t0 = time.time() - k * CTRL_DT                                  # keep the decision clock
            n_retract[0] += 1
            if n_retract[0] > 2:
                reason = "object kept moving away (3 retracts)"
                break
            continue
        # approach/press: present the object at the trained grasp height (both directions); after the attach the carry
        # uses true heights (a frozen negative shift kept the object low in s2_01f)
        if not attached:
            z_shift = max(0.0, TRAIN_GRASP_Z - (2 * float(ob.half[2]) + CUP_R))   # thin objects only; tall ones stall with a negative shift (s2_09)
            z_shift_frozen[0] = z_shift
        else:
            z_shift = max(0.0, z_shift_frozen[0])       # thin objects keep their shift through the carry; tall ones drop it
        obs, tcp, R = ob.build(q, qd, p_obj + [0, 0, z_shift], p_goal + [0, 0, z_shift], a_prev, q_virtual)
        obs[24] += z_shift                                   # tcp z (obs layout: q6 qd6 p_obj3 goal3 a_prev7 tcp3 ...)
        if obs_noise > 0:
            obs = obs + rng.normal(0, obs_noise, size=obs.shape).astype(np.float32)
        a = np.clip(policy(obs), -1, 1).astype(np.float32)
        if attached:                                          # carry/hold: filter the policy's 5 Hz dither, cap the step
            a[:6] = np.clip(CARRY_ALPHA * a[:6] + (1 - CARRY_ALPHA) * a_prev[:6], -CARRY_STEP, CARRY_STEP)
        if not attached and tau_base is not None:
            gz = ob.grasp_point(p_obj)[2]
            lateral = np.hypot(*(ob.grasp_point(p_obj)[:2] - tcp_now[:2]))
            if lateral < 0.02 and tcp_now[2] < gz - 0.008 and tau < 0.05 and ob.half[2] > 0.008:
                ob.half[2] -= 0.004; p_obj[2] = ob.half[2]          # no contact where the top should be: lower the estimate 4 mm
                n_hadapt[0] += 1
        d_land = np.linalg.norm(ob.grasp_point(p_obj) - tcp_now)
        soft = 0.6 if (d_land < 0.03 and not attached) else 1.0       # soft landing: <= 12 deg/s in the last 3 cm
        dq = a[:6] * dq_max * soft
        q_new = q_virtual + dq
        viol = guard.check_q(q_new)
        if viol:
            guard.n_clip += 1
            if verbose:
                print(f"[guard] step {k}: hold target ({'; '.join(viol)})")
            q_new = q_virtual.copy()
        # suction: same hysteresis as env_warp.step (a latched cup releases after 3 consecutive off commands)
        want = bool(a[6] > 0)
        if sealed_cmd and not want:
            rel_count += 1
            want = rel_count < 3
        else:
            rel_count = 0
        if want != sealed_cmd:
            if not touch_only:
                link.suction(want)
            sealed_cmd = want
            if verbose:
                print(f"[ctrl] step {k}: suction {'ON' if want else 'OFF'}{' (touch-only: not sent)' if touch_only else ''}")
        # contact from the drive torque; the baseline follows the posture until the cup is within 5 cm of the
        # grasp point (gravity torque at an extended reach differs from the home-pose baseline by > 0.1)
        tau = 0.0
        if tau_base is not None:
            tq = link.torque()
            tq_hist.append(tq)
            if not attached and np.linalg.norm(ob.grasp_point(p_obj) - tcp_now) > 0.05 and len(tq_hist) >= 5:
                tau_base = np.median(np.array(tq_hist[-5:]), axis=0)
            tau = abs(tq[1] - tau_base[1]) + W_J3 * abs(tq[2] - tau_base[2])
        abort_lvl = TAU_HARD if touch_only else (TAU_ABORT_ATTACHED if attached else TAU_ABORT)
        if tau >= abort_lvl:
            reason = f"HARD contact (tau {tau:.3f} >= {abort_lvl})"
            link.send_segment(q_send_prev, q, t_dec, seq=k)
            break
        hard_hold = (tau >= TAU_HARD) and not attached   # before the attach: hold, let the vacuum build, never press further
        # firm contact is only meaningful near the object (contact_detector's ARM gate): the first
        # acceleration from rest gave tau 0.084 on 2026-09-20 and ended a demo at step 1. Hard contact
        # (TAU_HARD) stays armed everywhere as the backstop.
        near = np.linalg.norm(ob.grasp_point(p_obj) - tcp_now) < 0.03
        # torque contact only counts at the object's height (a deceleration transient 1.6 cm above the cup fired it in s2_01)
        at_height = tcp_now[2] <= ob.grasp_point(p_obj)[2] + 0.015
        firm = ((tau >= TAU_FIRM) and near and (at_height or touch_only)) or hard_hold
        if firm and "contact" not in PHASE_T:
            PHASE_T["contact"] = time.time()          # phase boundary: approach -> press
        if tau >= TAU_FIRM and not near and tcp_now[2] < guard.z_floor + 0.02:
            reason = f"firm contact at the floor away from the object (tau {tau:.3f}, |tip-grasp| {100 * d_g:.0f} cm) -- stopping"
            link.send_segment(q_send_prev, q, t_dec, seq=k)
            break
        gp_now = ob.grasp_point(p_obj)
        lat_now = np.hypot(*(gp_now[:2] - tcp_now[:2]))
        pressed = sealed_cmd and not attached and lat_now < 0.025 and tcp_now[2] <= gp_now[2] + 0.015
        if sealed_cmd and not attached and lat_now < 0.03:
            min_tip_near[0] = min(min_tip_near[0], tcp_now[2])
        # lift-after-press: the policy only lifts once it believes the seal is on (sim semantics)
        if sealed_cmd and not attached and press_reached[0] and min_tip_near[0] <= gp_now[2] + 0.02 and tcp_now[2] > min_tip_near[0] + 0.015 and t_contact_on is not None and time.time() - t_contact_on >= ATTACH_AFTER:
            ob.half[2] = max(0.008, (min_tip_near[0] - CUP_R) / 2); p_obj[2] = ob.half[2]
            attached, attach_off = True, p_obj - tcp_now
            z_attach[0] = float(min_tip_near[0])
            PHASE_T.setdefault("attach", time.time())
            if verbose:
                print(f"[ctrl] step {k}: lift after press with suction on -> object assumed ATTACHED (pressed to z {min_tip_near[0]:.3f}, offset {np.round(attach_off, 3).tolist()})")
        if firm and touch_only:
            reason = f"TOUCH (tau {tau:.3f}) at tip {np.round(tcp_now, 4).tolist()} -- touch-only demo ends here"
            link.send_segment(q_send_prev, q, t_dec, seq=k)
            break
        if (firm or pressed) and sealed_cmd and not attached:
            t_contact_on = t_contact_on or time.time()
            if press_floor[0] > 5: press_floor[0] = min(tcp_now[2], gp_now[2]) - 0.015   # press up to 15 mm below the (calibrated) grasp height
            if press_reached[0] and t_press_ok[0] is None: t_press_ok[0] = time.time()
            at_floor = press_z[0] <= press_floor[0] + 0.0005
            if (t_press_ok[0] is not None and time.time() - t_press_ok[0] >= ATTACH_AFTER) or (at_floor and tau >= 0.05 and time.time() - t_contact_on >= ATTACH_AFTER + 0.5):
                ob.half[2] = max(0.008, (tcp_now[2] - CUP_R) / 2); p_obj[2] = ob.half[2]   # top = touch height (sim convention)
                attached, attach_off = True, p_obj - tcp_now
                z_attach[0] = float(tcp_now[2])
                PHASE_T.setdefault("attach", time.time())
                q_new = q.copy()                                          # forget the wound-up press target: lift from here
                if verbose:
                    print(f"[ctrl] step {k}: firm contact + suction for {ATTACH_AFTER} s -> object assumed ATTACHED (offset {np.round(attach_off, 3).tolist()}, top set to {2 * ob.half[2]:.3f}; height adapted {n_hadapt[0]}x)")
        elif not (firm or pressed):
            t_contact_on = None
            press_floor[0] = 9.9
            t_press_ok[0] = None
        if attached and not lifted_once[0]:
            lifted_once[0] = True
            q_here, _, _ = link.state(); tip_here, _ = ob.fk(q_here)
            q_up, e_up = ik_fn(ob.m, ik_data, "tcp", [float(tip_here[0]), float(tip_here[1]), float(tip_here[2]) + LIFT_AFTER_ATTACH], R_DOWN, q_here)
            q_up = np.array(q_up)
            if e_up < 0.005:
                if verbose:
                    print(f"[ctrl] step {k}: scripted lift {100 * LIFT_AFTER_ATTACH:.0f} cm after the attach, then the policy carries")
                PHASE_T["lift0"] = time.time()
                q_m = scripted_move(link, guard, ob, q_up, f"lift {100 * LIFT_AFTER_ATTACH:.0f} cm after attach",
                                    v_peak_deg=SCRIPT_VMAX_SLOW, min_T=0.8, verbose=verbose)
                PHASE_T["lift1"] = time.time()
                if q_m is not None:
                    q_virtual = q_m.copy(); q_send_prev = q_m.copy(); a_prev[:] = 0.0
                    t0 = time.time() - k * CTRL_DT
                    continue
        # reference actually streamed: bounded lead; hold on firm contact
        lead = np.clip(q_new - q, -lead_max, lead_max)
        holding = False
        if (firm or pressed) and not attached and sealed_cmd:
            # guarded press primitive (independent of the policy): descend vertically at 2 cm/s at the contact xy
            # until the contact metric reaches TAU_PRESS or 15 mm below first contact, then hold for the dwell.
            # (1 cm/s in series 2; 2 mm/decision halves the 2.3-5 s press phase, the floor rule is unchanged.)
            if press_xy[0] is None:
                # press at the ESTIMATED centre, not where the policy happened to land: the last runs attached 8-13 mm
                # off-centre (always +y) and the cup sat on the rim (user, 2026-09-20). The IK descent moves the cup
                # laterally by that much during the first press decisions, before the contact force builds.
                press_xy[0] = gp_now[:2].copy(); press_z[0] = float(tcp_now[2])
            if tau < TAU_PRESS and press_z[0] > press_floor[0]:
                press_z[0] -= 0.002
                qp, ep = ik_fn(ob.m, ik_data, "tcp", [float(press_xy[0][0]), float(press_xy[0][1]), press_z[0]], R_DOWN, q)
                q_send = np.array(qp) if (ep < 0.005 and not guard.check_q(np.array(qp))) else q
                press_reached[0] = False
            else:
                q_send = q; holding = True
                press_reached[0] = press_reached[0] or tau >= TAU_PRESS
            q_new = q.copy()                                   # the policy's target is frozen while the primitive presses
        else:
            q_send = q + lead
            if not attached:
                press_xy[0] = None
            elif ob.fk(q_send)[0][2] < z_attach[0] - 0.003:      # attached: never take the cup below the contact height
                q_send = q; holding = True
        link.send_segment(q_send_prev, q_send, t_dec, seq=k, hold=holding)
        if attached and not touch_only:
            d_goal_now = np.linalg.norm(p_obj - p_goal)
            if d_goal_now < 0.035:
                t_at_goal = t_at_goal or time.time()
                if time.time() - t_at_goal >= AT_GOAL_DWELL:
                    reason = f"object held at the goal for {AT_GOAL_DWELL} s ({100 * d_goal_now:.1f} cm) -> place"
                    link.send_segment(q_send_prev, q, t_dec, seq=k)
                    rows.append(dict(k=k, t=round(t_dec - t0, 3), q=np.round(q, 4).tolist(), qd=np.round(qd, 3).tolist(), tcp=np.round(tcp, 4).tolist(),
                                     a=np.round(a, 3).tolist(), q_virtual=np.round(q_new, 4).tolist(), q_send=np.round(q, 4).tolist(),
                                     suction=int(sealed_cmd), tau=round(tau, 4), attached=int(attached), p_obj=np.round(p_obj, 4).tolist(), age_ms=round(1000 * age, 1)))
                    break
            else:
                t_at_goal = None
        rows.append(dict(k=k, t=round(t_dec - t0, 3), q=np.round(q, 4).tolist(), qd=np.round(qd, 3).tolist(), tcp=np.round(tcp, 4).tolist(),
                         a=np.round(a, 3).tolist(), q_virtual=np.round(q_new, 4).tolist(), q_send=np.round(q_send, 4).tolist(),
                         suction=int(sealed_cmd), tau=round(tau, 4), attached=int(attached), p_obj=np.round(p_obj, 4).tolist(), age_ms=round(1000 * age, 1)))
        if verbose and k % 10 == 0:
            g = ob.grasp_point(p_obj)
            print(f"[ctrl] k={k:3d} tip {np.round(tcp, 3).tolist()} |tip-grasp| {np.linalg.norm(g - tcp) * 100:.1f} cm  suction {int(sealed_cmd)}  "
                  f"tau {tau:.3f}{' CONTACT' if firm else ''}{' ATTACHED' if attached else ''}  windup {np.degrees(np.abs(q_new - q)).max():.1f} deg  fb age {1000 * age:.0f} ms")
        q_virtual = q_new
        q_send_prev = q_send
        a_prev = a
    PHASE_T["ep_end"] = time.time()
    if log:
        json.dump(rows, open(log, "w"))
    print(f"[ctrl] episode end: {reason} after {len(rows)} decisions" + (f"  (object tracked: {n_track[0]} updates, final estimate {np.round(p_obj, 3).tolist()})" if tracker is not None else ""))
    return rows, sealed_cmd


def report_phases(log=None):
    """Print the per-phase wall-clock timeline of the run and, with --log, write <log>.phases.json.
    (The log itself is a list of rows, so the timeline cannot live inside it.)"""
    PHASE_T["end"] = time.time()
    t = PHASE_T

    def dur(a_key, b_key):
        if a_key in t and b_key in t and t[b_key] >= t[a_key]:
            return round(t[b_key] - t[a_key], 2)
        return None
    ph = {
        "calib": dur("calib0", "calib1"),
        "approach": dur("ep0", "contact"),
        "press_attach": dur("contact", "attach"),
        "lift": dur("lift0", "lift1"),
        "carry": dur("lift1", "ep_end") if "lift1" in t else dur("attach", "ep_end"),
        "place_settle": dur("place0", "place1"),
        "release_retract_home": dur("place1", "end"),
        "total": dur("run0", "end"),
    }
    print("[timeline] " + "  ".join(f"{k} {'-' if v is None else format(v, '.1f')}s" for k, v in ph.items()))
    if log:
        try:
            json.dump({"phases_s": ph, "stamps": {k: round(v, 3) for k, v in t.items()}}, open(str(log) + ".phases.json", "w"), indent=1)
            print(f"[timeline] written to {log}.phases.json")
        except OSError as e:  # noqa: BLE001
            print(f"[timeline] could not write {log}.phases.json: {e}")
    return ph


def main():
    global SCRIPT_VMAX_DEG
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(HERE, "weights", "dagger6_real_iter10"), help="stem of .plan/.pt (dagger6_real_iter10: 87.8 % in sim on the measured drive, 1024 episodes)")
    ap.add_argument("--obs_noise", type=float, default=OBS_NOISE, help="Gaussian noise added to the observation (training value; 0 makes the policy stall)")
    ap.add_argument("--lead_max", type=float, default=LEAD_MAX_DEG, help="max lead (deg) of the streamed reference over the measured joint")
    ap.add_argument("--no_contact", action="store_true", help="disable the torque contact guard / attach emulation")
    ap.add_argument("--no_extrap", action="store_true", help="fix-1 off: two-point segments (reference runs dry between chunks)")
    ap.add_argument("--track_bias", type=float, nargs=2, default=[0.0, 0.0], help="subtract this (bx, by) m from every tracker detection (measured 2026-09-20: +0.0007, -0.0169)")
    ap.add_argument("--touch_calib", action="store_true", help="measure the object top by a slow touch at the detected xy before the episode (camera tops are 1-2.5 cm low on cups)")
    ap.add_argument("--track", action="store_true", help="follow the object live from rl/d435_track.py (udp :9701); xy updated each decision until the cup is within 8 cm")
    ap.add_argument("--ref", default="spline", choices=["linear", "spline"],
                    help="fix 3: reference sent to the Pi -- linear (sampled chunk, velocity kinks at every segment joint) or spline (uniform cubic B-spline control points, C2)")
    ap.add_argument("--lead", type=float, default=0.045, help="fix 2: Pi samples the reference this far ahead (s) = drive dead-time; 0 = off")
    ap.add_argument("--k0", type=float, default=STREAM_GAINS["k0"], help="Pi stream law position gain (1/s)")
    ap.add_argument("--k1", type=float, default=STREAM_GAINS["k1"], help="Pi stream law velocity-error gain")
    ap.add_argument("--torch", action="store_true", help="use the torch checkpoint instead of the TensorRT engine")
    ap.add_argument("--pi", default="192.168.50.2")
    ap.add_argument("--obj", type=float, nargs=3, default=None, help="object CENTRE in the robot base frame (m)")
    ap.add_argument("--half", type=float, nargs=3, default=[0.025, 0.025, 0.02], help="object half extents (training box: 5x5x4 cm)")
    ap.add_argument("--obj_from_tcp", action="store_true", help="take the object top centre from the current cup tip (jog the cup onto the object first)")
    ap.add_argument("--goal", type=float, nargs=3, default=None, help="goal for the object centre (m); default: +10 cm z above the object, 12 cm toward -y")
    ap.add_argument("--dq_max", type=float, default=DQ_MAX_DEG, help="deg per decision (training 2.0)")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--go_home", action="store_true", help="stream to the training start pose before the episode")
    ap.add_argument("--script_vmax", type=float, default=SCRIPT_VMAX_DEG,
                    help="peak joint speed (deg/s) of the scripted min-jerk legs (home, calib, retract, place); "
                         "the drive faults above ~36 deg/s, object-carrying legs use 12 deg/s regardless")
    ap.add_argument("--keep_suction", action="store_true", help="leave suction on at the end (object stays on the cup)")
    ap.add_argument("--touch_only", action="store_true", help="demo mode: never activate suction; end at first firm contact, retract 5 cm and return home")
    ap.add_argument("--exec", action="store_true", help="actually send commands (default: dry run)")
    ap.add_argument("--force", action="store_true", help="skip the object-height / start-pose sanity checks")
    ap.add_argument("--log", default=None, help="write per-step JSON log here")
    ap.add_argument("--selftest", action="store_true", help="run the controller loop against the simulator instead of the robot")
    ap.add_argument("--episodes", type=int, default=8, help="selftest episodes")
    ap.add_argument("--drive", default="real", choices=["real", "ideal"], help="selftest drive model")
    a = ap.parse_args()
    SCRIPT_VMAX_DEG = float(a.script_vmax)
    if SCRIPT_VMAX_DEG > 34.0:
        raise SystemExit(f"[abort] --script_vmax {SCRIPT_VMAX_DEG} deg/s is at/over the drive's following-error ceiling (~36 deg/s)")

    xml = os.path.join(HERE, "scenes", "box_med.xml")
    ob = ObsBuilder(xml, a.half)
    policy = load_policy(a.policy, use_torch=a.torch)
    dq_max = math.radians(a.dq_max)

    if a.selftest:
        # N episodes in parallel worlds; per-world observation building, policy call and target
        # integration go through the SAME code as the robot path (ObsBuilder.build, tanh policy,
        # q_target += a*dq_max), only the drive/physics is the simulator's
        N = a.episodes
        link = SimLink(a.drive, nworld=N)
        p_obj, p_goal = link.obj(), link.goal()
        q, qd, _ = link.state()
        q_target = q.copy()
        a_prev = np.zeros((N, 7), np.float32)
        t_pol = 0.0
        for k in range(a.steps):
            q, qd, _ = link.state()
            p_obj = link.obj()
            obs = np.stack([ob.build(q[w], qd[w], p_obj[w], p_goal[w], a_prev[w], q_target[w])[0] for w in range(N)])
            obs_clean = obs
            if a.obs_noise > 0:
                obs = obs + np.random.default_rng(k).normal(0, a.obs_noise, size=obs.shape).astype(np.float32)
            ref = link.env_obs()
            err = np.abs(obs_clean - ref).max()
            if k == 0:
                print(f"[selftest] obs layout vs env.observe(): max abs diff {err:.2e} over {N} worlds")
                if err > 1e-3:
                    raise SystemExit("observation mismatch -- do not run on the robot")
                err_max, err_arg = 0.0, None
            if err > err_max:
                err_max, err_arg = err, (k, int(np.abs(obs_clean - ref).max(0).argmax()))
            t1 = time.time()
            act = np.stack([np.clip(policy(obs[w]), -1, 1) for w in range(N)]).astype(np.float32)
            t_pol += time.time() - t1
            q_target = q_target + act[:, :6] * dq_max
            a_prev = act
            if link.step(act):
                break
        placed = link.info["placed"]
        sealed = link.env.ever_sealed.cpu().numpy()
        lag = np.degrees(np.abs(q_target - link.state()[0])).max()
        print(f"[selftest] obs vs env.observe() over the episode: max abs diff {err_max:.2e} at (step, obs index) {err_arg}")
        print(f"[selftest] {N} episodes ({a.drive} drive): success {100 * placed.mean():.1f} %  sealed {100 * sealed.mean():.0f} %  "
              f"policy {1e6 * t_pol / (N * (k + 1)):.0f} us/call  final cmd-vs-meas lag {lag:.2f} deg")
        return

    link = PiLink(a.pi, a.exec, ref_mode=a.ref)
    link.extrapolate = not a.no_extrap
    print(f"[ctrl] reference mode: {a.ref}")
    STREAM_GAINS.update(lead=float(a.lead), k0=float(a.k0), k1=float(a.k1))
    t_wait = time.time()
    while not link.ok() and time.time() - t_wait < 6:
        time.sleep(0.1)
    q, qd, age = link.state()
    if q is None:
        raise SystemExit(f"no feedback from {a.pi}:9999 -- is robot_hal running on the Pi?")
    print(f"[robot] q (URDF deg) {np.round(np.degrees(q), 2).tolist()}  fb age {1000 * age:.0f} ms  clock offset {1000 * link.clock_offset:.1f} ms  cmd {'ok' if link.cmd_sock else 'DOWN'}")
    tcp, R = ob.fk(q)
    print(f"[robot] cup tip {np.round(tcp, 4).tolist()}  cup axis {np.round(R[:, 2], 3).tolist()}")

    if a.obj_from_tcp:
        p_obj = tcp.copy(); p_obj[2] -= a.half[2]
        print(f"[obj] from cup tip: centre {np.round(p_obj, 4).tolist()} (top z {tcp[2]:.4f})")
    elif a.obj is not None:
        p_obj = np.asarray(a.obj, float)
    else:
        raise SystemExit("give --obj x y z (object centre, base frame) or --obj_from_tcp")
    obj_top = p_obj[2] + a.half[2]
    if not (OBJ_TOP_TRAIN[0] <= obj_top <= OBJ_TOP_TRAIN[1]):
        msg = f"object top z {obj_top:.3f} m is outside the training range {OBJ_TOP_TRAIN} (sim table top z=0, object top 0.04); the policy has never seen this height"
        if not a.force:
            raise SystemExit("[abort] " + msg + " -- raise the object (riser) or pass --force")
        print("[warn] " + msg)
    p_goal = np.asarray(a.goal, float) if a.goal is not None else p_obj + np.array([0.0, -0.12, 0.10])
    print(f"[task] object centre {np.round(p_obj, 3).tolist()} half {a.half}  goal {np.round(p_goal, 3).tolist()}  grasp point {np.round(ob.grasp_point(p_obj), 3).tolist()}")
    p_obj0 = np.asarray(p_obj, float).copy()
    guard = Guard(ob, obj_top, force=a.force)

    def stop(*_):
        print("\n[ctrl] interrupted -> suction off, stop streaming (Pi holds)")
        try:
            link.suction(False)
        except Exception:  # noqa: BLE001
            pass
        try:
            report_phases(a.log)
        except Exception:  # noqa: BLE001
            pass
        link.close()
        sys.exit(1)
    signal.signal(signal.SIGINT, stop)

    PHASE_T["run0"] = time.time()
    if a.go_home:
        q_m = scripted_move(link, guard, ob, START_Q, "home (START_Q)", min_T=1.0)
        if q_m is None:
            raise SystemExit("[abort] path to START_Q violates the guard")
        q = q_m
    dev = np.degrees(np.abs(q - START_Q)).max()
    if dev > 6.0 and not a.force:
        raise SystemExit(f"[abort] arm is {dev:.1f} deg from the training start pose START_Q; use --go_home or --force")
    v0 = guard.check_q(q)
    if v0:
        raise SystemExit("[abort] start pose violates: " + "; ".join(v0))
    if a.touch_calib:
        # measure the object's top by a slow vertical touch at the detected xy (camera tops are 1-6 cm low on cups)
        PHASE_T["calib0"] = time.time()
        import mujoco
        demo = {"__file__": os.path.join(ROOT, "mjwarp_pick_demo.py")}
        exec(open(demo["__file__"]).read().split("if __name__")[0], demo)
        dik = mujoco.MjData(ob.m)
        q0c, _, _ = link.state()
        top_cam = 2 * float(ob.half[2])
        p_hover = [float(p_obj[0]), float(p_obj[1]), top_cam + 0.07]     # 7 cm: camera tops can be 6 cm low
        q_h, err = demo["ik"](ob.m, dik, "tcp", p_hover, demo["R_DOWN"], q0c); q_h = np.array(q_h)
        q_hov = scripted_move(link, guard, ob, q_h, "calib hover", min_T=1.0) if err < 0.005 else None
        if q_hov is not None:
            base_c = link.torque_baseline(0.8)
            qprev, _, _ = link.state(); z = p_hover[2]; contact_z = None; t_s = time.time(); kk = 0
            v_fast, v_slow = 0.04, 0.02                 # two-speed descent: 4 cm/s down to 2 cm above the camera top,
            z_slow = top_cam + 0.02                     # then 2 cm/s (the old descent was 2 cm/s the whole 11 cm)
            while z > top_cam - 0.04:
                t_dec = t_s + kk * CTRL_DT; d = t_dec - time.time()
                if d > 0: time.sleep(d)
                qm, _, _ = link.state(); tq = link.torque(); tau_c = abs(tq[1] - base_c[1]) + W_J3 * abs(tq[2] - base_c[2]); tipm, _ = ob.fk(qm)
                if tau_c >= TAU_FIRM:
                    contact_z = float(tipm[2]); break
                # contact safety: the camera top may be several cm low, so the object can already be inside the
                # fast stretch -- any torque rise at all drops the descent to 2 cm/s before the firm threshold.
                if tau_c >= 0.4 * TAU_FIRM:
                    z_slow = max(z_slow, z)
                z -= (v_slow if z <= z_slow else v_fast) * CTRL_DT
                qn, e2 = demo["ik"](ob.m, dik, "tcp", [p_hover[0], p_hover[1], z], demo["R_DOWN"], qprev); qn = np.array(qn)
                if e2 > 0.005 or guard.check_q(qn): break
                link.send_segment(qprev, qn, t_dec, seq=kk); qprev = qn; kk += 1
            qm, _, _ = link.state(); tipm, _ = ob.fk(qm)
            q_up, e_up = demo["ik"](ob.m, dik, "tcp", [p_hover[0], p_hover[1], float(tipm[2]) + 0.06], demo["R_DOWN"], qm); q_up = np.array(q_up)
            for q_to, lab in (((q_up, "calib retract") if e_up < 0.005 else (None, None)), (START_Q, "calib home")):
                if q_to is None:
                    continue
                if scripted_move(link, guard, ob, q_to, lab) is None:
                    break
            if contact_z is not None and (contact_z - CUP_R) < top_cam - 0.012:
                print(f"[calib] touch contact at tip z {contact_z:.3f} is {1000 * (top_cam - (contact_z - CUP_R)):.0f} mm below the camera top -> probably missed the object edge; keeping the camera height")
                contact_z = None
            if contact_z is not None:
                top_touch = contact_z - CUP_R
                print(f"[calib] touch: contact at tip z {contact_z:.3f} -> object top {top_touch:.3f} (camera {top_cam:.3f}, diff {1000 * (top_touch - top_cam):+.0f} mm)")
                ob.half[2] = max(0.008, top_touch / 2); p_obj[2] = ob.half[2]; p_obj0[2] = ob.half[2]
                guard.z_floor = top_touch - 0.03
            else:
                print("[calib] touch found no contact above the camera top - keeping the camera height")
        else:
            print(f"[calib] hover IK/guard failed (err {err:.4f}) - keeping the camera height")
        PHASE_T["calib1"] = time.time()
    tau_base = None
    if not a.no_contact:
        tau_base = link.torque_baseline(1.0)
        print(f"[contact] torque baseline (1 s median) {np.round(tau_base, 3).tolist()}  firm {TAU_FIRM} hard {TAU_HARD}")
    print(f"[ctrl] {'EXECUTING' if a.exec else 'DRY RUN'}: {a.steps} decisions at {1 / CTRL_DT:.0f} Hz, dq_max {a.dq_max} deg, lead_max {a.lead_max} deg, obs noise {a.obs_noise}, gains {STREAM_GAINS}")
    tracker = None
    if a.track:
        tracker = ObjectTracker()
        tracker.bias = tuple(a.track_bias)
        t_w = time.time()
        while tracker.poll() is None and time.time() - t_w < 3.0:
            time.sleep(0.1)
        d0 = tracker.poll()
        print(f"[track] live detection {'OK: ' + str({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d0.items()}) if d0 else 'NOT RECEIVED (is rl/d435_track.py running?)'}")
        if d0 is None:
            raise SystemExit("[abort] --track requested but no detections on udp :9701")
    rows, sealed = run_episode(link, policy, ob, guard, p_obj, p_goal, a.steps, dq_max, a.log, obs_noise=a.obs_noise, lead_max_deg=a.lead_max,
                               tau_base=tau_base, touch_only=a.touch_only, tracker=tracker)
    if a.touch_only:
        sealed = False
        link.suction(False)                       # belt and braces: the pin is never set in this mode
        q_now, _, _ = link.state()
        tip_now, _ = ob.fk(q_now)
        import mujoco
        demo = {"__file__": os.path.join(ROOT, "mjwarp_pick_demo.py")}
        exec(open(demo["__file__"]).read().split("if __name__")[0], demo)
        dik = mujoco.MjData(ob.m)
        q_up, err = demo["ik"](ob.m, dik, "tcp", [float(tip_now[0]), float(tip_now[1]), float(tip_now[2]) + 0.05], demo["R_DOWN"], q_now)
        # place pose: cup (empty) over the goal, as if carrying the object there
        p_place = [float(p_goal[0]), float(p_goal[1]), float(p_goal[2]) + float(ob.half[2]) + CUP_R]
        q_place, err_p = demo["ik"](ob.m, dik, "tcp", p_place, demo["R_DOWN"], np.array(q_up))
        legs = [(np.array(q_up), "retract 5 cm")]
        if err_p < 0.005:
            legs.append((np.array(q_place), f"to place pose {np.round(p_place, 3).tolist()}"))
        else:
            print(f"[ctrl] place pose IK failed (err {err_p:.4f}) -> skipping that leg")
        legs.append((START_Q, "home"))
        for q_to, lab in legs:
            if scripted_move(link, guard, ob, q_to, lab) is None:
                break
    if not sealed and not a.touch_only:
        link.suction(False)
        q_now, _, _ = link.state(); tip_now, _ = ob.fk(q_now)
        import mujoco
        demo = {"__file__": os.path.join(ROOT, "mjwarp_pick_demo.py")}
        exec(open(demo["__file__"]).read().split("if __name__")[0], demo)
        q_up, err = demo["ik"](ob.m, mujoco.MjData(ob.m), "tcp", [float(tip_now[0]), float(tip_now[1]), float(tip_now[2]) + 0.08], demo["R_DOWN"], q_now)
        for q_to, lab in ((np.array(q_up), "retract 8 cm"), (START_Q, "home")):
            if scripted_move(link, guard, ob, q_to, lab) is None:
                break
    if sealed and not a.keep_suction:
        # gentle place-down: lower the (assumed) held object onto the table at the current xy, then release
        q_now, _, _ = link.state()
        tip_now, _ = ob.fk(q_now)
        z_place = 2 * float(ob.half[2]) + CUP_R + 0.004          # object bottom on the table, cup still pressed on the top
        import mujoco
        demo = {"__file__": os.path.join(ROOT, "mjwarp_pick_demo.py")}
        exec(open(demo["__file__"]).read().split("if __name__")[0], demo)
        q_dn, err = demo["ik"](ob.m, mujoco.MjData(ob.m), "tcp", [float(tip_now[0]), float(tip_now[1]), max(z_place, 0.03)], demo["R_DOWN"], q_now)
        q_dn = np.array(q_dn)
        PHASE_T["place0"] = time.time()
        q_dn_meas = None
        if err < 0.005:
            print(f"[ctrl] place-down to z {max(z_place, 0.03):.3f} at ({tip_now[0]:.3f},{tip_now[1]:.3f})")
            q_dn_meas = scripted_move(link, guard, ob, q_dn, "place-down", v_peak_deg=SCRIPT_VMAX_SLOW, min_T=0.8)
        if q_dn_meas is not None:
            # guarded settle: keep descending 2 mm per decision until the object rests on the table (torque rise) or 20 mm
            base_p = link.torque_baseline(0.4)
            qprev, _, _ = link.state(); tipp, _ = ob.fk(qprev); zp = float(tipp[2]); t_s = time.time(); kk = 0; settled = False
            while zp > max(z_place, 0.03) - 0.020:
                t_dec = t_s + kk * CTRL_DT; d = t_dec - time.time()
                if d > 0: time.sleep(d)
                qm, _, _ = link.state(); tq = link.torque(); tau_p = abs(tq[1] - base_p[1]) + W_J3 * abs(tq[2] - base_p[2])
                if tau_p >= 0.05:
                    settled = True; break
                zp -= 0.002
                qn, e2 = demo["ik"](ob.m, mujoco.MjData(ob.m), "tcp", [float(tipp[0]), float(tipp[1]), zp], demo["R_DOWN"], qprev); qn = np.array(qn)
                if e2 > 0.005 or guard.check_q(qn): break
                link.send_segment(qprev, qn, t_dec, seq=kk); qprev = qn; kk += 1
            qm, _, _ = link.state(); tipm, _ = ob.fk(qm)
            print(f"[ctrl] guarded settle: {'object resting (torque rise)' if settled else 'no torque rise'} at tip z {tipm[2]:.3f} after {2 * kk} mm")
            time.sleep(0.15)
        else:
            print(f"[ctrl] place-down IK/guard failed (err {err:.4f}) -> releasing where it is")
        PHASE_T["place1"] = time.time()
        link.suction(False)
        print("[ctrl] suction off (released)")
        time.sleep(0.3)
        q_now, _, _ = link.state(); tip_now, _ = ob.fk(q_now)
        q_up, err = demo["ik"](ob.m, mujoco.MjData(ob.m), "tcp", [float(tip_now[0]), float(tip_now[1]), float(tip_now[2]) + 0.08], demo["R_DOWN"], q_now)
        for q_to, lab in ((np.array(q_up), "retract 8 cm"), (START_Q, "home")):
            if scripted_move(link, guard, ob, q_to, lab) is None:
                break
        # verify with the tracker once the arm is out of the way: is the object at the goal?
        if tracker is not None:
            det = None
            t_w = time.time()
            while det is None and time.time() - t_w < 3.0:
                det = tracker.poll(); time.sleep(0.1)
            if det is not None:
                # judge by the blob closest to the goal (other objects may be on the table)
                cands = det.get("cands") or [det]
                best = min(cands, key=lambda c: np.hypot(c["cx"] - p_goal[0], c["cy"] - p_goal[1]))
                moved = float(np.hypot(best["cx"] - p_obj0[0], best["cy"] - p_obj0[1]))
                d_goal = float(np.hypot(best["cx"] - p_goal[0], best["cy"] - p_goal[1]))
                det = best
                verdict = ("PICK AND PLACE OK" if d_goal < 0.06 else "carried, placed %.0f cm off" % (100 * d_goal)) if (moved > 0.10 and d_goal < 0.10) else ("pushed, not picked" if moved > 0.03 else "seal FAILED (object did not move)")
                print(f"[result] object now at ({det['cx']:.3f},{det['cy']:.3f}); {100 * d_goal:.1f} cm from the goal, displaced {100 * moved:.1f} cm -> {verdict}")
            else:
                print("[result] tracker sees no object after the run")
    print(f"[ctrl] done: {len(rows)} decisions, guard clips {guard.n_clip}, final tip {rows[-1]['tcp'] if rows else None}")
    report_phases(a.log)
    link.close()


if __name__ == "__main__":
    main()
