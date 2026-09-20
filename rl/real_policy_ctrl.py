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
STREAM_GAINS = dict(k0=20.0, k1=0.3, vmax=50.0, vel_scale=17.0, vff=1.0, lead=0.0)
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
ATTACH_AFTER = 0.5                  # s of firm contact with suction on before the object is assumed attached


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

    def __init__(self, host, exec_):
        self.host, self.exec = host, exec_
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

    def send_segment(self, q_from, q_to, t_start, dt=CTRL_DT, seq=0, tag=None):
        """Linear reference from q_from to q_to over dt, anchored at desktop time t_start.
        With `extrapolate`, a third point continues the same velocity for one more dt: the next chunk
        (anchored at t_start + dt) arrives a few ms late and would otherwise leave the welder past the
        end of the reference (v_ref = 0, velocity command dips once per decision = the 10 Hz ripple).
        The welder drops points at t >= the next anchor, so the extrapolated point is only followed for
        those few ms -- or for at most dt if the desktop stalls, after which the Pi holds."""
        pts = [[round(float(x), 4) for x in rad_to_linuxcnc_deg(q_from)], [round(float(x), 4) for x in rad_to_linuxcnc_deg(q_to)]]
        if self.extrapolate:
            q_ext = np.asarray(q_to, float) + (np.asarray(q_to, float) - np.asarray(q_from, float))
            pts.append([round(float(x), 4) for x in rad_to_linuxcnc_deg(q_ext)])
        self._send({"chunk": pts, "traj_dt": dt, "t_anchor": t_start - self.clock_offset, "seq": seq, "tag": tag or f"pol:{seq}",
                    "gains": STREAM_GAINS, "period_ms": PERIOD_MS, "vel_cmd_max": VEL_CMD_MAX, "log_stamp": "policy"})

    def send_path(self, qs, dt, t_start):
        pts = [[round(float(x), 4) for x in rad_to_linuxcnc_deg(q)] for q in qs]
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


# ---------------------------------------------------------------- controller loop
def run_episode(link, policy, ob, guard, p_obj, p_goal, n_steps, dq_max, log, obs_noise=OBS_NOISE, lead_max_deg=LEAD_MAX_DEG,
                tau_base=None, verbose=True, touch_only=False):
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
    t0 = time.time()
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
        obs, tcp, R = ob.build(q, qd, p_obj, p_goal, a_prev, q_virtual)
        if obs_noise > 0:
            obs = obs + rng.normal(0, obs_noise, size=obs.shape).astype(np.float32)
        a = np.clip(policy(obs), -1, 1).astype(np.float32)
        dq = a[:6] * dq_max
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
        # contact from the drive torque
        tau = 0.0
        if tau_base is not None:
            tq = link.torque()
            tau = abs(tq[1] - tau_base[1]) + W_J3 * abs(tq[2] - tau_base[2])
        if tau >= TAU_HARD:
            reason = f"HARD contact (tau {tau:.3f} >= {TAU_HARD})"
            link.send_segment(q_send_prev, q, t_dec, seq=k)
            break
        # firm contact is only meaningful near the object (contact_detector's ARM gate): the first
        # acceleration from rest gave tau 0.084 on 2026-09-20 and ended a demo at step 1. Hard contact
        # (TAU_HARD) stays armed everywhere as the backstop.
        near = np.linalg.norm(ob.grasp_point(p_obj) - tcp_now) < 0.06
        firm = (tau >= TAU_FIRM) and near
        if firm and touch_only:
            reason = f"TOUCH (tau {tau:.3f}) at tip {np.round(tcp_now, 4).tolist()} -- touch-only demo ends here"
            link.send_segment(q_send_prev, q, t_dec, seq=k)
            break
        if firm and sealed_cmd and not attached:
            t_contact_on = t_contact_on or time.time()
            if time.time() - t_contact_on >= ATTACH_AFTER:
                attached, attach_off = True, p_obj - tcp_now
                if verbose:
                    print(f"[ctrl] step {k}: firm contact + suction for {ATTACH_AFTER} s -> object assumed ATTACHED (offset {np.round(attach_off, 3).tolist()})")
        elif not firm:
            t_contact_on = None
        # reference actually streamed: bounded lead; hold on firm contact
        lead = np.clip(q_new - q, -lead_max, lead_max)
        q_send = q if (firm and not attached) else q + lead
        link.send_segment(q_send_prev, q_send, t_dec, seq=k)
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
    if log:
        json.dump(rows, open(log, "w"))
    print(f"[ctrl] episode end: {reason} after {len(rows)} decisions")
    return rows, sealed_cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(HERE, "weights", "dagger6_real_iter10"), help="stem of .plan/.pt (dagger6_real_iter10: 87.8 % in sim on the measured drive, 1024 episodes)")
    ap.add_argument("--obs_noise", type=float, default=OBS_NOISE, help="Gaussian noise added to the observation (training value; 0 makes the policy stall)")
    ap.add_argument("--lead_max", type=float, default=LEAD_MAX_DEG, help="max lead (deg) of the streamed reference over the measured joint")
    ap.add_argument("--no_contact", action="store_true", help="disable the torque contact guard / attach emulation")
    ap.add_argument("--no_extrap", action="store_true", help="fix-1 off: two-point segments (reference runs dry between chunks)")
    ap.add_argument("--torch", action="store_true", help="use the torch checkpoint instead of the TensorRT engine")
    ap.add_argument("--pi", default="192.168.50.2")
    ap.add_argument("--obj", type=float, nargs=3, default=None, help="object CENTRE in the robot base frame (m)")
    ap.add_argument("--half", type=float, nargs=3, default=[0.025, 0.025, 0.02], help="object half extents (training box: 5x5x4 cm)")
    ap.add_argument("--obj_from_tcp", action="store_true", help="take the object top centre from the current cup tip (jog the cup onto the object first)")
    ap.add_argument("--goal", type=float, nargs=3, default=None, help="goal for the object centre (m); default: +10 cm z above the object, 12 cm toward -y")
    ap.add_argument("--dq_max", type=float, default=DQ_MAX_DEG, help="deg per decision (training 2.0)")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--go_home", action="store_true", help="stream to the training start pose (5 deg/s) before the episode")
    ap.add_argument("--keep_suction", action="store_true", help="leave suction on at the end (object stays on the cup)")
    ap.add_argument("--touch_only", action="store_true", help="demo mode: never activate suction; end at first firm contact, retract 5 cm and return home")
    ap.add_argument("--exec", action="store_true", help="actually send commands (default: dry run)")
    ap.add_argument("--force", action="store_true", help="skip the object-height / start-pose sanity checks")
    ap.add_argument("--log", default=None, help="write per-step JSON log here")
    ap.add_argument("--selftest", action="store_true", help="run the controller loop against the simulator instead of the robot")
    ap.add_argument("--episodes", type=int, default=8, help="selftest episodes")
    ap.add_argument("--drive", default="real", choices=["real", "ideal"], help="selftest drive model")
    a = ap.parse_args()

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

    link = PiLink(a.pi, a.exec)
    link.extrapolate = not a.no_extrap
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
    guard = Guard(ob, obj_top, force=a.force)

    def stop(*_):
        print("\n[ctrl] interrupted -> suction off, stop streaming (Pi holds)")
        try:
            link.suction(False)
        except Exception:  # noqa: BLE001
            pass
        link.close()
        sys.exit(1)
    signal.signal(signal.SIGINT, stop)

    if a.go_home:
        dq_home = np.degrees(np.abs(START_Q - q)).max()
        T = max(1.0, dq_home / 5.0)
        n = int(T / 0.1) + 1
        qs = [q + (START_Q - q) * (i / (n - 1)) for i in range(n)]
        print(f"[home] streaming to START_Q over {T:.1f} s ({dq_home:.1f} deg max){'' if a.exec else '  [dry run]'}")
        link.send_path(qs, 0.1, time.time() + 0.2)
        time.sleep(T + 1.2)
        q, _, _ = link.state()
    dev = np.degrees(np.abs(q - START_Q)).max()
    if dev > 6.0 and not a.force:
        raise SystemExit(f"[abort] arm is {dev:.1f} deg from the training start pose START_Q; use --go_home or --force")
    v0 = guard.check_q(q)
    if v0:
        raise SystemExit("[abort] start pose violates: " + "; ".join(v0))
    tau_base = None
    if not a.no_contact:
        tau_base = link.torque_baseline(1.0)
        print(f"[contact] torque baseline (1 s median) {np.round(tau_base, 3).tolist()}  firm {TAU_FIRM} hard {TAU_HARD}")
    print(f"[ctrl] {'EXECUTING' if a.exec else 'DRY RUN'}: {a.steps} decisions at {1 / CTRL_DT:.0f} Hz, dq_max {a.dq_max} deg, lead_max {a.lead_max} deg, obs noise {a.obs_noise}, gains {STREAM_GAINS}")
    rows, sealed = run_episode(link, policy, ob, guard, p_obj, p_goal, a.steps, dq_max, a.log, obs_noise=a.obs_noise, lead_max_deg=a.lead_max,
                               tau_base=tau_base, touch_only=a.touch_only)
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
            q_from, _, _ = link.state()
            T = max(0.5, np.degrees(np.abs(q_to - q_from)).max() / 6.0)
            n = int(T / 0.1) + 1
            qs = [q_from + (q_to - q_from) * i / (n - 1) for i in range(n)]
            if any(guard.check_q(qq) for qq in qs):
                print(f"[guard] {lab} path violates limits -> stopping here"); break
            print(f"[ctrl] {lab}: {T:.1f} s{'' if a.exec else ' [dry run]'}")
            link.send_path(qs, 0.1, time.time() + 0.2)
            time.sleep(T + 0.8 + (1.0 if lab.startswith("to place") else 0.0))
    if sealed and not a.keep_suction:
        time.sleep(0.5)
        link.suction(False)
        print("[ctrl] suction off")
    print(f"[ctrl] done: {len(rows)} decisions, guard clips {guard.n_clip}, final tip {rows[-1]['tcp'] if rows else None}")
    link.close()


if __name__ == "__main__":
    main()
