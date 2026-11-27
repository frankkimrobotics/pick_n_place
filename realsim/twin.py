"""realsim.twin :: Pro 630 digital twin with the measured controller, drive and sensor delays.

`Pro630Twin` owns one MjModel/MjData (any project scene whose first 6 dofs are the arm with
torque motors, e.g. ~/pnp_dataset_zed/scene_*/scene.xml or rl/scenes/box_med.xml) and
replaces "write a target, PD to it" with the deployed chain (see realsim/config.py):

    host.send_spline(pts, t_anchor)  --tx latency-->  Pi stream law @10 ms on 1-tick-old posfb
    --> dead-time --> accel-limited drive velocity --> drive position loop --> physics @2 ms

Sensors read the PAST: `joint_feedback()` is the posfb stream as the desktop sees it,
`state_at(t)` gives the full scene state at an earlier time so a camera frame can be rendered
with its real capture age (`CameraModel`).
"""
from collections import deque

import mujoco
import numpy as np
from realsim.config import phys_substeps

NJ = 6


# ---------------------------------------------------------------- uniform cubic B-spline (mycobot_mpc/spline_ref.py)
def spline_eval(pts, t_anchor, dt, t):
    """(q, qd) of the uniform cubic B-spline over control points `pts` (n, 6), knot i at
    t_anchor + i*dt. Mirrors spline_ref._eval: segment index clamped to [0, n-2], local
    parameter clamped to [0, 1], end points repeated."""
    n = len(pts)
    u = (t - t_anchor) / dt
    i = int(np.floor(u))
    i = min(max(i, 0), n - 2)
    s = min(max(u - i, 0.0), 1.0)
    om = 1.0 - s
    b = np.array([om ** 3, 3 * s ** 3 - 6 * s * s + 4, -3 * s ** 3 + 3 * s * s + 3 * s + 1, s ** 3]) / 6.0
    db = np.array([-0.5 * om * om, (3 * s * s - 4 * s) * 0.5, (-3 * s * s + 2 * s + 1) * 0.5, 0.5 * s * s])
    idx = [min(max(j, 0), n - 1) for j in (i - 1, i, i + 1, i + 2)]
    P = pts[idx]
    return b @ P, (db @ P) / dt


class Pro630Twin:
    def __init__(self, m, d, cfg, rng=None, on_tick=None, hist_s=0.6):
        self.m, self.d, self.cfg = m, d, cfg
        self.rng = rng or np.random.default_rng(0)
        self.on_tick = on_tick                     # callback(twin) after every 10 ms tick (suction, metrics)
        # physics step must divide the Pi period (5 ms stack on 2 ms scenes -> 3 x 1.667 ms, not 4 ms ticks)
        self.sub, self.dt_phys = phys_substeps(cfg["cmd_dt"], float(m.opt.timestep))
        m.opt.timestep = self.dt_phys
        self._quat_adr = [int(m.jnt_qposadr[j]) + (3 if m.jnt_type[j] == 0 else 0)   # free: pos then quat; ball: quat
                          for j in range(m.njnt) if m.jnt_type[j] in (0, 1)]
        m.dof_armature[:NJ] = cfg["armature"]
        m.dof_damping[:NJ] = np.asarray(cfg["damping"], float)
        m.dof_frictionloss[:NJ] = np.asarray(cfg["frictionloss"], float)
        self.kp, self.kd = np.asarray(cfg["kp"], float), np.asarray(cfg["kd"], float)
        self.t = 0.0
        q0 = d.qpos[:NJ].copy()
        self.q_drive, self.v_drive = q0.copy(), np.zeros(NJ)
        self.n_dead = int(round(cfg["dead"] / self.dt_phys))          # dead-time at physics resolution
        self.v_buf = deque([np.zeros(NJ)] * (self.n_dead + 1), maxlen=self.n_dead + 1)
        self.fb = deque([(q0.copy(), np.zeros(NJ))] * (cfg["fb_ticks"] + 1), maxlen=cfg["fb_ticks"] + 1)
        self.q_prev_fb = q0.copy()
        self.ref = (np.repeat(q0[None], 5, 0), -1e9, 0.1)             # (pts, t_anchor, dt): hold
        self.inbox = []                                               # (arrive_t, ref)
        nh = int(round(hist_s / cfg["cmd_dt"])) + 2
        self.hist = deque(maxlen=nh)                                  # (t, qpos, qvel) per tick
        self._record()
        self.stats = dict(vsat=0, ticks=0, ferror=0, late=0)
        self.log = None                                               # list -> per-tick trace if set
        self.point_target, self.point_k0 = None, 6.0                  # point-move mode (validation only)

    # ---------------------------------------------------------- host -> Pi
    def _tx_delay(self):
        c = self.cfg
        if c["tx_spike_p"] > 0 and self.rng.random() < c["tx_spike_p"]:
            return self.rng.uniform(*c["tx_spike"])
        return c["tx_lat"]

    def send_spline(self, pts, t_anchor, dt=0.1):
        """Stream a control polygon (n, 6) URDF rad, knot 0 at t_anchor (sim time)."""
        arrive = self.t + self._tx_delay()
        self.inbox.append((arrive, (np.asarray(pts, float).copy(), float(t_anchor), float(dt))))

    def reset_drive(self):
        """Re-initialise the drive / reference state at the CURRENT joint pose (after a teleport)."""
        q = self.d.qpos[:NJ].copy()
        self.q_drive, self.v_drive = q.copy(), np.zeros(NJ)
        self.v_buf = deque([np.zeros(NJ)] * (self.n_dead + 1), maxlen=self.n_dead + 1)
        self.fb = deque([(q.copy(), np.zeros(NJ))] * (self.cfg["fb_ticks"] + 1), maxlen=self.cfg["fb_ticks"] + 1)
        self.q_prev_fb = q.copy()
        self.ref = (np.repeat(q[None], 5, 0), -1e9, 0.1)
        self.inbox.clear()
        self.hist.clear()
        self._record()

    # ---------------------------------------------------------- one 10 ms Pi tick
    def tick(self):
        c, m, d = self.cfg, self.m, self.d
        late = [x for x in self.inbox if x[0] <= self.t + 1e-9]
        if late:
            self.ref = max(late, key=lambda x: x[0])[1]          # newest arrival wins (Pi re-welds)
            self.inbox = [x for x in self.inbox if x[0] > self.t + 1e-9]
        pts, t_anchor, rdt = self.ref
        q_ref, v_ref = spline_eval(pts, t_anchor, rdt, self.t + c["lead"])
        if self.t + c["lead"] > t_anchor + (len(pts) - 1) * rdt:  # ran past the polygon: the Pi holds
            v_ref = np.zeros(NJ)
            self.stats["late"] += 1
        lead_p = c.get("lead_p")                                  # robot_hal 2026-11-14: position error against
        if lead_p is None or lead_p == c["lead"]:                 # q_ref(t + lead_p); absent = old law (= lead)
            q_ref_p, v_ref_p = q_ref, v_ref
        else:
            q_ref_p, v_ref_p = spline_eval(pts, t_anchor, rdt, self.t + lead_p)
            if self.t + lead_p > t_anchor + (len(pts) - 1) * rdt:
                v_ref_p = np.zeros(NJ)
        q_fb, qd_fb = self.fb[0]                                  # fb_ticks old
        if self.point_target is not None:                         # robot_hal point-move (mpc) law, steps
            v_cmd = -self.point_k0 * (q_fb - self.point_target)
        else:
            v_cmd = c["vff"] * v_ref - c["k0"] * (q_fb - q_ref_p) - c["k1"] * (qd_fb - v_ref_p)
            if c.get("taff", 0.0) > 0:                            # robot_hal acceleration feedforward u += taff * a_ref
                h = c["cmd_dt"]                                    # (envelope 2026-11-25: taff 0.015 halves the step error)
                _, v_nxt = spline_eval(pts, t_anchor, rdt, self.t + c["lead"] + h)
                if self.t + c["lead"] + h > t_anchor + (len(pts) - 1) * rdt:
                    v_nxt = np.zeros(NJ)
                v_cmd = v_cmd + c["taff"] * (v_nxt - v_ref) / h
            if c.get("vfric", 0.0) > 0:                           # breakaway / Coulomb compensation
                v_cmd = v_cmd + c["vfric"] * np.tanh(v_cmd / c.get("veps", np.radians(0.3)))
            q_ref = q_ref_p
        self.stats["vsat"] += int(np.any(np.abs(v_cmd) >= 0.98 * c["vmax"]))
        v_cmd = np.clip(v_cmd, -c["vmax"], c["vmax"])
        # drive as identified open-loop 2026-11-14: velocity gain (vel_scale miscalibration) and a breakaway band
        # (a resting joint ignores commands below ~0.3 deg/s)
        v_cmd = v_cmd * c.get("vgain", 1.0)
        brk = c.get("breakaway", 0.0)
        if brk > 0:
            v_cmd = np.where((np.abs(self.v_drive) < 0.5 * brk) & (np.abs(v_cmd) < brk), 0.0, v_cmd)
        da = c["amax"] * self.dt_phys
        for _ in range(self.sub):
            self.v_buf.append(v_cmd)
            v_eff = self.v_buf[0]                                 # written n_dead physics steps ago
            vlag = c.get("vlag", 0.0)                             # first-order velocity lag of the drive (bare fit 2026-11-14:
            step = v_eff - self.v_drive                           # 32-36 ms dead + 10-20 ms lag); 0 = pure dead time + amax
            if vlag > 0:
                step = step * min(1.0, self.dt_phys / vlag)
            self.v_drive = self.v_drive + np.clip(step, -da, da)
            self.q_drive = self.q_drive + self.v_drive * self.dt_phys
            qj = d.qpos[:NJ]
            lead = self.q_drive - qj
            over = np.abs(lead) > c["ferror"]
            if over.any():                                        # blocked joint: following-error clamp
                self.stats["ferror"] += 1
                self.q_drive = np.where(over, qj + np.clip(lead, -c["ferror"], c["ferror"]), self.q_drive)
                self.v_drive = np.where(over & (np.sign(self.v_drive) == np.sign(lead)), 0.0, self.v_drive)
            tau = self.kp * (self.q_drive - qj) - self.kd * (d.qvel[:NJ] - self.v_drive)
            if c["gravity_comp"]:
                tau = tau + d.qfrc_bias[:NJ]
            d.ctrl[:NJ] = np.clip(tau, -c["tau_max"], c["tau_max"])
            mujoco.mj_step(m, d)
        self.t += c["cmd_dt"]
        q = d.qpos[:NJ].copy()
        self.fb.append((q, (q - self.q_prev_fb) / c["cmd_dt"]))  # robot_hal differentiates posfb
        self.q_prev_fb = q
        self.stats["ticks"] += 1
        self._record()
        if self.log is not None:
            self.log.append(dict(t=self.t, q=q, q_ref=q_ref, v_cmd=v_cmd, q_drive=self.q_drive.copy(), tau=d.ctrl[:NJ].copy()))
        if self.on_tick is not None:
            self.on_tick(self)

    def advance(self, dur):
        n = int(round(dur / self.cfg["cmd_dt"]))
        for _ in range(n):
            self.tick()

    # ---------------------------------------------------------- sensors
    def _record(self):
        self.hist.append((self.t, self.d.qpos.copy(), self.d.qvel.copy()))

    def state_at(self, t_query):
        """(t, qpos, qvel) at t_query, linearly interpolated between the two recorded ticks around it (free-joint
        quaternions re-normalised). Camera capture times fall between Pi ticks; before the history: the oldest tick."""
        h = self.hist
        if t_query <= h[0][0] + 1e-9:
            return h[0]
        prev = h[0]
        for cur in h:
            if cur[0] >= t_query - 1e-9:
                if cur[0] - t_query < 1e-9:
                    return cur
                w = (t_query - prev[0]) / (cur[0] - prev[0])
                qpos = (1 - w) * prev[1] + w * cur[1]
                for a in self._quat_adr:
                    qpos[a:a + 4] /= max(np.linalg.norm(qpos[a:a + 4]), 1e-12)
                return t_query, qpos, (1 - w) * prev[2] + w * cur[2]
            prev = cur
        return prev

    def joint_feedback(self):
        """posfb as the desktop sees it: 100 Hz sample-and-hold + stream latency."""
        c = self.cfg
        t_s = self.t - c["fb_lat"]
        if c["fb_period"] > 0:
            t_s = np.floor(t_s / c["fb_period"] + 1e-9) * c["fb_period"]
        t, qpos, qvel = self.state_at(t_s)
        return qpos[:NJ].copy(), qvel[:NJ].copy(), self.t - t


class CameraModel:
    """Renders a named camera as the host would receive it: the frame captured at the last
    frame-period boundary before (now - latency), drawn from the twin's state history."""

    def __init__(self, twin, renderer, cam, lat, period, phase=0.0, noise=0.0, vopt=None, rng=None):
        self.twin, self.ren, self.cam = twin, renderer, cam
        self.lat, self.period, self.phase, self.noise = lat, period, phase, noise
        self.vopt = vopt
        self.rng = rng or np.random.default_rng(1)
        self.dr = mujoco.MjData(twin.m)
        self.age = 0.0

    def grab(self):
        tw = self.twin
        t_cap = tw.t - self.lat
        if self.period > 0:
            t_cap = np.floor((t_cap - self.phase) / self.period + 1e-9) * self.period + self.phase
        t, qpos, qvel = tw.state_at(t_cap)
        self.age = tw.t - t
        self.dr.qpos[:] = qpos
        self.dr.qvel[:] = qvel
        mujoco.mj_forward(tw.m, self.dr)
        self.ren.update_scene(self.dr, camera=self.cam, scene_option=self.vopt)
        img = self.ren.render()
        if self.noise > 0:
            img = np.clip(img.astype(np.float32) + self.rng.normal(0, self.noise, img.shape), 0, 255).astype(np.uint8)
        return img
