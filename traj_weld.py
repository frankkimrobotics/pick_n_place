"""traj_weld :: a velocity-continuous trajectory WELDER for online (streaming) control.

The online planner emits short trajectory CHUNKS (e.g. 16-20 waypoints) faster than
they execute. The welder holds a single live time-parametrized reference q_ref(t) and
blends each new chunk onto it so that BOTH position and velocity stay continuous at the
junction -- the robot never stops between chunks, it "welds" the new chunk to the motion
it is already executing.

Mechanism (the weld):
  - all times are absolute wall-clock; the chunk is anchored at t_anchor (slightly in the
    future, covering planner+comms latency) so the controller is still on the OLD ref there.
  - over [t_anchor, t_anchor+blend] the output cross-fades old->chunk with a smoothstep a(s)
    whose derivative is 0 at both ends => the result matches the OLD velocity at t_anchor and
    the CHUNK velocity at t_anchor+blend (C1 continuity), i.e. it leaves the current motion
    tangentially and arrives on the chunk tangentially. No velocity jump = no jerk spike.

The controller samples q_ref(t) at its servo rate. The planner keeps a MIRROR welder fed
with the same chunks, so it can predict the junction state (q, qdot) it must plan from.
"""
import numpy as np


def smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)        # a(0)=0,a(1)=1, a'(0)=a'(1)=0


class TrajectoryWelder:
    def __init__(self, dof=6, fine_dt=0.01):
        self.dof = int(dof)
        self.fine_dt = float(fine_dt)
        self.t = None          # absolute times [M]
        self.q = None          # [M, dof]
        self._last_t = None    # latest sampled time (the trim keeps the active region around it)

    def seed(self, q, t0, hold=5.0):
        """Initialise a resting reference at q from t0 (held for `hold` s)."""
        q = np.asarray(q, float).reshape(self.dof)
        self.t = np.array([t0, t0 + hold])
        self.q = np.vstack([q, q])

    def sample(self, t):
        """q at time t. Holds the endpoints (clamp) outside the reference span."""
        t = float(t)
        if self.t is None:
            return None
        self._last_t = t
        return np.array([np.interp(t, self.t, self.q[:, j]) for j in range(self.dof)])

    def velocity(self, t, h=None):
        h = self.fine_dt if h is None else h
        a, b = self.sample(t - h), self.sample(t + h)
        return (b - a) / (2.0 * h)

    def horizon_end(self):
        return None if self.t is None else float(self.t[-1])

    def weld(self, chunk_q, chunk_dt, t_anchor, blend=0.06):
        """Blend a new chunk (chunk_q[N,dof] @ chunk_dt) onto the live reference,
        anchored at absolute time t_anchor, velocity-continuous over `blend` seconds."""
        chunk_q = np.asarray(chunk_q, float)
        if chunk_q.ndim != 2 or chunk_q.shape[0] < 2:
            return
        n = chunk_q.shape[0]
        tc = t_anchor + np.arange(n) * float(chunk_dt)
        if self.t is None:
            self.seed(chunk_q[0], t_anchor - 1.0)

        # fine uniform timeline from the anchor to the chunk end
        tnew = np.arange(t_anchor, tc[-1] + 1e-9, self.fine_dt)
        if len(tnew) < 2:
            return
        chunk_s = np.column_stack([np.interp(tnew, tc, chunk_q[:, j]) for j in range(self.dof)])
        old_s = np.column_stack([np.interp(tnew, self.t, self.q[:, j]) for j in range(self.dof)])
        a = smoothstep((tnew - t_anchor) / max(blend, 1e-6))[:, None]
        blended = (1.0 - a) * old_s + a * chunk_s

        keep = self.t < t_anchor            # keep the recent past (for sampling/velocity)
        self.t = np.concatenate([self.t[keep], tnew])
        self.q = np.vstack([self.q[keep], blended])
        # trim only history BEFORE the last sampled time (keep the active + future ref;
        # keying off self.t[-1] wrongly discarded the present when chunks anchor far ahead).
        cut = (self._last_t - 2.0) if self._last_t is not None else (self.t[-1] - 30.0)
        m = self.t >= cut
        if m.sum() >= 2:
            self.t, self.q = self.t[m], self.q[m]

    def reached(self, q_goal, t, tol=0.02):
        s = self.sample(t)
        return s is not None and float(np.abs(s - np.asarray(q_goal)).max()) < tol


def retime_to_velocity(path, dt_in, v_start, v_cruise, fine_dt=0.01, ramp=0.4):
    """Re-time a position path so it LEAVES at per-joint speed ~v_start (rad/s, the current
    motion) and ramps to v_cruise -- used by the MPC-style mode to keep momentum across a
    rest-to-rest trajopt chunk. Returns (q[fine], fine_dt)."""
    path = np.asarray(path, float)
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if d[-1] < 1e-6:
        return path[:1].copy(), fine_dt
    sg = np.linspace(0, d[-1], 2000)
    vg = np.where(sg < ramp * d[-1],
                  v_start + (v_cruise - v_start) * (sg / max(ramp * d[-1], 1e-6)),
                  v_cruise)
    vg = np.maximum(vg, 1e-3)
    tg = np.r_[0.0, np.cumsum(np.diff(sg) / vg[:-1])]
    ts = np.arange(0, tg[-1], fine_dt)
    ss = np.interp(ts, tg, sg)
    out = np.column_stack([np.interp(ss, d, path[:, j]) for j in range(path.shape[1])])
    return out, fine_dt
