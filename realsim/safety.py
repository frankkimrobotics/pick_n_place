"""realsim.safety :: deterministic safety filter for the streamed joint targets (same code for twin and robot).

Checked on a private MjData (kinematics only), against STATIC geometry the robot also knows on hardware:
table/floor, bin walls, a virtual back wall (x < wall_x, own margin wall_margin), joint limits, joint speed. Objects are not used.

  q_out, info = filt(q_cmd, q_tgt, dt)
    1. speed:   |dq| / dt <= v_lim            (uniform scaling, keeps direction)
       accel:   |d(dq/dt)| <= a_lim * dt      (optional; removes velocity reversals at chunk joins)
    2. limits:  stay jl_margin inside jnt_range
    3. collision: largest step fraction alpha in (1, 1/2, 1/4, 0) whose pose keeps
                  clearance >= margin(speed)  OR  does not reduce the current clearance (escape is always allowed)
"""
import mujoco
import numpy as np

ARM_BODIES = ("link2", "link3", "link4", "link5", "link6", "camera_mount")
_CORN = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


class SafetyFilter:
    def __init__(self, m, wall_x=-0.30, margin=0.005, k_stop=0.06, v_lim=np.radians(30.0), jl_margin=np.radians(2.0),
                 dmax=0.10, a_lim=0.0, wall_margin=None):
        self.m, self.d = m, mujoco.MjData(m)
        self.wall_x, self.margin, self.k_stop, self.v_lim, self.jl, self.dmax = wall_x, margin, k_stop, v_lim, jl_margin, dmax
        self.wall_extra = 0.0 if wall_margin is None else max(0.0, wall_margin - margin)   # back wall kept wall_margin away
        gb = lambda g: m.body(m.geom_bodyid[g]).name
        self.arm = [g for g in range(m.ngeom) if gb(g) in ARM_BODIES]
        self.world = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0]
        self.tip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.lo, self.hi = m.jnt_range[:6, 0] + jl_margin, m.jnt_range[:6, 1] - jl_margin
        self._ft = np.zeros(6)
        self.a_lim, self.v_prev = a_lim, np.zeros(6)          # joint acceleration limit (rad/s^2), 0 = off
        self.n = dict(calls=0, speed=0, limit=0, slowed=0, blocked=0, accel=0)

    def reset(self):
        """call when the stream is held / re-anchored: the arm is (being brought) to rest"""
        self.v_prev = np.zeros(6)

    def clearance(self, q):
        """min distance arm-static world (incl. back wall), and the tip position"""
        m, d = self.m, self.d
        d.qpos[:6] = q
        mujoco.mj_kinematics(m, d)
        c = min(mujoco.mj_geomDistance(m, d, ga, gw, self.dmax, self._ft) for ga in self.arm for gw in self.world)
        for g in self.arm:
            R = d.geom_xmat[g].reshape(3, 3)
            xc = (d.geom_xpos[g] + R @ m.geom_aabb[g, :3])[0]
            c = min(c, float(xc + ((_CORN * m.geom_aabb[g, 3:]) @ R[0]).min()) - self.wall_x - self.wall_extra)
        return c, d.site_xpos[self.tip].copy()

    def __call__(self, q_cmd, q_tgt, dt):
        self.n["calls"] += 1
        dq = np.asarray(q_tgt, float) - q_cmd
        info = dict(alpha=1.0, why="")
        mx = float(np.abs(dq).max())
        if mx > self.v_lim * dt:
            dq *= self.v_lim * dt / mx
            self.n["speed"] += 1
            info["why"] = "speed"
        if self.a_lim > 0:                                    # rate-limit the change of joint velocity (uniform)
            dv = dq / dt - self.v_prev
            mv = float(np.abs(dv).max())
            if mv > self.a_lim * dt:
                dq = (self.v_prev + dv * self.a_lim * dt / mv) * dt
                self.n["accel"] += 1
                info["why"] = "accel"
        qn = np.clip(q_cmd + dq, np.minimum(self.lo, q_cmd), np.maximum(self.hi, q_cmd))   # never push further out
        if np.abs(qn - (q_cmd + dq)).max() > 1e-9:
            self.n["limit"] += 1
            info["why"] = "limit"
        dq = qn - q_cmd
        c0, p0 = self.clearance(q_cmd)
        info["clr0"] = c0
        for al in (1.0, 0.5, 0.25):
            c1, p1 = self.clearance(q_cmd + al * dq)
            need = self.margin + self.k_stop * float(np.linalg.norm(p1 - p0)) / dt
            if c1 >= need or c1 >= c0 + 1e-5:
                if al < 1.0:
                    self.n["slowed"] += 1
                    info["why"] = "slow"
                info.update(alpha=al, clr=c1)
                self.v_prev = al * dq / dt
                return q_cmd + al * dq, info
        self.n["blocked"] += 1
        info.update(alpha=0.0, clr=c0, why="blocked")
        self.v_prev = np.zeros(6)
        return np.array(q_cmd, float), info
