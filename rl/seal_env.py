#!/usr/bin/env python3
"""seal_env :: contact-phase RL environment for the suction seal controller (P0/P1).

One episode = the last ~1.5 s of a pick: the cup starts ~30 mm above a randomized
object and the policy controls the descent, the suction gate, and the lift trigger.
Approach and carry stay with the frozen CFM — this env trains ONLY the window where
the measured cycle-time tails and the seal failures live.

PRIVILEGED (sim-only; reward + critic, never the deployed actor):
    plunger deflection x, spring force F = k*x, true attach state, object params
DEPLOYABLE (exists on the real robot):
    q, qd, tracking error (qref - q), tip z estimate, suction cmd state,
    torque proxy: J^T * F mapped to J2/J3 with randomized gain/bias/noise/latency
    (models the drive torque channel the real contact guard thresholds today)

Probabilistic seal (replaces the always-true weld latch):
    P(seal | suction ON this step) = sig((F-Fmin)/s) * sig((Fmax-F)/s) * exp(-rim_dev/lam)
    Fmin/Fmax/lam randomized per episode (they are UNKNOWN until real calibration).

Action (3): vz in [-30, +30] mm/s, suction gate in {0,1}, lift trigger in {0,1}.
Episode ends: verified lift (success), drop, overforce abort, or 1.5 s timeout.
"""
import os
import sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from rim_seal import rim_seal                                      # noqa: E402

CTRL_DT = 0.004
DECIM = 5                       # policy acts at 50 Hz (native); the real splice runs a 50 Hz inner loop
K_SPRING = 2000.0
F_DAMAGE = 15.0                 # over-press abort (N)
T_MAX = 3.5                     # approach 1.5 + press/dwell + lift + one retry
LIFT_MM = 25.0                  # verified lift height


def _build_scene(rng, xml_path):
    """One randomized object under the cup (primitives spanning the dev range)."""
    kind = rng.choice(["cylinder", "cube", "disc", "tallcyl", "dome"])
    if kind == "dome":            # unsealable control case: a sphere sunk into the table
        r = rng.uniform(0.025, 0.04)
        spec = ("object0", "sphere" if False else "cylinder",      # emulate dome w/ offset cyl
                f"{r:.4f} {r*0.5:.4f}", None, r, r * 0.5)
    sizes = {
        "cylinder": (rng.uniform(0.018, 0.034), rng.uniform(0.015, 0.03)),
        "disc": (rng.uniform(0.03, 0.048), rng.uniform(0.008, 0.014)),
        "tallcyl": (rng.uniform(0.016, 0.026), rng.uniform(0.03, 0.045)),
        "cube": (rng.uniform(0.016, 0.03),) * 2,
        "dome": (rng.uniform(0.025, 0.04),) * 2,
    }[kind]
    x, y = 0.38 + rng.uniform(-0.03, 0.03), rng.uniform(-0.05, 0.05)
    if kind == "cube":
        s_ = sizes[0]
        obj = ("object0", "box", f"{s_:.4f} {s_:.4f} {s_:.4f}",
               f"{x:.4f} {y:.4f} {s_:.4f}", "0.8 0.3 0.2 1", "1 0 0 0")
        top = 2 * s_
    elif kind == "dome":
        r = sizes[0]
        obj = ("object0", "sphere", f"{r:.4f}",
               f"{x:.4f} {y:.4f} {r*0.55:.4f}", "0.8 0.5 0.2 1", "1 0 0 0")
        top = r * 1.55
    else:
        r, hh = sizes
        obj = ("object0", "cylinder", f"{r:.4f} {hh:.4f}",
               f"{x:.4f} {y:.4f} {hh:.4f}", "0.8 0.3 0.2 1", "1 0 0 0")
        top = 2 * hh
    import subprocess
    code = (f"import sys; sys.path.insert(0, {ROOT!r});\n"
            f"import sim_robot_mjcf as g;\n"
            f"g.build(warp=True, plunger=True, objects={[obj]!r}, out_path={xml_path!r})")
    subprocess.run([os.path.expanduser("~/miniconda3/envs/curobo2/bin/python"), "-c", code],
                   check=True, capture_output=True)
    return np.array([x, y]), top, kind


class SealEnv:
    def __init__(self, seed=0, scene_cache=None):
        self.rng = np.random.default_rng(seed)
        self.cache = scene_cache or f"/tmp/seal_scenes_{os.getpid()}"
        os.makedirs(self.cache, exist_ok=True)
        self._demo = {"__file__": os.path.join(ROOT, "mjwarp_pick_demo.py")}
        exec(open(self._demo["__file__"]).read().split("if __name__")[0], self._demo)
        self._scene_n = 0

    # ---------------- helpers
    def _torque_proxy(self, F):
        """privileged force -> what the real drive torque channel would show."""
        tau = F * self._jac_z                     # J^T mapping onto J2/J3 (approx lever)
        tau = tau * self._tp_gain + self._tp_bias
        self._tp_buf.append(tau + self.rng.normal(0, self._tp_noise, 2))
        return self._tp_buf.pop(0)                # latency = buffer length

    def reset(self):
        bank = os.path.expanduser("~/pnp_rl/seal_scene_bank/meta.json")
        if os.path.exists(bank):
            if not hasattr(self, "_bank"):
                import json as _json
                self._bank = _json.load(open(bank))
            m = self._bank[int(self.rng.integers(0, len(self._bank)))]
            xml = m["xml"]; self.obj_xy = np.array(m["obj_xy"]); self.top = m["top"]; self.kind = m["kind"]
        else:
            xml = os.path.join(self.cache, f"s{self._scene_n % 8}.xml")
            if self._scene_n < 8:
                self.obj_xy, self.top, self.kind = _build_scene(self.rng, xml)
                if not hasattr(self, "_scenes"): self._scenes = []
                self._scenes.append((xml, self.obj_xy, self.top, self.kind))
            else:
                xml, self.obj_xy, self.top, self.kind = self._scenes[self._scene_n % 8]
        self._scene_n += 1
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        demo = self._demo
        self.d.qpos[:6] = demo["Q_START"]
        mujoco.mj_forward(self.m, self.d)
        self.tip_sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.tcp_bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "tcp")
        self.obj_bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "object0")
        self.eq_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, "suction_object0")
        self.jadr = self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "cup_plunger")]
        self._dik = mujoco.MjData(self.m)
        # teleport the arm to the hand-off pose: 30 mm above the object (+- a few mm xy)
        aim = self.obj_xy + self.rng.normal(0, 0.004, 2)
        z0 = self.top + 0.008 + 0.020
        q, e = demo["ik"](self.m, self._dik, "tcp", [aim[0], aim[1], z0], demo["R_DOWN"],
                          self.d.qpos[:6])
        self.d.qpos[:6] = q
        self.d.qvel[:] = 0
        mujoco.mj_forward(self.m, self.d)
        self.qref = np.asarray(q, float).copy()
        # per-episode randomization (contact-phase DR)
        self.Fmin = self.rng.uniform(1.0, 4.0)
        self.Fmax = self.rng.uniform(8.0, 14.0)
        self.lam = self.rng.uniform(0.0015, 0.004)
        self._jac_z = self.rng.uniform(0.25, 0.45)         # N -> Nm effective lever
        self._tp_gain = self.rng.uniform(0.7, 1.3)
        self._tp_bias = self.rng.normal(0, 0.02, 2)
        self._tp_noise = self.rng.uniform(0.01, 0.04)
        self._tp_buf = [np.zeros(2)] * int(self.rng.integers(2, 6))   # 40-100 ms latency
        self.obj_mass0 = float(self.m.body_mass[self.obj_bid])
        self.m.body_mass[self.obj_bid] = self.obj_mass0 * self.rng.uniform(0.5, 4.0)
        self.suction = 0
        self.attached = False
        self._contact_paid = False
        self.t = 0.0
        self.z_latch = None
        # rim deviation at the aim point (the geometric sealability)
        tip = self.d.site_xpos[self.tip_sid].copy()
        ax = self.d.site_xmat[self.tip_sid].reshape(3, 3)[:, 2]
        ok, dev, _, _ = rim_seal(self.m, self.d, tip, ax, target_bid=self.obj_bid,
                                 exclude_bid=self.tcp_bid)
        self.rim_dev = min(dev, 0.05)
        return self._obs()

    def _obs(self):
        d = self.d
        q = d.qpos[:6].copy(); qd = d.qvel[:6].copy()
        terr = self.qref - q
        F = K_SPRING * float(d.qpos[self.jadr])
        tau = self._torque_proxy(F)
        tipz = float(d.site_xpos[self.tip_sid][2])
        dep = np.concatenate([q, qd, terr, tau, [tipz, float(self.suction), self.t / T_MAX]])
        priv = np.array([float(d.qpos[self.jadr]), F, float(self.attached),
                         self.rim_dev, self.Fmin, self.Fmax])
        return dep.astype(np.float32), priv.astype(np.float32)

    def step(self, act):
        """act = [vz_mm_s (-30..30), suction (0/1), lift (0/1)]"""
        demo = self._demo
        vz = float(np.clip(act[0], -30, 30)) / 1000.0
        want_suction = act[1] > 0.5
        want_lift = act[2] > 0.5
        rew = -0.02                                       # time cost per 20 ms decision
        # lift overrides descent
        dz = 0.004 if want_lift or self.attached and want_lift else vz * CTRL_DT * DECIM * -1
        tgt = self.d.site_xpos[self.tip_sid].copy()
        tgt[2] += (0.004 if want_lift else vz * CTRL_DT * DECIM)
        qn, e = demo["ik"](self.m, self._dik, "tcp", tgt.tolist(), demo["R_DOWN"], self.d.qpos[:6])
        if e < 0.01:
            ref = np.linspace(self.d.qpos[:6], np.asarray(qn), DECIM)
            qdr, tf, kp, kd = demo["_gains_and_ff"](self.m, self.d, ref)
            sch = dict(qref=ref, qdref=qdr, tau_ff=tf, kp=kp, kd=kd)
            for k in range(DECIM):
                for _ in range(1):
                    self.d.ctrl[:6] = demo["pd_tau"](sch, k, self.d.qpos[:6], self.d.qvel[:6])
                    mujoco.mj_step(self.m, self.d, nstep=1)
            self.qref = np.asarray(qn, float)
        self.t += CTRL_DT * DECIM
        F = K_SPRING * float(self.d.qpos[self.jadr])
        if F > 0.5 and not self._contact_paid:
            rew += 0.3; self._contact_paid = True          # one-time: reach the object
        # suction gate -> probabilistic seal (one Bernoulli draw per rising edge in contact)
        if want_suction and not self.suction:
            self.suction = 1
        if self.suction and not self.attached and F > 0.3:
            p = (1 / (1 + np.exp(-(F - self.Fmin) / 0.8))
                 * 1 / (1 + np.exp((F - self.Fmax) / 0.8))
                 * np.exp(-self.rim_dev / self.lam))
            # hazard is a per-SECOND rate so the seal probability tracks dwell TIME, not the
            # decision count: at 50 Hz (dt 0.02) this is the original 0.25/decision; at 10 Hz
            # (dt 0.1) one decision now carries 5x the probability of one 50 Hz decision, so a
            # 0.5 s dwell seals the same whether it is 25 decisions or 5. (Was p*0.25/decision,
            # which silently made 10 Hz dwells 5x less likely to seal -> sac13 0/150 demos.)
            if self.rng.random() < p * (12.5 * CTRL_DT * DECIM):
                demo["_latch_weld"](self.m, self.d, self.eq_id, self.tcp_bid, self.obj_bid)
                self.attached = True
                self.z_latch = float(self.d.site_xpos[self.tip_sid][2])
                rew += 1.0
        if not want_suction and self.suction:
            self.suction = 0
            if self.attached:
                self.d.eq_active[self.eq_id] = 0
                self.attached = False
        done = False; info = {}
        if F > F_DAMAGE:
            rew -= 2.0; done = True; info["end"] = "overforce"
        if self.attached and self.z_latch is not None:
            if float(self.d.site_xpos[self.tip_sid][2]) - self.z_latch > LIFT_MM / 1000.0:
                # verified lift: object still attached after raising 25 mm
                obj_z = float(self.d.xpos[self.obj_bid][2])
                if obj_z > self.top * 0.5 + 0.010:
                    rew += 3.0; done = True; info["end"] = "lifted"
                else:
                    self.attached = False; rew -= 0.5
        if self.t >= T_MAX and not done:
            done = True; info["end"] = "timeout"
        return self._obs(), rew, done, info
