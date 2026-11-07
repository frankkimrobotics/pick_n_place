#!/usr/bin/env python3
"""wrench_pick :: tightly-packed block array, unknown heights, and a
wrench-set suction attachment.

Scene: 3x3 array of blocks (2 mm gaps) with per-scene random heights; the
TARGET is the centre block. Heights are only BELIEVED (noisy) — the policy
does not know how far to pull before the block clears its neighbours.

Suction attachment = admissible wrench set in the cup frame (vacuum force
F_V, friction MU, cup radius R_CUP):
    pull                  <= F_V                    (tension limit)
    |shear|               <= MU * (F_V - pull)      (friction, less when lifting hard)
    |peel moment|         <= 0.6 * R_EFF * (F_V - pull)
    |torsion|             <= (2/3) * MU * R_EFF * (F_V - pull)
The transmitted wrench is CLAMPED to the set; demanding beyond it for
BREAK_TICKS consecutive substeps DETACHES the block. The last interface
wrench + its margin are policy observations.

    python3 wrench_pick.py --demo                 # wrench model demo (fast vs slow lift)
    python3 wrench_pick.py --make_data --scenes 24
    python3 wrench_pick.py --train
    python3 wrench_pick.py --eval --scenes 6 --render [--naive]
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("MUJOCO_GL", "egl")
import warp as wp                                          # noqa: E402

OUT = os.path.expanduser("~/pnp_rl/wrench_pick")
DT = 0.1
ARRAY_C = (0.34, 0.0)
PITCH = 0.042                     # 4 cm blocks + 2 mm gap
BLOCK_HALF_XY = 0.020
BOX_C = (0.08, 0.40)

# ---- wrench set parameters (physical vacuum cup) ----
F_V = 18.0                        # N vacuum holding force
MU = 0.5
R_EFF = 0.008                     # m effective cup radius
BREAK_TICKS = 100                 # consecutive violating substeps (=200 ms)


# ------------------------------------------------------------------ scene
SCENE_TMPL = """
<mujoco model="wrench_pick">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <compiler angle="radian" meshdir="{meshdir}" autolimits="true"/>
  <include file="{incfile}"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" pos="0 0 -0.12" rgba="0.35 0.37 0.40 1"/>
    <geom name="table" type="box" pos="0.38 0.00 -0.01" size="0.20 0.22 0.01" rgba="0.55 0.44 0.35 1"/>
    <geom name="pbx_floor" type="box" pos="{bx} {by} 0.025" size="0.125 0.125 0.025" rgba="0.2 0.55 0.3 1"/>
    <geom name="pbx_xm" type="box" pos="{xm} {by} 0.125" size="0.006 0.137 0.125" rgba="0.2 0.55 0.3 0.5"/>
    <geom name="pbx_xp" type="box" pos="{xp} {by} 0.125" size="0.006 0.137 0.125" rgba="0.2 0.55 0.3 0.5"/>
    <geom name="pbx_ym" type="box" pos="{bx} {ym} 0.125" size="0.137 0.006 0.125" rgba="0.2 0.55 0.3 0.5"/>
    <geom name="pbx_yp" type="box" pos="{bx} {yp} 0.125" size="0.137 0.006 0.125" rgba="0.2 0.55 0.3 0.5"/>
    {objects}
    <camera name="iso" pos="1.05 -0.55 0.78" xyaxes="0.45 0.89 0 -0.5 0.25 0.83"/>
    <camera name="front" pos="0.38 -0.85 0.45" xyaxes="1 0 0 0 0.45 0.89"/>
  </worldbody>
  <equality>
    <weld name="suction_object0" body1="tcp" body2="object0" active="false" solref="0.005 1"/>
  </equality>
  <contact>
    <!-- once sealed, the wrench-set spring IS the cup/object interface;
         the rigid cup contact double-models it and fights the spring
         (measured: 8-9 N sustained preload, 26 N shear spikes -> breaks) -->
    <exclude body1="tcp" body2="object0"/>
  </contact>
</mujoco>
"""
OBJ = ('<body name="object{n}" pos="{x} {y} {z}">'
       '<joint type="free" name="obj{n}_free" damping="0.6"/>'
       # overdamped contact (dampratio 2): zero restitution, energy-
       # dissipating -- pressed/dropped blocks settle instead of exploding
       '<geom name="g_object{n}" type="box" size="{hx} {hy} {hz}" friction="0.7 0.005 0.0001" '
       'solref="0.02 2" solimp="0.9 0.95 0.002" '
       'rgba="{rgba}" mass="0.07"/></body>')


def make_scene(rng, out_xml):
    """3x3 array; object0 = CENTRE block. Returns per-block meta."""
    import diffog_pnp as DP
    DP.extract_arm_include()
    # soften the cup_tip contact: a suction cup interacts through vacuum,
    # not rigid impact. A stiff cup-vs-block contact while plunging into
    # the tight array is the solver-NaN source. Compliant contact (soft
    # solref, low solimp, contact gap) registers touch WITHOUT exploding.
    inc_path = os.path.join(os.path.expanduser("~/pnp_rl/diffog"), "_arm_include.xml")
    inc = open(inc_path).read()
    if 'name="cup_tip"' in inc and "solref" not in inc.split('name="cup_tip"')[1][:200]:
        inc = inc.replace(
            '<geom name="cup_tip" type="sphere" size="0.008"',
            '<geom name="cup_tip" type="sphere" size="0.008" '
            'solref="0.03 1" solimp="0.5 0.85 0.01" gap="0.002"')
        open(inc_path, "w").write(inc)
    heights = rng.uniform(0.030, 0.060, 9)
    order = [4] + [i for i in range(9) if i != 4]      # centre first -> object0
    objs, meta = [], []
    for n, cell in enumerate(order):
        r, c = divmod(cell, 3)
        x = ARRAY_C[0] + (r - 1) * PITCH
        y = ARRAY_C[1] + (c - 1) * PITCH
        hz = heights[cell] / 2
        rgba = "0.85 0.2 0.15 1" if n == 0 else "0.4 0.55 0.75 1"
        objs.append(OBJ.format(n=n, x=x, y=y, z=hz + 0.001,
                               hx=BLOCK_HALF_XY, hy=BLOCK_HALF_XY, hz=hz,
                               rgba=rgba))
        meta.append(dict(pos=[x, y], half=[BLOCK_HALF_XY, BLOCK_HALF_XY, hz]))
    hi = 0.125
    xml = SCENE_TMPL.format(
        meshdir=os.path.join(HERE, "scenes", "meshes"),
        incfile=os.path.join(os.path.expanduser("~/pnp_rl/diffog"), "_arm_include.xml"),
        bx=BOX_C[0], by=BOX_C[1],
        xm=BOX_C[0] - hi - 0.006, xp=BOX_C[0] + hi + 0.006,
        ym=BOX_C[1] - hi - 0.006, yp=BOX_C[1] + hi + 0.006,
        objects="\n    ".join(objs))
    open(out_xml, "w").write(xml)
    return meta


# ------------------------------------------------------------------ env
def build_env(seed):
    from env_warp import PickEnv
    rng = np.random.default_rng(seed)
    os.makedirs(OUT, exist_ok=True)
    xml = os.path.join(OUT, f"scene_{seed}.xml")
    meta = make_scene(rng, xml)
    env = PickEnv(nworld=1, mode="pnp", dr=False, xml=xml)
    env.auto_reset = False
    # --- wrench-set suction state ---
    env.wr_violation = torch.zeros(1, dtype=torch.long, device=env.device)
    env.wr_last = torch.zeros(1, 6, device=env.device)      # cup-frame wrench
    env.wr_margin = torch.ones(1, device=env.device)
    _install_wrench_model(env)
    return env, meta, rng


def _install_wrench_model(env):
    """Replace the F_MAX cap with the admissible wrench set (clamp inside,
    detach on persistent demand outside)."""
    import types
    from env_warp import TAU_CUP, TAU_TORSION

    def _apply(self):
        if not self.sealed.any():
            self.xfrc[:, self.bid_obj] = 0.0
            self.wr_last[:] = 0.0
            self.wr_margin[:] = 1.0
            return
        tcp, R = self._tcp()
        target = tcp + torch.einsum("nij,nj->ni", R, self.anchor)
        op = self._obj_pos()
        ov = self.qvel[:, self.vadr_obj:self.vadr_obj + 3]
        # damp RELATIVE cup-object motion, not absolute object velocity: a
        # world-frame damper drags C*|v| (15 N at a 0.5 m/s carry) through
        # the shear budget (8.6 N) and every transport breaks the seal
        tv = (tcp - getattr(self, "_tcp_prev", tcp)) / self._dt_phys
        self._tcp_prev = tcp.clone()
        tv = tv.clamp(-2.0, 2.0)                 # teleport guard (resets)
        F_raw = self.K_SUCTION * (target - op) - \
            self._c_suction()[:, None] * (ov - tv)
        # orientation spring (as before, raw)
        R_tgt = torch.einsum("nij,njk->nik", R, self.anchor_R)
        E = torch.einsum("nij,nkj->nik", R_tgt, self._obj_R())
        err = 0.5 * torch.stack([E[:, 2, 1] - E[:, 1, 2],
                                 E[:, 0, 2] - E[:, 2, 0],
                                 E[:, 1, 0] - E[:, 0, 1]], -1)
        T_raw = 20.0 * err
        # ---- decompose in the cup frame ----
        # cup axis pointing cup->object: the tool z-axis (R z) already points
        # DOWN toward the object (R_DOWN convention).  The former -R[:,:,2]
        # inverted the axis, swapping tension and compression -- true vacuum
        # tension was never counted against F_V (found 2026-09-28).
        n_ax = R[:, :, 2]
        f_ax = (F_raw * n_ax).sum(-1)            # + = pushing object away = compression
        pull = (-f_ax).clamp(min=0.0)            # tension the vacuum must carry
        F_t = F_raw - f_ax[:, None] * n_ax
        shear = F_t.norm(dim=-1)
        t_ax = (T_raw * n_ax).sum(-1)
        torsion = t_ax.abs()
        M_p = T_raw - t_ax[:, None] * n_ax
        peel = M_p.norm(dim=-1)
        # ---- admissible set (coupled: margin shrinks as pull grows) ----
        head = (F_V - pull).clamp(min=0.0)
        sh_lim = MU * head + 1e-6
        pe_lim = 0.6 * R_EFF * head * 40.0 + 1e-6    # scaled: spring units
        to_lim = (2.0 / 3.0) * MU * R_EFF * head * 40.0 + 1e-6
        vio = ((pull > F_V) | (shear > sh_lim) |
               (peel > pe_lim) | (torsion > to_lim))
        margin = torch.stack([
            1.0 - pull / F_V, 1.0 - shear / sh_lim,
            1.0 - peel / pe_lim, 1.0 - torsion / to_lim], -1).amin(-1)
        # ---- clamp transmitted wrench INTO the set ----
        pull_c = pull.clamp(max=F_V)
        f_ax_c = torch.where(f_ax < 0, -pull_c, f_ax)
        F_t_c = F_t * (sh_lim / shear.clamp(min=1e-6)).clamp(max=1.0)[:, None]
        F = f_ax_c[:, None] * n_ax + F_t_c
        M_p_c = M_p * (pe_lim / peel.clamp(min=1e-6)).clamp(max=1.0)[:, None]
        t_ax_c = t_ax.clamp(-to_lim, to_lim)
        T = M_p_c + t_ax_c[:, None] * n_ax
        m = self.sealed.float()[:, None]
        self.xfrc[:, self.bid_obj, :3] = F * m
        self.xfrc[:, self.bid_obj, 3:] = T * m
        # angular damping as before
        va = self.vadr_obj
        self.qvel[:, va + 3:va + 6] = torch.where(
            self.sealed[:, None], self.qvel[:, va + 3:va + 6] * 0.55,
            self.qvel[:, va + 3:va + 6])
        # ---- persistence counter + telemetry ----
        self.wr_violation = torch.where(vio & self.sealed,
                                        self.wr_violation + 1,
                                        torch.zeros_like(self.wr_violation))
        cf = torch.stack([f_ax, shear, torch.zeros_like(shear),
                          peel, torsion, torch.zeros_like(shear)], -1)
        self.wr_last = torch.where(self.sealed[:, None], cf, self.wr_last)
        self.wr_margin = torch.where(self.sealed, margin, torch.ones_like(margin))

    def _check(self):
        broke = self.sealed & (self.wr_violation >= BREAK_TICKS)
        if broke.any():
            idx = torch.nonzero(broke).squeeze(-1)
            self.sealed[idx] = False
            self.xfrc[idx, self.bid_obj] = 0.0
            self.wr_violation[idx] = 0
        return broke

    env._apply_suction_force = types.MethodType(_apply, env)
    env._check_break = types.MethodType(_check, env)


# ------------------------------------------------------------------ policy obs
def policy_obs(env, believed, sigma, phase_flags):
    """q(6) qd(6) tcp(3) Rz(3) rel-obj(3) believed_target_h(1) sigma(1)
    believed_neighbor_max_h(1) wrench(5: pull, shear, peel, torsion, margin)
    sealed(1) phase(3)  -> 33-D"""
    tcp, R = env._tcp()
    op = env._obj_pos()
    w = env.wr_last[0]
    obs = torch.cat([
        env.qpos[0, :6], env.qvel[0, :6], tcp[0], R[0, :, 2],
        (op - tcp)[0],
        torch.tensor([believed[0]["half"][2] * 2, sigma[0],
                      max(b["half"][2] * 2 for b in believed[1:])],
                     device=env.device),
        torch.stack([-w[0], w[1], w[3], w[4], env.wr_margin[0]]),
        env.sealed.float(),
        torch.tensor(phase_flags, dtype=torch.float32, device=env.device),
    ])
    return obs


class Policy(torch.nn.Module):
    def __init__(self, obs_dim=33, act_dim=7, h=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, h), torch.nn.ELU(),
            torch.nn.Linear(h, h), torch.nn.ELU(),
            torch.nn.Linear(h, 128), torch.nn.ELU(),
            torch.nn.Linear(128, act_dim))

    def forward(self, o):
        return torch.tanh(self.net(o))


# ------------------------------------------------------------------ teacher
class ExtractionTeacher:
    """Wrench- and belief-aware scripted expert.

    Strategy: hover -> slow press (touch event-hold) -> seal + height
    correction -> VERTICAL extraction with wrench-modulated speed (margin
    low -> slow down; margin high -> speed up) until the block bottom
    clears believed-neighbour-max + 3 sigma -> transit -> drop in the box.
    In naive mode: fixed fast extraction to a fixed height (no wrench, no
    belief) — the ablation baseline."""

    def __init__(self, env, believed, sigma, naive=False, rest_z=None):
        import mujoco
        from mjwarp_pick_demo import ik, R_DOWN
        self.env = env
        self.ik = ik
        self.R_DOWN = R_DOWN
        self.dik = mujoco.MjData(env.mjm)
        self.believed = believed
        self.sigma = sigma
        self.naive = naive
        self.phase = "hover"
        self.CUP = 0.008     # sim tcp sits CUP_R above the block top (env grasp convention)
        m = believed[0]
        self.oxy = m["pos"]
        self.q_wp = {}
        self._plan_leg("hover", [*self.oxy, m["half"][2] * 2 + self.CUP + 0.06])
        self.h_corr = None
        self.lift_v = 0.006 if naive else 0.0025     # m per decision, base
        self.t_phase = 0
        self.rest_z = rest_z                          # obj0 resting z (contact sensor)

    def _plan_leg(self, name, xyz):
        q0 = self.env.qpos[0, :6].cpu().numpy()
        q, _ = self.ik(self.env.mjm, self.dik, "tcp", xyz, self.R_DOWN, q0)
        self.q_wp[name] = np.asarray(q, float)

    def act(self):
        env = self.env
        tcp, R = env._tcp()
        q_now = env.qpos[0, :6].cpu().numpy()
        m = self.believed[0]
        bel_top = m["half"][2] * 2 if self.h_corr is None else self.h_corr
        want = 0.0
        self.t_phase += 1
        if self.phase == "hover":
            tgt = self.q_wp["hover"]
            if np.abs(q_now - tgt).max() < math.radians(1.5):
                self._plan_leg("press", [*self.oxy, bel_top + self.CUP - 0.030])
                self.phase = "press"
                self.t_phase = 0
        elif self.phase == "press":
            want = 1.0
            tgt = self.q_wp["press"]
            # retry: if no seal after 45 ticks, lift 3 cm and re-approach
            if self.t_phase > 100 and getattr(self, "retries", 0) < 3:
                self.retries = getattr(self, "retries", 0) + 1
                if self.retries < 3:
                    self._plan_leg("hover", [*self.oxy, bel_top + self.CUP + 0.05])
                    self.phase = "hover"
                    self.t_phase = 0
                    tgt = self.q_wp["hover"]
                else:   # final attempt: press deeper (belief may be high-side)
                    self._plan_leg("press", [*self.oxy, bel_top + self.CUP - 0.045])
                    self.t_phase = 0
                    tgt = self.q_wp["press"]
            gp = env._grasp_point()
            # contact = tip within 5 mm of the grasp point (impossible
            # without touching); hold there and let the latch fire
            if float((tcp - gp).norm()) < 0.0035:
                tgt = q_now                        # event-hold at touch
            if env.sealed.any():
                self.h_corr = float(tcp[0, 2]) - self.CUP
                nb_max = max(b["half"][2] * 2 for b in self.believed[1:])
                clear = (0.09 if self.naive else
                         nb_max + 3 * self.sigma[0] + 0.02)
                self.z_clear = self.h_corr + clear
                self.phase = "extract"
                self.t_phase = 0
                self.q_ref = q_now.copy()
        elif self.phase == "extract":
            want = 1.0
            # vertical wrench-modulated ascent via IK z-steps
            margin = float(env.wr_margin[0])
            step = self.lift_v
            if not hasattr(self, "_stall"):
                self._stall, self._last_z = 0, float(tcp[0, 2])
            if float(tcp[0, 2]) < self._last_z + 0.001:
                self._stall += 1
            else:
                self._stall, self._last_z = 0, float(tcp[0, 2])
            forced = self._stall > 15
            if not self.naive and not forced:
                if margin < 0.2:
                    step = -0.002                  # back off: relax the spring
                elif margin < 0.4:
                    step = 0.0
                elif margin < 0.6:
                    step = self.lift_v * 0.5
                elif margin > 0.8:
                    step = self.lift_v * 1.8
            if forced:
                step = self.lift_v * 2.0
            z_next = min(float(tcp[0, 2]) + step, self.z_clear + self.CUP)
            self._plan_leg("ext", [*self.oxy, z_next])
            tgt = self.q_wp["ext"]
            if float(tcp[0, 2]) >= self.z_clear + self.CUP - 0.004:
                self._plan_leg("over", [BOX_C[0], BOX_C[1], max(0.34, self.z_clear + 0.10)])
                self.phase = "transit"
                self.t_phase = 0
        elif self.phase == "transit":
            want = 1.0
            tgt = self.q_wp["over"]
            if np.abs(q_now - tgt).max() < math.radians(2.0):
                h_obj = (self.h_corr if self.h_corr else bel_top)
                self._plan_leg("drop", [BOX_C[0], BOX_C[1], 0.05 + h_obj + self.CUP + 0.02])
                self.phase = "drop"
                self.t_phase = 0
        elif self.phase == "drop":
            want = 1.0
            tgt = self.q_wp["drop"]
            if np.abs(q_now - tgt).max() < math.radians(1.5) or self.t_phase > 45:
                self.phase = "release"
                self.t_phase = 0
        elif self.phase == "release":
            want = 0.0
            tgt = q_now
            if self.t_phase > 8:
                self._plan_leg("retreat", [BOX_C[0], BOX_C[1], 0.30])
                self.phase = "retreat"
        else:
            want = 0.0
            tgt = self.q_wp.get("retreat", q_now)
        dq = np.clip(tgt - q_now, -math.radians(2.0), math.radians(2.0))
        if self.phase in ("transit", "drop"):
            dq = dq * 0.45                     # gentle carry: wrench headroom
        if self.phase == "press":
            # decelerate on approach: full speed >4 cm out, crawl at contact
            gp = env._grasp_point()
            dist = float((tcp - gp).norm())
            dq = dq * min(1.0, max(0.22, dist / 0.04))
        a = np.zeros(7, np.float32)
        a[:6] = dq / math.radians(2.0)
        a[6] = 1.0 if want else -1.0
        return a

    def phase_flags(self):
        return [1.0 if self.phase in ("hover", "press") else 0.0,
                1.0 if self.phase == "extract" else 0.0,
                1.0 if self.phase in ("transit", "drop", "release", "retreat") else 0.0]


# ------------------------------------------------------------------ episode
def cup_touches_target(env):
    """Solver contact between cup_tip and object0 this substep."""
    import mujoco
    if not hasattr(env, "_gid_cup"):
        env._gid_cup = mujoco.mj_name2id(env.mjm, mujoco.mjtObj.mjOBJ_GEOM, "cup_tip")
        env._gid_o0 = mujoco.mj_name2id(env.mjm, mujoco.mjtObj.mjOBJ_GEOM, "g_object0")
    n = int(wp.to_torch(env.d.nacon).flatten()[0])
    if n == 0:
        return False
    g = wp.to_torch(env.d.contact.geom)[:n]
    m = (((g[:, 0] == env._gid_cup) & (g[:, 1] == env._gid_o0)) |
         ((g[:, 0] == env._gid_o0) & (g[:, 1] == env._gid_cup)))
    return bool(m.any())


def touch_latch(env, rest_z=None):
    """Touch-based latch; contact = grasp-point gate OR block pressed."""
    dev = env.device
    tcp, R = env._tcp()
    gp = env._grasp_point()
    if (cup_touches_target(env) and
            float(env.qvel[0, :6].abs().max()) < 1.0 and
            float(R[0, 2, 2]) < -0.85):
        idx = torch.tensor([0], device=dev)
        op = env._obj_pos()[idx]
        env.anchor[idx] = torch.einsum("nij,nj->ni", R[idx].transpose(1, 2), op - tcp[idx])
        env.anchor_R[idx] = torch.einsum("nij,njk->nik", R[idx].transpose(1, 2), env._obj_R()[idx])
        env.sealed[idx] = True
        env.ever_sealed[idx] = True
        env.wr_violation[idx] = 0


def restore(env, meta):
    import mujoco_warp as mjw
    import config as C
    qa = env.jadr_obj
    env.qpos[0, qa:qa + 3] = torch.tensor(
        [*meta[0]["pos"], meta[0]["half"][2] + 0.001], device=env.device)
    env.qpos[0, qa + 3:qa + 7] = torch.tensor([1.0, 0, 0, 0], device=env.device)
    env.qvel[0, env.vadr_obj:env.vadr_obj + 6] = 0.0
    env.sealed[:] = False
    env.ever_sealed[:] = False
    qt = torch.as_tensor(np.asarray(C.START_Q, float), dtype=torch.float32,
                         device=env.device)
    env.qpos[0, :6] = qt
    env.qvel[0, :6] = 0.0
    for attr in ("q_target", "q_target_prev", "q_drive", "q_meas_lag"):
        getattr(env, attr)[0] = qt
    env.q_hist[0] = qt[None]
    env.v_drive[0] = 0.0
    env.qd_meas_lag[0] = 0.0
    mjw.step(env.m, env.d)


def run_episode(env, meta, rng, actor=None, naive=False, record=None,
                log_rows=None, max_t=560):
    believed = [dict(pos=list(m["pos"]),
                     half=[m["half"][0], m["half"][1],
                           float(abs(m["half"][2] + rng.normal(0, 0.008)))])
                for m in meta]
    sigma = np.full(9, 0.010)
    restore(env, meta)
    # measure the block's true settled height (it sinks ~2 mm into table
    # compliance): 10 settle decisions, then read it from the sim
    import mujoco_warp as mjw
    for _ in range(10):
        env._step_real_drive()
    rest_z = float(env._obj_pos()[0, 2])
    teacher = ExtractionTeacher(env, believed, sigma, naive=naive,
                                rest_z=rest_z)
    detaches = 0
    # VIRTUAL reference: integrate dq on our own q_ref, never rebase on the
    # measured joints (rebasing feeds contact vibration into the reference
    # -> solver resonance -> NaN). Lead-clamped +-3 deg for safety.
    q_ref = env.qpos[0, :6].clone()
    for t in range(max_t):
        obs = policy_obs(env, believed, sigma, teacher.phase_flags())
        a_teach = teacher.act()
        if actor is not None and teacher.phase == "extract":
            with torch.no_grad():
                a = actor(obs[None])[0].cpu().numpy()
        else:
            a = a_teach
        if log_rows is not None:
            log_rows.append((obs.cpu().numpy().astype(np.float32),
                             a_teach.astype(np.float32)))
        dq = a[:6] * math.radians(2.0)
        q_ref = q_ref + torch.as_tensor(dq, dtype=torch.float32,
                                        device=env.device)
        q_ref = env.qpos[0, :6] + (q_ref - env.qpos[0, :6]).clamp(
            -math.radians(3.0), math.radians(3.0))
        env.q_target = q_ref[None]
        was_sealed = bool(env.sealed[0])
        env._step_real_drive()
        if a[6] > 0 and not env.sealed.any():
            touch_latch(env, rest_z=rest_z)
        broke = env._check_break()
        if was_sealed and bool(broke[0]):
            detaches += 1
        if a[6] < 0 and env.sealed.any():
            env.sealed[:] = False
            env.xfrc[:, env.bid_obj] = 0.0
        if record is not None:
            record.append((env.qpos[0].cpu().numpy().copy(),
                           bool(env.sealed[0]), teacher.phase,
                           float(env.wr_margin[0])))
        if teacher.phase == "retreat" and teacher.t_phase > 15:
            break
    op = env._obj_pos()[0].cpu().numpy()
    in_box = (abs(op[0] - BOX_C[0]) < 0.125 and abs(op[1] - BOX_C[1]) < 0.125
              and 0.04 < op[2] < 0.3)
    # neighbour disturbance: max displacement of blocks 1..8
    disturb = 0.0
    for n in range(1, 9):
        bid = env.mjm.body(f"object{n}").id
        p = wp.to_torch(env.d.xpos)[0, bid].cpu().numpy()
        disturb = max(disturb, float(np.linalg.norm(p[:2] - meta[n]["pos"])))
    return dict(in_box=bool(in_box), sealed_ever=bool(env.ever_sealed[0]),
                detaches=detaches, h_corr=teacher.h_corr,
                true_h=meta[0]["half"][2] * 2, disturb_mm=disturb * 1000,
                phase_end=teacher.phase)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--make_data", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--naive", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    wp.init()

    if a.demo:
        # fast vs slow extraction on the same scene: wrench model demo
        for naive, nm in [(True, "NAIVE fast"), (False, "wrench-aware")]:
            env, meta, rng = build_env(a.seed0)
            st = run_episode(env, meta, rng, naive=naive)
            print(f"[{nm}] in_box={st['in_box']} detaches={st['detaches']} "
                  f"true_h={st['true_h']:.3f} corrected={st['h_corr']} "
                  f"disturb {st['disturb_mm']:.1f} mm end={st['phase_end']}")
            del env
            torch.cuda.empty_cache()

    if a.make_data:
        rows = []
        for s in range(a.scenes):
            env, meta, rng = build_env(a.seed0 + s)
            st = run_episode(env, meta, rng, log_rows=rows)
            print(f"[data {s}] in_box={st['in_box']} detach={st['detaches']} "
                  f"disturb {st['disturb_mm']:.0f}mm", flush=True)
            del env
            torch.cuda.empty_cache()
        torch.save(rows, os.path.join(OUT, "bc_data.pt"))
        print(f"[data] {len(rows)} pairs")

    if a.train:
        rows = torch.load(os.path.join(OUT, "bc_data.pt"), weights_only=False)
        dev = "cuda"
        X = torch.tensor(np.stack([r[0] for r in rows]), device=dev)
        Y = torch.tensor(np.stack([r[1] for r in rows]), device=dev)
        pol = Policy(obs_dim=X.shape[1]).to(dev)
        opt = torch.optim.AdamW(pol.parameters(), lr=3e-4)
        N = len(X)
        print(f"[train] {N} pairs, obs {X.shape[1]}")
        for ep in range(a.epochs):
            perm = torch.randperm(N, device=dev)
            tot = 0.0
            for i in range(0, N, 512):
                idx = perm[i:i + 512]
                loss = ((pol(X[idx]) - Y[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
            if ep % 5 == 0 or ep == a.epochs - 1:
                print(f"  epoch {ep}: {tot/N:.5f}", flush=True)
        torch.save(pol.state_dict(), os.path.join(OUT, "policy.pt"))
        print("[train] saved policy.pt")

    if a.eval:
        dev = "cuda"
        actor = None
        tag = "naive" if a.naive else "teacher"
        if not a.naive and os.path.exists(os.path.join(OUT, "policy.pt")):
            actor = Policy().to(dev)
            actor.load_state_dict(torch.load(os.path.join(OUT, "policy.pt")))
            actor.eval()
            tag = "policy"
            print("[eval] trained policy drives the EXTRACT phase")
        res, vids = [], []
        for s in range(a.scenes):
            env, meta, rng = build_env(a.seed0 + 100 + s)
            rec = [] if a.render else None
            st = run_episode(env, meta, rng, actor=actor, naive=a.naive,
                             record=rec)
            res.append(st)
            print(f"[eval {s}] in_box={st['in_box']} detach={st['detaches']} "
                  f"true_h {st['true_h']*1000:.0f}mm corr "
                  f"{(st['h_corr'] or 0)*1000:.0f}mm disturb {st['disturb_mm']:.0f}mm",
                  flush=True)
            if rec:
                vids.append((a.seed0 + 100 + s, rec, st))
            del env
            torch.cuda.empty_cache()
        print(f"\n[{tag}] in-box {sum(r['in_box'] for r in res)}/{len(res)}  "
              f"detaches {sum(r['detaches'] for r in res)}  "
              f"mean disturb {np.mean([r['disturb_mm'] for r in res]):.0f} mm")
        json.dump(res, open(os.path.join(OUT, f"eval_{tag}.json"), "w"), indent=1)
        if vids:
            render(vids, tag)


def render(vids, tag):
    import cv2
    import mujoco
    outp = os.path.join(OUT, f"video_{tag}.mp4")
    vw = None
    for seed, rec, st in vids:
        m = mujoco.MjModel.from_xml_path(os.path.join(OUT, f"scene_{seed}.xml"))
        d = mujoco.MjData(m)
        ren = mujoco.Renderer(m, height=360, width=480)
        for k, (qpos, sealed, phase, margin) in enumerate(rec):
            if k % 2:
                continue
            d.qpos[:len(qpos)] = qpos
            mujoco.mj_forward(m, d)
            ims = []
            for cam in ("iso", "front"):
                ren.update_scene(d, camera=cam)
                ims.append(cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR))
            canvas = np.zeros((420, 960, 3), np.uint8)
            canvas[60:] = np.hstack(ims)
            cv2.putText(canvas, f"{tag} seed{seed} {phase}  "
                        f"{'IN BOX' if st['in_box'] else 'miss'}  "
                        f"detach:{st['detaches']}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            # wrench margin bar
            w = int(300 * max(0.0, min(1.0, margin)))
            col = (0, 200, 0) if margin > 0.5 else ((0, 200, 255) if margin > 0.25 else (0, 0, 255))
            cv2.rectangle(canvas, (620, 12), (620 + w, 36), col, -1)
            cv2.rectangle(canvas, (620, 12), (920, 36), (200, 200, 200), 1)
            cv2.putText(canvas, "wrench margin", (620, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            if sealed:
                cv2.rectangle(canvas, (0, 60), (959, 419), (0, 180, 255), 3)
            if vw is None:
                vw = cv2.VideoWriter(outp.replace(".mp4", "_raw.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 10, (960, 420))
            vw.write(canvas)
        for _ in range(5):
            vw.write(canvas)
    vw.release()
    os.system(f"ffmpeg -y -loglevel error -i {outp.replace('.mp4','_raw.mp4')} "
              f"-c:v libx264 -pix_fmt yuv420p {outp}")
    os.remove(outp.replace(".mp4", "_raw.mp4"))
    print(f"[video] {outp}")


if __name__ == "__main__":
    main()
