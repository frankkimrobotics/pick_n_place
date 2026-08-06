"""env_warp :: batched single-pick suction environment on mujoco_warp.

N parallel worlds of ONE compiled model (table + bin + 1 box/cylinder
object + arm). Object dims are model-level, so size randomization happens
by rebuilding the model between phases (call `PickEnv.build(...)` again);
object POSE is randomized per world at reset.

Decisions at 10 Hz (50 physics substeps @ 2 ms). Actions (7):
  a[0:6]  delta-q joint target (clamped ±2 deg/step, tracked by PD)
  a[6]    suction command logit (>0 = ON)

Suction model (RL-hardened vs the BC dataset's instant weld):
  latch requires: cup tip within SEAL_DIST of object top, pressing depth
  reached, cup axis tilt < SEAL_TILT, AND relative speed < SEAL_VEL
  (low-energy latch: kills the impact-jolt exploit); while latched a
  force estimate m*(|dv/dt| + g) > F_MAX breaks the seal (20 N cup limit).

Rewards: staged, potential-based (see reward() docstring / RL_SAC_PLAN).

Smoke test:   python rl/env_warp.py --nworld 64 --steps 20   (mjwarp env)
"""
import os
import subprocess
import sys
import time  # noqa

import mujoco
import numpy as np
import torch
import warp as wp

import mujoco_warp as mjw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

CUROBO_PY = "/home/lisc-frank/miniconda3/envs/curobo2/bin/python"
CTRL_HZ = 10
CUP_R = 0.008
PRESS_M = 0.012
SEAL_DIST = 0.012          # cup tip to object-top center (m)
SEAL_TILT = 25.0           # deg
SEAL_VEL = 0.08            # m/s relative speed gate (low-energy latch)
F_MAX = 20.0               # N suction holding limit
TAU_CUP = 0.4              # N*m peel-torque limit (~F_MAX x cup radius)
PEEL_COS = 0.878           # cos(~28.6 deg): tilt beyond this = cup peeling
BIN_XY = np.array([0.10, 0.40])
BIN_HALF = 0.145
TABLE_X = (0.24, 0.52)
TABLE_Y = (-0.16, 0.16)
DQ_MAX = np.radians(2.0)   # per-decision joint delta clamp
EP_LEN = 150
EP_LEN_ATTACH = 40
EP_LEN_PNP = 100
PLACE_TOL = 0.035

W = dict(approach=1.0, align=0.3, press=0.5, seal=5.0, lift=4.0,
         transport=6.0, place=20.0, drop=-0.5, chatter=-0.05,
         act=-0.01, time=-0.005, table_slam=-0.5, off_table=-2.0,
         descend=4.0, tilt_pen=-0.4, rel_mask=-0.08)


def build_scene_xml(half_extents, kind, out_xml, seed=0):
    """One-object scene via the repo's MJCF generator (curobo2 env)."""
    hx, hy, hz = half_extents
    if kind == "cylinder":
        size = f"{hx:.4f} {hz:.4f}"
        shape = "cylinder"
    else:
        size = f"{hx:.4f} {hy:.4f} {hz:.4f}"
        shape = "box"
    tup = [("object0", shape, size, f"0.38 0.0 {hz:.4f}", "0.8 0.3 0.3 1", "1 0 0 0")]
    code = (f"import sys; sys.path.insert(0, {ROOT!r});\n"
            f"import sim_robot_mjcf as g;\n"
            f"g.build(warp=True, objects={tup!r}, light_pos=(0.3, 0.1, 1.5), "
            f"out_path={out_xml!r})")
    subprocess.run([CUROBO_PY, "-c", code], check=True, capture_output=True)


class PickEnv:
    def __init__(self, nworld=1024, device="cuda:0", seed=0,
                 xml=None, half_extents=(0.025, 0.025, 0.02), kind="box",
                 mode="full", dr=False, target_max=0.30,
                 lift_req=0.0, speed_bonus=0.0, release_mask=False,
                 mask_h=0.05):
        """mode='attach': staged sub-task -- episodes START with the cup
        hovering 2-4 cm above the (jittered) grasp point; success = seal +
        hold + 2 cm lift within a 40-step episode. mode='full': whole task."""
        self.nworld = nworld
        self.device = device
        self.mode = mode
        self.dr = dr
        self.target_max = target_max
        self.lift_req = lift_req          # required MAX lift for pnp success
        self.speed_bonus = speed_bonus    # terminal bonus * (1 - t/EP_LEN)
        self.release_mask = release_mask  # hold seal if release cmd high
        self.mask_h = mask_h              # anneal 0.05 -> 0.015
        self.rng = np.random.default_rng(seed)
        if xml is None:
            xml = os.path.join(HERE, "_scene_rl.xml")
            build_scene_xml(half_extents, kind, xml)
        self.half = np.asarray(half_extents, float)
        self.mjm = mujoco.MjModel.from_xml_path(xml)
        mjd = mujoco.MjData(self.mjm)
        # ids
        M, name2id = self.mjm, mujoco.mj_name2id
        self.bid_obj = name2id(M, mujoco.mjtObj.mjOBJ_BODY, "object0")
        self.sid_tcp = name2id(M, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.eq_suction = name2id(M, mujoco.mjtObj.mjOBJ_EQUALITY, "suction_object0")
        self.jadr_obj = M.jnt_qposadr[M.body_jntadr[self.bid_obj]]
        self.vadr_obj = M.jnt_dofadr[M.body_jntadr[self.bid_obj]]
        self.obj_mass = float(M.body_mass[self.bid_obj])
        import config as C
        self.q_home = np.asarray(C.START_Q, float)
        # canonical hover for attach mode (CPU IK once)
        demo = {"__file__": os.path.join(ROOT, "mjwarp_pick_demo.py")}
        exec(open(demo["__file__"]).read().split("if __name__")[0], demo)
        dik = mujoco.MjData(self.mjm)
        self.q_hover, _e = demo["ik"](
            self.mjm, dik, "tcp",
            [0.38, 0.0, float(half_extents[2]) + CUP_R + 0.05],
            demo["R_DOWN"], self.q_home)
        # HOVER GRID: pre-solved IK across the whole table so resets (and
        # therefore training picks) cover the full workspace, not one spot
        grid = []
        z_h = float(half_extents[2]) + CUP_R + 0.05
        q_seed = self.q_hover
        for gx in np.linspace(TABLE_X[0] + 0.03, TABLE_X[1] - 0.03, 9):
            for gy in np.linspace(TABLE_Y[0] + 0.03, TABLE_Y[1] - 0.03, 7):
                q_g, e_g = demo["ik"](self.mjm, dik, "tcp",
                                      [float(gx), float(gy), z_h],
                                      demo["R_DOWN"], q_seed)
                if e_g < 0.005:
                    grid.append(q_g)
                    q_seed = q_g
        self.hover_grid = np.stack(grid)
        print(f"[env] hover grid: {len(grid)} reachable cells", flush=True)
        # CARRY GRID: same lattice at carry height (for place-curriculum
        # resets that start sealed mid-carry above the target)
        cgrid = []
        z_c = 0.30 + 2 * float(half_extents[2]) + CUP_R + 0.004
        q_seed = self.q_hover
        for gx in np.linspace(TABLE_X[0] + 0.03, TABLE_X[1] - 0.03, 9):
            for gy in np.linspace(TABLE_Y[0] + 0.03, TABLE_Y[1] - 0.03, 7):
                q_g, e_g = demo["ik"](self.mjm, dik, "tcp",
                                      [float(gx), float(gy), z_c],
                                      demo["R_DOWN"], q_seed)
                if e_g < 0.005:
                    cgrid.append(q_g)
                    q_seed = q_g
        self.carry_grid = np.stack(cgrid) if cgrid else self.hover_grid
        print(f"[env] carry grid: {len(cgrid)} reachable cells", flush=True)
        mjd.qpos[:6] = self.q_home
        mujoco.mj_forward(self.mjm, mjd)
        # PD gains (fixed diagonal; simpler than per-tick gain scheduling)
        self.kp = torch.tensor([600, 900, 600, 200, 80, 40], device=device,
                               dtype=torch.float32)
        self.kd = torch.tensor([40, 60, 40, 12, 4, 2], device=device,
                               dtype=torch.float32)
        self.tau_max = 100.0
        self.gain_scale = torch.ones(nworld, 1, device=device)
        self.seal_dist_w = torch.full((nworld,), SEAL_DIST, device=device)
        self.seal_vel_w = torch.full((nworld,), SEAL_VEL, device=device)
        self.delay_mask = torch.zeros(nworld, dtype=torch.bool, device=device)
        self.prev_action = torch.zeros(nworld, 7, device=device)
        # batched warp model/data
        self.m = mjw.put_model(self.mjm)
        self.d = mjw.put_data(self.mjm, mjd, nworld=nworld)
        self.substeps = int(round(1.0 / (CTRL_HZ * self.mjm.opt.timestep)))
        # torch views over warp arrays (zero copy, on device)
        self.qpos = wp.to_torch(self.d.qpos)          # (N, nq)
        self.qvel = wp.to_torch(self.d.qvel)
        self.ctrl = wp.to_torch(self.d.ctrl)
        self.xpos = wp.to_torch(self.d.xpos)          # (N, nbody, 3)
        self.xmat_site = wp.to_torch(self.d.site_xmat)  # (N, nsite, 3, 3)
        self.site_xpos = wp.to_torch(self.d.site_xpos)
        self.xfrc = wp.to_torch(self.d.xfrc_applied)    # (N, nbody, 6)
        self.xquat = wp.to_torch(self.d.xquat)          # (N, nbody, 4) wxyz
        self.bid_ped = name2id(M, mujoco.mjtObj.mjOBJ_BODY, "pedestal")
        self.mocap_ped = int(M.body_mocapid[self.bid_ped]) if self.bid_ped >= 0 else -1
        self.mocap_pos = wp.to_torch(self.d.mocap_pos) if self.mocap_ped >= 0 else None
        # episode state (torch, device)
        N = nworld
        self.t_step = torch.zeros(N, dtype=torch.long, device=device)
        self.sealed = torch.zeros(N, dtype=torch.bool, device=device)
        self.ever_sealed = torch.zeros(N, dtype=torch.bool, device=device)
        self.q_target = torch.zeros(N, 6, device=device)
        self.prev_obj_vel = torch.zeros(N, 3, device=device)
        self.anchor = torch.zeros(N, 3, device=device)
        self.sat_count = torch.zeros(N, dtype=torch.long, device=device)
        self.phi_approach = torch.zeros(N, device=device)
        self.phi_transport = torch.zeros(N, device=device)
        self.phi_lift = torch.zeros(N, device=device)
        self.phi_desc = torch.zeros(N, device=device)
        self.RKEYS = ["approach", "align", "press", "seal", "lift",
                      "transport", "place", "drop", "chatter", "act",
                      "time", "table_slam", "off_table", "descend",
                      "tilt_pen", "rel_mask"]
        self.ep_comp = torch.zeros(N, len(self.RKEYS), device=device)
        self.max_lift = torch.zeros(N, device=device)
        self.target_h = torch.zeros(N, device=device)   # place surface height
        self.max_tilt = torch.zeros(N, device=device)    # rad, while sealed
        self.release_h = torch.zeros(N, device=device)   # rest_h at release
        self.bin_pos = torch.tensor([*BIN_XY, 0.0], device=device, dtype=torch.float32)
        self.place_target = self.bin_pos[:2].expand(N, 2).clone()
        self.reset(torch.ones(N, dtype=torch.bool, device=device))

    # ---------------- helpers ----------------
    def _obj_pos(self):
        return self.xpos[:, self.bid_obj]

    def _tcp(self):
        p = self.site_xpos[:, self.sid_tcp]
        R = self.xmat_site[:, self.sid_tcp]
        return p, R

    def _grasp_point(self):
        g = self._obj_pos().clone()
        g[:, 2] += self.half[2] + CUP_R
        return g

    # ---------------- reset ----------------
    def reset(self, mask):
        idx = torch.nonzero(mask).squeeze(-1)
        if idx.numel() == 0:
            return
        n = idx.numel()
        xy = torch.tensor(self.rng.uniform(
            [TABLE_X[0] + 0.03, TABLE_Y[0] + 0.03],
            [TABLE_X[1] - 0.03, TABLE_Y[1] - 0.03], size=(n, 2)),
            device=self.device, dtype=torch.float32)
        yaw = torch.tensor(self.rng.uniform(0, np.pi, size=n),
                           device=self.device, dtype=torch.float32)
        qa = self.jadr_obj
        self.qpos[idx, :6] = torch.tensor(
            self.q_home + self.rng.normal(0, 0.02, size=(n, 6)),
            device=self.device, dtype=torch.float32)
        self.qvel[idx, :] = 0.0
        self.qpos[idx, qa:qa + 2] = xy
        self.qpos[idx, qa + 2] = float(self.half[2]) + 0.001
        self.qpos[idx, qa + 3] = torch.cos(yaw / 2)
        self.qpos[idx, qa + 4:qa + 6] = 0.0
        self.qpos[idx, qa + 6] = torch.sin(yaw / 2)
        self.xfrc[idx, self.bid_obj] = 0.0
        self.t_step[idx] = 0
        self.sealed[idx] = False
        self.ever_sealed[idx] = False
        self.ep_comp[idx] = 0.0
        self.max_lift[idx] = 0.0
        self.max_tilt[idx] = 0.0
        self.release_h[idx] = 0.0
        if self.dr:
            n_ = idx.numel()
            self.gain_scale[idx] = torch.tensor(
                self.rng.uniform(0.8, 1.2, size=(n_, 1)), device=self.device,
                dtype=torch.float32)
            self.seal_dist_w[idx] = SEAL_DIST * torch.tensor(
                self.rng.uniform(0.8, 1.2, size=n_), device=self.device,
                dtype=torch.float32)
            self.seal_vel_w[idx] = SEAL_VEL * torch.tensor(
                self.rng.uniform(0.8, 1.2, size=n_), device=self.device,
                dtype=torch.float32)
            self.delay_mask[idx] = torch.tensor(
                self.rng.random(n_) < 0.5, device=self.device)
            self.prev_action[idx] = 0.0
        self.q_target[idx] = self.qpos[idx, :6]
        if self.mode in ("attach", "pnp"):
            # place the ARM near the grasp: per-world CPU IK is too slow, so
            # use a canonical pre-solved hover configuration for the object at
            # scene center and correct the object to sit UNDER the cup instead:
            # sample cup-relative offsets and put the object there.
            cells = self.rng.integers(0, len(self.hover_grid), size=idx.numel())
            self.qpos[idx, :6] = torch.tensor(
                self.hover_grid[cells] + self.rng.normal(0, 0.015, size=(idx.numel(), 6)),
                device=self.device, dtype=torch.float32)
            mjw.forward(self.m, self.d)
            tcp, _ = self._tcp()
            off = torch.tensor(self.rng.uniform(
                [-0.015, -0.015], [0.015, 0.015], size=(idx.numel(), 2)),
                device=self.device, dtype=torch.float32)
            hover = torch.tensor(self.rng.uniform(0.02, 0.04, size=idx.numel()),
                                 device=self.device, dtype=torch.float32)
            qa = self.jadr_obj
            self.qpos[idx, qa:qa + 2] = tcp[idx, :2] + off
            self.qpos[idx, qa + 2] = (tcp[idx, 2] - hover
                                      - float(self.half[2]) - CUP_R).clamp(
                min=float(self.half[2]) + 0.001)
            self.q_target[idx] = self.qpos[idx, :6]
        if self.mode == "pnp":
            # random set-down target, guaranteed >= 8 cm from the object
            # curriculum: target at distance U[0.05, target_max] from object
            qa0 = self.jadr_obj
            objxy = self.qpos[idx, qa0:qa0 + 2]
            ang = torch.tensor(self.rng.uniform(0, 2 * np.pi, size=idx.numel()),
                               device=self.device, dtype=torch.float32)
            rad = torch.tensor(self.rng.uniform(0.05, self.target_max,
                                                size=idx.numel()),
                               device=self.device, dtype=torch.float32)
            tgt = objxy + torch.stack([rad * torch.cos(ang),
                                       rad * torch.sin(ang)], -1)
            tgt[:, 0] = tgt[:, 0].clamp(TABLE_X[0] + 0.04, TABLE_X[1] - 0.04)
            tgt[:, 1] = tgt[:, 1].clamp(TABLE_Y[0] + 0.04, TABLE_Y[1] - 0.04)
            qa = self.jadr_obj
            near = torch.norm(tgt - self.qpos[idx, qa:qa + 2], dim=-1) < 0.08
            tgt[near] = torch.where(
                tgt[near] > 0.0, tgt[near] - 0.12, tgt[near] + 0.12).clamp(
                torch.tensor([TABLE_X[0] + 0.04, TABLE_Y[0] + 0.04], device=self.device),
                torch.tensor([TABLE_X[1] - 0.04, TABLE_Y[1] - 0.04], device=self.device))
            self.place_target[idx] = tgt
            if self.mocap_ped >= 0:
                n2 = idx.numel()
                elevated = torch.tensor(self.rng.random(n2) < 0.6,
                                        device=self.device)
                hh = torch.tensor(self.rng.uniform(0.04, 0.12, size=n2),
                                  device=self.device, dtype=torch.float32)
                hh = torch.where(elevated, hh, torch.zeros_like(hh))
                self.target_h[idx] = hh
                self.mocap_pos[idx, self.mocap_ped, 0] = tgt[:, 0]
                self.mocap_pos[idx, self.mocap_ped, 1] = tgt[:, 1]
                self.mocap_pos[idx, self.mocap_ped, 2] = torch.where(
                    elevated, hh - 0.06,
                    torch.full_like(hh, -0.5))    # park under world if table
            else:
                self.target_h[idx] = 0.0
        if self.mode == "place":
            # PLACE CURRICULUM: start SEALED mid-carry above the target so
            # every step of experience is descend -> contact -> release
            # (the attach-mode trick applied to the end of the task)
            n2 = idx.numel()
            cells = self.rng.integers(0, len(self.carry_grid), size=n2)
            self.qpos[idx, :6] = torch.tensor(
                self.carry_grid[cells] + self.rng.normal(0, 0.02, size=(n2, 6)),
                device=self.device, dtype=torch.float32)
            self.qvel[idx, :6] = 0.0
            mjw.forward(self.m, self.d)
            tcp, R = self._tcp()
            qa = self.jadr_obj
            self.qpos[idx, qa] = tcp[idx, 0]
            self.qpos[idx, qa + 1] = tcp[idx, 1]
            self.qpos[idx, qa + 2] = tcp[idx, 2] - 0.004 - float(self.half[2])
            self.qpos[idx, qa + 3] = 1.0
            self.qpos[idx, qa + 4:qa + 7] = 0.0
            self.qvel[idx, self.vadr_obj:self.vadr_obj + 6] = 0.0
            mjw.forward(self.m, self.d)
            op = self._obj_pos()
            self.anchor[idx] = torch.einsum(
                "nij,nj->ni", R[idx].transpose(1, 2), op[idx] - tcp[idx])
            self.sealed[idx] = True
            self.ever_sealed[idx] = True
            self.sat_count[idx] = 0
            self.max_lift[idx] = self.lift_req + 0.02   # already carried
            tgt = op[idx, :2] + torch.tensor(
                self.rng.uniform(-0.06, 0.06, size=(n2, 2)),
                device=self.device, dtype=torch.float32)
            tgt[:, 0] = tgt[:, 0].clamp(TABLE_X[0] + 0.04, TABLE_X[1] - 0.04)
            tgt[:, 1] = tgt[:, 1].clamp(TABLE_Y[0] + 0.04, TABLE_Y[1] - 0.04)
            self.place_target[idx] = tgt
            if self.mocap_ped >= 0:
                elevated = torch.tensor(self.rng.random(n2) < 0.6,
                                        device=self.device)
                hh = torch.tensor(self.rng.uniform(0.04, 0.12, size=n2),
                                  device=self.device, dtype=torch.float32)
                hh = torch.where(elevated, hh, torch.zeros_like(hh))
                self.target_h[idx] = hh
                self.mocap_pos[idx, self.mocap_ped, 0] = tgt[:, 0]
                self.mocap_pos[idx, self.mocap_ped, 1] = tgt[:, 1]
                self.mocap_pos[idx, self.mocap_ped, 2] = torch.where(
                    elevated, hh - 0.06, torch.full_like(hh, -0.5))
            else:
                self.target_h[idx] = 0.0
            self.q_target[idx] = self.qpos[idx, :6]
        mjw.forward(self.m, self.d)
        tcp, _ = self._tcp()
        if self.mode == "place":
            # phi_lift consistent with the seeded carry (no free first-step
            # lift payout); phi tracks MAX lift
            lift_cap = max(self.lift_req, 0.10)
            self.phi_lift[idx] = self.max_lift[idx].clamp(0, lift_cap) / lift_cap
        self.phi_approach[idx] = -torch.norm(
            tcp[idx] - self._grasp_point()[idx], dim=-1)
        self.phi_transport[idx] = -torch.norm(
            self._obj_pos()[idx, :2] - self.place_target[idx], dim=-1)
        if self.mode != "place":
            self.phi_lift[idx] = 0.0
        op_i = self._obj_pos()[idx]
        rh_i = (op_i[:, 2] - float(self.half[2]) - self.target_h[idx])
        d_i = torch.norm(op_i[:, :2] - self.place_target[idx], dim=-1)
        self.phi_desc[idx] = -rh_i.clamp(min=0.0, max=0.40) * \
            torch.exp(-(d_i ** 2) / (2 * 0.07 ** 2))

    # ---------------- suction ----------------
    # Suction = per-world compliant spring-damper (xfrc_applied) with a
    # hard force cap: |F| clamps at F_MAX; sustained saturation breaks the
    # seal. Physically: a force-limited compliant cup (no weld snapping,
    # no jolt, no unbreakable carry -- the RL-hardened suction model).
    K_SUCTION = 1500.0        # N/m  (~1.3 mm sag under a 2 N payload)
    C_SUCTION = 30.0          # N s/m
    BREAK_STEPS = 20          # substeps of saturation before break

    def _try_latch(self, want):
        tcp, R = self._tcp()
        gp = self._grasp_point()
        dist = torch.norm(tcp - gp, dim=-1)
        pressing = tcp[:, 2] < gp[:, 2] - 0.003
        tilt_ok = R[:, 2, 2] < -np.cos(np.radians(SEAL_TILT))  # cup z down
        ov = self.qvel[:, self.vadr_obj:self.vadr_obj + 3]
        tv = (tcp - self._last_tcp) * CTRL_HZ if hasattr(self, "_last_tcp") \
            else torch.zeros_like(tcp)
        rel_speed = torch.norm(tv - ov, dim=-1)
        gate = want & ~self.sealed & (dist < self.seal_dist_w) & pressing \
            & tilt_ok & (rel_speed < self.seal_vel_w)
        if gate.any():
            idx = torch.nonzero(gate).squeeze(-1)
            # anchor: object top point in tcp frame at latch
            op = self._obj_pos()[idx]
            self.anchor[idx] = torch.einsum(
                "nij,nj->ni", R[idx].transpose(1, 2), op - tcp[idx])
            self.sat_count[idx] = 0
            self.sealed[idx] = True
            self.ever_sealed[idx] = True
        return gate

    def _obj_zaxis(self):
        q = self.xquat[:, self.bid_obj]                  # wxyz
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return torch.stack([2 * (x * z + w * y),
                            2 * (y * z - w * x),
                            1 - 2 * (x * x + y * y)], -1)

    def _obj_w_world(self):
        """Free-joint angular velocity (qvel) is BODY-frame; rotate to world."""
        w_b = self.qvel[:, self.vadr_obj + 3:self.vadr_obj + 6]
        q = self.xquat[:, self.bid_obj]
        qw, qv = q[:, :1], q[:, 1:]
        t = 2 * torch.cross(qv, w_b, dim=-1)
        return w_b + qw * t + torch.cross(qv, t, dim=-1)

    K_ROT = 4.0               # N*m/rad righting stiffness
    C_ROT = 0.05              # N*m*s/rad

    def _apply_suction_force(self):
        """Called each physics substep: spring-damper pulling the object's
        latch anchor to its latched pose in the (moving) tcp frame."""
        if not self.sealed.any():
            self.xfrc[:, self.bid_obj] = 0.0
            return
        tcp, R = self._tcp()
        target = tcp + torch.einsum("nij,nj->ni", R, self.anchor)
        op = self._obj_pos()
        ov = self.qvel[:, self.vadr_obj:self.vadr_obj + 3]
        F = self.K_SUCTION * (target - op) - self.C_SUCTION * ov
        Fmag = torch.norm(F, dim=-1, keepdim=True)
        scale = (F_MAX / Fmag.clamp(min=1e-6)).clamp(max=1.0)
        F = F * scale
        m = self.sealed.float()[:, None]
        self.xfrc[:, self.bid_obj, :3] = F * m
        # rotational spring: right the object's z-axis toward the CUP axis,
        # capped at the peel-torque limit (single-cup wrench limit)
        objz = self._obj_zaxis()
        cupz = -R[:, :, 2]                       # cup presses along -z
        err = torch.cross(objz, cupz, dim=-1)
        # spring capped at the cup's holding limit; damping is applied as
        # direct exponential decay of the object's angular velocity below
        # (torque-based damping is either starved by the cap -- undamped
        # 70deg ringing -- or explicit-unstable for the box's tiny inertia)
        Ts = self.K_ROT * err
        Tmag = torch.norm(Ts, dim=-1, keepdim=True)
        tscale = (TAU_CUP / Tmag.clamp(min=1e-6)).clamp(max=1.0)
        self.xfrc[:, self.bid_obj, 3:] = Ts * tscale * m
        va = self.vadr_obj
        self.qvel[:, va + 3:va + 6] = torch.where(
            self.sealed[:, None], self.qvel[:, va + 3:va + 6] * 0.82,
            self.qvel[:, va + 3:va + 6])
        # peel = object actually hanging far off the cup axis (the capped
        # righting spring can no longer recover it), NOT mere cap saturation
        cosang = (objz * cupz).sum(-1)
        sat_t = self.sealed & (cosang < PEEL_COS)
        sat = self.sealed & ((Fmag.squeeze(-1) * scale.squeeze(-1) >= F_MAX - 1e-4)
                             | sat_t)
        self.sat_count = torch.where(sat, self.sat_count + 1,
                                     torch.zeros_like(self.sat_count))

    def _check_break(self):
        broke = self.sealed & (self.sat_count >= self.BREAK_STEPS)
        if broke.any():
            idx = torch.nonzero(broke).squeeze(-1)
            self.sealed[idx] = False
            self.xfrc[idx, self.bid_obj] = 0.0
            self.sat_count[idx] = 0
        ov = self.qvel[:, self.vadr_obj:self.vadr_obj + 3]
        self.prev_obj_vel = ov.clone()
        return broke

    # ---------------- step ----------------
    def step(self, action):
        """action (N,7) in [-1,1]. Returns obs, reward, done, info."""
        a = action.clamp(-1, 1)
        if self.dr:                     # 1-step action latency on half the worlds
            eff = torch.where(self.delay_mask[:, None], self.prev_action, a)
            self.prev_action = a.clone()
            a = eff
        self.q_target = self.q_target + a[:, :6] * DQ_MAX
        want = a[:, 6] > 0
        if self.release_mask:
            # training aid: the cup does not let go >5 cm above the
            # target surface -- deletes the drop-from-height optimum.
            # Commanding a masked release is penalized (rel_mask) so
            # the policy learns to TIME release, not lean on the mask.
            op_z = self.qpos[:, self.jadr_obj + 2]
            rest_now = op_z - float(self.half[2]) - self.target_h
            self._masked_rel = self.sealed & ~want & (rest_now > self.mask_h)
            want = want | self._masked_rel
        else:
            self._masked_rel = torch.zeros_like(self.sealed)
        tcp_before, _ = self._tcp()
        obj_before = self._obj_pos().clone()

        first_possible = ~self.ever_sealed.clone()
        latched_now = self._try_latch(want)
        first_latch = latched_now & first_possible
        released = self.sealed & ~want
        if released.any():
            idx = torch.nonzero(released).squeeze(-1)
            self.sealed[idx] = False
            self.xfrc[idx, self.bid_obj] = 0.0
        for _ in range(self.substeps):
            tau = (self.gain_scale * (self.kp * (self.q_target - self.qpos[:, :6])
                   - self.kd * self.qvel[:, :6])).clamp(-self.tau_max, self.tau_max)
            self.ctrl[:, :6] = tau
            self._apply_suction_force()
            mjw.step(self.m, self.d)
        broke = self._check_break()
        self._last_tcp = self._tcp()[0].clone()
        self.t_step += 1

        r, done, info = self.reward(want, first_latch, released, broke,
                                    tcp_before, obj_before, a)
        obs = self.observe()
        if done.any() and getattr(self, "auto_reset", True):
            self.reset(done)
        return obs, r, done, info

    # ---------------- observation (privileged, 41-D) ----------------
    def observe(self):
        tcp, R = self._tcp()
        op = self._obj_pos()
        rel_goal = self._grasp_point() - tcp
        return torch.cat([
            self.qpos[:, :6], self.qvel[:, :6],
            tcp, R[:, :, 0], R[:, :, 1],
            op - tcp, rel_goal,
            torch.tensor(self.half, device=self.device, dtype=torch.float32).expand(self.nworld, 3),
            self.sealed.float()[:, None],
            (op[:, 2] - self.half[2])[:, None],          # lift height
            (op[:, :2] - self.place_target),              # to-target xy
            self.target_h[:, None],                        # place surface z
            (op[:, 2] - float(self.half[2]) - self.target_h)[:, None],
            self._obj_zaxis()[:, 2:3],                     # uprightness
        ], dim=-1) + (torch.randn(self.nworld, 37, device=self.device) * 0.005
                      if self.dr else 0.0)

    # ---------------- reward ----------------
    def reward(self, want, latched_now, released, broke, tcp_before, obj_before, a):
        """Staged, potential-based pick-and-place reward. See RL_SAC_PLAN.md."""
        N = self.nworld
        C = {}
        C["time"] = torch.full((N,), W["time"], device=self.device)
        tcp, R = self._tcp()
        op = self._obj_pos()
        gp = self._grasp_point()
        lift_h = (op[:, 2] - float(self.half[2])).clamp(
            0, max(self.lift_req, 0.10) + 0.05)

        # 1 approach (potential delta, pre-seal only)
        phi_a = -torch.norm(tcp - gp, dim=-1)
        C["approach"] = W["approach"] * torch.where(self.sealed, torch.zeros_like(phi_a),
                                                    phi_a - self.phi_approach)
        self.phi_approach = phi_a
        # 2 alignment near contact (within 3 cm, pre-seal)
        near = (~self.sealed) & (torch.norm(tcp - gp, dim=-1) < 0.03)
        align = (-R[:, 2, 2]).clamp(0, 1)               # 1 = cup facing down
        C["align"] = W["align"] * near.float() * align / CTRL_HZ
        # 3 press quality while commanding suction near the object
        press = near & want & (tcp[:, 2] < gp[:, 2])
        C["press"] = W["press"] * press.float() / CTRL_HZ
        # 4 seal event (one-time)
        C["seal"] = W["seal"] * latched_now.float()
        # 5 lift while sealed -- potential on MAX height reached (monotone):
        # a current-height potential paid -4 for sealed descent while
        # RELEASED descent froze the potential and was free, systematically
        # bribing the policy to drop instead of lower-and-place
        lift_cap = max(self.lift_req, 0.10)
        self.max_lift = torch.maximum(self.max_lift, lift_h)
        phi_l = self.max_lift.clamp(0, lift_cap) / lift_cap
        C["lift"] = W["lift"] * self.sealed.float() * (phi_l - self.phi_lift)
        self.phi_lift = torch.where(self.sealed, phi_l, self.phi_lift)
        # 6 transport while sealed and lifted
        phi_t = -torch.norm(op[:, :2] - self.place_target, dim=-1)
        carrying = self.sealed & (lift_h > 0.015)   # low-carry counts
        C["transport"] = W["transport"] * carrying.float() * (phi_t - self.phi_transport)
        self.phi_transport = phi_t
        # 7 drop penalty: released or broke while far from bin and airborne
        if self.mode in ("pnp", "place"):
            over_bin = torch.norm(op[:, :2] - self.place_target, dim=-1) < PLACE_TOL
        else:
            over_bin = (torch.abs(op[:, 0] - self.bin_pos[0]) < BIN_HALF) & \
                       (torch.abs(op[:, 1] - self.bin_pos[1]) < BIN_HALF)
        rest_h = op[:, 2] - float(self.half[2]) - self.target_h
        tilt = torch.arccos(self._obj_zaxis()[:, 2].clamp(-1, 1))
        # track tilt only while the CUP alone supports the object (clear of
        # table and target surface) -- press/set-down contact transients pin
        # the object and spike tilt in ways the policy cannot control
        lift_h_now = op[:, 2] - float(self.half[2])
        airborne = self.sealed & (lift_h_now > 0.012) & (rest_h > 0.012)
        self.max_tilt = torch.where(airborne,
                                    torch.maximum(self.max_tilt, tilt),
                                    self.max_tilt)
        # record height at release AND at seal break (else break-at-height
        # would score full contact-release credit)
        self.release_h = torch.where(released | broke, rest_h.clamp(min=0),
                                     self.release_h)
        # 6b DENSE descend-to-surface: potential -rest_h weighted by a
        # smooth over-target gate -- the xy transport shaping never pulls
        # the object DOWN, so contact-release had no per-step gradient
        d_now = torch.norm(op[:, :2] - self.place_target, dim=-1)
        phi_d = -rest_h.clamp(min=0.0, max=0.40) * \
            torch.exp(-(d_now ** 2) / (2 * 0.07 ** 2))
        C["descend"] = W["descend"] * self.sealed.float() * \
            (phi_d - self.phi_desc)
        self.phi_desc = torch.where(self.sealed, phi_d, self.phi_desc)
        # 6c DENSE tilt discipline while the cup alone carries the object
        C["tilt_pen"] = W["tilt_pen"] * airborne.float() * \
            (tilt - 0.30).clamp(min=0)
        if self.mode in ("pnp", "place"):
            # penalize only AERIAL drops (relative to the TARGET surface)
            bad_drop = (released | broke) & (rest_h > 0.02)
        else:
            bad_drop = (released | broke) & (lift_h > 0.02) & ~over_bin
        C["drop"] = W["drop"] * bad_drop.float()
        # 8 suction chatter: commanding far from the object
        C["chatter"] = W["chatter"] * (want & ~self.sealed &
                                       (torch.norm(tcp - gp, dim=-1) > 0.03)).float()
        C["rel_mask"] = W["rel_mask"] * self._masked_rel.float()
        # 9 action penalty
        C["act"] = W["act"] * a[:, :6].pow(2).sum(-1)
        # 10 table slam: cup below table plane proxy
        C["table_slam"] = W["table_slam"] * (tcp[:, 2] < 0.004).float()

        # terminal conditions
        placed = self.ever_sealed & ~self.sealed & over_bin & \
            (op[:, 2] < float(self.half[2]) + 0.06) & \
            (torch.norm(self.qvel[:, self.vadr_obj:self.vadr_obj + 3], dim=-1) < 0.05)
        off = (op[:, 0] < TABLE_X[0] - 0.08) | (op[:, 0] > TABLE_X[1] + 0.08) | \
              (op[:, 1] < TABLE_Y[0] - 0.10) | (op[:, 1] > TABLE_Y[1] + 0.10)
        off = off & ~over_bin
        if self.mode == "attach":
            placed = self.sealed & (lift_h > 0.02)      # success = seal + lift
            timeout = self.t_step >= EP_LEN_ATTACH
        elif self.mode in ("pnp", "place"):
            # GRADED placement: any gentle set-down (release at <1 cm lift,
            # object slow) ENDS the episode with reward 20*exp(-d^2/2s^2),
            # s = 5 cm -- smooth gradient toward the target from anywhere.
            # 'placed' (the success METRIC) still requires d <= PLACE_TOL.
            spd = torch.norm(self.qvel[:, self.vadr_obj:self.vadr_obj + 3],
                             dim=-1)
            setdown = self.ever_sealed & ~self.sealed & \
                (rest_h.abs() < 0.012) & (spd < 0.08) & \
                (released | (self.t_step > 1))
            # GENTLE-LANDING grade: full credit at rest, fades with impact speed
            gentle = torch.exp(-(spd ** 2) / (2 * 0.04 ** 2))
            # CONTACT-RELEASE grade, Laplace kernel: the warm-start policy
            # releases from ~25 cm, and any Gaussian tight enough to mean
            # "contact" is a reward cliff from there (e^-50).  exp(-h/8cm)
            # keeps a monotone gradient over the whole range: 25 cm -> 0.04,
            # 5 cm -> 0.54, 0 -> 1.  The strict placed gate stays at 1 cm.
            contact_rel = torch.exp(-self.release_h / 0.08)
            # TILT-DISCIPLINE grade, Laplace: warm start carries at ~80 deg
            # peak tilt (never penalized before); a Gaussian is e^-24 there
            tilt_g = torch.exp(-(self.max_tilt - 0.26).clamp(min=0) / 0.35)
            d_tgt = torch.norm(op[:, :2] - self.place_target, dim=-1)
            if self.lift_req > 0:
                lift_ok = self.max_lift >= self.lift_req
                # GRADED lift factor (quadratic ramp): every cm of carry
                # height raises the set-down payout; all-or-nothing gating
                # re-created the never-release trap (run 6a)
                lift_fac = (self.max_lift / self.lift_req).clamp(0, 1) ** 2
            else:
                lift_ok = torch.ones_like(setdown)
                lift_fac = torch.ones_like(self.max_lift)
            placed = setdown & (d_tgt < PLACE_TOL) & lift_ok & \
                (self.release_h < 0.010) & (self.max_tilt < 0.44)  # 1cm, 25deg
            dist_g = torch.exp(-(d_tgt ** 2) / (2 * 0.05 ** 2))
            # cube root: three multiplicative near-zero grades starve PPO of
            # any terminal signal (product ~2e-4 from the warm start); the
            # geometric mean keeps every factor mandatory at a usable scale
            self._graded = setdown.float() * lift_fac * gentle * \
                (contact_rel * tilt_g * dist_g).clamp(min=0) ** (1.0 / 3.0)
            if self.speed_bonus > 0:
                self._graded = self._graded * (
                    1.0 + self.speed_bonus *
                    (1.0 - self.t_step.float() / EP_LEN_PNP))
            timeout = self.t_step >= EP_LEN_PNP
        else:
            timeout = self.t_step >= EP_LEN
        if self.mode in ("pnp", "place"):
            C["place"] = W["place"] * getattr(self, "_graded",
                                              placed.float())
            done_setdown = self._graded > 0
        else:
            C["place"] = W["place"] * placed.float()
            done_setdown = placed
        C["off_table"] = W["off_table"] * off.float()
        done = done_setdown | placed | off | timeout
        comp = torch.stack([C[k] for k in self.RKEYS], dim=-1)
        self.ep_comp += comp
        r = comp.sum(-1)
        info = dict(placed=placed, sealed=self.sealed.clone(),
                    ever_sealed=self.ever_sealed.clone(), off=off,
                    timeout=timeout, ep_comp=self.ep_comp.clone(),
                    ep_len=self.t_step.clone(),
                    max_lift=self.max_lift.clone(),
                    final_d=torch.norm(op[:, :2] - self.place_target, dim=-1),
                    final_spd=torch.norm(
                        self.qvel[:, self.vadr_obj:self.vadr_obj + 3], dim=-1),
                    target_h=self.target_h.clone(),
                    max_tilt=self.max_tilt.clone(),
                    release_h=self.release_h.clone())
        return r, done, info


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    wp.init()
    env = PickEnv(nworld=args.nworld)
    obs = env.observe()
    print(f"obs dim {obs.shape}, nworld {args.nworld}, substeps {env.substeps}")
    t0 = time.time()
    R = torch.zeros(args.nworld, device=env.device)
    for k in range(args.steps):
        a = torch.rand(args.nworld, 7, device=env.device) * 2 - 1
        obs, r, done, info = env.step(a)
        R += r
    dt = time.time() - t0
    sps = args.nworld * args.steps / dt
    print(f"{args.steps} decisions x {args.nworld} worlds in {dt:.2f}s "
          f"= {sps:,.0f} env-steps/s ({sps * env.substeps:,.0f} physics steps/s)")
    print(f"reward mean {R.mean():.3f} std {R.std():.3f} | done {int(done.sum())}")
