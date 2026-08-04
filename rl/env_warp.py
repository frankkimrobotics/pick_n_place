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
BIN_XY = np.array([0.10, 0.40])
BIN_HALF = 0.145
TABLE_X = (0.24, 0.52)
TABLE_Y = (-0.16, 0.16)
DQ_MAX = np.radians(2.0)   # per-decision joint delta clamp
EP_LEN = 150
EP_LEN_ATTACH = 40

W = dict(approach=1.0, align=0.3, press=0.5, seal=5.0, lift=2.0,
         transport=1.5, place=10.0, drop=-2.0, chatter=-0.05,
         act=-0.01, time=-0.005, table_slam=-0.5, off_table=-2.0)


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
                 mode="full"):
        """mode='attach': staged sub-task -- episodes START with the cup
        hovering 2-4 cm above the (jittered) grasp point; success = seal +
        hold + 2 cm lift within a 40-step episode. mode='full': whole task."""
        self.nworld = nworld
        self.device = device
        self.mode = mode
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
        mjd.qpos[:6] = self.q_home
        mujoco.mj_forward(self.mjm, mjd)
        # PD gains (fixed diagonal; simpler than per-tick gain scheduling)
        self.kp = torch.tensor([600, 900, 600, 200, 80, 40], device=device,
                               dtype=torch.float32)
        self.kd = torch.tensor([40, 60, 40, 12, 4, 2], device=device,
                               dtype=torch.float32)
        self.tau_max = 100.0
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
        self.RKEYS = ["approach", "align", "press", "seal", "lift",
                      "transport", "place", "drop", "chatter", "act",
                      "time", "table_slam", "off_table"]
        self.ep_comp = torch.zeros(N, len(self.RKEYS), device=device)
        self.bin_pos = torch.tensor([*BIN_XY, 0.0], device=device, dtype=torch.float32)
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
        self.q_target[idx] = self.qpos[idx, :6]
        if self.mode == "attach":
            # place the ARM near the grasp: per-world CPU IK is too slow, so
            # use a canonical pre-solved hover configuration for the object at
            # scene center and correct the object to sit UNDER the cup instead:
            # sample cup-relative offsets and put the object there.
            self.qpos[idx, :6] = torch.tensor(
                self.q_hover + self.rng.normal(0, 0.015, size=(idx.numel(), 6)),
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
        mjw.forward(self.m, self.d)
        tcp, _ = self._tcp()
        self.phi_approach[idx] = -torch.norm(
            tcp[idx] - self._grasp_point()[idx], dim=-1)
        self.phi_transport[idx] = -torch.norm(
            self._obj_pos()[idx, :2] - self.bin_pos[:2], dim=-1)
        self.phi_lift[idx] = 0.0

    # ---------------- suction ----------------
    # Suction = per-world compliant spring-damper (xfrc_applied) with a
    # hard force cap: |F| clamps at F_MAX; sustained saturation breaks the
    # seal. Physically: a force-limited compliant cup (no weld snapping,
    # no jolt, no unbreakable carry -- the RL-hardened suction model).
    K_SUCTION = 1500.0        # N/m  (~1.3 mm sag under a 2 N payload)
    C_SUCTION = 30.0          # N s/m
    BREAK_STEPS = 10          # substeps of saturation before break

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
        gate = want & ~self.sealed & (dist < SEAL_DIST) & pressing & tilt_ok \
            & (rel_speed < SEAL_VEL)
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
        self.xfrc[:, self.bid_obj, 3:] = 0.0
        sat = self.sealed & (Fmag.squeeze(-1) * scale.squeeze(-1) >= F_MAX - 1e-4)
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
        self.q_target = self.q_target + a[:, :6] * DQ_MAX
        want = a[:, 6] > 0
        tcp_before, _ = self._tcp()
        obj_before = self._obj_pos().clone()

        latched_now = self._try_latch(want)
        released = self.sealed & ~want
        if released.any():
            idx = torch.nonzero(released).squeeze(-1)
            self.sealed[idx] = False
            self.xfrc[idx, self.bid_obj] = 0.0
        for _ in range(self.substeps):
            tau = (self.kp * (self.q_target - self.qpos[:, :6])
                   - self.kd * self.qvel[:, :6]).clamp(-self.tau_max, self.tau_max)
            self.ctrl[:, :6] = tau
            self._apply_suction_force()
            mjw.step(self.m, self.d)
        broke = self._check_break()
        self._last_tcp = self._tcp()[0].clone()
        self.t_step += 1

        r, done, info = self.reward(want, latched_now, released, broke,
                                    tcp_before, obj_before, a)
        obs = self.observe()
        if done.any():
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
            (op[:, :2] - self.bin_pos[:2]),               # to-bin xy
        ], dim=-1)

    # ---------------- reward ----------------
    def reward(self, want, latched_now, released, broke, tcp_before, obj_before, a):
        """Staged, potential-based pick-and-place reward. See RL_SAC_PLAN.md."""
        N = self.nworld
        C = {}
        C["time"] = torch.full((N,), W["time"], device=self.device)
        tcp, R = self._tcp()
        op = self._obj_pos()
        gp = self._grasp_point()
        lift_h = (op[:, 2] - float(self.half[2])).clamp(0, 0.10)

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
        # 5 lift while sealed (potential on capped height)
        phi_l = lift_h / 0.10
        C["lift"] = W["lift"] * self.sealed.float() * (phi_l - self.phi_lift)
        self.phi_lift = torch.where(self.sealed, phi_l, self.phi_lift)
        # 6 transport while sealed and lifted
        phi_t = -torch.norm(op[:, :2] - self.bin_pos[:2], dim=-1)
        carrying = self.sealed & (lift_h > 0.05)
        C["transport"] = W["transport"] * carrying.float() * (phi_t - self.phi_transport)
        self.phi_transport = phi_t
        # 7 drop penalty: released or broke while far from bin and airborne
        over_bin = (torch.abs(op[:, 0] - self.bin_pos[0]) < BIN_HALF) & \
                   (torch.abs(op[:, 1] - self.bin_pos[1]) < BIN_HALF)
        bad_drop = (released | broke) & (lift_h > 0.02) & ~over_bin
        C["drop"] = W["drop"] * bad_drop.float()
        # 8 suction chatter: commanding far from the object
        C["chatter"] = W["chatter"] * (want & ~self.sealed &
                                       (torch.norm(tcp - gp, dim=-1) > 0.03)).float()
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
        else:
            timeout = self.t_step >= EP_LEN
        C["place"] = W["place"] * placed.float()
        C["off_table"] = W["off_table"] * off.float()
        done = placed | off | timeout
        comp = torch.stack([C[k] for k in self.RKEYS], dim=-1)
        self.ep_comp += comp
        r = comp.sum(-1)
        info = dict(placed=placed, sealed=self.sealed.clone(),
                    ever_sealed=self.ever_sealed.clone(), off=off,
                    timeout=timeout, ep_comp=self.ep_comp.clone(),
                    ep_len=self.t_step.clone())
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
