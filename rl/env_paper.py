#!/usr/bin/env python3
"""env_paper :: pick-and-place environment following
   Arafat, Kheyal, Das, Ali, "Efficient Parallel PPO-Based Reinforcement Learning for
   Generalized Pick and Place with Dense Reward Shaping", IEEE QPAIN 2026
   (DOI 10.1109/QPAIN69676.2026.11545619),
re-hosted on this repo's mujoco_warp Pro 630 twin (same physics, suction model and the
hardware-calibrated drive model of env_warp.PickEnv).

Paper setup                                  -> here
  SO-101 arm + gripper, one cube             -> Pro 630 + suction cup, one box (suction ON = "gripper closed")
  randomised object pose + goal pose         -> object xy/yaw random on the table; goal = random 3-D point
                                                (xy within --target_max of the object, z in [z_lo, z_hi])
  start: arm at home, must reach             -> start="home" (env_warp's hover start is available with start="hover")
  obs  o_t = [q, qdot, p_obj, p_goal, a_{t-1}]  (25-D; + q_target-q if obs_lag)
  act  a_t = [a_arm (6), a_grip (1, binary)]   -> joint-delta targets (env_warp convention) + suction logit
  r_t  = r_task + lambda(t) * r_reg
     r_reach = 1 - tanh(|p_obj - p_ee| / sigma_reach)
     r_lift  = 1[z_obj > h_min]
     r_track = 1[z_obj > h_min] * sum_k w_k (1 - tanh(d_goal / sigma_k)),  k in {coarse, fine}
     r_reg   = -(|a_t - a_{t-1}|^2 + |qdot|^2),  lambda ramps 0 -> lambda_max (curriculum)
  termination: time limit (paper 5 s) or failure (robot root height) -> time limit + object off table
  success: object within --succ_tol of the goal at episode end (paper reports 1.8 cm mean)
The paper gives no numeric weights / sigmas; the defaults below are documented assumptions
(reach 1 @ sigma 0.25 m -- our home pose is ~0.45 m from the objects, the paper's 0.10 m kernel is flat there --
 lift 2, track coarse 2 @ 0.10 m + fine 4 @ 0.02 m, lambda_max 0.02, h_min 2 cm).
There is NO release phase in the paper: "placement" = holding the lifted object at the goal.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import env_warp as E  # noqa: E402
from env_warp import PickEnv, TABLE_X, TABLE_Y, CTRL_HZ  # noqa: E402


class PaperPickEnv(PickEnv):
    RKEYS_PAPER = ["reach", "lift", "track_c", "track_f", "reg_act", "reg_vel", "fail", "press", "seal",
                   "speed", "time"]

    def __init__(self, nworld=1024, device="cuda:0", seed=0, xml=None, dr=False, drive="real",
                 dq_max_deg=None, obs_lag=None, target_max=0.30, start="home",
                 ep_len=100, w_reach=1.0, w_lift=2.0, w_track_c=2.0, w_track_f=4.0,
                 sigma_reach=0.25, sigma_c=0.10, sigma_f=0.02, h_min=0.02,
                 lambda_max=0.02, goal_z=(0.05, 0.25), succ_tol=0.035, grasp_shaping=True,
                 w_press=0.5, w_seal=2.0, obs_ee=True, reach_target="grasp", lift_dense=True,
                 w_time=0.0, obs_obj_err=False, obj_err_xy=0.015, obj_err_jit=0.005,
                 obj_err_top=(-0.06, 0.02), obj_err_p=1.0,
                 place_phase=False, place_dwell=5, place_rate=0.015, place_dq_deg=1.2,
                 place_floor=0.004, place_settle=5, place_tol=0.035, **kw):
        self.paper = dict(ep_len=int(ep_len), w_reach=w_reach, w_lift=w_lift, w_track_c=w_track_c,
                          w_track_f=w_track_f, sigma_reach=sigma_reach, sigma_c=sigma_c, sigma_f=sigma_f,
                          h_min=h_min, lambda_max=lambda_max, goal_z=tuple(goal_z), succ_tol=succ_tol,
                          start=start, grasp_shaping=bool(grasp_shaping), w_press=w_press, w_seal=w_seal,
                          obs_ee=bool(obs_ee), reach_target=reach_target, lift_dense=bool(lift_dense),
                          w_time=float(w_time), obs_obj_err=bool(obs_obj_err),
                          obj_err_xy=float(obj_err_xy), obj_err_jit=float(obj_err_jit),
                          obj_err_top=tuple(obj_err_top), obj_err_p=float(obj_err_p),
                          place_phase=bool(place_phase), place_dwell=int(place_dwell),
                          place_rate=float(place_rate), place_dq=float(np.radians(place_dq_deg)),
                          place_floor=float(place_floor), place_settle=int(place_settle),
                          place_tol=float(place_tol))
        self.reg_lambda = 0.0                     # set by the trainer: lambda(t) curriculum
        self._paper_ready = False
        super().__init__(nworld=nworld, device=device, seed=seed, xml=xml, mode="pnp", dr=dr,
                         target_max=target_max, drive=drive, dq_max_deg=dq_max_deg, obs_lag=obs_lag, **kw)
        # parent __init__ already called reset once (before our tensors existed) -> finish setup now
        N = nworld
        self.goal = torch.zeros(N, 3, device=device)
        self.a_prev = torch.zeros(N, 7, device=device)
        # speed bookkeeping: decision index of the first seal / of first arrival at the goal
        # (ep_len when it never happens), so the trainer and the evaluators can log task TIME.
        self.t_seal = torch.full((N,), float(self.paper["ep_len"]), device=device)
        self.t_goal = torch.full((N,), float(self.paper["ep_len"]), device=device)
        self.ep_comp_p = torch.zeros(N, len(self.RKEYS_PAPER), device=device)
        # TRACKER-ERROR DR (off by default; PLAN_QPLANNING "NEW tracker error DR").
        # Per-episode 3-D error of the OBSERVED object position only -- the physics, the seal
        # test and the success metric all keep using the true pose.  Real numbers it stands in
        # for: the colour/depth tracker's +-16 mm xy bias (FINDINGS 21) and camera tops that
        # read 1-6 cm low (FINDINGS 20/23).
        self.obj_err = torch.zeros(N, 3, device=device)
        self.noise_gen = None       # optional torch.Generator for obs noise / jitter (see qplan)
        # SCRIPTED PLACE PHASE (off by default).  Mirrors what rl/real_policy_ctrl.py does on the
        # robot once the policy has the object at the goal: dwell, straight-down descent at the
        # CURRENT tcp xy (the arm places where the object actually is, not where the goal is),
        # release on contact with the table, settle.  With it on, "success" is the object RESTING
        # on the table within place_tol of the goal xy after the cup let go -- so the arrival
        # height and the lateral error cost real decisions and the critic's time head has
        # something to optimise.  pstate: 0 policy, 1 descending, 2 settling, 3 done.
        self.pstate = torch.zeros(N, dtype=torch.long, device=device)
        self.goal_run = torch.zeros(N, dtype=torch.long, device=device)
        self.settle_run = torch.zeros(N, dtype=torch.long, device=device)
        self.desc_run = torch.zeros(N, dtype=torch.long, device=device)
        self.t_placed = torch.full((N,), float(self.paper["ep_len"]), device=device)
        self.placed_ok = torch.zeros(N, dtype=torch.bool, device=device)
        self.place_err = torch.zeros(N, device=device)
        self._paper_ready = True
        self.reset(torch.ones(N, dtype=torch.bool, device=device))

    # ---------------- reset: home start + 3-D goal ----------------
    def reset(self, mask):
        super().reset(mask)                       # object pose, DR, place_target xy, drive state
        if not self._paper_ready:
            return
        idx = torch.nonzero(mask).squeeze(-1)
        if idx.numel() == 0:
            return
        n = idx.numel()
        if self.paper["start"] == "home":
            self.qpos[idx, :6] = torch.tensor(self.q_home + self.rng.normal(0, 0.02, size=(n, 6)),
                                              device=self.device, dtype=torch.float32)
            self.qvel[idx, :6] = 0.0
            E.mjw.forward(self.m, self.d)
            self.q_target[idx] = self.qpos[idx, :6]
            self.q_target_prev[idx] = self.qpos[idx, :6]
            self.q_hist[idx] = self.qpos[idx, None, :6]
            self.q_drive[idx] = self.qpos[idx, :6]
            self.v_drive[idx] = 0.0
            self.v_buf[idx] = 0.0
            self.q_meas_lag[idx] = self.qpos[idx, :6]
            self.qd_meas_lag[idx] = 0.0
        # 3-D goal: parent sampled place_target xy within target_max of the object; add a height
        gz = torch.tensor(self.rng.uniform(*self.paper["goal_z"], size=n), device=self.device, dtype=torch.float32)
        self.goal[idx, :2] = self.place_target[idx]
        self.goal[idx, 2] = gz + float(self.half[2])   # goal for the object CENTRE
        self.a_prev[idx] = 0.0
        self.ep_comp_p[idx] = 0.0
        self.t_seal[idx] = float(self.paper["ep_len"])
        self.t_goal[idx] = float(self.paper["ep_len"])
        self.pstate[idx] = 0
        self.goal_run[idx] = 0
        self.settle_run[idx] = 0
        self.desc_run[idx] = 0
        self.t_placed[idx] = float(self.paper["ep_len"])
        self.placed_ok[idx] = False
        self.place_err[idx] = 0.0
        if self.paper["obs_obj_err"]:
            P = self.paper
            b = self.rng.uniform(-P["obj_err_xy"], P["obj_err_xy"], size=(n, 2))
            tz = self.rng.uniform(P["obj_err_top"][0], P["obj_err_top"][1], size=(n, 1))
            on = (self.rng.random((n, 1)) < P["obj_err_p"]).astype(np.float32)
            self.obj_err[idx] = torch.tensor(np.concatenate([b, tz], axis=1) * on,
                                             device=self.device, dtype=torch.float32)
        else:
            self.obj_err[idx] = 0.0

    # ---------------- tracker-error DR helper ----------------
    def _randn(self, shape):
        g = getattr(self, "noise_gen", None)
        if g is None:
            return torch.randn(shape, device=self.device)
        return torch.randn(shape, device=self.device, generator=g)

    def _obj_err_now(self):
        """Per-episode bias + per-step jitter of the OBSERVED object position (m)."""
        j = self.paper["obj_err_jit"]
        e = self.obj_err
        if j > 0:
            e = e + self._randn(e.shape) * j
        return e

    # ---------------- scripted place phase ----------------
    def _site_jac(self, eps=1e-3):
        """Finite-difference FULL (position + orientation) Jacobian of the tcp site, (N, 6, 6).

        mujoco_warp exposes no jacobian array and a CPU IK per world is out of the question at
        4096 worlds, so the six columns are read off six extra `mjw.forward` calls on perturbed
        qpos (forward is kinematics only -- ~200x cheaper than a decision's physics) and the
        state is restored with a final forward before anything else touches `d`.

        The ORIENTATION rows matter: with position alone the null space rotates the wrist during
        the place descent and the object -- welded a cup radius + half a box below the tcp --
        swings sideways (measured: 4.4 cm median placement error, 38 % placed).  Holding the cup
        orientation fixed is also what the deployed controller does (it streams a fixed R_DOWN).
        """
        q0 = self.qpos[:, :6].clone()
        p0 = self.site_xpos[:, self.sid_tcp].clone()
        R0 = self.xmat_site[:, self.sid_tcp].clone()
        cols = []
        for j in range(6):
            self.qpos[:, j] = q0[:, j] + eps
            E.mjw.forward(self.m, self.d)
            dp = (self.site_xpos[:, self.sid_tcp] - p0) / eps
            dR = torch.einsum("nij,nkj->nik", self.xmat_site[:, self.sid_tcp], R0)
            w = torch.stack([dR[:, 2, 1] - dR[:, 1, 2],
                             dR[:, 0, 2] - dR[:, 2, 0],
                             dR[:, 1, 0] - dR[:, 0, 1]], -1) / (2 * eps)
            cols.append(torch.cat([dp, w], -1))
            self.qpos[:, j] = q0[:, j]
        E.mjw.forward(self.m, self.d)                      # restore
        return torch.stack(cols, dim=-1)                   # (N, 6, 6)

    def _place_action(self, a):
        """Override the policy action on worlds that are in the scripted place phase."""
        P = self.paper
        N = self.nworld
        op = self._obj_pos()
        tcp, _R = self._tcp()
        d_goal_xy = torch.norm(op[:, :2] - self.goal[:, :2], dim=-1)
        lift_h = op[:, 2] - float(self.half[2])
        at_goal = (lift_h > P["h_min"]) & (torch.norm(op - self.goal, dim=-1) < P["succ_tol"])
        self.goal_run = torch.where(at_goal, self.goal_run + 1, torch.zeros_like(self.goal_run))
        start = (self.pstate == 0) & self.sealed & (self.goal_run >= P["place_dwell"])
        self.pstate = torch.where(start, torch.ones_like(self.pstate), self.pstate)
        desc = self.pstate == 1
        if not bool((self.pstate > 0).any()):
            return a
        out = a.clone()
        if desc.any():
            # damped least squares for a pure -z tcp motion at the CURRENT xy
            J = self._site_jac()
            # weight the orientation rows so "keep the cup pointing where it points" is enforced
            # as hard as the descent itself (0.05 m of position error ~ 1 rad of tilt)
            wgt = torch.tensor([1.0, 1.0, 1.0, 0.05, 0.05, 0.05], device=self.device)
            J = J * wgt[None, :, None]
            v = torch.zeros(N, 6, device=self.device)
            v[:, 2] = -P["place_rate"]
            A = torch.einsum("nij,nkj->nik", J, J) + 1e-4 * torch.eye(6, device=self.device)
            dq = torch.einsum("nji,njk->nik", J, torch.linalg.solve(A, v[..., None]))[..., 0]
            # UNIFORM scaling, never a per-joint clamp: clamping each joint separately rotates
            # the Cartesian direction, and a place-down that drifts sideways lands the object
            # off the goal (measured: median placement error 3.0 cm with a per-joint clamp,
            # 61 % placed; see also FINDINGS "per-joint clamp direction chatter").
            sc = (P["place_dq"] / dq.abs().amax(-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.0)
            dq = dq * sc
            out = torch.where(desc[:, None],
                              torch.cat([(dq / self.dq_max).clamp(-1, 1),
                                         torch.ones(N, 1, device=self.device)], -1), out)
            # release: object bottom on the table, or it stopped descending while pressed down
            self.desc_run = torch.where(desc, self.desc_run + 1, self.desc_run)
            bottom = op[:, 2] - float(self.half[2])
            # contact proxy: the object stopped descending although the cup is still being
            # driven down (only after 3 decisions, so the drive dead-time cannot fake it)
            stalled = (self.desc_run > 3) & (self.qvel[:, self.vadr_obj + 2].abs() < 0.005)
            rel = desc & ((bottom <= P["place_floor"]) | stalled)
            self.pstate = torch.where(rel, torch.full_like(self.pstate, 2), self.pstate)
        set_ = self.pstate == 2
        if set_.any():
            self.settle_run = torch.where(set_, self.settle_run + 1, self.settle_run)
            hold = torch.cat([torch.zeros(N, 6, device=self.device),
                              -torch.ones(N, 1, device=self.device)], -1)
            out = torch.where(set_[:, None], hold, out)
            done_now = set_ & (self.settle_run >= P["place_settle"])
            if done_now.any():
                spd = torch.norm(self.qvel[:, self.vadr_obj:self.vadr_obj + 3], dim=-1)
                ok = done_now & (~self.sealed) & (spd < 0.02) & (d_goal_xy < P["place_tol"]) & \
                     ((op[:, 2] - float(self.half[2])) < 0.02)
                self.placed_ok = self.placed_ok | ok
                self.place_err = torch.where(done_now, d_goal_xy, self.place_err)
                self.t_placed = torch.where(ok & (self.t_placed >= P["ep_len"]),
                                            self.t_step.float(), self.t_placed)
                self.pstate = torch.where(done_now, torch.full_like(self.pstate, 3), self.pstate)
        fin = self.pstate == 3
        if fin.any():
            hold = torch.cat([torch.zeros(N, 6, device=self.device),
                              -torch.ones(N, 1, device=self.device)], -1)
            out = torch.where(fin[:, None], hold, out)
        return out

    def step(self, action):
        if self.paper["place_phase"]:
            action = self._place_action(action.clamp(-1, 1))
        return super().step(action)

    # ---------------- observation (paper eq. 1) ----------------
    def observe(self):
        # tracker-error DR: the same 3-D error shifts the observed object centre AND the observed
        # grasp point (a camera that reads the top 5 cm low puts the whole object 5 cm low).
        # The goal is COMMANDED, not perceived, so it is left alone.
        e = self._obj_err_now() if (self.paper["obs_obj_err"] and self._paper_ready) else None
        op = self._obj_pos()
        parts = [self.qpos[:, :6], self.qvel[:, :6], op if e is None else op + e,
                 self.goal, self.a_prev]
        if self.paper["obs_ee"]:
            # OBSERVATION EXTENSION (not in the paper): end-effector position, cup axis and the
            # object-relative vector. The paper's 5-DoF SO-101 policy learns FK implicitly; with
            # [q, p_obj] alone our 6-DoF policies plateaued ~10 cm from the object (paper2_*).
            tcp, R = self._tcp()
            gp = self._grasp_point()
            parts += [tcp, R[:, :, 2], (gp if e is None else gp + e) - tcp]
        if self.obs_lag:
            parts.append(self.q_target - self.qpos[:, :6])
        obs = torch.cat(parts, dim=-1)
        if self.dr:
            obs = obs + self._randn(obs.shape) * 0.005
        return obs

    # ---------------- reward (paper eq. 3-7) + termination ----------------
    def reward(self, want, latched_now, released, broke, tcp_before, obj_before, a):
        P = self.paper
        N = self.nworld
        tcp, R = self._tcp()
        op = self._obj_pos()
        # reach distance: the paper uses the object CENTRE (a gripper encloses the cube from the
        # side). For a suction cup that pulls the tcp down BESIDE the box (replay of paper3_ideal:
        # tcp below the object top, cup tilted 30-43 deg). "grasp" = top centre + cup radius.
        d_obj = torch.norm((self._grasp_point() if P["reach_target"] == "grasp" else op) - tcp, dim=-1)
        lift_h = op[:, 2] - float(self.half[2])
        lifted = lift_h > P["h_min"]
        d_goal = torch.norm(op - self.goal, dim=-1)
        C = {}
        C["reach"] = P["w_reach"] * (1 - torch.tanh(d_obj / P["sigma_reach"])) / CTRL_HZ
        # lift: the paper's indicator 1[z > h_min] has no gradient below h_min; sealed policies
        # kept pressing and never rose (paper4_* replay: sealed 55 steps, max lift 0.0 cm). A dense
        # ramp to the threshold (same value at and above h_min) makes the first centimetres pay.
        if P["lift_dense"]:
            C["lift"] = P["w_lift"] * (lift_h / P["h_min"]).clamp(0.0, 1.0) / CTRL_HZ
        else:
            C["lift"] = P["w_lift"] * lifted.float() / CTRL_HZ
        C["track_c"] = P["w_track_c"] * lifted.float() * (1 - torch.tanh(d_goal / P["sigma_c"])) / CTRL_HZ
        C["track_f"] = P["w_track_f"] * lifted.float() * (1 - torch.tanh(d_goal / P["sigma_f"])) / CTRL_HZ
        # EMBODIMENT ADAPTATION (not in the paper): a parallel gripper closing on a cube grasps
        # trivially; a suction cup must be pressed 3 mm into the top at low speed with suction
        # on. Without a dense press term and a one-time seal bonus neither drive discovered a
        # single seal in 2-4M steps (paper_real / paper_ideal, 2026-09-17). Same terms as
        # env_warp's attach shaping; disable with grasp_shaping=False for the pure paper reward.
        gp = self._grasp_point()
        near = (~self.ever_sealed) & (torch.norm(tcp - gp, dim=-1) < 0.03)
        press = near & want & (tcp[:, 2] < gp[:, 2])
        if P["grasp_shaping"]:
            C["press"] = P["w_press"] * press.float() / CTRL_HZ
            C["seal"] = P["w_seal"] * latched_now.float()
        else:
            C["press"] = torch.zeros(N, device=self.device)
            C["seal"] = torch.zeros(N, device=self.device)
        lam = float(self.reg_lambda)
        C["reg_act"] = -lam * (a - self.a_prev).pow(2).sum(-1) / CTRL_HZ
        C["reg_vel"] = -lam * self.qvel[:, :6].pow(2).sum(-1) / CTRL_HZ
        # DRIVE-ENVELOPE penalty (not in the paper): hinge on joint speed/accel above
        # the soft caps, so the learnt motion stays inside the Pro 630's 36 deg/s
        # following-error ceiling.  Off unless w_speed/w_acc are set (PickEnv helper).
        C["speed"] = self._speed_penalty()
        # TIME penalty (not in the paper): a flat cost per decision until the object is lifted AND
        # at the goal.  The reward is otherwise a rate (everything is /CTRL_HZ), so a policy that
        # arrives 1 s earlier only gains the tracking rate it collects for that extra second --
        # far too weak to trade against the risk of moving faster.  -w/CTRL_HZ per decision makes
        # *finishing* pay, which is what the real robot's cycle time cares about.
        at_goal = lifted & (d_goal < P["succ_tol"])
        C["time"] = -P["w_time"] * (~at_goal).float() / CTRL_HZ
        tnow = self.t_step.float()
        self.t_seal = torch.where(latched_now & (self.t_seal >= P["ep_len"]), tnow, self.t_seal)
        self.t_goal = torch.where(at_goal & (self.t_goal >= P["ep_len"]), tnow, self.t_goal)
        self.a_prev = a.clone()
        off = (op[:, 0] < TABLE_X[0] - 0.08) | (op[:, 0] > TABLE_X[1] + 0.08) | \
              (op[:, 1] < TABLE_Y[0] - 0.10) | (op[:, 1] > TABLE_Y[1] + 0.10)
        C["fail"] = -1.0 * off.float()
        timeout = self.t_step >= P["ep_len"]
        done = timeout | off
        # success METRIC (paper: final object-goal distance): lifted and within tolerance at episode end
        placed = timeout & lifted & (d_goal < P["succ_tol"])
        if P["place_phase"]:
            # the object must be RESTING on the table within place_tol of the goal xy, with the
            # cup released; worlds still mid-descent at the timeout are judged on the same test.
            d_goal_xy = torch.norm(op[:, :2] - self.goal[:, :2], dim=-1)
            spd_o = torch.norm(self.qvel[:, self.vadr_obj:self.vadr_obj + 3], dim=-1)
            at_timeout = timeout & (~self.sealed) & (spd_o < 0.02) & (d_goal_xy < P["place_tol"]) \
                & (lift_h < 0.02)
            placed = self.placed_ok | at_timeout
            self.place_err = torch.where(timeout & (self.pstate < 3), d_goal_xy, self.place_err)
            d_goal = d_goal_xy
        comp = torch.stack([C[k] for k in self.RKEYS_PAPER], dim=-1)
        self.ep_comp_p += comp
        r = comp.sum(-1)
        # bookkeeping shared with the trainer/plots
        self.max_lift = torch.maximum(self.max_lift, lift_h)
        info = dict(placed=placed, sealed=self.sealed.clone(), ever_sealed=self.ever_sealed.clone(),
                    off=off, timeout=timeout, ep_comp=self.ep_comp_p.clone(), ep_len=self.t_step.clone(),
                    max_lift=self.max_lift.clone(), final_d=d_goal.clone(),
                    final_spd=torch.norm(self.qvel[:, self.vadr_obj:self.vadr_obj + 3], dim=-1),
                    target_h=self.goal[:, 2].clone(), max_tilt=self.max_tilt.clone(),
                    release_h=self.release_h.clone(), peak_qd=self.peak_qd.clone(),
                    t_seal=self.t_seal.clone(), t_goal=self.t_goal.clone(),
                    peak_qdd=self.peak_qdd.clone(), at_goal=at_goal.clone(),
                    lifted=lifted.clone(), obj_err=self.obj_err.clone(),
                    wmode=torch.zeros(N, dtype=torch.long, device=self.device))
        if P["place_phase"]:
            info.update(t_placed=self.t_placed.clone(), placed_now=self.placed_ok.clone(),
                        place_err=self.place_err.clone(), pstate=self.pstate.clone())
        return r, done, info

    @property
    def RKEYS(self):
        return self.RKEYS_PAPER

    @RKEYS.setter
    def RKEYS(self, v):        # parent assigns its own list in __init__; ignore
        pass


if __name__ == "__main__":
    import argparse
    import time
    import warp as wp
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--drive", default="real", choices=["real", "ideal"])
    args = ap.parse_args()
    wp.init()
    env = PaperPickEnv(nworld=args.nworld, drive=args.drive, xml=os.path.join(HERE, "scenes", "box_med.xml"))
    obs = env.observe()
    print(f"obs dim {obs.shape}, nworld {args.nworld}, substeps {env.substeps}, RKEYS {env.RKEYS}")
    t0 = time.time()
    Rsum = torch.zeros(args.nworld, device=env.device)
    for k in range(args.steps):
        a = torch.rand(args.nworld, 7, device=env.device) * 2 - 1
        obs, r, done, info = env.step(a)
        Rsum += r
    dt = time.time() - t0
    print(f"{args.steps} x {args.nworld} in {dt:.2f}s = {args.nworld * args.steps / dt:,.0f} env-steps/s | reward mean {Rsum.mean():.3f} | done {int(done.sum())}")
