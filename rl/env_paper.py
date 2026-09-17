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
    RKEYS_PAPER = ["reach", "lift", "track_c", "track_f", "reg_act", "reg_vel", "fail", "press", "seal"]

    def __init__(self, nworld=1024, device="cuda:0", seed=0, xml=None, dr=False, drive="real",
                 dq_max_deg=None, obs_lag=None, target_max=0.30, start="home",
                 ep_len=100, w_reach=1.0, w_lift=2.0, w_track_c=2.0, w_track_f=4.0,
                 sigma_reach=0.25, sigma_c=0.10, sigma_f=0.02, h_min=0.02,
                 lambda_max=0.02, goal_z=(0.05, 0.25), succ_tol=0.035, grasp_shaping=True,
                 w_press=0.5, w_seal=2.0, obs_ee=True, reach_target="grasp", **kw):
        self.paper = dict(ep_len=int(ep_len), w_reach=w_reach, w_lift=w_lift, w_track_c=w_track_c,
                          w_track_f=w_track_f, sigma_reach=sigma_reach, sigma_c=sigma_c, sigma_f=sigma_f,
                          h_min=h_min, lambda_max=lambda_max, goal_z=tuple(goal_z), succ_tol=succ_tol,
                          start=start, grasp_shaping=bool(grasp_shaping), w_press=w_press, w_seal=w_seal,
                          obs_ee=bool(obs_ee), reach_target=reach_target)
        self.reg_lambda = 0.0                     # set by the trainer: lambda(t) curriculum
        self._paper_ready = False
        super().__init__(nworld=nworld, device=device, seed=seed, xml=xml, mode="pnp", dr=dr,
                         target_max=target_max, drive=drive, dq_max_deg=dq_max_deg, obs_lag=obs_lag, **kw)
        # parent __init__ already called reset once (before our tensors existed) -> finish setup now
        N = nworld
        self.goal = torch.zeros(N, 3, device=device)
        self.a_prev = torch.zeros(N, 7, device=device)
        self.ep_comp_p = torch.zeros(N, len(self.RKEYS_PAPER), device=device)
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

    # ---------------- observation (paper eq. 1) ----------------
    def observe(self):
        parts = [self.qpos[:, :6], self.qvel[:, :6], self._obj_pos(), self.goal, self.a_prev]
        if self.paper["obs_ee"]:
            # OBSERVATION EXTENSION (not in the paper): end-effector position, cup axis and the
            # object-relative vector. The paper's 5-DoF SO-101 policy learns FK implicitly; with
            # [q, p_obj] alone our 6-DoF policies plateaued ~10 cm from the object (paper2_*).
            tcp, R = self._tcp()
            parts += [tcp, R[:, :, 2], self._grasp_point() - tcp]
        if self.obs_lag:
            parts.append(self.q_target - self.qpos[:, :6])
        obs = torch.cat(parts, dim=-1)
        if self.dr:
            obs = obs + torch.randn_like(obs) * 0.005
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
        self.a_prev = a.clone()
        off = (op[:, 0] < TABLE_X[0] - 0.08) | (op[:, 0] > TABLE_X[1] + 0.08) | \
              (op[:, 1] < TABLE_Y[0] - 0.10) | (op[:, 1] > TABLE_Y[1] + 0.10)
        C["fail"] = -1.0 * off.float()
        timeout = self.t_step >= P["ep_len"]
        done = timeout | off
        # success METRIC (paper: final object-goal distance): lifted and within tolerance at episode end
        placed = timeout & lifted & (d_goal < P["succ_tol"])
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
                    release_h=self.release_h.clone(), wmode=torch.zeros(N, dtype=torch.long, device=self.device))
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
