#!/usr/bin/env python3
"""wrench_extract :: batched RL for the unknown-height extraction sub-skill,
built ON the proven env_warp.PickEnv (stable batched physics + measured Pro
630 drive model) instead of a from-scratch scene.

Task: episode STARTS sealed on the centre block of a tight 3x3 array whose
neighbour heights are unknown (the policy sees a noisy belief). The policy
must extract vertically until the block clears the (true) neighbours, carry
it to the 0.25 m box at (0.08, 0.40) and release inside — while keeping the
transmitted wrench inside the suction cup's admissible set (violating it for
BREAK_TICKS detaches the block).

Why start sealed: threading hover->press through the tight array is a separate
(scripted-teacher-solvable) skill; the part that needs learning is the
belief-aware, wrench-limited extraction, which the scripted teacher failed.

    python3 wrench_extract.py --smoke
    python3 wrench_extract.py --train --nworld 512 --variants 4 --steps 4000000
    python3 wrench_extract.py --eval --scenes 6 --render
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("MUJOCO_GL", "egl")
import warp as wp                                          # noqa: E402
import mujoco_warp as mjw                                  # noqa: E402

from env_warp import PickEnv, CUP_R                        # noqa: E402
import wrench_pick                                         # noqa: E402
from wrench_pick import (make_scene, _install_wrench_model,  # noqa: E402
                         F_V, ARRAY_C, BOX_C)

# a real cup lip deforms before peeling: 300 ms of persistent violation to
# detach (200 ms made the swing of a carried block terminal ~70% of carries)
wrench_pick.BREAK_TICKS = 150

OUT = os.path.expanduser("~/pnp_rl/wrench_extract")
OBS_DIM, ACT_DIM = 33, 7          # act = PickEnv native: dq(6) + suction logit
EP = 165                          # 16.5 s: home -> approach + press + extract + carry + drop
SIG_H = 0.006                     # belief noise on heights (m, 1 sigma)
CLEAR_MARGIN = 0.004              # block bottom must top neighbours by this


def variant_env(variant_seed, nworld, device="cuda:0", dr=True):
    """Per-variant geometry randomization (mjwarp geometry is model-level,
    so dims/positions vary across VARIANTS, poses/beliefs per world):
    block half-xy 15-26 mm, gap 2-5 mm, array centre x 0.30-0.40 y +-0.06,
    heights 30-60 mm (inside make_scene)."""
    os.makedirs(OUT, exist_ok=True)
    xml = os.path.join(OUT, f"scene_{variant_seed}.xml")
    mpath = xml + ".meta.json"
    if os.path.exists(xml) and os.path.exists(mpath):
        meta = json.load(open(mpath))
    else:
        g = np.random.default_rng(variant_seed + 7777)
        half = float(g.uniform(0.015, 0.026))
        wrench_pick.BLOCK_HALF_XY = half
        wrench_pick.PITCH = 2 * half + float(g.uniform(0.002, 0.005))
        wrench_pick.ARRAY_C = (float(g.uniform(0.30, 0.40)),
                               float(g.uniform(-0.06, 0.06)))
        meta = make_scene(np.random.default_rng(variant_seed), xml)
        json.dump(meta, open(mpath, "w"))
    return ExtractEnv(xml, meta, nworld=nworld, seed=variant_seed,
                      device=device, dr=dr)


class ExtractEnv(PickEnv):
    def __init__(self, xml, meta, nworld, seed=0, device="cuda:0", dr=True):
        self.meta = meta
        # 9 free blocks in mutual contact + cup + table overflow the mujoco_warp
        # defaults (64/48) -- a dropped constraint corrupts the solve into NaN
        self.data_kw = dict(njmax=512, nconmax=256)
        super().__init__(nworld=nworld, mode="pnp", dr=dr, xml=xml,
                         seed=seed, device=device, dq_max_deg=3.2)
        N, dev = nworld, device
        # wrench-set suction (batched; replaces the F_MAX cap)
        self.wr_violation = torch.zeros(N, dtype=torch.long, device=dev)
        self.wr_last = torch.zeros(N, 6, device=dev)
        self.wr_margin = torch.ones(N, device=dev)
        _install_wrench_model(self)
        # true geometry (privileged, reward-side only)
        self.true_h0 = meta[0]["half"][2] * 2
        self.true_nb = max(b["half"][2] * 2 for b in meta[1:])
        self.clear_z = self.true_nb + 0.001 + CLEAR_MARGIN  # world z of "cleared"
        import mujoco
        b2i = lambda n: mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, n)
        nb_jadr = [self.mjm.jnt_qposadr[self.mjm.body_jntadr[b2i(f"object{k}")]]
                   for k in range(1, 9)]
        self.nb_ix = torch.tensor(nb_jadr, dtype=torch.long, device=dev)
        self.nb_rest = torch.tensor([b["pos"] for b in meta[1:]],
                                    dtype=torch.float32, device=dev)   # (8,2)
        self.box_c = torch.tensor(BOX_C, dtype=torch.float32, device=dev)
        # XML rest pose of every free body (arm dofs excluded)
        self._q_rest = torch.tensor(self.mjm.qpos0[6:], dtype=torch.float32,
                                    device=dev)
        # per-variant array centre (geometry is randomized per variant)
        self.ac = [float(meta[0]["pos"][0]), float(meta[0]["pos"][1])]
        self.ac_t = torch.tensor(self.ac, dtype=torch.float32, device=dev)
        # per-variant seal joint pose: tcp pressed onto the centre block top
        from mjwarp_pick_demo import ik, R_DOWN
        d_ik = mujoco.MjData(self.mjm)
        tcp_z = self.true_h0 + 0.001 + CUP_R - 0.004      # pressing depth
        q, err = ik(self.mjm, d_ik, "tcp",
                    [self.ac[0], self.ac[1], tcp_z], R_DOWN, self.q_home)
        assert err < 0.005, f"seal IK failed ({err * 1000:.1f} mm)"
        self.q_seal = torch.tensor(np.asarray(q, np.float32), device=dev)
        # mix-curriculum start poses (env_warp "mix" pattern: value from the
        # easy over-box starts propagates back to teach the carry, which
        # pure exploration never found -- v9 probe climbed to z=0.69)
        self.q_start = {}
        # carry altitude 0.33: the box walls top out at z=0.25 and the block
        # hangs ~0.05 below the tcp -- a lower leg drags the block into the
        # wall (it snags, shear breaks the seal; found via teacher trace)
        mid_xy = [(self.ac[0] + BOX_C[0]) / 2, (self.ac[1] + BOX_C[1]) / 2]
        for name, xyz in (("mid", [*mid_xy, 0.33]),
                          ("box", [BOX_C[0], BOX_C[1], 0.33])):
            qc, ec = ik(self.mjm, d_ik, "tcp", xyz, R_DOWN, self.q_home)
            assert ec < 0.01, f"{name} IK failed ({ec * 1000:.1f} mm)"
            self.q_start[name] = torch.tensor(np.asarray(qc, np.float32),
                                              device=dev)
        # scripted-teacher waypoints: vertical extract, then carry via mid
        # to over-box.  q-space legs; the env's margin governor supplies the
        # wrench modulation.
        q_up, e_up = ik(self.mjm, d_ik, "tcp",
                        [self.ac[0], self.ac[1], 0.30], R_DOWN, self.q_home)
        assert e_up < 0.01
        self.q_wp = torch.stack([
            torch.tensor(np.asarray(q_up, np.float32), device=dev),
            self.q_start["mid"],
            self.q_start["box"],
        ])                                                     # (3, 6)
        # approach: hover above the array + deepest-possible press target
        # (the geometric latch stops the descent at the TRUE block height;
        # the policy/teacher only know the noisy belief)
        q_hov, e_h = ik(self.mjm, d_ik, "tcp",
                        [self.ac[0], self.ac[1], 0.14], R_DOWN, self.q_home)
        assert e_h < 0.01
        self.q_hover = torch.tensor(np.asarray(q_hov, np.float32), device=dev)
        q_dp, e_d = ik(self.mjm, d_ik, "tcp",
                       [self.ac[0], self.ac[1], 0.030 + 0.001 + CUP_R - 0.004],
                       R_DOWN, self.q_home)
        assert e_d < 0.01
        self.q_deep = torch.tensor(np.asarray(q_dp, np.float32), device=dev)
        q_pp, e_p = ik(self.mjm, d_ik, "tcp",
                       [self.ac[0], self.ac[1], 0.095], R_DOWN, self.q_home)
        assert e_p < 0.01
        self.q_pre = torch.tensor(np.asarray(q_pp, np.float32), device=dev)
        # beliefs + task state
        self.bel_h = torch.zeros(N, device=dev)
        self.bel_nb = torch.zeros(N, device=dev)
        self.cleared = torch.zeros(N, dtype=torch.bool, device=dev)
        self.start_mode = torch.zeros(N, dtype=torch.long, device=dev)
        self.post_rel = torch.full((N,), -1, dtype=torch.long, device=dev)
        self.ep_broke = torch.zeros(N, dtype=torch.bool, device=dev)
        self.phi_c = torch.zeros(N, device=dev)
        self.phi_t = torch.zeros(N, device=dev)
        self.phi_d = torch.zeros(N, device=dev)
        self.phi_h = torch.zeros(N, device=dev)
        self.phi_a = torch.zeros(N, device=dev)
        self.prev_disp = torch.zeros(N, device=dev)
        self.time_pen = 0.01          # per-decision cost (speed round: 0.04)
        self.speed_bonus = 0.0        # extra place bonus * (1 - t/EP)
        self._ext_ready = True
        self.reset(torch.ones(N, dtype=torch.bool, device=dev))

    # ---------------- reset: start SEALED on the centre block --------------
    def reset(self, mask):
        if not getattr(self, "_ext_ready", False):
            return super().reset(mask)         # parent-init path
        super().reset(mask)                    # DR draws + counters (poses overwritten)
        idx = torch.nonzero(mask).squeeze(-1)
        if idx.numel() == 0:
            return
        n = idx.numel()
        # mix curriculum: 40% hover (full task incl. approach + press),
        # 20% sealed in the array, 15% mid-carry, 25% over the box
        u = self.rng.random(n)
        u_t = torch.tensor(u, device=self.device)
        g_hover = idx[u_t < 0.4]
        g_mid = idx[(u_t >= 0.6) & (u_t < 0.75)]
        g_box = idx[u_t >= 0.75]
        g_carry = idx[u_t >= 0.6]
        qn = torch.tensor(self.rng.normal(0, 0.006, size=(n, 6)),
                          device=self.device, dtype=torch.float32)
        self.qpos[idx, :6] = self.q_seal[None, :] + qn
        # full-task worlds start at the robot's HOME pose (START_Q), exactly
        # like a real run -- transit to the array is part of the episode
        qn_h = torch.tensor(
            self.q_home + self.rng.normal(0, 0.02, size=(len(g_hover), 6)),
            device=self.device, dtype=torch.float32)
        self.qpos[g_hover, :6] = qn_h
        qn_c = torch.tensor(self.rng.normal(0, 0.02, size=(len(g_mid), 6)),
                            device=self.device, dtype=torch.float32)
        self.qpos[g_mid, :6] = self.q_start["mid"][None, :] + qn_c
        qn_b = torch.tensor(self.rng.normal(0, 0.02, size=(len(g_box), 6)),
                            device=self.device, dtype=torch.float32)
        self.qpos[g_box, :6] = self.q_start["box"][None, :] + qn_b
        self.qpos[idx, 6:] = self._q_rest[None, :]
        self.qvel[idx, :] = 0.0
        self.xfrc[idx] = 0.0
        mjw.forward(self.m, self.d)
        tcp, R = self._tcp()
        # carry-start worlds: block hangs on the cup, not in the array
        if g_carry.numel():
            qa = self.jadr_obj
            self.qpos[g_carry, qa] = tcp[g_carry, 0]
            self.qpos[g_carry, qa + 1] = tcp[g_carry, 1]
            self.qpos[g_carry, qa + 2] = tcp[g_carry, 2] - 0.004 - float(self.half[2])
            self.qpos[g_carry, qa + 3] = 1.0
            self.qpos[g_carry, qa + 4:qa + 7] = 0.0
            mjw.forward(self.m, self.d)
        op = self._obj_pos()
        self.anchor[idx] = torch.einsum(
            "nij,nj->ni", R[idx].transpose(1, 2), op[idx] - tcp[idx])
        self.anchor_R[idx] = torch.einsum(
            "nij,njk->nik", R[idx].transpose(1, 2), self._obj_R()[idx])
        self.sealed[idx] = True
        self.ever_sealed[idx] = True
        self.sealed[g_hover] = False          # hover worlds must earn the seal
        self.ever_sealed[g_hover] = False
        self.wr_violation[idx] = 0
        self.wr_last[idx] = 0.0
        self.wr_margin[idx] = 1.0
        # measured-drive state must start AT the seal pose
        self.q_target[idx] = self.qpos[idx, :6]
        self.q_target_prev[idx] = self.qpos[idx, :6]
        self.q_hist[idx] = self.qpos[idx, None, :6]
        self.q_drive[idx] = self.qpos[idx, :6]
        self.v_drive[idx] = 0.0
        self.v_buf[idx] = 0.0
        self.q_meas_lag[idx] = self.qpos[idx, :6]
        self.qd_meas_lag[idx] = 0.0
        self.prev_dq[idx] = 0.0
        self.prev_qd[idx] = 0.0
        # noisy height beliefs (what the policy is told)
        bh = self.true_h0 + self.rng.normal(0, SIG_H, size=n)
        bn = self.true_nb + self.rng.normal(0, SIG_H, size=n)
        self.bel_h[idx] = torch.tensor(np.clip(bh, 0.025, 0.070),
                                       device=self.device, dtype=torch.float32)
        self.bel_nb[idx] = torch.tensor(np.clip(bn, 0.025, 0.070),
                                        device=self.device, dtype=torch.float32)
        self.cleared[idx] = False
        self.cleared[g_carry] = True
        self.start_mode[idx] = 0
        self.start_mode[g_mid] = 1
        self.start_mode[g_box] = 2
        self.start_mode[g_hover] = 3
        self.post_rel[idx] = -1
        self.ep_broke[idx] = False
        # potentials at the start state
        bottom = op[idx, 2] - float(self.half[2])
        self.phi_c[idx] = (bottom - self.clear_z).clamp(min=-0.06, max=0.025)
        d_box = torch.norm(op[idx, :2] - self.box_c, dim=-1)
        self.phi_t[idx] = -d_box.clamp(max=0.6)
        self.phi_d[idx] = -(op[idx, 2] - 0.06).clamp(min=0, max=0.4) * \
            torch.exp(-(d_box ** 2) / (2 * 0.08 ** 2))
        self.phi_h[idx] = -(op[idx, 2] - 0.30).clamp(min=0.0)
        gp_bel = torch.stack([
            torch.full((n,), self.ac[0], device=self.device),
            torch.full((n,), self.ac[1], device=self.device),
            self.bel_h[idx] + 0.001 + CUP_R], -1)
        self.phi_a[idx] = -torch.norm(tcp[idx] - gp_bel, dim=-1)
        self.prev_disp[idx] = 0.0
        # relative-velocity damper: forget pre-teleport cup positions
        self._tcp_prev = self._tcp()[0].clone()

    # ---------------- step: gate release until extracted --------------------
    def step(self, action):
        # A release before the block has cleared the array is never useful,
        # and (found at v7's 63%-clear plateau) neither is releasing right
        # after clearing: "drop it where it is, take -4" beat the risky
        # 0.45 m carry, so success was never discovered.  Gate release until
        # the block is cleared AND over the box (a real executor would gate
        # the pump the same way); a masked release command costs a small
        # timing penalty.  Only exits now: break, timeout, or place.
        a = action.clone()
        # margin governor: scale joint deltas down as the wrench margin
        # erodes (the teacher's "wrench-modulated speed", made structural).
        # v11 probe: free-speed carries detached 70% within 0.24 m.
        gov = torch.where(self.sealed, self.wr_margin.clamp(0.25, 1.0),
                          torch.ones_like(self.wr_margin))
        a[:, :6] = a[:, :6] * gov[:, None]
        op = self._obj_pos()
        near_box = torch.norm(op[:, :2] - self.box_c, dim=-1) < 0.12
        allow = self.cleared & near_box
        self._rel_early = (a[:, 6] <= 0) & ~allow & self.sealed
        a[:, 6] = torch.where(allow, a[:, 6], torch.ones_like(a[:, 6]))
        # gate replaces the 3-consecutive-off hysteresis: a gated release is
        # safe by construction, and the hysteresis made release a 3-action
        # commitment random exploration threads only 6% of the time (v10:
        # mean release logit never went negative, succ was sampling noise)
        imm = allow & (a[:, 6] <= 0)
        self.rel_count = torch.where(
            imm, torch.full_like(self.rel_count, 5), self.rel_count)
        return super().step(a)

    # ---------------- observation (33-D, belief-based) ---------------------
    def observe(self):
        tcp, R = self._tcp()
        op = self._obj_pos()
        w = self.wr_last
        bel_bottom = op[:, 2] - self.bel_h / 2
        clear_bel = bel_bottom - (self.bel_nb + 0.001 + CLEAR_MARGIN)
        # tcp is expressed relative to the ARRAY CENTRE: absolute workspace
        # coordinates keyed the policy to each variant's layout and killed
        # generalization to unseen array placements (v8 9/12, PPO-ft 4/12)
        tcp_rel = tcp - torch.cat(
            [self.ac_t, torch.zeros(1, device=self.device)])[None, :]
        parts = [
            self.qpos[:, :6], self.qvel[:, :6],
            tcp_rel, R[:, :, 2], op - tcp,
            self.box_c[None, :] - tcp[:, :2],
            self.bel_h[:, None], self.bel_nb[:, None], clear_bel[:, None],
            (-w[:, 0:1]) / F_V, w[:, 1:2] / F_V,
            w[:, 3:4] * 10.0, w[:, 4:5] * 10.0,
            self.wr_margin[:, None],
            self.sealed.float()[:, None],
            (self.t_step.float() / EP)[:, None],
        ]
        obs = torch.cat(parts, dim=-1)
        if self.dr:
            obs = obs + torch.randn_like(obs) * 0.005
        return obs

    # ---------------- reward -----------------------------------------------
    def reward(self, want, latched_now, released, broke, tcp_before,
               obj_before, a):
        N = self.nworld
        op = self._obj_pos()
        sealed = self.sealed
        r = torch.zeros(N, device=self.device)
        # 0) approach + seal (hover starts): pull the cup to the BELIEVED
        # grasp point; one-time bonus when the latch actually fires
        tcp_now, _ = self._tcp()
        gp_bel = torch.stack([
            torch.full((N,), self.ac[0], device=self.device),
            torch.full((N,), self.ac[1], device=self.device),
            self.bel_h + 0.001 + CUP_R], -1)
        phi_a = -torch.norm(tcp_now - gp_bel, dim=-1)
        r += 5.0 * (phi_a - self.phi_a) * (~self.ever_sealed).float()
        self.phi_a = phi_a
        r += 3.0 * latched_now.float()
        # 1) clearance progress (privileged true neighbour height)
        bottom = op[:, 2] - float(self.half[2])
        phi_c = (bottom - self.clear_z).clamp(min=-0.06, max=0.025)
        r += 50.0 * (phi_c - self.phi_c) * sealed.float()
        self.phi_c = phi_c
        newly_cleared = sealed & (bottom > self.clear_z) & ~self.cleared
        r += 4.0 * newly_cleared.float()       # one-time extraction bonus
        self.cleared |= newly_cleared
        # 2) transport (only once cleared — no reward for dragging through).
        # 18x: at 8x the v8 policy never discovered the carry (probe: it
        # extracted then climbed vertically to z=0.49 until timeout)
        d_box = torch.norm(op[:, :2] - self.box_c, dim=-1)
        phi_t = -d_box.clamp(max=0.6)
        gate = (self.cleared & sealed).float()
        r += 18.0 * (phi_t - self.phi_t) * gate
        self.phi_t = phi_t
        # ceiling: carrying above 0.30 m is never useful; potential-based
        # so hovering high costs once, not per-step
        phi_h = -(op[:, 2] - 0.30).clamp(min=0.0)
        r += 3.0 * (phi_h - self.phi_h) * sealed.float()
        self.phi_h = phi_h
        # 3) lower toward the box floor when above it
        phi_d = -(op[:, 2] - 0.06).clamp(min=0, max=0.4) * \
            torch.exp(-(d_box ** 2) / (2 * 0.08 ** 2))
        r += 4.0 * (phi_d - self.phi_d) * gate
        self.phi_d = phi_d
        # 4) wrench prudence: stay inside the admissible set
        r -= 0.05 * (1.0 - self.wr_margin.clamp(0.0, 1.0)) * sealed.float()
        # 5) neighbour disturbance (penalize increases only)
        nb_x = self.qpos[:, self.nb_ix]          # (N,8)
        nb_y = self.qpos[:, self.nb_ix + 1]
        disp = ((nb_x - self.nb_rest[None, :, 0]) ** 2 +
                (nb_y - self.nb_rest[None, :, 1]) ** 2).sqrt().clamp(max=0.04)
        disp = disp.sum(-1)
        r -= 35.0 * (disp - self.prev_disp).clamp(min=0)
        self.prev_disp = disp
        # 6) time + break; releasing before the block is even extracted is
        # abandoning the job -- as bad as a break, else "give up instantly"
        # is a local optimum (observed: ep len collapsed to ~20 at 300k)
        r -= self.time_pen
        r -= 6.0 * broke.float()
        r -= 0.05 * getattr(self, "_rel_early",
                            torch.zeros_like(broke)).float()
        self.ep_broke |= broke
        # 7) release / break -> settle countdown -> landing evaluation
        start = (released | broke) & (self.post_rel < 0)
        self.post_rel = torch.where(
            start, torch.full_like(self.post_rel, 8), self.post_rel)
        tick = self.post_rel > 0
        self.post_rel = torch.where(tick, self.post_rel - 1, self.post_rel)
        ev = self.post_rel == 0
        in_box = ev & (torch.abs(op[:, 0] - self.box_c[0]) < 0.105) \
            & (torch.abs(op[:, 1] - self.box_c[1]) < 0.105) \
            & (op[:, 2] > 0.03) & (op[:, 2] < 0.25)
        r += torch.where(
            in_box,
            25.0 + self.speed_bonus * (1.0 - self.t_step.float() / EP),
            torch.zeros_like(r))
        r -= torch.where(ev & ~in_box, torch.full_like(r, 4.0),
                         torch.zeros_like(r))
        off = op[:, 2] < -0.05
        r -= 5.0 * off.float()
        # a pending settle countdown may finish past EP -- cutting it off
        # scored a release at t=133 with the block mid-fall over the box
        # as a failure (eval 101/104/105, all steps==EP)
        done = ev | off | ((self.t_step >= EP) & (self.post_rel < 0))
        info = dict(placed=in_box, broke=self.ep_broke.clone(),
                    disturb=disp, cleared=self.cleared.clone(),
                    mode=self.start_mode.clone(), t_done=self.t_step.clone())
        return r, done, info


def wp_init(env):
    """Waypoint leg for each start mode (hover starts at the extract leg,
    which it enters after the press stage seals)."""
    m = env.start_mode
    return torch.where(m == 1, 1, torch.where(m == 2, 2, 0)).to(env.device)


def _track(q, wp, dq_max, speed):
    """Direction-preserving tracking step: the largest joint moves at
    `speed` (in dq_max units), the rest proportionally.  A per-joint box
    clamp saturates joints unevenly and bows the tcp path sideways --
    measured 11-17 mm lateral miss at the latch gate."""
    err = (wp - q) / dq_max
    m = err.abs().max(-1, keepdim=True).values.clamp(min=1e-6)
    return err * (speed / m).clamp(max=1.0)


def teacher_action(env, wp_idx):
    """Batched scripted expert.  Unsealed worlds: slow press toward the
    deepest-possible block top -- the geometric latch stops them at the TRUE
    height.  Sealed: track q-space waypoints (extract-up -> mid -> over-box),
    release once over the box.  Returns (action (N,7) in [-1,1], wp_idx)."""
    N = env.nworld
    q = env.q_target                                  # track own reference
    wp = env.q_wp[wp_idx]                             # (N,6)
    err = wp - q
    close = err.abs().max(-1).values < math.radians(4.0)
    wp_idx = torch.where(close & (wp_idx < 2) & env.sealed, wp_idx + 1, wp_idx)
    wp = env.q_wp[wp_idx]
    dq = _track(q, wp, env.dq_max, 1.0)
    # press = two stages.  Stage 1: fast to the PRE-PRESS point just above
    # the tallest possible block (z=0.075) -- this absorbs the hover-pose
    # noise while still clear of everything.  Stage 2: slow straight descent
    # to the deepest press; xy cannot drift meaningfully over those 4 cm, so
    # the tip crosses the true block top inside the 12 mm latch gate.
    tcp_now, _ = env._tcp()
    # transit stage: from HOME (or anywhere laterally off the array) go to
    # the hover point above the array first, then pre-press, then descend.
    # Descend ONLY while laterally aligned (<4 mm): a height-based switch
    # left tall blocks (top near the pre-press z) only ~10 mm of travel to
    # converge xy, and the tip crossed their top outside the latch gate.
    xy_off = torch.norm(tcp_now[:, :2] - env.ac_t[None, :], dim=-1)
    far = xy_off > 0.04
    # smooth height-gated press (sharp alignment gates clone badly: the
    # regression smooths the discontinuity into neither behaviour).  The
    # pre-press point sits at z=0.095 so even a 60 mm block (top+cup at
    # ~0.068) gets ~35 mm of slow, converging runway before its top.
    phase1 = tcp_now[:, 2] > 0.105
    dq_hov = _track(q, env.q_hover[None, :], env.dq_max, 0.9)
    dq_pre = _track(q, env.q_pre[None, :], env.dq_max, 0.9)
    # 0.25, not slower: press actions below ~0.2 sink under the clone's
    # regression noise floor (v13: best MSE ever, 0/12 -- the tiny descent
    # labels were drowned by the large transit actions in the loss)
    dq_deep = _track(q, env.q_deep[None, :], env.dq_max, 0.25)
    dq_press = torch.where(far[:, None], dq_hov,
                           torch.where(phase1[:, None], dq_pre, dq_deep))
    dq = torch.where(env.sealed[:, None], dq, dq_press)
    op = env._obj_pos()
    near = torch.norm(op[:, :2] - env.box_c, dim=-1) < 0.10
    settled = near & (wp_idx == 2) & close
    a = torch.zeros(N, 7, device=env.device)
    a[:, :6] = dq
    a[:, 6] = torch.where(settled & env.cleared, -1.0, 1.0)
    return a, wp_idx


# ------------------------------------------------------------------ PPO
class AC(nn.Module):
    def __init__(self, obs=OBS_DIM, act=ACT_DIM, h=256, std_max=-0.3):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(obs, h), nn.ELU(), nn.Linear(h, h),
                                nn.ELU(), nn.Linear(h, act))
        self.v = nn.Sequential(nn.Linear(obs, h), nn.ELU(), nn.Linear(h, h),
                               nn.ELU(), nn.Linear(h, 1))
        self.log_std = nn.Parameter(torch.full((act,), -0.7))
        self.std_max = std_max

    def dist(self, o):
        # bounded std: collapse guard (run v3 blew up at 540k with free std)
        return torch.distributions.Normal(
            self.pi(o), self.log_std.clamp(-2.5, self.std_max).exp())

    def value(self, o):
        return self.v(o).squeeze(-1)


def bc(a):
    """Behaviour-clone the (now 128/128) scripted teacher: roll it out with
    exploration noise on the dq channels, regress pi on the teacher's clean
    action.  ~10 min for 400k pairs; PPO then polishes from this init."""
    dev = "cuda"
    os.makedirs(OUT, exist_ok=True)
    wp.init()
    per = a.nworld // a.variants
    envs = [variant_env(v, per, device=dev, dr=True)
            for v in range(a.variants)]
    wpi = [wp_init(e) for e in envs]
    pol = None
    if a.dagger and os.path.exists(os.path.join(OUT, a.dagger)):
        pol = AC(h=a.h, std_max=a.std_max).to(dev)
        pol.load_state_dict(torch.load(os.path.join(OUT, a.dagger),
                                       map_location=dev, weights_only=True))
        pol.eval()
        print(f"[bc] DAgger round: half the worlds executed by {a.dagger}",
              flush=True)
    OBs, ACs = [], []
    pairs = 0
    while pairs < a.bc_pairs:
        for i, e in enumerate(envs):
            # done worlds got fresh modes on auto-reset: re-derive their leg
            fresh = e.t_step == 0
            wpi[i] = torch.where(fresh, wp_init(e), wpi[i])
            obs = e.observe()
            ta, wpi[i] = teacher_action(e, wpi[i])
            OBs.append(obs)
            ACs.append(ta.clone())
            if pol is not None and i % 2 == 0:
                # on-policy states, teacher labels (DAgger)
                with torch.no_grad():
                    ex = torch.tanh(pol.pi(obs))
                ex[:, :6] = (ex[:, :6] +
                             torch.randn_like(ex[:, :6]) * 0.10).clamp(-1, 1)
            else:
                # phase-scaled exploration noise: 0.25 flat drowned the
                # 0.25-scale press action at dq_max 3.2 deg (demos stopped
                # sealing -> the clone inherited the failure)
                ex = ta.clone()
                # DART: enough press noise that the teacher DEMONSTRATES
                # lateral corrections (0.05 taught open-loop tracking only,
                # and the clone drifted through the 12 mm latch gate)
                ns = torch.where(e.sealed[:, None], 0.20, 0.12)
                ex[:, :6] = (ex[:, :6] +
                             torch.randn_like(ex[:, :6]) * ns).clamp(-1, 1)
            _, _, dn, nf = e.step(ex)
            if dn.any():
                di = torch.nonzero(dn).squeeze(-1)
                bc._n = getattr(bc, "_n", 0) + di.numel()
                bc._p = getattr(bc, "_p", 0) + int(nf["placed"][di].sum())
            pairs += per
        if pairs % 50000 < a.nworld:
            print(f"[bc] {pairs} pairs", flush=True)
    print(f"[bc] demo quality: {getattr(bc, '_p', 0)}/{getattr(bc, '_n', 0)} "
          f"episodes placed during collection", flush=True)
    OB = torch.cat(OBs)
    AC_t = torch.cat(ACs)
    ac = AC(h=a.h, std_max=a.std_max).to(dev)
    opt = torch.optim.Adam(ac.pi.parameters(), lr=1e-3)
    n = OB.shape[0]
    for ep in range(a.bc_epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for k in range(0, n, 8192):
            mb = perm[k:k + 8192]
            loss = ((ac.pi(OB[mb]) - AC_t[mb]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * mb.numel()
        print(f"[bc] epoch {ep} mse {tot / n:.4f}", flush=True)
    torch.save(ac.state_dict(), os.path.join(OUT, "policy_bc.pt"))
    print(f"[bc] saved policy_bc.pt ({n} pairs)")


def train(a):
    dev = "cuda"
    os.makedirs(OUT, exist_ok=True)
    wp.init()
    per = a.nworld // a.variants
    envs = [variant_env(v, per, device=dev, dr=a.dr)
            for v in range(a.variants)]
    for e in envs:
        e.time_pen, e.speed_bonus = a.time_pen, a.speed_bonus
    ac = AC(h=a.h, std_max=a.std_max).to(dev)
    if a.warm and os.path.exists(a.warm):
        ac.load_state_dict(torch.load(a.warm, map_location=dev))
        print(f"[train] warm start from {a.warm}", flush=True)
    opt = torch.optim.Adam(ac.parameters(), lr=a.lr)
    T, N = a.rollout, per * a.variants
    step, t0, best = 0, time.time(), -1e9
    log = open(os.path.join(OUT, "log.jsonl"), "a")
    while step < a.steps:
        OB = torch.zeros(T, N, OBS_DIM, device=dev)
        AA = torch.zeros(T, N, ACT_DIM, device=dev)
        LP = torch.zeros(T, N, device=dev)
        RW = torch.zeros(T, N, device=dev)
        DN = torch.zeros(T, N, device=dev)
        VL = torch.zeros(T + 1, N, device=dev)
        obs = torch.cat([e.observe() for e in envs], 0)
        ep_place = ep_n = ep_broke = ep_clear = 0
        ep_place_a = ep_n_a = 0          # full-task (home-start) episodes
        ep_dist = 0.0
        ep_steps_p = ep_np = 0           # steps-to-place (cycle time)
        with torch.no_grad():
            for t in range(T):
                dist = ac.dist(obs)
                raw = dist.sample()
                OB[t], AA[t] = obs, raw
                LP[t] = dist.log_prob(raw).sum(-1)
                VL[t] = ac.value(obs)
                act = torch.tanh(raw)
                outs = [e.step(act[i * per:(i + 1) * per])
                        for i, e in enumerate(envs)]
                obs = torch.cat([o[0] for o in outs], 0)
                RW[t] = torch.cat([o[1] for o in outs], 0)
                DN[t] = torch.cat([o[2].float() for o in outs], 0)
                for (_, _, done, info) in outs:
                    if done.any():
                        di = torch.nonzero(done).squeeze(-1)
                        ep_n += di.numel()
                        ep_place += int(info["placed"][di].sum())
                        ep_broke += int(info["broke"][di].sum())
                        ep_clear += int(info["cleared"][di].sum())
                        ep_dist += float(info["disturb"][di].sum())
                        da = di[info["mode"][di] == 3]   # full task from home
                        ep_n_a += da.numel()
                        ep_place_a += int(info["placed"][da].sum())
                        dp = da[info["placed"][da]]
                        ep_np += dp.numel()
                        ep_steps_p += int(envs[0].t_step.new_zeros(1) .sum()) if dp.numel() == 0 else int(info["t_done"][dp].sum())
            VL[T] = ac.value(obs)
            adv = torch.zeros(T, N, device=dev)
            gae = torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                delta = RW[t] + a.gamma * (1 - DN[t]) * VL[t + 1] - VL[t]
                gae = delta + a.gamma * a.lam * (1 - DN[t]) * gae
                adv[t] = gae
            ret = adv + VL[:T]
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        step += N * T
        fo, fa = OB.reshape(-1, OBS_DIM), AA.reshape(-1, ACT_DIM)
        fl, fad, fr = LP.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        kl_stop = False
        for _ in range(a.epochs):
            if kl_stop:
                break
            perm = torch.randperm(fo.shape[0], device=dev)
            for k in range(0, fo.shape[0], a.minibatch):
                mb = perm[k:k + a.minibatch]
                dist = ac.dist(fo[mb])
                lp = dist.log_prob(fa[mb]).sum(-1)
                logr = lp - fl[mb]
                # KL early stop: the update that killed run v3 walked the
                # policy far off-distribution inside a single iteration
                approx_kl = ((logr.exp() - 1) - logr).mean()
                if approx_kl > a.kl:
                    kl_stop = True
                    break
                ratio = logr.exp()
                s1 = ratio * fad[mb]
                s2 = ratio.clamp(0.8, 1.2) * fad[mb]
                pl = -torch.min(s1, s2).mean()
                vl = ((ac.value(fo[mb]) - fr[mb]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pl + 0.5 * vl - a.ent * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
                opt.step()
        sr = ep_place / max(ep_n, 1)
        cl = ep_clear / max(ep_n, 1)
        md = ep_dist / max(ep_n, 1) * 1000
        std = float(ac.log_std.detach().clamp(-2.5, a.std_max).exp().mean())
        sr_a = ep_place_a / max(ep_n_a, 1)   # full-task (home-start) success
        cyc = ep_steps_p / max(ep_np, 1)     # mean decisions to place (home)
        row = dict(step=step, succ=sr, succ_arr=sr_a, clear=cl, disturb_mm=md,
                   broke=ep_broke / max(ep_n, 1), eps=ep_n, eps_arr=ep_n_a,
                   cycle=cyc, std=std, sps=int(step / (time.time() - t0)))
        print(f"[{step:>9}] succ {sr:.1%} succ_home {sr_a:.1%} clear {cl:.1%} "
              f"cycle {cyc:.0f} broke {row['broke']:.1%} eps {ep_n} "
              f"std {std:.2f} {row['sps']} sps", flush=True)
        json.dump(row, log)
        log.write("\n")
        log.flush()
        # best = task score, not raw success (success is 0 for a long time,
        # so gating on it saved every iteration incl. the collapsed ones)
        score = sr_a * 200.0 + sr * 100.0 + cl * 10.0 - row["broke"] \
            + (20.0 * (1.0 - cyc / EP) if ep_np > 0 else 0.0)
        if score > best:
            best = score
            torch.save(ac.state_dict(), os.path.join(OUT, "policy_best.pt"))
    torch.save(ac.state_dict(), os.path.join(OUT, "policy_final.pt"))
    print(f"[train] done, best succ {best:.1%}")


# ------------------------------------------------------------------ eval
def evaluate(a):
    import cv2
    import mujoco
    dev = "cuda"
    wp.init()
    ac = AC(h=a.h).to(dev)
    ckpt = os.path.join(OUT, a.ckpt)
    ac.load_state_dict(torch.load(ckpt, map_location=dev))
    ac.eval()
    rows, clips = [], []
    for s in range(a.scenes):
        vseed = 100 + s                       # UNSEEN variants
        env = variant_env(vseed, 1, device=dev, dr=False)
        for _ in range(64):     # eval = FULL task: hover start (approach incl.)
            if int(env.start_mode[0]) == 3:
                break
            env.reset(torch.ones(1, dtype=torch.bool, device=dev))
        obs = env.observe()
        rec, done = [], torch.zeros(1, dtype=torch.bool)
        info = {}
        for t in range(EP + 12):
            with torch.no_grad():
                act = torch.tanh(ac.pi(obs))
            if a.dq_gain != 1.0:
                act = act.clone()
                act[:, :6] = (act[:, :6] * a.dq_gain).clamp(-1, 1)
            env.auto_reset = False
            obs, r, done, info = env.step(act)
            rec.append((env.qpos[0].cpu().numpy().copy(),
                        bool(env.sealed[0]), float(env.wr_margin[0])))
            if done.any():
                break
        rows.append(dict(seed=vseed, placed=bool(info["placed"][0]),
                         broke=bool(info["broke"][0]),
                         cleared=bool(info["cleared"][0]),
                         disturb_mm=float(info["disturb"][0]) * 1000,
                         steps=len(rec)))
        print(f"[eval {vseed}] placed={rows[-1]['placed']} "
              f"cleared={rows[-1]['cleared']} broke={rows[-1]['broke']} "
              f"disturb={rows[-1]['disturb_mm']:.1f}mm", flush=True)
        if a.render:
            clips.append((os.path.join(OUT, f"scene_{vseed}.xml"), rec))
    json.dump(rows, open(os.path.join(OUT, "eval.json"), "w"), indent=1)
    if a.render and clips:
        outp = os.path.join(OUT, "video_policy.mp4")
        vw = None
        for xml, rec in clips:
            m = mujoco.MjModel.from_xml_path(xml)
            d = mujoco.MjData(m)
            ren = mujoco.Renderer(m, height=480, width=640)
            for qp, sealed, margin in rec:
                d.qpos[:] = qp
                mujoco.mj_forward(m, d)
                ren.update_scene(d, camera="iso")
                im = cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR)
                col = (60, 220, 60) if sealed else (40, 40, 230)
                cv2.putText(im, f"sealed={int(sealed)} margin={margin:.2f}",
                            (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                if vw is None:
                    vw = cv2.VideoWriter(outp.replace(".mp4", "_raw.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         10, (im.shape[1], im.shape[0]))
                vw.write(im)
            ren.close()
        vw.release()
        os.system(f"ffmpeg -y -loglevel error -i {outp.replace('.mp4', '_raw.mp4')} "
                  f"-c:v libx264 -pix_fmt yuv420p {outp}")
        os.remove(outp.replace(".mp4", "_raw.mp4"))
        print(f"[eval] video -> {outp}")
    n = len(rows)
    print(f"[eval] placed {sum(r['placed'] for r in rows)}/{n}  "
          f"cleared {sum(r['cleared'] for r in rows)}/{n}  "
          f"broke {sum(r['broke'] for r in rows)}/{n}")


def smoke():
    wp.init()
    env = variant_env(0, 32, dr=True)
    obs = env.observe()
    assert obs.shape == (32, OBS_DIM), obs.shape
    print(f"[smoke] obs {tuple(obs.shape)} sealed {int(env.sealed.sum())}/32")
    for t in range(30):
        a = torch.zeros(32, ACT_DIM, device="cuda")
        a[:, 2] = -0.5          # crude: joint 3 up-ish
        a[:, 6] = 1.0           # keep suction
        obs, r, done, info = env.step(a)
        assert not torch.isnan(obs).any(), f"NaN at t{t}"
        if t % 6 == 0:
            op = env._obj_pos()
            print(f"t{t:02d} o0z {float(op[0, 2]):.3f} sealed "
                  f"{int(env.sealed.sum())}/32 margin "
                  f"{float(env.wr_margin.mean()):.2f} r {float(r.mean()):.3f} "
                  f"done {int(done.sum())}")
    print("[smoke] OK — no NaN over 30 decisions x 32 worlds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--bc", action="store_true")
    ap.add_argument("--bc_pairs", type=int, default=400_000)
    ap.add_argument("--bc_epochs", type=int, default=12)
    ap.add_argument("--dagger", default="",
                    help="existing ckpt in OUT: execute it on half the worlds "
                         "(on-policy states), teacher labels")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--ckpt", default="policy_best.pt")
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--dq_gain", type=float, default=1.0)
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--nworld", type=int, default=512)
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4_000_000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--kl", type=float, default=0.06)
    ap.add_argument("--ent", type=float, default=0.004)
    ap.add_argument("--std_max", type=float, default=-0.3,
                    help="upper clamp on log_std")
    ap.add_argument("--warm", default="")
    ap.add_argument("--time_pen", type=float, default=0.01)
    ap.add_argument("--speed_bonus", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--dr", action="store_true", default=True)
    a = ap.parse_args()
    if a.smoke:
        smoke()
    if a.bc:
        bc(a)
    if a.train:
        train(a)
    if a.eval:
        evaluate(a)


if __name__ == "__main__":
    main()
