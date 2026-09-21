#!/usr/bin/env python3
"""env_v3 :: ObstacleEnv -- env_v2.DiverseEnv + STATIC DISTRACTOR OBSTACLES the carry has to
avoid, on an enlarged workspace.

Everything is env_v2 (diverse objects, big table, walls, measured drive) except:

  (a) SPAWN enlarged to x in [0.20, 0.56], y in [-0.35, 0.35] (v2: x [0, 0.5], y [-0.3, 0.3]).
      The reach cache / rejection sampling of DiverseEnv are untouched, so an enlarged box
      simply re-solves the top-down-cup IK grid once and caches it under a new tag.
      GOALS are re-sampled here with v3 bounds (x [0.18, 0.52], y +- 0.33 -> >= 10 cm from
      every wall) and a MINIMUM carry distance `goal_min` (0.18 m), so that "the straight
      corridor between the object and the goal" is long enough to hold an obstacle that is
      >= 6 cm clear of both ends.

  (b) 1-3 DISTRACTOR bodies per world (`obstacles=True`; OFF by default so the class is a
      drop-in for DiverseEnv).  Each is a box or an upright cylinder; with probability
      `p_post` (0.5) one of them is a TALL POST 0.10-0.25 m high.  Placement is rejection
      sampling: on the table, clear of the base and the bin, >= 6 cm from the target and
      >= 6 cm from the goal xy, >= 4 cm from the other distractors, and -- in `p_corridor`
      (0.80) of the worlds -- at least one of them sits ON the straight target->goal
      corridor (perpendicular offset <= 6 cm, 22-78 % along the segment, clipped to the band
      that can also keep the size-aware end clearance).  The post takes that slot whenever
      there is one.  `p_corridor` is the INTENT: about 12 % of those worlds have no feasible
      corridor spot at all (the segment is short and the bodies are large), so 0.80 is what
      delivers the 70 % MEASURED occupancy the task specifies (measured 68.7 % at 1024
      worlds; p_corridor 0.70 delivered only 61.3 %).
      Like the target object in env_v2, the shape is selected per world from TWO geoms on
      ONE body (box + cylinder) by shrinking the inactive one to a 1 mm stub; sizes, mass,
      inertia and colour are written through the batched warp Model fields.  An INACTIVE
      slot is parked at (2.0 + 0.15 k, 2.0) -- on the collision plane but far outside the
      walls, so it can never touch anything.
      The bodies are FREE (a freejoint each): they stand still on their own, but they CAN
      be knocked over, which is what makes "displaced > 1 cm" a meaningful test.

  (c) FAILURE (`info["obst_hit"]`, mirrors env_v2's wall_hit exactly): any arm proxy / cup
      tip / target-object geom in contact with a distractor geom, OR an active distractor
      displaced > `disp_tol` (1 cm) from where it was placed.  Charged as the paper
      reward's -1 "fail" term once per episode, ends the episode, and clears `placed`.

  (d) OBSERVATION = env_v2's 47 + 7 = 54:
      [dx, dy, dz of the NEAREST active distractor's centre relative to the tcp,
       its half extents hx, hy, hz,  n_active / 3].  All zeros when there is none.

  (e) njmax / nconmax raised (3 free bodies more; env_v2's pattern, now a kwarg).

    $PY rl/env_v3.py --nworld 64 --steps 20 --obstacles 1
"""
import os
import sys

import numpy as np
import torch
import warp as wp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import build_scene_v2 as B  # noqa: E402
import env_v2 as V2  # noqa: E402
import env_warp as E  # noqa: E402
from env_v2 import DiverseEnv  # noqa: E402

SCENE_V3 = B.OUT_XML_V3

# ---- (a) enlarged workspace --------------------------------------------------
SPAWN_V3 = (0.20, 0.56, -0.35, 0.35)
GOAL_X_V3 = (0.18, 0.52)          # back wall x = -0.30 -> >= 0.48 m clear
GOAL_Y_V3 = (-0.33, 0.33)         # side walls y = +-0.50 -> >= 0.17 m clear
GOAL_MIN = 0.20                   # minimum |goal - object| xy (a corridor needs the room)
TARGET_MAX_V3 = 0.32              # maximum |goal - object| xy

# ---- (b) distractors ---------------------------------------------------------
N_DIST = B.N_DIST                 # bodies in the scene
# ">= 6 cm from the target / from the goal xy" (spec), but centre-to-centre 6 cm is NOT
# enough once the bodies have size: a 5 cm-radius distractor 6 cm from the goal leaves the
# CARRIED object (radius up to 4.5 cm) overlapping it, and cuRobo then rejects every carry
# plan (measured: 0/21 plans succeeded before this was size-aware).  The clearance is
# therefore max(6 cm, r_distractor + r_object + SIZE_PAD).
DIST_CLEAR_TGT = 0.06             # >= 6 cm from the target (centre to centre)
DIST_CLEAR_GOAL = 0.06            # >= 6 cm from the goal xy
SIZE_PAD = 0.03                   # surface clearance added on top of the two radii
DIST_CLEAR_EACH = 0.04            # >= 4 cm between distractors (plus their radii)
CORRIDOR_OFF = 0.06               # "on the corridor" = perpendicular offset <= 6 cm
CORRIDOR_T = (0.22, 0.78)         # fraction along the target->goal segment
DISP_TOL = 0.01                   # displaced > 1 cm = failure


class ObstacleEnv(DiverseEnv):
    """DiverseEnv + distractor obstacles + an enlarged workspace.  obs = 54."""

    def __init__(self, nworld=1024, device="cuda:0", seed=0, xml=None,
                 spawn=SPAWN_V3, obstacles=False, n_dist=(1, 3), p_post=0.5,
                 p_corridor=0.80, goal_min=GOAL_MIN, disp_tol=DISP_TOL,
                 njmax=256, nconmax=160, target_max=TARGET_MAX_V3, **kw):
        self._v3_ready = False
        self.obstacles = bool(obstacles)
        self.n_dist_range = (int(n_dist[0]), int(n_dist[1])) if not isinstance(n_dist, int) else (int(n_dist), int(n_dist))
        self.p_post = float(p_post)
        self.p_corridor = float(p_corridor)
        self.goal_min = float(goal_min)
        self.disp_tol = float(disp_tol)
        if xml is None:
            xml = SCENE_V3
            if not os.path.exists(xml):
                B.build(xml, n_dist=N_DIST)
        # tensors the observation needs before the parent's first reset()
        self.dist_half = torch.zeros(nworld, N_DIST, 3, device=device)
        self.dist_post = torch.zeros(nworld, N_DIST, dtype=torch.bool, device=device)
        self.dist_act = torch.zeros(nworld, N_DIST, dtype=torch.bool, device=device)
        self.dist_xy0 = torch.zeros(nworld, N_DIST, 3, device=device)
        self.n_act = torch.zeros(nworld, device=device)
        self.obst_hit_ep = torch.zeros(nworld, dtype=torch.bool, device=device)
        super().__init__(nworld=nworld, device=device, seed=seed, xml=xml, spawn=spawn,
                         njmax=njmax, nconmax=nconmax, target_max=target_max, **kw)
        # ---- distractor ids -------------------------------------------------
        import mujoco
        M, nid = self.mjm, mujoco.mj_name2id
        self.bid_dist, self.gid_dist, self.jadr_dist, self.vadr_dist = [], [], [], []
        for k in range(N_DIST):
            b = nid(M, mujoco.mjtObj.mjOBJ_BODY, B.dist_body_name(k))
            assert b >= 0, f"scene {xml} has no {B.dist_body_name(k)} (rebuild with --n_dist 3)"
            gb, gc = (nid(M, mujoco.mjtObj.mjOBJ_GEOM, n) for n in B.dist_geom_names(k))
            self.bid_dist.append(b)
            self.gid_dist.append((gb, gc))
            self.jadr_dist.append(int(M.jnt_qposadr[M.body_jntadr[b]]))
            self.vadr_dist.append(int(M.jnt_dofadr[M.body_jntadr[b]]))
        self.bid_dist_t = torch.tensor(self.bid_dist, device=device, dtype=torch.long)
        # xipos = the body's INERTIAL frame origin in world coords; body_ipos is (0, 0, hh),
        # so this is the distractor's geometric CENTRE.  A toppling post barely moves its
        # bottom-centre origin but swings its centre by hh*sin(tilt) -- so the displacement
        # test has to be on the centre, not on xpos.
        self.xipos = wp.to_torch(self.d.xipos)
        self._d_iw_ref = self.mw["body_invweight0"][0, self.bid_dist[0]].clone()
        self._d_dw_ref = self.mw["dof_invweight0"][0, self.vadr_dist[0]:self.vadr_dist[0] + 6].clone()
        self._d_m_ref = float(M.body_mass[self.bid_dist[0]])
        self._d_I_ref = float(np.mean(M.body_inertia[self.bid_dist[0]]))
        # contact masks: distractor geoms vs (arm proxies + cup tip + the target object)
        self.dist_mask = torch.zeros(M.ngeom + 1, dtype=torch.bool, device=device)
        # geom -> distractor SLOT, so a contact can be ignored when that slot is parked
        self.geom_slot = torch.full((M.ngeom + 1,), -1, dtype=torch.long, device=device)
        for k, (gb, gc) in enumerate(self.gid_dist):
            self.dist_mask[[gb, gc]] = True
            self.geom_slot[[gb, gc]] = k
        self.hitter_mask = self.robot_mask.clone()          # proxies + cup_tip (from env_v2)
        self.hitter_mask[self.gid_shape] = True             # + the (held) target object
        self._v3_ready = True
        self.reset(torch.ones(nworld, dtype=torch.bool, device=device))

    # ================= sampling =================
    def _sample_dist_specs(self, n):
        """Per world: n_active, and per slot shape / half extents / mass / post flag."""
        rng = self.rng
        n_act = rng.integers(self.n_dist_range[0], self.n_dist_range[1] + 1, size=n)
        act = np.arange(N_DIST)[None, :] < n_act[:, None]                 # (n, N_DIST)
        is_post = np.zeros((n, N_DIST), bool)
        # with probability p_post exactly ONE active slot (slot 0, which is also the
        # corridor slot) is a tall post
        is_post[:, 0] = (rng.random(n) < self.p_post) & act[:, 0]
        shape = rng.integers(0, 2, size=(n, N_DIST))                      # 0 box, 1 cylinder
        hw = np.where(is_post, rng.uniform(*B.POST_HW, size=(n, N_DIST)),
                      rng.uniform(*B.DIST_BOX_HW, size=(n, N_DIST)))
        hw2 = np.where(is_post, hw, rng.uniform(*B.DIST_BOX_HW, size=(n, N_DIST)))
        hw2 = np.where(shape == 1, hw, hw2)                               # cylinder: hy = hx = r
        hh = np.where(is_post, rng.uniform(*B.POST_HH, size=(n, N_DIST)),
                      rng.uniform(*B.DIST_BOX_HH, size=(n, N_DIST)))
        mass = rng.uniform(*B.DIST_MASS, size=(n, N_DIST))
        rgba = np.concatenate([rng.uniform(0.10, 0.60, size=(n, N_DIST, 3)),
                               np.ones((n, N_DIST, 1))], -1)
        rad = np.hypot(hw, hw2)                                           # circumradius
        return dict(act=act, n_act=n_act, is_post=is_post, shape=shape,
                    hw=hw, hw2=hw2, hh=hh, mass=mass, rgba=rgba, rad=rad)

    def _place_distractors(self, spec, obj_xy, obj_rad, goal_xy, tries=40):
        """(n, N_DIST, 2) distractor centres by rejection sampling.  Slot 0 is the CORRIDOR
        slot in a `p_corridor` fraction of the worlds; the rest are free on the table."""
        rng, n = self.rng, len(obj_xy)
        act, rad = spec["act"], spec["rad"]
        xy = np.tile(np.array([[B.DIST_PARK[0], B.DIST_PARK[1]]]), (n, N_DIST, 1))
        xy[:, :, 0] += 0.15 * np.arange(N_DIST)[None, :]
        want_corr = (rng.random(n) < self.p_corridor) & act[:, 0]
        seg = goal_xy - obj_xy
        L = np.linalg.norm(seg, axis=-1).clip(1e-6)
        u = seg / L[:, None]
        perp = np.stack([-u[:, 1], u[:, 0]], -1)
        n_corr = 0
        for k in range(N_DIST):
            live = act[:, k]
            if not live.any():
                continue
            r = rad[:, k]
            if k == 0:
                # corridor candidates first, free candidates as the tail (fallback).
                # The fraction along the segment is clipped to the band that can satisfy the
                # size-aware end clearance: a fixed U(0.3, 0.7) put most candidates inside the
                # target's or the goal's keep-out and dropped corridor occupancy to 17 %.
                pad_l = (np.maximum(DIST_CLEAR_TGT, r + obj_rad + SIZE_PAD) + 0.005) / L
                lo = np.clip(np.maximum(CORRIDOR_T[0], pad_l), 0.05, 0.5)
                hi = np.clip(np.minimum(CORRIDOR_T[1], 1.0 - pad_l), 0.5, 0.95)
                lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
                nc = 3 * tries                      # the corridor slot gets more attempts
                t = lo[:, None] + (hi - lo)[:, None] * rng.random((n, nc))
                o = rng.uniform(-CORRIDOR_OFF, CORRIDOR_OFF, size=(n, nc))
                cx = obj_xy[:, 0:1] + t * seg[:, 0:1] + o * perp[:, 0:1]
                cy = obj_xy[:, 1:2] + t * seg[:, 1:2] + o * perp[:, 1:2]
                fx = rng.uniform(B.TABLE_X2[0], B.TABLE_X2[1], size=(n, tries))
                fy = rng.uniform(B.TABLE_Y2[0], B.TABLE_Y2[1], size=(n, tries))
                cand_x = np.concatenate([np.where(want_corr[:, None], cx, np.tile(fx, (1, 3))), fx], 1)
                cand_y = np.concatenate([np.where(want_corr[:, None], cy, np.tile(fy, (1, 3))), fy], 1)
            else:
                cand_x = rng.uniform(B.TABLE_X2[0], B.TABLE_X2[1], size=(n, tries))
                cand_y = rng.uniform(B.TABLE_Y2[0], B.TABLE_Y2[1], size=(n, tries))
            ok = self._static_ok(cand_x, cand_y, r[:, None], in_box=False)
            d_obj = np.hypot(cand_x - obj_xy[:, 0:1], cand_y - obj_xy[:, 1:2])
            d_goal = np.hypot(cand_x - goal_xy[:, 0:1], cand_y - goal_xy[:, 1:2])
            pad = r[:, None] + obj_rad[:, None] + SIZE_PAD
            ok &= d_obj >= np.maximum(DIST_CLEAR_TGT, pad)
            ok &= d_goal >= np.maximum(DIST_CLEAR_GOAL, pad)
            for j in range(k):
                dj = np.hypot(cand_x - xy[:, j, 0:1], cand_y - xy[:, j, 1:2])
                ok &= ~act[:, j][:, None] | (dj >= DIST_CLEAR_EACH + r[:, None] + rad[:, j][:, None])
            first = np.argmax(ok, 1)
            got = ok[np.arange(n), first]
            sel = live & got
            xy[sel, k, 0] = cand_x[np.arange(n), first][sel]
            xy[sel, k, 1] = cand_y[np.arange(n), first][sel]
            spec["act"][live & ~got, k] = False              # no legal spot -> park this slot
            if k == 0:
                n_corr = int((sel & want_corr & (first < 3 * tries)).sum())
        spec["n_act"] = spec["act"].sum(1)
        self._last_corr_frac = n_corr / max(1, n)
        return xy

    # ================= model writes =================
    def _write_distractors(self, idx, spec, xy):
        dev, f32 = self.device, torch.float32
        t = lambda v, d=f32: torch.as_tensor(np.ascontiguousarray(v), device=dev, dtype=d)  # noqa: E731
        act = spec["act"]
        stub = B.STUB
        for k in range(N_DIST):
            gb, gc = self.gid_dist[k]
            a = act[:, k]
            hw = np.where(a, spec["hw"][:, k], stub)
            hw2 = np.where(a, spec["hw2"][:, k], stub)
            hh = np.where(a, spec["hh"][:, k], stub)
            box = a & (spec["shape"][:, k] == 0)
            cyl = a & (spec["shape"][:, k] == 1)
            hzt = t(hh)
            for g in (gb, gc):                                # both to stub at the centroid
                self.mw["geom_size"][idx, g] = stub
                self.mw["geom_rbound"][idx, g] = float(np.sqrt(3) * stub)
                self.mw["geom_aabb"][idx, g, 0] = 0.0
                self.mw["geom_aabb"][idx, g, 1] = stub
                self.mw["geom_pos"][idx, g, 0] = 0.0
                self.mw["geom_pos"][idx, g, 1] = 0.0
                self.mw["geom_pos"][idx, g, 2] = hzt
                self.mw["geom_rgba"][idx, g] = t(spec["rgba"][:, k])
            for g, m, size in ((gb, box, np.stack([hw, hw2, hh], -1)),
                               (gc, cyl, np.stack([hw, hh, np.zeros_like(hh)], -1))):
                sel = np.nonzero(m)[0]
                if not len(sel):
                    continue
                ii = idx[sel]
                self.mw["geom_size"][ii, g] = t(size[sel])
                ext = np.stack([hw[sel], hw2[sel], hh[sel]], -1)
                self.mw["geom_rbound"][ii, g] = t(np.linalg.norm(ext, axis=-1))
                self.mw["geom_aabb"][ii, g, 0] = 0.0
                self.mw["geom_aabb"][ii, g, 1] = t(ext)
            # inertial: origin at the bottom face, centroid at +hh
            mass = np.where(a, spec["mass"][:, k], self._d_m_ref)
            I = np.stack([mass / 12 * ((2 * hw2) ** 2 + (2 * hh) ** 2),
                          mass / 12 * ((2 * hw) ** 2 + (2 * hh) ** 2),
                          mass / 12 * ((2 * hw) ** 2 + (2 * hw2) ** 2)], -1)
            b = self.bid_dist[k]
            self.mw["body_mass"][idx, b] = t(mass)
            self.mw["body_subtreemass"][idx, b] = t(mass)
            self.mw["body_ipos"][idx, b, 0] = 0.0
            self.mw["body_ipos"][idx, b, 1] = 0.0
            self.mw["body_ipos"][idx, b, 2] = hzt
            self.mw["body_inertia"][idx, b] = t(I)
            sm = t(self._d_m_ref / mass)
            si = t(self._d_I_ref / np.mean(I, axis=-1).clip(1e-9))
            self.mw["body_invweight0"][idx, b, 0] = self._d_iw_ref[0] * sm
            self.mw["body_invweight0"][idx, b, 1] = self._d_iw_ref[1] * si
            va = self.vadr_dist[k]
            for c in range(3):
                self.mw["dof_invweight0"][idx, va + c] = self._d_dw_ref[c] * sm
                self.mw["dof_invweight0"][idx, va + 3 + c] = self._d_dw_ref[3 + c] * si
            # pose
            qa = self.jadr_dist[k]
            yaw = self.rng.uniform(0, np.pi, size=len(xy))
            self.qpos[idx, qa:qa + 2] = t(xy[:, k])
            self.qpos[idx, qa + 2] = 0.0
            self.qpos[idx, qa + 3] = t(np.cos(yaw / 2))
            self.qpos[idx, qa + 4:qa + 6] = 0.0
            self.qpos[idx, qa + 6] = t(np.sin(yaw / 2))
            self.qvel[idx, self.vadr_dist[k]:self.vadr_dist[k] + 6] = 0.0
            self.xfrc[idx, b] = 0.0
            self.dist_half[idx, k] = t(np.stack([hw, hw2, hh], -1))
            self.dist_act[idx, k] = t(a, torch.bool)
            self.dist_post[idx, k] = t(a & spec["is_post"][:, k], torch.bool)

    # ================= reset =================
    def reset(self, mask):
        if not self._v3_ready:
            return super().reset(mask)
        super().reset(mask)                      # DiverseEnv: object variant, spawn, v2 goal
        idx_t = torch.nonzero(mask).squeeze(-1)
        if idx_t.numel() == 0:
            return
        idx = idx_t.cpu().numpy()
        n = len(idx)
        qa = self.jadr_obj
        obj_xy = self.qpos[idx_t, qa:qa + 2].cpu().numpy().astype(np.float64)
        obj_rad = np.hypot(self.half_w[idx_t, 0].cpu().numpy(), self.half_w[idx_t, 1].cpu().numpy())
        # ---- (a) v3 goal: own bounds + a minimum carry distance ---------------
        goal_xy = self._sample_goal(obj_xy)
        gz = np.clip(self.rng.uniform(*self.paper["goal_z"], size=n), *V2.GOAL_Z)
        g = torch.as_tensor(goal_xy, device=self.device, dtype=torch.float32)
        self._assign_target(idx_t, g)
        self.goal[idx_t, :2] = g
        self.goal[idx_t, 2] = torch.as_tensor(gz, device=self.device, dtype=torch.float32)
        # ---- (b) distractors --------------------------------------------------
        spec = self._sample_dist_specs(n)
        if not self.obstacles:
            spec["act"][:] = False
            spec["n_act"][:] = 0
            self._last_corr_frac = 0.0
        xy = self._place_distractors(spec, obj_xy, obj_rad, goal_xy)
        self._write_distractors(idx_t, spec, xy)
        self.n_act[idx_t] = torch.as_tensor(spec["n_act"], device=self.device, dtype=torch.float32)
        self.obst_hit_ep[idx_t] = False
        for t_ in self._solver_state:
            t_[idx_t] = 0.0
        self.qpos[idx_t] = torch.nan_to_num(self.qpos[idx_t])
        self.qvel[idx_t] = 0.0
        E.mjw.forward(self.m, self.d)
        self.dist_xy0[idx_t] = self.xipos[idx_t][:, self.bid_dist_t]
        tcp, _ = self._tcp()
        self.phi_approach[idx_t] = -torch.norm(tcp[idx_t] - self._grasp_point()[idx_t], dim=-1)

    def _sample_goal(self, obj_xy, tries=32):
        """(n, 2) goal xy: inside GOAL_X_V3 x GOAL_Y_V3, |goal - obj| in [goal_min, target_max]."""
        n = len(obj_xy)
        ang = self.rng.uniform(0, 2 * np.pi, size=(n, tries))
        rr = self.rng.uniform(self.goal_min, max(self.goal_min + 0.02, self.target_max), size=(n, tries))
        gx = obj_xy[:, 0:1] + rr * np.cos(ang)
        gy = obj_xy[:, 1:2] + rr * np.sin(ang)
        ok = ((gx >= GOAL_X_V3[0]) & (gx <= GOAL_X_V3[1])
              & (gy >= GOAL_Y_V3[0]) & (gy <= GOAL_Y_V3[1]))
        first = np.argmax(ok, 1)
        ar = np.arange(n)
        out = np.stack([gx[ar, first], gy[ar, first]], -1)
        bad = ~ok[ar, first]
        if bad.any():                                    # aim back at the table centre
            c = np.array([0.5 * (GOAL_X_V3[0] + GOAL_X_V3[1]), 0.0])
            d = c - obj_xy[bad]
            d = d / np.linalg.norm(d, axis=-1, keepdims=True).clip(1e-6)
            out[bad] = obj_xy[bad] + d * self.goal_min
            out[bad, 0] = np.clip(out[bad, 0], *GOAL_X_V3)
            out[bad, 1] = np.clip(out[bad, 1], *GOAL_Y_V3)
        self._last_goal_reject = float(bad.mean())
        return out

    # ================= observation =================
    def observe(self):
        obs = super().observe()                                  # 47
        tcp, _ = self._tcp()
        c = self.xpos[:, self.bid_dist_t].clone()                # (N, K, 3) bottom centre
        c[:, :, 2] = c[:, :, 2] + self.dist_half[:, :, 2]        # -> geometric centre
        rel = c - tcp[:, None, :]
        d = torch.norm(rel, dim=-1)
        d = torch.where(self.dist_act, d, torch.full_like(d, 1e6))
        j = torch.argmin(d, dim=1)
        ar = torch.arange(self.nworld, device=self.device)
        any_act = self.dist_act.any(1, keepdim=True).float()
        near = torch.cat([rel[ar, j], self.dist_half[ar, j]], -1) * any_act
        obs = torch.cat([obs, near, (self.n_act / N_DIST)[:, None]], -1)
        return torch.nan_to_num(obs, nan=0.0, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)

    # ================= obstacle failure =================
    def _dist_contacts(self):
        """(N,) bool: an arm proxy / cup tip / the target object touches a distractor."""
        con = self.d.contact
        g = wp.to_torch(con.geom).long()
        w = wp.to_torch(con.worldid).long()
        dist = wp.to_torch(con.dist)
        if self._con_ar is None or self._con_ar.numel() != g.shape[0]:
            self._con_ar = torch.arange(g.shape[0], device=self.device)
        nac = wp.to_torch(self.d.nacon).reshape(-1)[:1].long()
        valid = self._con_ar < nac
        ng = self.mjm.ngeom
        g0, g1 = g[:, 0].clamp(0, ng), g[:, 1].clamp(0, ng)
        a0 = self.dist_mask[g0] & self.hitter_mask[g1]
        a1 = self.dist_mask[g1] & self.hitter_mask[g0]
        slot = torch.where(a0, self.geom_slot[g0], self.geom_slot[g1]).clamp(0, N_DIST - 1)
        wc = w.clamp(0, self.nworld - 1)
        hit = valid & (a0 | a1) & (dist < 0.0) & self.dist_act[wc, slot]
        out = torch.zeros(self.nworld, device=self.device)
        out.index_add_(0, w.clamp(0, self.nworld - 1), hit.float())
        return out > 0

    def _dist_moved(self):
        """(N,) bool: an ACTIVE distractor's centre has moved more than disp_tol."""
        d = torch.norm(self.xipos[:, self.bid_dist_t] - self.dist_xy0, dim=-1)
        return ((d > self.disp_tol) & self.dist_act).any(1)

    # ================= reward =================
    def reward(self, want, latched_now, released, broke, tcp_before, obj_before, a):
        r, done, info = super().reward(want, latched_now, released, broke, tcp_before, obj_before, a)
        if not self.obstacles:
            info["obst_hit"] = torch.zeros(self.nworld, dtype=torch.bool, device=self.device)
            info["n_dist"] = self.n_act.clone()
            return r, done, info
        hit_now = self._dist_contacts() | self._dist_moved()
        new = hit_now & ~self.obst_hit_ep & ~info["off"] & ~info["wall_hit"]
        self.obst_hit_ep |= hit_now
        if new.any():
            r = r - new.float()                      # same -1 as the paper's off-table fail
            self.ep_comp_p[:, self.RKEYS_PAPER.index("fail")] -= new.float()
            info["ep_comp"] = self.ep_comp_p.clone()
        done = done | self.obst_hit_ep
        info["placed"] = info["placed"] & ~self.obst_hit_ep
        info["obst_hit"] = self.obst_hit_ep.clone()
        info["obst_hit_now"] = hit_now
        info["n_dist"] = self.n_act.clone()
        return r, done, info

    # ================= divergence guard =================
    def _diverged(self):
        bad = super()._diverged()
        bad |= self.xpos[:, self.bid_dist_t].abs().amax(-1).amax(-1) > self.DIVERGE_POS
        return bad

    def step(self, action):
        obs, r, done, info = super().step(action)
        done = done | self.obst_hit_ep
        info["placed"] = info["placed"] & ~self.obst_hit_ep
        return obs, r, done, info


if __name__ == "__main__":
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--drive", default="real", choices=["real", "ideal"])
    ap.add_argument("--obstacles", type=int, default=1)
    args = ap.parse_args()
    wp.init()
    env = ObstacleEnv(nworld=args.nworld, drive=args.drive, dr=True, ep_len=150,
                      obstacles=bool(args.obstacles))
    obs = env.observe()
    print(f"obs dim {obs.shape[-1]} | nworld {args.nworld} | substeps {env.substeps} | "
          f"corridor {100 * env._last_corr_frac:.0f} % | mean n_dist {float(env.n_act.mean()):.2f}")
    t0 = time.time()
    for _ in range(args.steps):
        a = torch.rand(args.nworld, 7, device=env.device) * 2 - 1
        obs, r, done, info = env.step(a)
    dt = time.time() - t0
    print(f"{args.steps} x {args.nworld} in {dt:.2f}s = {args.nworld * args.steps / dt:,.0f} env-steps/s "
          f"| reward {r.mean():.3f} | wall_hit {int(info['wall_hit'].sum())} "
          f"| obst_hit {int(info['obst_hit'].sum())}")
