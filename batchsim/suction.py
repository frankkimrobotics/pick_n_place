"""batchsim.suction :: vacuum cup for every world (simplified, vectorised).

Valve timing as in realsim.config: a seal can form `suction_onset` s after the valve opens and the object is held
`suction_release` s after it closes. A seal forms when the lip sits on the TOP FACE of an object: tip within `gap`
of the top, the whole 7.5 mm lip inside the face, cup within `tilt` of the face normal (upright objects only).
A sealed object is carried by a spring-damper to its pose in the cup frame, applied every physics step.
NOT modelled here (the scalar twin has them): rim conformance on curved tops, the sprung plunger, seal break force.
"""
import json
import os

import numpy as np
import torch

LIP_R = 0.0075


def quat_conj(q):
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device)


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def quat_rot(q, v):
    w, u = q[..., :1], q[..., 1:]
    return v + 2 * torch.cross(u, torch.cross(u, v, dim=-1) + w * v, dim=-1)


class BatchSuction:
    def __init__(self, world, cfg, meta=None, gap=0.004, tilt_deg=20.0, k=600.0, f_max=25.0):
        self.w, self.c = world, cfg
        N, dev, K = world.N, world.device, len(world.obj_bid)
        self.N, self.K, self.dev = N, K, dev
        self.bid = torch.tensor(world.obj_bid, device=dev)
        half = np.full((K, 3), 0.02)
        round_ = np.zeros(K, bool)
        if meta is not None:                                   # scene meta.json: half extents per object
            objs = {o["name"]: o for o in (json.load(open(meta)) if isinstance(meta, str) else meta)["objects"]}
            import mujoco
            for i, b in enumerate(world.obj_bid):
                o = objs.get(mujoco.mj_id2name(world.mjm, mujoco.mjtObj.mjOBJ_BODY, b))
                if o:
                    half[i] = o["half_extents"]
                    round_[i] = o.get("geom", "box") in ("cylinder", "sphere", "capsule")
        self.half = torch.tensor(half, dtype=torch.float32, device=dev)
        self.round = torch.tensor(round_, device=dev)
        self.mass = torch.tensor(world.mjm.body_mass[world.obj_bid], dtype=torch.float32, device=dev)
        self.gap, self.cos_tilt, self.k, self.f_max = gap, float(np.cos(np.radians(tilt_deg))), k, f_max
        self.valve = torch.zeros(N, dtype=torch.bool, device=dev)
        self.t_on = torch.full((N,), -1e9, device=dev)
        self.t_off = torch.full((N,), -1e9, device=dev)
        self.held = torch.full((N,), -1, dtype=torch.long, device=dev)      # object slot or -1
        self.anchor_p = torch.zeros(N, 3, device=dev)                       # object pose in the tcp frame at latch
        self.anchor_q = torch.zeros(N, 4, device=dev)
        self.ar = torch.arange(N, device=dev)
        self.tcp_bid = int(world.mjm.site_bodyid[world.sid_tcp])

    def reset(self, mask=None):
        m = torch.ones(self.N, dtype=torch.bool, device=self.dev) if mask is None else mask
        self.valve[m] = False
        self.t_on[m] = -1e9
        self.t_off[m] = -1e9
        self.held[m] = -1

    def set_valve(self, on, t, mask=None):
        m = torch.ones(self.N, dtype=torch.bool, device=self.dev) if mask is None else mask
        on = on.bool()
        opened, closed = m & on & ~self.valve, m & ~on & self.valve
        self.t_on[opened] = t
        self.t_off[closed] = t
        self.valve = torch.where(m, on, self.valve)

    def update(self, t):
        """Once per Pi tick: form / drop seals."""
        w = self.w
        tcp, R = w.tcp()
        ax = R[:, :, 2]                                                      # cup axis (points out of the lip)
        op = w.xpos[:, self.bid]                                             # [N, K, 3]
        oq = w.xquat[:, self.bid]
        up = quat_rot(oq, torch.tensor([0.0, 0.0, 1.0], device=self.dev).expand(self.N, self.K, 3))
        upright = up[..., 2] > 0.97
        top = op[..., 2] + self.half[None, :, 2]
        rel = tcp[:, None, :] - op
        loc = quat_rot(quat_conj(oq), rel)[..., :2].abs()                    # lip centre in the object's face frame
        on_face = torch.where(self.round[None], loc.norm(dim=-1) + LIP_R <= self.half[None, :, 0],
                              (loc[..., 0] + LIP_R <= self.half[None, :, 0]) & (loc[..., 1] + LIP_R <= self.half[None, :, 1]))
        g = tcp[:, None, 2] - top
        cand = upright & on_face & (g < self.gap) & (g > -0.02) & (ax[:, None, 2] < -self.cos_tilt)
        ready = self.valve & (t - self.t_on >= self.c["suction_onset"]) & (self.held < 0)
        pick = torch.where(cand, -g.abs(), torch.full_like(g, -1e9)).argmax(1)
        latch = ready & cand.any(1)
        if latch.any():
            a, s = self.ar[latch], pick[latch]
            tq = w.xquat[a, self.tcp_bid]
            self.anchor_p[a] = quat_rot(quat_conj(tq), op[a, s] - tcp[a])
            self.anchor_q[a] = quat_mul(quat_conj(tq), oq[a, s])
            self.held[a] = s
        drop = (self.held >= 0) & ~self.valve & (t - self.t_off >= self.c["suction_release"])
        lost = (self.held >= 0) & ((w.xpos[self.ar, self.bid[self.held.clamp(min=0)]] - tcp).norm(dim=-1) > 0.15)
        self.held[drop | lost] = -1
        return latch

    def apply(self):
        """Every physics step: carry the sealed object."""
        w = self.w
        w.xfrc[:, self.bid] = 0.0
        h = self.held >= 0
        if not h.any():
            return
        a, s = self.ar[h], self.held[h]
        b = self.bid[s]
        tcp = w.site_xpos[a, w.sid_tcp]
        tq = w.xquat[a, self.tcp_bid]
        tgt = tcp + quat_rot(tq, self.anchor_p[a])
        m = self.mass[s][:, None]
        va = w.obj_vadr_t[s]
        v = w.qvel[a[:, None], va[:, None] + torch.arange(6, device=self.dev)[None]]
        F = self.k * (tgt - w.xpos[a, b]) - 2.0 * torch.sqrt(self.k * m) * v[:, :3]
        F = F * (self.f_max / F.norm(dim=-1, keepdim=True).clamp(min=self.f_max))
        qe = quat_mul(quat_mul(tq, self.anchor_q[a]), quat_conj(w.xquat[a, b]))
        qe = qe * torch.sign(qe[:, :1])
        Tq = 1.0 * qe[:, 1:] - 0.02 * v[:, 3:]
        w.xfrc[a, b, :3] = F
        w.xfrc[a, b, 3:] = Tq
