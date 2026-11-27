"""batchsim.twin :: BatchTwin = N digital twins of the Pro 630 cell stepping together on one GPU.

    policy / RL agent
        | observe(): joint feedback and camera frames AS THE HOST RECEIVES THEM (each with its own age)
        | command(): a joint control polygon + valve state; reaches the Pi after compute + transport delay
        v
    DriveChain (Pi stream law, dead time, accel limit, inner loop, joint friction)  ->  mujoco_warp physics
        ^ StateHistory (0.6 s of state per tick) feeds the delayed sensors and the batched renderer

Time advances only in advance(n): n Pi periods of cfg['cmd_dt'] (10 ms). All latencies are per world; dr=True
draws them from realsim.config.DR. Configuration is realsim.config (level 'real' | 'ideal' | ..., joint_model=1 for
the torque-fitted joint physics), so the scalar twin and this one cannot drift apart.
"""
import os

import numpy as np
import torch

from realsim.config import make_cfg, STREAMS
from . import latency
from .drive import DriveChain
from .sensors import BatchCamera, JointFeedback, StateHistory
from .suction import BatchSuction
from .world import BatchWorld, NJ


class BatchTwin:
    def __init__(self, xml, nworld=256, device="cuda:0", level="real", dr=False, stream="tuned", cameras=(), cam_res=(432, 240),
                 seed=0, max_pts=8, nconmax=None, njmax=None, **over):
        if stream in STREAMS and level != "ideal":             # "tuned" | "fast" | "fast5" (5 ms stack: timing="stack_5ms")
            over = {**STREAMS[stream], **over}
        self.cfg = cfg = make_cfg(level, **over)
        self.N, self.dev = nworld, device
        A = lambda v: np.full(NJ, v) if np.isscalar(v) else np.asarray(v, float)
        phys = dict(armature=A(cfg["armature"]), damping=A(cfg["damping"]), frictionloss=A(cfg["frictionloss"]))
        self.world = w = BatchWorld(os.path.expanduser(xml), nworld, device, joint_physics=phys, nconmax=nconmax, njmax=njmax,
                                    cmd_dt=cfg["cmd_dt"])
        w.obj_vadr_t = torch.tensor(w.obj_vadr, device=device)
        self.lat = lat = latency.sample(cfg, nworld, device, dr=dr, gen=torch.Generator().manual_seed(seed))
        if dr:                                                    # per-world reflected inertia spread
            w.set_joint_physics(armature=torch.tensor(phys["armature"], dtype=torch.float32, device=device)[None] * lat["phys_scale"][:, None])
        self.drive = DriveChain(w, cfg, lat, max_pts=max_pts)
        self.hist = StateHistory(w, cfg["cmd_dt"])
        self.fb = JointFeedback(self.hist, cfg, lat)
        self.clock = latency.PolicyClock(cfg, lat, nworld, device, seed=seed + 1)
        meta = os.path.join(os.path.dirname(os.path.expanduser(xml)), "meta.json")
        self.suction = BatchSuction(w, cfg, meta if os.path.exists(meta) else None)
        cam_cfg = dict(zed=("zed_lat", cfg["zed_period"]), wrist_d405=("wrist_lat", cfg["wrist_period"]))
        self.cams = {}
        for c in cameras:
            key, period = cam_cfg.get(c, ("zed_lat", cfg["zed_period"]))
            self.cams[c] = BatchCamera(w, self.hist, c, lat[key], period, phase=lat["cam_phase"], res=cam_res, noise=cfg["cam_noise"], seed=seed + 2)
        self.pend_valve = torch.zeros(nworld, dtype=torch.bool, device=device)
        self.pend_t = torch.full((nworld,), float("inf"), device=device)
        self.last_latency = torch.zeros(nworld, device=device)

    @property
    def t(self):
        return self.drive.t

    # ---------------------------------------------------------------- episode control
    def reset(self, mask=None, qpos=None):
        self.world.reset(mask, qpos)
        self.drive.reset(mask)
        self.hist.reset(mask)
        self.suction.reset(mask)
        m = torch.ones(self.N, dtype=torch.bool, device=self.dev) if mask is None else mask
        self.pend_t[m] = float("inf")

    # ---------------------------------------------------------------- host side
    def observe(self, images=True):
        t = self.t
        q, qd, age = self.fb.read(t)
        obs = dict(t=t, q=q, qd=qd, q_age=age, held=self.suction.held.clone(), img={}, img_age={})
        if images:
            for name, cam in self.cams.items():
                obs["img"][name] = cam.grab(t)
                obs["img_age"][name] = cam.age.clone()
        return obs

    def command(self, pts, knot_dt, t_anchor=None, valve=None, mask=None, compute=None):
        """Send what the policy decided from the observation taken NOW. pts [N, p, 6] joint control polygon (URDF rad),
        knot 0 at t_anchor (default: now), knots knot_dt apart. It reaches the Pi after this decision's compute time
        (sampled per world, or `compute` [N] s if the caller measured its own) plus transport; the valve switches when
        the compute is done. Returns the total command latency [N]."""
        comp, tx = self.clock.delay()
        if compute is not None:
            comp = torch.as_tensor(compute, dtype=torch.float32, device=self.dev).expand(self.N) + self.cfg["host_proc"]
        t = self.t
        self.drive.send(pts, t if t_anchor is None else t_anchor, knot_dt, t + comp + tx, mask=mask)
        if valve is not None:
            m = torch.ones(self.N, dtype=torch.bool, device=self.dev) if mask is None else mask
            self.pend_valve = torch.where(m, valve.bool(), self.pend_valve)
            self.pend_t = torch.where(m, t + comp, self.pend_t)
        self.last_latency = comp + tx
        return self.last_latency

    # ---------------------------------------------------------------- time
    def advance(self, n=1):
        for _ in range(n):
            due = self.pend_t <= self.t + 1e-6
            if due.any():
                self.suction.set_valve(self.pend_valve, self.t, due)
                self.pend_t[due] = float("inf")
            self.drive.tick(on_substep=self.suction.apply)
            self.suction.update(self.t)
            self.hist.record()

    def healthy(self):
        return self.world.healthy()
