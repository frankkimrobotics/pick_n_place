"""batchsim.sensors :: what the host receives, when it receives it.

`StateHistory` keeps the last ~0.6 s of full simulator state per Pi tick. The sensors read the PAST from it:
  JointFeedback  posfb as the desktop sees it: sample-and-hold at fb_period, fb_lat old
  BatchCamera    the frame captured at the last frame boundary before (now - latency) -- state interpolated between
                 Pi ticks, so 60 fps ages are continuous -- rendered for all worlds at once with the mujoco_warp ray-traced renderer from each world's own delayed state
Per-world latencies and camera phases come from latency.sample().
"""
import torch
import warp as wp
import mujoco_warp as mw

from .latency import RingHistory

NJ = 6


class StateHistory:
    def __init__(self, world, cmd_dt, hist_s=0.6):
        self.w, self.cmd_dt = world, cmd_dt
        self.L = int(round(hist_s / cmd_dt)) + 2
        self.qpos = RingHistory(world.N, world.qpos.shape[1], self.L, world.device)
        self.qvel = RingHistory(world.N, world.qvel.shape[1], self.L, world.device)
        m = world.mjm                                                     # quaternion slots in qpos (free: after xyz; ball)
        adr = [int(m.jnt_qposadr[j]) + (3 if m.jnt_type[j] == 0 else 0) for j in range(m.njnt) if m.jnt_type[j] in (0, 1)]
        self.quat_idx = torch.tensor([[a + i for i in range(4)] for a in adr], dtype=torch.long, device=world.device).reshape(-1, 4)
        self.reset()

    def reset(self, mask=None):
        self.qpos.fill(self.w.qpos, mask)
        self.qvel.fill(self.w.qvel, mask)

    def record(self):
        self.qpos.push(self.w.qpos)
        self.qvel.push(self.w.qvel)

    def ticks_ago(self, age):
        """Index of the newest recorded tick that is at least `age` seconds old ([N] seconds -> [N] ticks)."""
        return torch.ceil(age / self.cmd_dt - 1e-4).long().clamp(0, self.L - 1)

    def at_age(self, age):
        """(qpos, qvel, age) `age` [N] seconds in the past, linearly interpolated between the two recorded ticks around
        it (free-joint quaternions re-normalised): 60 fps capture times fall between the Pi ticks. Older than the
        history: the oldest tick, with its true age."""
        u = (age / self.cmd_dt).clamp(0.0, self.L - 1.0)
        k0 = torch.floor(u + 1e-4).long().clamp(max=self.L - 2)
        w = (u - k0).clamp(0.0, 1.0)[:, None]
        qp = (1 - w) * self.qpos.at(k0) + w * self.qpos.at(k0 + 1)
        qv = (1 - w) * self.qvel.at(k0) + w * self.qvel.at(k0 + 1)
        if self.quat_idx.numel():
            qq = qp[:, self.quat_idx]                                     # [N, nq, 4]
            qp[:, self.quat_idx] = qq / qq.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return qp, qv, u * self.cmd_dt


class JointFeedback:
    def __init__(self, hist, cfg, lat):
        self.h, self.c, self.lat = hist, cfg, lat

    def read(self, t):
        """(q [N, 6], qd [N, 6], age [N] s) at sim time t."""
        t_s = t - self.lat["fb_lat"]
        if self.c["fb_period"] > 0:
            t_s = torch.floor(t_s / self.c["fb_period"] + 1e-4) * self.c["fb_period"]
        qp, qv, age = self.h.at_age(t - t_s)
        return qp[:, :NJ], qv[:, :NJ], age


class BatchCamera:
    """One named camera rendered for every world. `grab(t)` -> uint8 [N, H, W, 3] on the device, plus the frame age."""

    def __init__(self, world, hist, cam, lat, period, phase=None, res=(432, 240), geom_groups=(0, 1, 2, 3), noise=0.0,
                 shadows=False, seed=0):
        self.w, self.h, self.cam = world, hist, cam
        self.lat, self.period = lat, float(period)
        self.phase = torch.zeros(world.N, device=world.device) if phase is None else phase * self.period
        self.W, self.H = res
        self.noise = noise
        self.g = torch.Generator(device=world.device).manual_seed(seed)
        with wp.ScopedDevice(world.device):
            self.d = mw.put_data(world.mjm, world.mjd, nworld=world.N, nconmax=world.nconmax, njmax=world.njmax)   # scratch state
            self.rc = mw.create_render_context(world.mjm, nworld=world.N, cam_res=(self.W, self.H), render_rgb=True, render_depth=False,
                                               enabled_geom_groups=list(geom_groups), cam_active=[cam], use_shadows=shadows)
            self.out = wp.zeros((world.N, self.H, self.W), dtype=wp.vec3f)
        self.qpos = wp.to_torch(self.d.qpos)
        self.rgb = wp.to_torch(self.out)
        self.age = torch.zeros(world.N, device=world.device)

    def grab(self, t):
        t_cap = t - self.lat
        if self.period > 0:
            t_cap = torch.floor((t_cap - self.phase) / self.period + 1e-4) * self.period + self.phase
        qp, _, self.age = self.h.at_age(t - t_cap)                        # capture between Pi ticks: interpolated
        self.qpos[:] = qp
        mw.kinematics(self.w.m, self.d)
        mw.camlight(self.w.m, self.d)
        mw.refit_bvh(self.w.m, self.d, self.rc)
        mw.render(self.w.m, self.d, self.rc)
        mw.get_rgb(self.rc, 0, self.out)
        img = self.rgb.clamp(0.0, 1.0) * 255.0
        if self.noise > 0:
            img = (img + self.noise * torch.randn(img.shape, generator=self.g, device=img.device)).clamp(0, 255)
        return img.to(torch.uint8)
