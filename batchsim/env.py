"""batchsim.env :: a vectorised RL task on BatchTwin (state observations): seal one object and lift it.

The agent acts at 10 Hz through the SAME path as the deployed executor: it moves a joint target, the last targets form
a cubic B-spline control polygon, and that polygon reaches the Pi after the (sampled) inference + transport delay.
What it observes is what the host would have: joint feedback one sample old, the object position as a camera would
report it (ZED latency), its own previous action. Nothing in the observation is newer than the real system allows.

    env = PickLiftEnv(xml, nworld=1024, dr=True)
    obs = env.reset(); obs, rew, done, info = env.step(action)      # action [N, 7] in [-1, 1]: 6 joint steps + valve
"""
import numpy as np
import torch

from .twin import BatchTwin
from .world import NJ

HOME = [0.0, -0.3491, 1.3963, 0.1745, -1.5708, 0.0]


class PickLiftEnv:
    obs_dim, act_dim = 6 + 6 + 6 + 7 + 3 + 3 + 3, 7

    def __init__(self, xml, nworld=1024, device="cuda:0", dr=True, decision_dt=0.1, horizon=80, v_max_deg=30.0, seed=0, **twin_kw):
        self.tw = BatchTwin(xml, nworld, device, dr=dr, seed=seed, **twin_kw)
        self.N, self.dev = nworld, device
        self.n_tick = int(round(decision_dt / self.tw.cfg["cmd_dt"]))
        self.dt, self.horizon = decision_dt, horizon
        self.dq = float(np.radians(v_max_deg) * decision_dt)
        w = self.tw.world
        self.jlo = torch.tensor(w.mjm.jnt_range[:NJ, 0], dtype=torch.float32, device=device)
        self.jhi = torch.tensor(w.mjm.jnt_range[:NJ, 1], dtype=torch.float32, device=device)
        self.home = torch.tensor(HOME, dtype=torch.float32, device=device)
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        self.K = len(w.obj_bid)
        self.obj_qadr = torch.tensor(w.obj_qadr, device=device)
        self.half = self.tw.suction.half
        self.tgt = torch.zeros(nworld, dtype=torch.long, device=device)
        self.step_i = torch.zeros(nworld, dtype=torch.long, device=device)
        self.targets = torch.zeros(nworld, 4, NJ, device=device)             # last four joint targets (control points)
        self.prev_a = torch.zeros(nworld, 7, device=device)
        self.z0 = torch.zeros(nworld, device=device)
        self.ar = torch.arange(nworld, device=device)
        self.ep_ret = torch.zeros(nworld, device=device)
        self.reset()

    # ---------------------------------------------------------------- helpers
    def _obj_pos_true(self):
        return self.tw.world.xpos[self.ar, self.tw.suction.bid[self.tgt]]

    def _obj_pos_seen(self):
        """Target object position as the fixed camera would report it: zed_lat old, frame-quantised."""
        cfg, lat, h = self.tw.cfg, self.tw.lat, self.tw.hist
        t_cap = self.tw.t - lat["zed_lat"]
        t_cap = torch.floor(t_cap / cfg["zed_period"] + 1e-4) * cfg["zed_period"]
        qp = h.at_age(self.tw.t - t_cap)[0]
        a = self.obj_qadr[self.tgt]
        return qp[self.ar[:, None], a[:, None] + torch.arange(3, device=self.dev)[None]]

    def _reset_worlds(self, m):
        if not m.any():
            return
        n = int(m.sum())
        qp = self.tw.world.qpos0[None].expand(self.N, -1).clone()
        qp[:, :NJ] = self.home + 0.05 * (torch.rand(self.N, NJ, generator=self.g).to(self.dev) - 0.5)
        self.tw.reset(m, qp)
        self.tgt[m] = torch.randint(0, self.K, (n,), generator=self.g).to(self.dev)
        self.step_i[m] = 0
        self.targets[m] = self.tw.world.qpos[m, :NJ][:, None, :]
        self.prev_a[m] = 0.0
        self.z0[m] = self._obj_pos_true()[m, 2]
        self.ep_ret[m] = 0.0

    def reset(self):
        self._reset_worlds(torch.ones(self.N, dtype=torch.bool, device=self.dev))
        self.tw.advance(2)                                                   # let the sensors see the new scene
        return self.observe()

    def observe(self):
        o = self.tw.observe(images=False)
        q, qd = o["q"], o["qd"]
        op = self._obj_pos_seen()
        held = (o["held"] == self.tgt).float()
        other = ((o["held"] >= 0) & (o["held"] != self.tgt)).float()
        return torch.nan_to_num(torch.cat([q, qd * 0.5, (self.targets[:, -1] - q) * 10.0, self.prev_a, op, self.half[self.tgt] * 10.0,
                                           torch.stack([held, other, self.step_i / self.horizon], 1)], 1), 0.0, 10.0, -10.0).clamp(-10.0, 10.0)

    # ---------------------------------------------------------------- step
    def step(self, action):
        a = action.clamp(-1.0, 1.0)
        tcp_b = self.tw.world.tcp()[0].clone()
        q_seen = self.tw.fb.read(self.tw.t)[0]
        new = self.targets[:, -1] + a[:, :NJ] * self.dq
        lead = self.tw.cfg["lead_max"]
        new = torch.minimum(torch.maximum(new, q_seen - lead), q_seen + lead)   # executor: target never runs away from the arm
        new = torch.minimum(torch.maximum(new, self.jlo + 0.05), self.jhi - 0.05)
        self.targets = torch.cat([self.targets[:, 1:], new[:, None]], 1)
        ext = new + (new - self.targets[:, -2])
        pts = torch.cat([self.targets, ext[:, None]], 1)                     # control point j sits at knot t_j + dt
        self.tw.command(pts, self.dt, t_anchor=self.tw.t - 2 * self.dt, valve=a[:, 6] > 0)
        self.tw.advance(self.n_tick)
        self.step_i += 1
        self.prev_a = a
        # reward from the TRUE state
        tcp = self.tw.world.tcp()[0]
        op = self._obj_pos_true()
        top = op + torch.stack([torch.zeros_like(self.z0), torch.zeros_like(self.z0), self.half[self.tgt, 2]], 1)
        d_b, d = (tcp_b - top).norm(dim=-1), (tcp - top).norm(dim=-1)
        held = self.tw.suction.held == self.tgt
        lift = (op[:, 2] - self.z0).clamp(0.0, 0.15)
        rew = 4.0 * (d_b - d) + 0.05 * held.float() + 2.0 * lift * held.float() - 0.002 * (a[:, :NJ] ** 2).sum(1)
        bad = ~self.tw.healthy() | ~torch.isfinite(rew)
        rew = torch.where(bad, torch.zeros_like(rew), rew)
        success = held & (lift > 0.10)
        done = (self.step_i >= self.horizon) | bad | success
        rew = rew + 5.0 * success.float() - 1.0 * bad.float()
        self.ep_ret += rew
        info = dict(success=success, held=held, bad=bad, ep_ret=self.ep_ret.clone(), dist=d, latency=self.tw.last_latency)
        self._reset_worlds(done)
        if done.any():
            self.tw.advance(0)
        return self.observe(), rew, done, info
