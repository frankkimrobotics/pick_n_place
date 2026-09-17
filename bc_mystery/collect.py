"""collect :: batched human-like demonstration collection (BC-mystery style).

Per-episode randomized Catmull-Rom spline in task space (waypoint jitter,
speed variability, optional mid-transport wobble, suction-timing jitter),
tracked closed-loop by a batched DLS controller with a cup-vertical
orientation task. Jacobians are computed analytically per world from mjwarp
state (xaxis/xanchor; verified vs mj_jacSite). Actions recorded are the
env's native 10 Hz interface (dq +-2 deg + suction logit), so any policy
trained on this data plugs straight into PickEnv.

    python bc_mystery/collect.py --episodes 2048 --out ~/pnp_bc/shard0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "rl"))
sys.path.insert(0, os.path.dirname(HERE))
import warp as wp                                          # noqa: E402

T_EP = 100                 # ticks (10 s at 10 Hz)
DQ_MAX = np.radians(2.0)


class SplineExpert:
    """Per-world randomized task-space spline + suction schedule."""
    def __init__(self, env, rng):
        self.env = env
        self.rng = rng
        N = env.nworld
        self.dev = env.device
        self.jids = list(range(6))

    def plan(self):
        """(Re)build splines for all worlds from current env state."""
        env, rng = self.env, self.rng
        N = env.nworld
        tcp, _ = env._tcp()
        obj = env._obj_pos()
        tgt = env.place_target
        th = env.target_h
        hz = float(env.half[2])
        dev = self.dev

        def U(lo, hi, *shape):
            return torch.tensor(rng.uniform(lo, hi, shape if shape else (N,)),
                                dtype=torch.float32, device=dev)

        # waypoints (N, 10, 3): dwell pairs at press (2,3) and set-down
        # (7,8) give the latch/release gates time at near-zero velocity
        # (SEAL_VEL = 8 cm/s; release hysteresis = 3 off-ticks)
        wps = []
        wps.append(tcp.clone())                                        # 0 start
        hover = torch.stack([obj[:, 0] + U(-0.01, 0.01),
                             obj[:, 1] + U(-0.01, 0.01),
                             obj[:, 2] + hz + U(0.04, 0.07)], -1)
        wps.append(hover)                                              # 1
        press = torch.stack([obj[:, 0] + U(-0.003, 0.003),
                             obj[:, 1] + U(-0.003, 0.003),
                             obj[:, 2] + hz - U(0.010, 0.020)], -1)
        wps.append(press)                                              # 2
        wps.append(press + U(-0.001, 0.001, N, 3))                     # 3 dwell
        # the ascent OVERLAPS the transport (a pure vertical 0.31 m object
        # lift alone costs ~30 ticks at +-2 deg/tick): waypoint 4 is a
        # partial lift, the mid via-point carries the tcp height that puts
        # the OBJECT above the 0.31 m gate (obj lift = tcp_z - CUP_R - 2hz)
        z_carry = 0.318 + 2 * hz + U(0.015, 0.045)
        lift = torch.stack([obj[:, 0] + U(-0.03, 0.03),
                            obj[:, 1] + U(-0.03, 0.03),
                            0.5 * z_carry + U(0.0, 0.04)], -1)
        wps.append(lift)                                               # 4
        # mid-transport via-point ("wobble" laterally, peak height here)
        mid = torch.stack([(obj[:, 0] + tgt[:, 0]) / 2 + U(-0.06, 0.06),
                           (obj[:, 1] + tgt[:, 1]) / 2 + U(-0.06, 0.06),
                           z_carry], -1)
        use_mid = torch.tensor(rng.random(N) < 0.5, device=dev)
        mid[:, :2] = torch.where(use_mid[:, None], mid[:, :2],
                                 (lift[:, :2] + mid[:, :2]) / 2)
        wps.append(mid)                                                # 5
        phigh = torch.stack([tgt[:, 0] + U(-0.01, 0.01),
                             tgt[:, 1] + U(-0.01, 0.01),
                             th + U(0.05, 0.10) + 2 * hz], -1)
        wps.append(phigh)                                              # 6
        pset = torch.stack([tgt[:, 0] + U(-0.004, 0.004),
                            tgt[:, 1] + U(-0.004, 0.004),
                            th + 2 * hz + 0.006 + U(0.0, 0.006)], -1)
        wps.append(pset)                                               # 7
        wps.append(pset + U(-0.001, 0.001, N, 3))                      # 8 dwell
        wps.append(pset + torch.tensor([0, 0, 0.12], device=dev))      # 9 retreat
        self.W = torch.stack(wps, 1)                       # (N, 10, 3)
        # Catmull-Rom tangents, ZEROED at contact waypoints: an upward
        # end-tangent at press/set-down makes the cubic dip ~2 cm below
        # the waypoint mid-segment -> table slam
        Wp = torch.cat([self.W[:, :1], self.W[:, :-1]], 1)
        Wn = torch.cat([self.W[:, 1:], self.W[:, -1:]], 1)
        ten = torch.full((10,), 0.5, device=dev)
        ten[0] = ten[2] = ten[3] = ten[7] = ten[8] = 0.0
        self.M = ten[None, :, None] * (Wn - Wp)            # (N, 10, 3)

        # segment durations (ticks), randomized speeds; base sum 75 of the
        # 100-tick episode (jitter 0.8-1.25 -> 60-94) leaves slack for the
        # clock stalls below
        base = np.array([4.0, 7, 4, 8, 8, 8, 8, 4, 4])     # 9 segments
        durs = np.stack([base * rng.uniform(0.8, 1.15, 9) for _ in range(N)])
        durs = np.maximum(np.round(durs), 3)
        starts = np.concatenate([np.zeros((N, 1)), np.cumsum(durs, 1)], 1)
        self.seg_start = torch.tensor(starts, dtype=torch.float32, device=dev)
        self.seg_dur = torch.tensor(durs, dtype=torch.float32, device=dev)
        self.n_seg = 9
        # suction: ON entering the press dwell, OFF entering the set-down
        # dwell (+3-tick release hysteresis lands inside the dwell)
        # on-jitter must be <= 0: the press hold pins tau AT seg_start[2],
        # a t_on beyond the pin would deadlock (suction never engages)
        self.t_on = self.seg_start[:, 2] + torch.tensor(
            rng.integers(-2, 1, N), dtype=torch.float32, device=dev)
        # t_off at (or just before) the set-down dwell start: the dwell
        # pin freezes tau AT seg_start[7], a later t_off would deadlock;
        # release-time variability comes from the rest-height event gate
        self.t_off = self.seg_start[:, 7] - torch.tensor(
            rng.uniform(0.0, 1.0, N), dtype=torch.float32, device=dev)
        # per-world spline clock: stalls when tracking error is large so
        # the reference never outruns the +-2 deg/tick joint budget
        self.tau = torch.zeros(N, device=dev)
        # release commit latch (set once the set-down actually happened)
        self.rel_commit = torch.zeros(N, dtype=torch.bool, device=dev)

    def target(self, tt):
        """Catmull-Rom position target per world at times tt (N,) float."""
        seg = torch.clamp(torch.searchsorted(
            self.seg_start, tt[:, None], right=True).squeeze(-1) - 1,
            0, self.n_seg - 1)
        s0 = torch.gather(self.seg_start, 1, seg[:, None]).squeeze(-1)
        du = torch.gather(self.seg_dur, 1, seg[:, None]).squeeze(-1)
        u = torch.clamp((tt - s0) / du, 0.0, 1.0)
        def pick(T, i):
            return torch.gather(T, 1, i[:, None, None].expand(-1, 1, 3)).squeeze(1)
        P1, P2 = pick(self.W, seg), pick(self.W, seg + 1)
        m1, m2 = pick(self.M, seg), pick(self.M, seg + 1)
        u = u[:, None]
        h00 = 2 * u**3 - 3 * u**2 + 1
        h10 = u**3 - 2 * u**2 + u
        h01 = -2 * u**3 + 3 * u**2
        h11 = u**3 - u**2
        return h00 * P1 + h10 * m1 + h01 * P2 + h11 * m2

    def act(self, t):
        """Env action (N,7) tracking the spline via batched DLS.

        The spline clock self.tau advances 1 tick/tick while tracking is
        tight and stalls (down to 0.25x) when the arm falls behind, so the
        reference stays reachable under the +-2 deg/tick action clamp."""
        env = self.env
        N = env.nworld
        xax = wp.to_torch(env.d.xaxis)[:, :6]              # arm joints 0..5
        xan = wp.to_torch(env.d.xanchor)[:, :6]
        tcp, R = env._tcp()
        err = torch.norm(self.target(self.tau) - tcp, dim=-1)
        # segment-aware stall: precision matters at press/set-down, so the
        # clock waits for the arm there; during transit it only slows a
        # little (lag along a straight transit doesn't change the path)
        seg = torch.clamp(torch.searchsorted(
            self.seg_start, self.tau[:, None], right=True).squeeze(-1) - 1,
            0, self.n_seg - 1)
        tight = (seg >= 1) & (seg <= 2) | (seg >= 7)       # press + set-down
        desc = seg == 6                                    # place descent
        thresh = torch.where(tight, 0.02, torch.where(desc, 0.03, 0.05))
        floor = torch.where(tight, 0.25, torch.where(desc, 0.35, 0.65)) \
            * torch.ones_like(err)
        rate = torch.clamp(thresh / (err + 1e-6), min=None, max=1.0)
        rate = torch.maximum(rate, floor)
        # safety-net hold at the mid via-point: don't descend toward the
        # place approach until the carry height actually cleared the lift
        # requirement (sealed worlds only; rarely engages -- the ascent
        # normally completes during the transit)
        hold = (self.tau >= self.seg_start[:, 5]) \
            & (self.tau < self.seg_start[:, 6]) \
            & env.sealed & (env.max_lift < 0.31)
        # press hold: the latch gate needs want & dist & 3 mm pressing depth
        # to line up for one tick -- pin the reference at the press dwell
        # until the seal actually happened
        hold_press = (self.tau >= self.seg_start[:, 2]) \
            & (self.tau < self.seg_start[:, 3]) & ~env.ever_sealed
        # set-down pin: don't retreat until the release actually committed
        # (the 0.25 stall floor alone lets tau cross the dwell while the
        # arm is still ~4 cm high -> object carried back up, never placed)
        hold_set = (self.tau >= self.seg_start[:, 7]) \
            & (self.tau < self.seg_start[:, 8]) \
            & env.sealed & ~self.rel_commit
        rate = torch.where(hold | hold_press | hold_set,
                           torch.zeros_like(rate), rate)
        self.tau = self.tau + rate
        x_des = self.target(self.tau)
        dx = torch.clamp(x_des - tcp, -0.035, 0.035)
        Jp = torch.cross(xax, (tcp[:, None, :] - xan), dim=-1).transpose(1, 2)
        Jr = xax.transpose(1, 2)                           # (N,3,6)
        cupz = R[:, :, 2]
        e_rot = torch.cross(cupz, torch.tensor([0.0, 0.0, -1.0],
                            device=tcp.device).expand(N, 3), dim=-1)
        dw = torch.clamp(1.5 * e_rot, -0.15, 0.15)
        WR = 0.8
        Jt = torch.cat([Jp, WR * Jr], 1)                   # (N,6,6)
        xt = torch.cat([dx, WR * dw], 1)[:, :, None]
        A = Jt @ Jt.transpose(1, 2) + 2e-3 * torch.eye(6, device=tcp.device)
        dq = (Jt.transpose(1, 2) @ torch.linalg.solve(A, xt)).squeeze(-1)
        # uniform scale-down (per-joint clamping bends the task-space path)
        scale = (DQ_MAX / dq.abs().max(dim=-1, keepdim=True).values.clamp(
            min=1e-9)).clamp(max=1.0)
        dq = dq * scale
        a = torch.zeros(N, 7, device=tcp.device)
        a[:, :6] = dq / DQ_MAX
        # release is EVENT-gated: t_off alone fires while the arm is still
        # descending (tau stalls but t_off is a tau timestamp), dropping
        # the object early -- also require the object at rest height
        op = env._obj_pos()
        rest_h = op[:, 2] - float(env.half[2]) - env.target_h
        self.rel_commit |= (self.tau >= self.t_off) \
            & (~env.sealed | (rest_h < 0.012))
        a[:, 6] = torch.where((self.tau >= self.t_on) & ~self.rel_commit,
                              1.0, -1.0)
        return a


class TermTracker:
    """Snapshot env info at each world's FIRST done tick (episodes are
    fixed-length T=100 = EP_LEN_PNP, auto_reset off, so placed worlds keep
    simulating; the metrics that matter are the ones at termination).
    Ladder identical to rl/eval_bench.py."""
    FIELDS = ["final_d", "max_lift", "release_h", "max_tilt", "final_spd"]

    def __init__(self, N, dev):
        self.seen = torch.zeros(N, dtype=torch.bool, device=dev)
        self.setd = torch.zeros(N, dtype=torch.bool, device=dev)
        self.seal = torch.zeros(N, dtype=torch.bool, device=dev)
        self.vals = {f: torch.zeros(N, device=dev) for f in self.FIELDS}

    def update(self, done, info):
        new = done & ~self.seen
        if new.any():
            self.setd[new] = ~(info["timeout"][new] | info["off"][new])
            self.seal[new] = info["ever_sealed"][new]
            for f in self.FIELDS:
                self.vals[f][new] = info[f][new]
            self.seen |= new

    def table(self):
        """(N, 8) float: seal setd v1 v2 v3 final_d max_lift final_spd."""
        d, lift = self.vals["final_d"], self.vals["max_lift"]
        spd, rel = self.vals["final_spd"], self.vals["release_h"]
        tilt = self.vals["max_tilt"]
        v1 = self.setd & (d < 0.035)
        v2 = v1 & (lift >= 0.30) & (spd < 0.08)
        v3 = v2 & (rel < 0.010) & (tilt < 0.44)
        return torch.stack([self.seal.float(), self.setd.float(), v1.float(),
                            v2.float(), v3.float(), d, lift, spd], -1)


TABLE_COLS = ["seal", "setd", "v1", "v2", "v3", "final_d", "max_lift",
              "final_spd"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2048)
    ap.add_argument("--nworld", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_bc/shard0"))
    ap.add_argument("--scene", default=os.path.join(
        os.path.dirname(HERE), "rl", "scenes", "box_med_ped.xml"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    wp.init()
    from env_warp import PickEnv
    rng = np.random.default_rng(a.seed)
    env = PickEnv(nworld=a.nworld, mode="pnp", dr=False, xml=a.scene,
                  lift_req=0.30)
    env.auto_reset = False
    expert = SplineExpert(env, rng)
    dev = env.device

    all_obs, all_act, all_met = [], [], []
    n_rounds = (a.episodes + a.nworld - 1) // a.nworld
    for rd in range(n_rounds):
        env.reset(torch.ones(a.nworld, dtype=torch.bool, device=dev))
        expert.plan()
        obs = env.observe()
        O = torch.zeros(T_EP, a.nworld, obs.shape[1], device=dev)
        A = torch.zeros(T_EP, a.nworld, 7, device=dev)
        trk = TermTracker(a.nworld, dev)
        for t in range(T_EP):
            act = expert.act(t)
            O[t] = obs
            A[t] = act
            obs, r, done, info = env.step(act)
            trk.update(done, info)
        met = trk.table()
        all_obs.append(O.transpose(0, 1).cpu().numpy().astype(np.float32))
        all_act.append(A.transpose(0, 1).cpu().numpy().astype(np.float32))
        all_met.append(met.cpu().numpy().astype(np.float32))
        print(f"[collect] round {rd+1}/{n_rounds}: seal {met[:,0].mean():.1%} "
              f"setd {met[:,1].mean():.1%} V1 {met[:,2].mean():.1%} "
              f"V2 {met[:,3].mean():.1%} V3 {met[:,4].mean():.1%} "
              f"d_p50 {met[:,5].median()*100:.1f} cm", flush=True)

    obs_np = np.concatenate(all_obs)[:a.episodes]
    act_np = np.concatenate(all_act)[:a.episodes]
    met_np = np.concatenate(all_met)[:a.episodes]
    np.savez_compressed(
        os.path.join(a.out, f"demos_{a.seed}.npz"),
        obs=obs_np, act=act_np, metrics=met_np,
        metric_names=np.array(TABLE_COLS))
    json.dump(dict(episodes=int(obs_np.shape[0]), T=T_EP,
                   obs_dim=int(obs_np.shape[2]),
                   v1=float(met_np[:, 2].mean()),
                   v2=float(met_np[:, 3].mean()),
                   v3=float(met_np[:, 4].mean()),
                   seed=a.seed),
              open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"[collect] saved {obs_np.shape[0]} episodes -> {a.out} "
          f"(V1 {met_np[:, 2].mean():.1%} V2 {met_np[:, 3].mean():.1%} "
          f"V3 {met_np[:, 4].mean():.1%})")


if __name__ == "__main__":
    main()
