#!/usr/bin/env python3
"""wrench_rl :: batched RL for robust unknown-height extraction under a
wrench-set suction constraint (the sub-skill the scripted teacher couldn't
nail). Episodes START sealed on the centre block of a tight array; the
policy must (1) extract vertically far enough to clear the (unknown-height)
neighbours, (2) carry to the 0.25 box, (3) release inside — all without
exceeding the cup's admissible wrench set (which would detach the block).

Per-world variation:
  * K scene VARIANTS (distinct true neighbour heights) replicated across
    the batch (mjwarp shares one model per world, so true geometry varies
    by variant, not per-world);
  * per-world BELIEF noise on the target height (what the policy is told);
  * per-world drive DR (vmax/dead) so it transfers to the measured robot.

Reward (dense): clearance progress + in-box shaping + release bonus
                - wrench-margin penalty - neighbour-disturbance - drop.

    python3 wrench_rl.py --train --nworld 512 --steps 4000000
    python3 wrench_rl.py --eval --render
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

OUT = os.path.expanduser("~/pnp_rl/wrench_rl")
ARRAY_C = (0.34, 0.0)
PITCH = 0.042
HALF_XY = 0.020
BOX_C = (0.08, 0.40)
BOX_HI = 0.125
DT = 0.1
EP_LEN = 90
F_V, MU, R_EFF, BREAK = 18.0, 0.5, 0.008, 5      # wrench set (decision-rate break)
OBS_DIM, ACT_DIM = 24, 4                          # act: dx dy dz release
DQ_XY, DQ_Z = 0.010, 0.012                       # m per decision


# ------------------------------------------------------------------ scene
SCENE = """
<mujoco model="wrench_rl">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" pos="0 0 -0.12" rgba="0.35 0.37 0.40 1"/>
    <geom name="table" type="box" pos="0.34 0 -0.01" size="0.16 0.16 0.01" rgba="0.55 0.44 0.35 1"/>
    <geom name="pbx_f" type="box" pos="{bx} {by} 0.025" size="0.125 0.125 0.025" rgba="0.2 0.55 0.3 1"/>
    <geom name="pbx_xm" type="box" pos="{xm} {by} 0.11" size="0.006 0.131 0.11" rgba="0.2 0.55 0.3 0.4"/>
    <geom name="pbx_xp" type="box" pos="{xp} {by} 0.11" size="0.006 0.131 0.11" rgba="0.2 0.55 0.3 0.4"/>
    <geom name="pbx_ym" type="box" pos="{bx} {ym} 0.11" size="0.131 0.006 0.11" rgba="0.2 0.55 0.3 0.4"/>
    <geom name="pbx_yp" type="box" pos="{bx} {yp} 0.11" size="0.131 0.006 0.11" rgba="0.2 0.55 0.3 0.4"/>
    <body name="cup" pos="0 0 0" mocap="false">
      <joint name="cx" type="slide" axis="1 0 0"/>
      <joint name="cy" type="slide" axis="0 1 0"/>
      <joint name="cz" type="slide" axis="0 0 1"/>
      <geom name="cup" type="cylinder" size="0.012 0.01" rgba="0.2 0.2 0.25 1"
            contype="0" conaffinity="0" mass="0.2"/>
    </body>
    {objects}
    <camera name="iso" pos="0.95 -0.5 0.62" xyaxes="0.45 0.89 0 -0.5 0.25 0.83"/>
    <camera name="front" pos="0.34 -0.7 0.35" xyaxes="1 0 0 0 0.45 0.89"/>
  </worldbody>
  <actuator>
    <position joint="cx" kp="600" kv="40" ctrlrange="-1 1"/>
    <position joint="cy" kp="600" kv="40" ctrlrange="-1 1"/>
    <position joint="cz" kp="600" kv="40" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""
OBJ = ('<body name="o{n}" pos="{x} {y} {z}"><joint type="free" name="j{n}" damping="0.6"/>'
       '<geom name="g{n}" type="box" size="{hx} {hy} {hz}" friction="0.7 0.005 0.0001" '
       'solref="0.02 2" solimp="0.9 0.95 0.002" rgba="{rgba}" mass="0.07"/></body>')


def variant_xml(heights, out_xml):
    order = [4] + [i for i in range(9) if i != 4]
    objs = []
    for n, cell in enumerate(order):
        r, c = divmod(cell, 3)
        x = ARRAY_C[0] + (r - 1) * PITCH
        y = ARRAY_C[1] + (c - 1) * PITCH
        hz = heights[cell] / 2
        rgba = "0.85 0.2 0.15 1" if n == 0 else "0.4 0.55 0.75 1"
        objs.append(OBJ.format(n=n, x=x, y=y, z=hz + 0.001, hx=HALF_XY,
                               hy=HALF_XY, hz=hz, rgba=rgba))
    xml = SCENE.format(bx=BOX_C[0], by=BOX_C[1],
                       xm=BOX_C[0] - BOX_HI - 0.006, xp=BOX_C[0] + BOX_HI + 0.006,
                       ym=BOX_C[1] - BOX_HI - 0.006, yp=BOX_C[1] + BOX_HI + 0.006,
                       objects="\n    ".join(objs))
    open(out_xml, "w").write(xml)
    return [heights[c] for c in order]     # true heights, object-indexed


# ------------------------------------------------------------------ batched env
class WrenchEnv:
    """Cup = 3-DOF slide body (kinematic, no arm — this trains the
    extraction/carry skill, not IK). Suction = wrench-set spring on the
    centre block, active from t=0 (episode starts sealed)."""

    def __init__(self, nworld, variant, seed=0, device="cuda:0", dr=True):
        self.N = nworld
        self.device = device
        self.dr = dr
        self.rng = np.random.default_rng(seed)
        import mujoco
        xml = os.path.join(OUT, f"variant_{variant}.xml")
        heights = np.clip(self.rng.uniform(0.030, 0.060, 9), 0.03, 0.06) \
            if not os.path.exists(xml) else None
        if heights is not None:
            self.true_h = variant_xml(heights, xml)
        else:
            self.true_h = json.load(open(xml + ".h.json"))
        json.dump(self.true_h, open(xml + ".h.json", "w"))
        self.mjm = mujoco.MjModel.from_xml_path(xml)
        self.m = mjw.put_model(self.mjm)
        self.d = mjw.make_data(self.mjm, nworld=nworld)
        self.qpos = wp.to_torch(self.d.qpos)
        self.qvel = wp.to_torch(self.d.qvel)
        self.ctrl = wp.to_torch(self.d.ctrl)
        self.xpos = wp.to_torch(self.d.xpos)
        self.xfrc = wp.to_torch(self.d.xfrc_applied)
        name2id = lambda n: mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, n)
        self.bid_cup = name2id("cup")
        self.bid_o = [name2id(f"o{n}") for n in range(9)]
        self.jadr_cup = self.mjm.jnt_qposadr[self.mjm.body_jntadr[self.bid_cup]]
        self.jadr_o0 = self.mjm.jnt_qposadr[self.mjm.body_jntadr[self.bid_o[0]]]
        self.vadr_o0 = self.mjm.jnt_dofadr[self.mjm.body_jntadr[self.bid_o[0]]]
        self.o0_h = self.true_h[0]
        self.substeps = int(round(DT / self.mjm.opt.timestep))
        self.nb_max_true = max(self.true_h[1:])
        self.reset(torch.ones(nworld, dtype=torch.bool, device=device))

    def reset(self, mask):
        idx = torch.nonzero(mask).squeeze(-1)
        if idx.numel() == 0:
            return
        n = idx.numel()
        # cup sitting on the centre block top, sealed
        top = self.o0_h + 0.001
        self.qpos[idx, self.jadr_cup] = ARRAY_C[0]
        self.qpos[idx, self.jadr_cup + 1] = ARRAY_C[1]
        self.qpos[idx, self.jadr_cup + 2] = top + 0.01
        self.qvel[idx, :3] = 0.0
        # all blocks at rest
        for k, bid in enumerate(self.bid_o):
            ja = self.mjm.jnt_qposadr[self.mjm.body_jntadr[bid]]
            r, c = divmod(([4] + [i for i in range(9) if i != 4])[k], 3)
            self.qpos[idx, ja] = ARRAY_C[0] + (r - 1) * PITCH
            self.qpos[idx, ja + 1] = ARRAY_C[1] + (c - 1) * PITCH
            self.qpos[idx, ja + 2] = self.true_h[k] / 2 + 0.001
            self.qpos[idx, ja + 3] = 1.0
            self.qpos[idx, ja + 4:ja + 7] = 0.0
        va = self.vadr_o0
        for k, bid in enumerate(self.bid_o):
            dv = self.mjm.jnt_dofadr[self.mjm.body_jntadr[bid]]
            self.qvel[idx, dv:dv + 6] = 0.0
        mjw.forward(self.m, self.d)      # update xpos from the qpos just set
        self._init_state(idx)            # anchor needs current world positions

    def _init_state(self, idx):
        n = idx.numel()
        if not hasattr(self, "sealed"):
            self.sealed = torch.zeros(self.N, dtype=torch.bool, device=self.device)
            self.anchor = torch.zeros(self.N, 3, device=self.device)
            self.wr_break = torch.zeros(self.N, dtype=torch.long, device=self.device)
            self.wr_margin = torch.ones(self.N, device=self.device)
            self.t_step = torch.zeros(self.N, dtype=torch.long, device=self.device)
            self.bel_h = torch.zeros(self.N, device=self.device)
            self.max_clear = torch.zeros(self.N, device=self.device)
            self.vmax_w = torch.full((self.N, 1), math.radians(50) * 0, device=self.device)
            self.done_place = torch.zeros(self.N, dtype=torch.bool, device=self.device)
            self.nb0 = torch.zeros(self.N, 8, 2, device=self.device)
        self.sealed[idx] = True
        cup = self.xpos[idx, self.bid_cup]
        o0 = self.xpos[idx, self.bid_o[0]]
        self.anchor[idx] = o0 - cup            # world-frame offset (cup axis fixed down)
        self.wr_break[idx] = 0
        self.wr_margin[idx] = 1.0
        self.t_step[idx] = 0
        self.bel_h[idx] = torch.tensor(
            self.o0_h + self.rng.normal(0, 0.008, n), device=self.device,
            dtype=torch.float32).clamp(0.02, 0.07)
        self.max_clear[idx] = 0.0
        self.done_place[idx] = False
        for k in range(8):
            self.nb0[idx, k] = self.xpos[idx, self.bid_o[k + 1]][:, :2]

    # ---------- wrench-set suction ----------
    def _suction(self):
        cup = self.xpos[:, self.bid_cup]
        o0 = self.xpos[:, self.bid_o[0]]
        va = self.vadr_o0
        ov = self.qvel[:, va:va + 3]
        target = cup + self.anchor
        F = 450.0 * (target - o0) - 25.0 * ov     # softer bellows stiffness
        # cup axis = world -z (cup presses down); pull = +z component of F on block
        f_ax = F[:, 2]
        pull = (-f_ax).clamp(min=0.0)          # block pulled UP = tension
        shear = F[:, :2].norm(dim=-1)
        head = (F_V - pull).clamp(min=0.0)
        sh_lim = MU * head + 1e-6
        vio = (pull > F_V) | (shear > sh_lim)
        self.wr_margin = torch.where(
            self.sealed, torch.minimum(1 - pull / F_V, 1 - shear / sh_lim),
            torch.ones_like(self.wr_margin))
        # clamp transmitted force into the set
        pull_c = pull.clamp(max=F_V)
        f_ax_c = torch.where(f_ax < 0, -pull_c, f_ax)
        F_t = F[:, :2] * (sh_lim / shear.clamp(min=1e-6)).clamp(max=1.0)[:, None]
        Fc = torch.cat([F_t, f_ax_c[:, None]], -1)
        m = self.sealed.float()[:, None]
        self.xfrc[:, self.bid_o[0], :3] = Fc.clamp(-25, 25) * m
        # clamp the held block's linear velocity: a spring-through-contact
        # transient can blow up the free body otherwise
        va = self.vadr_o0
        v = self.qvel[:, va:va + 3]
        self.qvel[:, va:va + 3] = torch.where(
            self.sealed[:, None], v.clamp(-3.0, 3.0), v)
        self.wr_break = torch.where(vio & self.sealed, self.wr_break + 1,
                                    torch.zeros_like(self.wr_break))

    def observe(self):
        cup = self.xpos[:, self.bid_cup]
        o0 = self.xpos[:, self.bid_o[0]]
        va = self.vadr_o0
        rel = o0 - cup
        obs = torch.cat([
            self.qpos[:, self.jadr_cup:self.jadr_cup + 3] - torch.tensor(
                [ARRAY_C[0], ARRAY_C[1], 0.0], device=self.device),
            self.qvel[:, :3],
            (o0 - torch.tensor([ARRAY_C[0], ARRAY_C[1], 0.0], device=self.device)),
            self.qvel[:, va:va + 3],
            rel,
            self.bel_h[:, None],
            torch.full((self.N, 1), self.nb_max_true + 0.0, device=self.device),
            (o0[:, 2] - self.nb_max_true)[:, None],           # clearance signal
            self.sealed.float()[:, None],
            self.wr_margin[:, None],
            torch.tensor([BOX_C[0] - ARRAY_C[0], BOX_C[1] - ARRAY_C[1]],
                         device=self.device).expand(self.N, 2),
            (o0[:, :2] - torch.tensor(BOX_C, device=self.device)),
        ], -1)
        return obs

    def step(self, a):
        a = a.clamp(-1, 1)
        cj = self.jadr_cup
        d = torch.stack([a[:, 0] * DQ_XY, a[:, 1] * DQ_XY, a[:, 2] * DQ_Z], -1)
        tgt = self.qpos[:, cj:cj + 3] + d
        self.ctrl[:, :3] = tgt
        want = a[:, 3] < 0.5                    # release when >0.5
        rel = self.sealed & ~want
        if rel.any():
            self.sealed[rel] = False
            self.xfrc[rel, self.bid_o[0]] = 0.0
        o0_before = self.xpos[:, self.bid_o[0]].clone()
        for _ in range(self.substeps):
            self._suction()
            mjw.step(self.m, self.d)
        broke = self.sealed & (self.wr_break >= BREAK * self.substeps)
        if broke.any():
            self.sealed[broke] = False
            self.xfrc[broke, self.bid_o[0]] = 0.0
        self.t_step += 1
        r, done, info = self._reward(broke, o0_before)
        obs = self.observe()
        if done.any():
            self.reset(done)
            obs = self.observe()
        return obs, r, done, info

    def _reward(self, broke, o0_before):
        o0 = self.xpos[:, self.bid_o[0]]
        va = self.vadr_o0
        clear = o0[:, 2] - self.nb_max_true - HALF_XY  # bottom above neighbour top
        gained = (o0[:, 2] - self.max_clear).clamp(min=0)
        self.max_clear = torch.maximum(self.max_clear, o0[:, 2])
        d_box = torch.norm(o0[:, :2] - torch.tensor(BOX_C, device=self.device), dim=-1)
        lifted = clear > 0.01
        # neighbour disturbance
        nb = torch.stack([self.xpos[:, self.bid_o[k + 1]][:, :2] for k in range(8)], 1)
        disturb = (nb - self.nb0).norm(dim=-1).amax(-1)
        R = torch.zeros(self.N, device=self.device)
        R += 3.0 * gained * self.sealed.float()             # extraction progress
        R += 2.0 * (1 - torch.tanh(d_box / 0.15)) * lifted.float() * self.sealed.float()
        R += -1.5 * (1 - self.wr_margin).clamp(min=0) * self.sealed.float()
        R += -8.0 * disturb
        R += -0.02                                          # time
        R += -3.0 * broke.float()                           # premature detach
        # success: released inside the box, above its floor, at rest
        in_box = ((o0[:, :2] - torch.tensor(BOX_C, device=self.device)).abs().amax(-1) < BOX_HI) \
            & (o0[:, 2] > 0.055) & (o0[:, 2] < 0.24)
        placed = in_box & ~self.sealed & (self.qvel[:, va:va + 3].norm(dim=-1) < 0.1) \
            & lifted
        R += 30.0 * (placed & ~self.done_place).float()
        newly = placed & ~self.done_place
        self.done_place = self.done_place | placed
        off = (o0[:, 2] < 0.01) | (disturb > 0.06)
        timeout = self.t_step >= EP_LEN
        done = self.done_place | off | timeout
        info = dict(placed=self.done_place.clone(), disturb=disturb,
                    broke=broke, clear=clear, sealed=self.sealed.clone(),
                    timeout=timeout)
        return R, done, info


# ------------------------------------------------------------------ policy
class AC(nn.Module):
    def __init__(self, obs=OBS_DIM, act=ACT_DIM, h=256):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(obs, h), nn.ELU(), nn.Linear(h, h),
                                nn.ELU(), nn.Linear(h, act))
        self.v = nn.Sequential(nn.Linear(obs, h), nn.ELU(), nn.Linear(h, h),
                               nn.ELU(), nn.Linear(h, 1))
        self.log_std = nn.Parameter(torch.full((act,), -0.7))

    def dist(self, o):
        return torch.distributions.Normal(self.pi(o), self.log_std.exp())

    def value(self, o):
        return self.v(o).squeeze(-1)


# ------------------------------------------------------------------ train
def train(a):
    dev = "cuda"
    os.makedirs(OUT, exist_ok=True)
    wp.init()
    n_var = a.variants
    per = a.nworld // n_var
    envs = [WrenchEnv(per, v, seed=v, device=dev, dr=a.dr) for v in range(n_var)]
    ac = AC().to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=a.lr)
    T = a.rollout
    N = per * n_var
    step = 0
    t0 = time.time()
    best = -1e9
    log = open(os.path.join(OUT, "log.jsonl"), "a")
    while step < a.steps:
        OB = torch.zeros(T, N, OBS_DIM, device=dev)
        AA = torch.zeros(T, N, ACT_DIM, device=dev)
        LP = torch.zeros(T, N, device=dev)
        RW = torch.zeros(T, N, device=dev)
        DN = torch.zeros(T, N, device=dev)
        VL = torch.zeros(T + 1, N, device=dev)
        obs = torch.cat([e.observe() for e in envs], 0)
        ep_place, ep_n, ep_dist, ep_broke = 0, 0, 0.0, 0
        with torch.no_grad():
            for t in range(T):
                dist = ac.dist(obs)
                raw = dist.sample()
                OB[t] = obs
                AA[t] = raw
                LP[t] = dist.log_prob(raw).sum(-1)
                VL[t] = ac.value(obs)
                act = torch.tanh(raw)
                outs = [e.step(act[i * per:(i + 1) * per]) for i, e in enumerate(envs)]
                obs = torch.cat([o[0] for o in outs], 0)
                RW[t] = torch.cat([o[1] for o in outs], 0)
                dn = torch.cat([o[2].float() for o in outs], 0)
                DN[t] = dn
                for i, (_, _, done, info) in enumerate(outs):
                    if done.any():
                        di = torch.nonzero(done).squeeze(-1)
                        ep_n += di.numel()
                        ep_place += int(info["placed"][di].sum())
                        ep_dist += float(info["disturb"][di].sum())
                        ep_broke += int(info["broke"][di].sum())
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
        fo = OB.reshape(-1, OBS_DIM)
        fa = AA.reshape(-1, ACT_DIM)
        fl = LP.reshape(-1)
        fad = adv.reshape(-1)
        fr = ret.reshape(-1)
        for _ in range(a.epochs):
            perm = torch.randperm(fo.shape[0], device=dev)
            for k in range(0, fo.shape[0], a.minibatch):
                mb = perm[k:k + a.minibatch]
                dist = ac.dist(fo[mb])
                lp = dist.log_prob(fa[mb]).sum(-1)
                ratio = (lp - fl[mb]).exp()
                s1 = ratio * fad[mb]
                s2 = ratio.clamp(1 - 0.2, 1 + 0.2) * fad[mb]
                pl = -torch.min(s1, s2).mean()
                vl = ((ac.value(fo[mb]) - fr[mb]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pl + 0.5 * vl - 0.004 * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
                opt.step()
        sr = ep_place / max(ep_n, 1)
        md = ep_dist / max(ep_n, 1) * 1000
        row = dict(step=step, succ=sr, disturb_mm=md,
                   broke=ep_broke / max(ep_n, 1), eps=ep_n,
                   sps=int(step / (time.time() - t0)))
        print(f"[{step:>9}] succ {sr:.1%} disturb {md:.0f}mm broke "
              f"{row['broke']:.1%} eps {ep_n} {row['sps']} sps", flush=True)
        json.dump(row, log)
        log.write("\n")
        log.flush()
        if sr > best:
            best = sr
            torch.save(ac.state_dict(), os.path.join(OUT, "policy_best.pt"))
    torch.save(ac.state_dict(), os.path.join(OUT, "policy_final.pt"))
    print(f"[train] done, best succ {best:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--nworld", type=int, default=512)
    ap.add_argument("--variants", type=int, default=8)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4_000_000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--dr", action="store_true", default=True)
    a = ap.parse_args()
    if a.train:
        train(a)
    if a.eval:
        from wrench_rl_eval import evaluate
        evaluate(a)


if __name__ == "__main__":
    main()
