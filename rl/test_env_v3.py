#!/usr/bin/env python3
"""test_env_v3 :: acceptance checks for rl/env_v3.py (ObstacleEnv).

  (a) build ObstacleEnv(nworld=512, obstacles=True, drive="real", dr=True, ep_len=150) and
      print obs_dim, the distractor distributions (count, post fraction, heights), the
      corridor fraction and the measured clearances to the target / goal / walls;
  (b) 300 random-action steps -- no NaN in obs / reward / qpos / qvel -- reporting the
      obstacle- and wall-hit rates;
  (c) OBSTACLE COLLISION PROBE: on a few worlds a 0.25 m post is planted directly under the
      tcp and the arm is driven straight down onto it (damped-least-squares -z Cartesian
      motion, the same primitive env_paper uses for the scripted place); info["obst_hit"]
      must fire on exactly those worlds and on no other.

    $PY rl/test_env_v3.py
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import warp as wp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import build_scene_v2 as B  # noqa: E402
import env_v3 as V3  # noqa: E402
import env_warp as E  # noqa: E402
from env_v3 import ObstacleEnv  # noqa: E402


def mm(name, v, unit="m"):
    v = np.asarray(v, float)
    if v.size == 0:
        print(f"    {name:<24s} (empty)")
        return
    print(f"    {name:<24s} min {v.min():+.4f}  max {v.max():+.4f}  mean {v.mean():+.4f}  [{unit}]")


def part_a(env):
    print("\n=== (a) build + distractor distributions " + "=" * 34)
    obs = env.observe()
    print(f"  obs_dim {obs.shape[-1]}  (env_v2 47 + 7 obstacle floats)")
    print(f"  nworld {env.nworld}  substeps {env.substeps}  ep_len {env.paper['ep_len']}  "
          f"drive {env.drive}  njmax {env.data_kw['njmax']} nconmax {env.data_kw['nconmax']}")
    qa = env.jadr_obj
    xy = env.qpos[:, qa:qa + 2].cpu().numpy()
    mm("spawn x", xy[:, 0]); mm("spawn y", xy[:, 1])
    g = env.goal.cpu().numpy()
    mm("goal x", g[:, 0]); mm("goal y", g[:, 1]); mm("goal z", g[:, 2])
    carry = np.linalg.norm(g[:, :2] - xy, axis=-1)
    mm("carry distance", carry)
    print(f"  goal wall clearance: back {g[:, 0].min() - B.WALL_BACK_X:.3f} m, "
          f"sides {B.WALL_SIDE_Y - np.abs(g[:, 1]).max():.3f} m  "
          f"[{'PASS' if (g[:, 0].min() - B.WALL_BACK_X >= 0.10 and B.WALL_SIDE_Y - np.abs(g[:, 1]).max() >= 0.10) else 'FAIL'}] >= 0.10 m")
    act = env.dist_act.cpu().numpy()
    half = env.dist_half.cpu().numpy()
    n_act = act.sum(1)
    print(f"  distractors per world: " + "  ".join(f"{k}: {100 * (n_act == k).mean():.1f} %" for k in range(4))
          + f"   mean {n_act.mean():.2f}")
    h = half[act]
    mm("distractor half width", h[:, 0]); mm("distractor half height", h[:, 2])
    post = env.dist_post.cpu().numpy()
    hp = half[post]
    print(f"  tall posts: {100 * post.any(1).mean():.1f} % of worlds (target {100 * env.p_post:.0f} %)  "
          f"height {2 * hp[:, 2].min():.3f}..{2 * hp[:, 2].max():.3f} m")
    # clearances + corridor occupancy, measured (not assumed)
    p = env.xipos[:, env.bid_dist_t].cpu().numpy()[:, :, :2]
    d_tgt = np.linalg.norm(p - xy[:, None, :], axis=-1)
    d_goal = np.linalg.norm(p - g[:, None, :2], axis=-1)
    print(f"  min |distractor - target| {d_tgt[act].min():.4f} m   "
          f"min |distractor - goal| {d_goal[act].min():.4f} m  "
          f"[{'PASS' if min(d_tgt[act].min(), d_goal[act].min()) >= V3.DIST_CLEAR_TGT - 1e-3 else 'FAIL'}] >= 0.06 m")
    seg = g[:, :2] - xy
    L = np.linalg.norm(seg, axis=-1).clip(1e-6)
    u = seg / L[:, None]
    w = p - xy[:, None, :]
    t = (w * u[:, None, :]).sum(-1) / L[:, None]
    off = np.abs(w[:, :, 0] * (-u[:, None, 1]) + w[:, :, 1] * u[:, None, 0])
    on_corr = act & (off <= V3.CORRIDOR_OFF) & (t > 0) & (t < 1)
    print(f"  worlds with a distractor ON the corridor (offset <= {V3.CORRIDOR_OFF} m): "
          f"{100 * on_corr.any(1).mean():.1f} %  (spec 70 %; p_corridor INTENT "
          f"{100 * env.p_corridor:.0f} %, ~12 % of those have no feasible spot)")
    print(f"  ... of which the corridor body is a post: "
          f"{100 * (on_corr & post).any(1).mean():.1f} % of all worlds")
    # obstacle observation block
    nb = obs[:, -7:].cpu().numpy()
    none = ~act.any(1)
    print(f"  obs[-7:] = [rel xyz, half xyz, n/3]; worlds with no distractor: {none.sum()} "
          f"(max |obs[-7:]| there = {np.abs(nb[none]).max() if none.any() else 0:.3f}, must be 0)")
    ok_zero = (not none.any()) or float(np.abs(nb[none]).max()) == 0.0
    # settling check
    hold = torch.zeros(env.nworld, 7, device=env.device)
    hold[:, 6] = -1.0
    for _ in range(20):
        env.step(hold)
    moved = torch.norm(env.xipos[:, env.bid_dist_t] - env.dist_xy0, dim=-1).cpu().numpy()
    print(f"  distractor settle drift after 20 idle steps: max {moved[act].max() * 1000:.2f} mm "
          f"(tolerance {env.disp_tol * 1000:.0f} mm)  "
          f"spurious obst_hit {100 * env.obst_hit_ep.float().mean().item():.2f} %")
    ok_settle = float(env.obst_hit_ep.float().mean()) < 0.01
    print(f"  [{'PASS' if ok_settle else 'FAIL'}] idle worlds do not trip the displacement test")
    return ok_zero and ok_settle


def part_b(env, steps=300):
    print(f"\n=== (b) {steps} random steps, finite obs/reward " + "=" * 27)
    env.reset(torch.ones(env.nworld, dtype=torch.bool, device=env.device))
    gen = torch.Generator(device=env.device).manual_seed(0)
    bad = 0
    t0 = time.time()
    for k in range(steps):
        a = torch.rand(env.nworld, 7, device=env.device, generator=gen) * 2 - 1
        obs, r, done, info = env.step(a)
        for nm, v in (("obs", obs), ("reward", r), ("qpos", env.qpos), ("qvel", env.qvel)):
            if not torch.isfinite(v).all():
                print(f"    NaN/Inf in {nm} at step {k}")
                bad += 1
                break
        if bad:
            break
    dt = time.time() - t0
    print(f"  {steps} x {env.nworld} steps in {dt:.1f}s ({env.nworld * steps / dt:,.0f} env-steps/s)")
    print(f"  obst_hit {100 * info['obst_hit'].float().mean():.1f} %   "
          f"wall_hit {100 * info['wall_hit'].float().mean():.1f} %   "
          f"off {100 * info['off'].float().mean():.1f} %   diverged {env.n_diverged}")
    print(f"  [{'PASS' if bad == 0 else 'FAIL'}] no NaN/Inf in obs / reward / qpos / qvel")
    return bad == 0


def plant_post(env, worlds, xy, hh=0.125, hw=0.030):
    """Put a box post of half height `hh` at `xy` on `worlds` (and park every other slot)."""
    idx = torch.nonzero(worlds).squeeze(-1)
    n = idx.numel()
    dev = env.device
    for k in range(V3.N_DIST):
        gb, gc = env.gid_dist[k]
        qa = env.jadr_dist[k]
        if k == 0:
            ext = torch.tensor([hw, hw, hh], device=dev)
            env.mw["geom_size"][idx, gb] = ext
            env.mw["geom_rbound"][idx, gb] = float(torch.norm(ext))
            env.mw["geom_aabb"][idx, gb, 0] = 0.0
            env.mw["geom_aabb"][idx, gb, 1] = ext
            env.mw["geom_pos"][idx, gb, 2] = hh
            env.mw["body_ipos"][idx, env.bid_dist[k], 2] = hh
            env.qpos[idx, qa:qa + 2] = xy
            env.dist_half[idx, k] = ext
            env.dist_act[idx, k] = True
        else:
            env.qpos[idx, qa] = B.DIST_PARK[0] + 0.15 * k
            env.qpos[idx, qa + 1] = B.DIST_PARK[1]
            env.dist_act[idx, k] = False
        env.qpos[idx, qa + 2] = 0.0
        env.qpos[idx, qa + 3] = 1.0
        env.qpos[idx, qa + 4:qa + 7] = 0.0
        env.qvel[idx, env.vadr_dist[k]:env.vadr_dist[k] + 6] = 0.0
    env.n_act[idx] = env.dist_act[idx].sum(1).float()
    E.mjw.forward(env.m, env.d)
    env.dist_xy0[idx] = env.xipos[idx][:, env.bid_dist_t]
    return n


def part_c(env, n_probe=8, steps=120):
    print("\n=== (c) obstacle collision probe " + "=" * 42)
    env.auto_reset = False
    env.reset(torch.ones(env.nworld, dtype=torch.bool, device=env.device))
    probe = torch.zeros(env.nworld, dtype=torch.bool, device=env.device)
    probe[:n_probe] = True
    tcp, _ = env._tcp()
    # probe worlds: a 0.25 m post exactly under the tcp.  control worlds: no distractor at
    # all, so "obst_hit anywhere else" can only be a bug.
    plant_post(env, probe, tcp[probe, :2].clone())
    park = torch.tensor([[B.DIST_PARK[0], B.DIST_PARK[1]]], device=env.device).repeat(int((~probe).sum()), 1)
    plant_post(env, ~probe, park)
    env.dist_act[~probe] = False
    env.n_act[~probe] = 0
    z0 = float(tcp[0, 2])
    print(f"  {n_probe} probe worlds: post 0.25 m tall at the tcp xy, tcp starts {z0:.3f} m up; "
          f"{int((~probe).sum())} control worlds have no distractor")
    fired = None
    cause = ""
    for k in range(steps):
        # straight-down Cartesian motion (env_paper's place-descent primitive)
        J = env._site_jac()
        wgt = torch.tensor([1.0, 1.0, 1.0, 0.05, 0.05, 0.05], device=env.device)
        Jw = J * wgt[None, :, None]
        v = torch.zeros(env.nworld, 6, device=env.device)
        v[:, 2] = -0.02
        A = torch.einsum("nij,nkj->nik", Jw, Jw) + 1e-4 * torch.eye(6, device=env.device)
        dq = torch.einsum("nji,njk->nik", Jw, torch.linalg.solve(A, v[..., None]))[..., 0]
        sc = (np.radians(1.5) / dq.abs().amax(-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.0)
        a = torch.cat([(dq * sc / env.dq_max).clamp(-1, 1),
                       -torch.ones(env.nworld, 1, device=env.device)], -1)
        obs, r, done, info = env.step(a)
        if fired is None and bool(info["obst_hit"][probe].any()):
            fired = k
            cause = ("contact" if bool(env._dist_contacts()[probe].any()) else "") + \
                    ("+moved" if bool(env._dist_moved()[probe].any()) else "")
    hit = info["obst_hit"]
    tcp2, _ = env._tcp()
    print(f"  tcp descended to z = {float(tcp2[0, 2]):.3f} m; obst_hit on probes "
          f"{int(hit[probe].sum())}/{n_probe} (first at step {fired}, cause {cause or 'n/a'}); "
          f"controls {int(hit[~probe].sum())}/{int((~probe).sum())}")
    print(f"  reward charged on probes: fail component "
          f"{float(info['ep_comp'][probe][:, env.RKEYS_PAPER.index('fail')].mean()):+.2f} "
          f"(controls {float(info['ep_comp'][~probe][:, env.RKEYS_PAPER.index('fail')].mean()):+.2f})")
    print(f"  placed cleared on probes: {int(info['placed'][probe].sum())} (must be 0)")
    ok = bool(hit[probe].all()) and not bool(hit[~probe].any())
    print(f"  [{'PASS' if ok else 'FAIL'}] obst_hit fires on exactly the worlds driven into a post")
    env.auto_reset = True
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworld", type=int, default=512)
    ap.add_argument("--steps", type=int, default=300)
    a = ap.parse_args()
    wp.init()
    if not os.path.exists(V3.SCENE_V3):
        B.build(V3.SCENE_V3, n_dist=V3.N_DIST)
    t0 = time.time()
    env = ObstacleEnv(nworld=a.nworld, drive="real", dr=True, ep_len=150, obstacles=True)
    print(f"[test_env_v3] env built in {time.time() - t0:.1f}s")
    ok_a = part_a(env)
    ok_b = part_b(env, a.steps)
    ok_c = part_c(env)
    print("\n=== summary " + "=" * 62)
    print(f"  (a) build + distributions      : {'PASS' if ok_a else 'FAIL'}")
    print(f"  (b) {a.steps} random steps, finite : {'PASS' if ok_b else 'FAIL'}")
    print(f"  (c) obstacle collision probe   : {'PASS' if ok_c else 'FAIL'}")
    sys.exit(0 if (ok_a and ok_b and ok_c) else 1)


if __name__ == "__main__":
    main()
