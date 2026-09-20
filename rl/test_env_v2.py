#!/usr/bin/env python3
"""test_env_v2 :: acceptance checks for rl/env_v2.py (DiverseEnv).

  (a) build DiverseEnv(nworld=512, drive="real", dr=True, ep_len=150); print obs_dim, the
      min/max of the sampled dims / mass / spawn positions and the fraction of each shape;
  (b) 150 random-action steps -- no NaN anywhere -- then drive joint 1 into the back wall on
      a few worlds and check info["wall_hit"] fires;
  (c) one IK-teacher batch through rl/bc_curobo.py --env v2 --no_planner, reporting teacher
      success / seal per shape class.

    $PY rl/test_env_v2.py                # all three
    $PY rl/test_env_v2.py --skip_teacher # (a) + (b) only
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import torch
import warp as wp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import build_scene_v2 as B  # noqa: E402
import env_v2 as V  # noqa: E402
from env_v2 import DiverseEnv  # noqa: E402

PY = sys.executable
OUT = os.path.expanduser("~/pnp_rl/v2_teacher_check")


def mm(name, v, unit="m"):
    v = np.asarray(v, float)
    print(f"    {name:<22s} min {v.min():+.4f}  max {v.max():+.4f}  mean {v.mean():+.4f}  [{unit}]")


def part_a(env):
    print("\n=== (a) build + sampled distributions " + "=" * 36)
    obs = env.observe()
    print(f"  obs_dim {obs.shape[-1]}  (paper obs {obs.shape[-1] - 7} + 7-D object descriptor)")
    print(f"  nworld {env.nworld}  substeps {env.substeps}  ep_len {env.paper['ep_len']}  drive {env.drive}")
    half = env.half_w.cpu().numpy()
    shp = env.shape_w.cpu().numpy()
    qa = env.jadr_obj
    xy = env.qpos[:, qa:qa + 2].cpu().numpy()
    print("  object dims / mass / friction:")
    for s_i, nm in enumerate(V.SHAPES):
        m = shp == s_i
        if not m.any():
            continue
        h = half[m]
        lab = "hx/hy" if s_i == 0 else "radius"
        print(f"    [{nm}] n={m.sum():4d}  {lab} {h[:, 0].min():.4f}..{h[:, 0].max():.4f}  "
              f"hz {h[:, 2].min():.4f}..{h[:, 2].max():.4f}  top {2 * h[:, 2].min():.4f}..{2 * h[:, 2].max():.4f}")
    mm("half x", half[:, 0]); mm("half y", half[:, 1]); mm("half z", half[:, 2])
    mm("mass", env.mass_w.cpu().numpy(), "kg")
    mm("friction", env.fric_w.cpu().numpy(), "-")
    mm("spawn x", xy[:, 0]); mm("spawn y", xy[:, 1])
    mm("goal x", env.goal[:, 0].cpu().numpy()); mm("goal y", env.goal[:, 1].cpu().numpy())
    mm("goal z", env.goal[:, 2].cpu().numpy())
    frac = {nm: float((shp == i).mean()) for i, nm in enumerate(V.SHAPES)}
    print("  shape fractions: " + "  ".join(f"{k} {100 * v:.1f} %" for k, v in frac.items()))
    print(f"  requested spawn box x{env.spawn[:2]} y{env.spawn[2:]}: "
          f"{100 * env.static_frac:.1f} % on table + base/bin clear, of which "
          f"{100 * env.ik_frac:.1f} % IK-reachable over the 4-12 cm grasp band "
          f"-> {100 * env.spawn_frac:.1f} % of the requested area usable")
    print(f"  spawn rejection fallback rate: {100 * env._last_reject:.2f} %")
    # resting check: bottom-origin convention means qpos z ~ 0 once settled
    hold = torch.zeros(env.nworld, 7, device=env.device)
    hold[:, 6] = -1.0                       # arm still, suction off
    for _ in range(10):
        env.step(hold)
    z = env.qpos[:, qa + 2].cpu().numpy()
    print(f"  settled object bottom z: min {z.min():+.5f}  max {z.max():+.5f} (0 = on the table)")
    ok = abs(z).max() < 0.004
    print(f"  [{'PASS' if ok else 'WARN'}] resting penetration/float < 4 mm")
    return ok


def part_b(env):
    print("\n=== (b) 150 random steps + wall-hit probe " + "=" * 32)
    env.reset(torch.ones(env.nworld, dtype=torch.bool, device=env.device))
    g = torch.Generator(device=env.device).manual_seed(0)
    bad = 0
    t0 = time.time()
    for k in range(150):
        a = torch.rand(env.nworld, 7, device=env.device, generator=g) * 2 - 1
        obs, r, done, info = env.step(a)
        for nm, v in (("obs", obs), ("reward", r), ("qpos", env.qpos), ("qvel", env.qvel)):
            if not torch.isfinite(v).all():
                print(f"    NaN/Inf in {nm} at step {k}")
                bad += 1
                break
        if bad:
            break
    dt = time.time() - t0
    print(f"  150 x {env.nworld} steps in {dt:.1f}s ({env.nworld * 150 / dt:,.0f} env-steps/s)"
          f"  wall_hit {100 * info['wall_hit'].float().mean():.1f} %  off {100 * info['off'].float().mean():.1f} %")
    print(f"  [{'PASS' if bad == 0 else 'FAIL'}] no NaN/Inf in obs / reward / qpos / qvel")
    # ---- drive joint 1 toward -x on a few worlds ----------------------------
    env.auto_reset = False
    env.reset(torch.ones(env.nworld, dtype=torch.bool, device=env.device))
    probe = torch.zeros(env.nworld, dtype=torch.bool, device=env.device)
    probe[:8] = True
    fired = None
    for k in range(150):
        a = torch.zeros(env.nworld, 7, device=env.device)
        a[:, 6] = -1.0
        a[probe, 0] = 1.0            # joint 1: swing the arm round behind the base (-x)
        obs, r, done, info = env.step(a)
        if fired is None and bool(info["wall_hit"][probe].any()):
            fired = k
    hit = info["wall_hit"]
    j1 = env.qpos[0, 0].item()
    print(f"  probe worlds 0-7: joint1 swung to {np.degrees(j1):+.0f} deg, "
          f"wall_hit {int(hit[probe].sum())}/8 (first at step {fired}); "
          f"untouched worlds {int(hit[~probe].sum())}/{int((~probe).sum())}")
    ok = bool(hit[probe].all()) and not bool(hit[~probe].any())
    print(f"  [{'PASS' if ok else 'FAIL'}] wall_hit fires only on the worlds driven into the wall")
    env.auto_reset = True
    return bad == 0 and ok


def part_c(nworld=256):
    print("\n=== (c) IK teacher (bc_curobo --env v2 --no_planner) " + "=" * 21)
    cmd = [PY, os.path.join(HERE, "bc_curobo.py"), "--env", "v2", "--drive", "real", "--no_planner",
           "--nworld", str(nworld), "--teacher_batches", "1", "--dagger_iters", "0",
           "--press_depth", "0.02", "--ep_len", "150", "--out", OUT]
    print("  $ " + " ".join(cmd), flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("[dagger]") or ln.startswith("[env")]
    for ln in lines:
        print("  " + ln)
    if p.returncode != 0:
        print("  STDERR tail:\n" + "\n".join(p.stderr.splitlines()[-25:]))
    print(f"  [{'PASS' if p.returncode == 0 else 'FAIL'}] bc_curobo --env v2 ran (rc {p.returncode})")
    return p.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworld", type=int, default=512)
    ap.add_argument("--teacher_nworld", type=int, default=256)
    ap.add_argument("--skip_teacher", action="store_true")
    a = ap.parse_args()
    wp.init()
    if not os.path.exists(B.OUT_XML):
        B.build()
    t0 = time.time()
    env = DiverseEnv(nworld=a.nworld, drive="real", dr=True, ep_len=150)
    print(f"[test_env_v2] env built in {time.time() - t0:.1f}s")
    ok_a = part_a(env)
    ok_b = part_b(env)
    ok_c = True
    if not a.skip_teacher:
        del env
        torch.cuda.empty_cache()
        ok_c = part_c(a.teacher_nworld)
    print("\n=== summary " + "=" * 62)
    print(f"  (a) build + distributions : {'PASS' if ok_a else 'WARN'}")
    print(f"  (b) rollout + wall_hit    : {'PASS' if ok_b else 'FAIL'}")
    print(f"  (c) IK teacher batch      : {'PASS' if ok_c else 'FAIL'}")
    sys.exit(0 if (ok_b and ok_c) else 1)


if __name__ == "__main__":
    main()
