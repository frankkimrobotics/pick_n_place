#!/usr/bin/env python3
"""eval_speed :: drive-envelope evaluation of a (residual) policy on the measured drive.

Rolls `--nworld` deterministic episodes of the paper pick task in the mujoco_warp twin
with the measured Pro 630 drive model (`drive=real`, `dr=True`) and reports, next to the
task metrics (success / seal), the distribution of the PER-EPISODE peak joint speed and
acceleration -- the quantities the firmware's following-error protection trips on
(36 deg/s ceiling, ~600 deg/s^2 design acceleration cap; see planner_sweep/urdf_audit.md).

The joint trajectory is sampled at the drive's own 100 Hz command rate by monkey-patching
`env._step_real_drive(log=...)` (the same hook rl/rl_vs_planner.py uses), so |qd| and |qdd|
are the real drive-chain numbers, not the 10 Hz decision deltas.

Usage
-----
  PY=/home/lisc-frank/miniconda3/envs/mjwarp/bin/python
  # residual checkpoint on top of its DAgger base
  CUDA_VISIBLE_DEVICES=1 $PY rl/eval_speed.py \
      --ckpt ~/pnp_rl/resid1_real/best.pt \
      --base ~/pnp_rl/dagger6_real/bc_iter10.pt
  # the base policy alone
  CUDA_VISIBLE_DEVICES=1 $PY rl/eval_speed.py --ckpt ~/pnp_rl/dagger6_real/bc_iter10.pt
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

V_HARD = 36.0        # deg/s   firmware following-error ceiling
A_HARD = 600.0       # deg/s^2 design acceleration cap


def logged(env, sink):
    """Monkey-patch _step_real_drive so every 10 ms command tick lands in `sink`."""
    raw = env._step_real_drive

    def wrapped(log=None, _raw=raw, _sink=sink):
        return _raw(log=_sink)
    env._step_real_drive = wrapped
    return raw


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default=os.path.expanduser("~/pnp_rl/resid1_real/best.pt"),
                    help="policy to evaluate; if it carries residual_base it is fused with --base")
    ap.add_argument("--base", default=None,
                    help="frozen base policy (default: the ckpt's own residual_base)")
    ap.add_argument("--bound", type=float, default=None, help="residual bound (default: the ckpt's)")
    ap.add_argument("--nworld", type=int, default=512, help="episodes (one per world)")
    ap.add_argument("--ep_len", type=int, default=150)
    ap.add_argument("--scene", default=os.path.join(HERE, "scenes", "box_med.xml"))
    ap.add_argument("--arch", default="paper", choices=["default", "paper"])
    ap.add_argument("--dq_max", type=float, default=2.0)
    ap.add_argument("--drive", default="real", choices=["real", "ideal"])
    ap.add_argument("--no_dr", action="store_true", help="disable domain randomisation (default: on)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target_max", type=float, default=0.30)
    ap.add_argument("--w_reach", type=float, default=0.5)
    ap.add_argument("--w_track_c", type=float, default=4.0)
    ap.add_argument("--w_track_f", type=float, default=8.0)
    ap.add_argument("--out", default=None, help="write the summary as JSON here")
    a = ap.parse_args()

    import torch
    import warp as wp
    wp.init()
    from env_paper import PaperPickEnv
    from ppo import AC

    dev = "cuda:0"
    torch.manual_seed(a.seed)
    env = PaperPickEnv(nworld=a.nworld, device=dev, seed=a.seed, xml=a.scene,
                       dr=not a.no_dr, drive=a.drive, dq_max_deg=a.dq_max,
                       target_max=a.target_max, start="home", ep_len=a.ep_len,
                       grasp_shaping=True, obs_ee=True, reach_target="grasp", lift_dense=True,
                       w_reach=a.w_reach, w_track_c=a.w_track_c, w_track_f=a.w_track_f)
    env.auto_reset = False
    obs_dim = env.observe().shape[-1]

    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    base_path = a.base or ck.get("residual_base")
    bound = float(a.bound if a.bound is not None else ck.get("residual_bound", 0.3))

    def _ac(path, extra=0):
        m = AC(obs_dim=obs_dim, arch=a.arch, critic_extra=extra).to(dev)
        c = torch.load(path, map_location=dev, weights_only=False)
        m.load_state_dict(c["ac"] if "ac" in c else c)
        m.eval()
        return m

    pol = _ac(a.ckpt, extra=(5 if ck.get("critic_priv") else 0))
    if base_path:
        base = _ac(base_path)
        print(f"[eval] residual {a.ckpt} (bound {bound}) on base {base_path}", flush=True)

        def act(o):
            return (torch.tanh(base.pi(o)) + bound * torch.tanh(pol.pi(o))).clamp(-1.0, 1.0)
    else:
        print(f"[eval] plain policy {a.ckpt}", flush=True)

        def act(o):
            return torch.tanh(pol.pi(o))

    sink = []
    logged(env, sink)
    info = None
    with torch.no_grad():
        for _ in range(a.ep_len):
            o = env.observe()
            _, _, _, info = env.step(act(o))

    placed = info["placed"].float().mean().item()
    sealed = info["ever_sealed"].float().mean().item()
    lift = info["max_lift"].cpu().numpy()

    # (T, N, 6) joint positions at the drive's 100 Hz command rate
    q = torch.stack(sink, dim=0).cpu().numpy()
    dt = 0.01
    qd = np.degrees(np.gradient(q, dt, axis=0))
    qdd = np.gradient(qd, dt, axis=0)            # deg/s -> deg/s^2
    vpk = np.abs(qd).max(0)          # (N, 6) per-episode peak |qd| per joint, deg/s
    apk = np.abs(qdd).max(0)         # (N, 6) per-episode peak |qdd| per joint, deg/s^2

    def table(name, X, hard, unit):
        print(f"\n[{name}] per-episode peak, {unit}   (hard limit {hard:g})")
        print(f"  joint |   mean     p95     max   | %ep over limit")
        for j in range(6):
            x = X[:, j]
            print(f"   j{j + 1}   | {x.mean():7.1f} {pct(x, 95):7.1f} {x.max():7.1f}   | "
                  f"{100.0 * float((x > hard).mean()):5.1f} %")
        allj = X.max(-1)
        print(f"   any  | {allj.mean():7.1f} {pct(allj, 95):7.1f} {allj.max():7.1f}   | "
              f"{100.0 * float((allj > hard).mean()):5.1f} %")

    print(f"\n[eval] {a.nworld} episodes x {a.ep_len} decisions, drive={a.drive} "
          f"dr={not a.no_dr} dq_max={a.dq_max} deg")
    print(f"[eval] success {placed:.2%}   seal {sealed:.2%}   mean max-lift {lift.mean() * 100:.1f} cm")
    table("qd", vpk, V_HARD, "deg/s")
    table("qdd", apk, A_HARD, "deg/s^2")

    out = dict(ckpt=os.path.abspath(a.ckpt), base=base_path, bound=bound,
               nworld=a.nworld, ep_len=a.ep_len, drive=a.drive, dr=not a.no_dr,
               success=placed, seal=sealed, mean_max_lift=float(lift.mean()),
               qd=dict(mean=vpk.mean(0).round(2).tolist(),
                       p95=np.percentile(vpk, 95, axis=0).round(2).tolist(),
                       max=vpk.max(0).round(2).tolist(),
                       any_mean=float(vpk.max(-1).mean()), any_max=float(vpk.max()),
                       frac_ep_over_limit=float((vpk.max(-1) > V_HARD).mean())),
               qdd=dict(mean=apk.mean(0).round(1).tolist(),
                        p95=np.percentile(apk, 95, axis=0).round(1).tolist(),
                        max=apk.max(0).round(1).tolist(),
                        any_mean=float(apk.max(-1).mean()), any_max=float(apk.max()),
                        frac_ep_over_limit=float((apk.max(-1) > A_HARD).mean())))
    print("\n[eval] summary " + json.dumps(out))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"[eval] wrote {a.out}")


if __name__ == "__main__":
    main()
