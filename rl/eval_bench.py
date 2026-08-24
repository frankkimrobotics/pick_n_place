"""eval_bench :: standardized policy benchmark on the CURRENT environment.

Runs each checkpoint deterministically (mean action) for --episodes pnp
episodes in the current physics (full-orientation suction, hysteresis,
DR off) and scores a criteria LADDER from the episode-terminal info:

  SEAL   ever sealed
  SETD   set-down reached (gentle release at target height)
  V1     placed <= 3.5 cm at rest              (ppo4-era spec)
  V2     V1 + max_lift >= 0.30 + landing < 8 cm/s   (ppo7-era spec)
  V3     V2 + release_h < 1 cm + obj tilt < 25 deg  (contact-release spec)

plus placement-error percentiles. Obs growth handled by zero-padding.

    python rl/eval_bench.py --episodes 256
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import warp as wp                                          # noqa: E402

CKPTS = [
    ("ppo4    (table spec)", "~/pnp_rl/ppo4/final.pt"),
    ("ppo5    (workspace)", "~/pnp_rl/ppo5/final.pt"),
    ("ppo6c   (lift+speed)", "~/pnp_rl/ppo6c/final.pt"),
    ("ppo7_ped(teacher)", "~/pnp_rl/ppo7_ped/final.pt"),
    ("ppodr   (from scratch)", "~/pnp_rl/ppodr/final.pt"),
    ("ppo13_mix (hysteresis)", "~/pnp_rl/ppo13_mix/final.pt"),
]


def load_padded(ac, path, dev):
    ck = torch.load(os.path.expanduser(path), map_location=dev,
                    weights_only=False)
    sd = ck["ac"]
    own = ac.state_dict()
    for k in list(sd.keys()):
        if k in own and own[k].shape != sd[k].shape:
            pad = torch.zeros_like(own[k])
            sl = tuple(slice(0, s) for s in sd[k].shape)
            pad[sl] = sd[k]
            sd[k] = pad
    ac.load_state_dict(sd)
    ac.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--nworld", type=int, default=256)
    ap.add_argument("--scene", default=os.path.join(HERE, "scenes",
                                                    "box_med_ped.xml"))
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/bench"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda:0"
    wp.init()
    from env_warp import PickEnv
    from ppo import AC
    env = PickEnv(nworld=a.nworld, mode="pnp", dr=False, xml=a.scene,
                  lift_req=0.30)
    rows = []
    hdr = (f"{'policy':24s} {'eps':>4} {'seal':>6} {'setd':>6} {'V1':>6} "
           f"{'V2':>6} {'V3':>6} {'d_p50':>6} {'d_p90':>6}")
    print(hdr)
    print("-" * len(hdr))
    for name, path in CKPTS:
        if not os.path.exists(os.path.expanduser(path)):
            print(f"{name:24s} MISSING {path}")
            continue
        ac = AC().to(dev)
        try:
            load_padded(ac, path, dev)
        except Exception as e:
            print(f"{name:24s} LOAD FAILED: {str(e)[:40]}")
            continue
        env.reset(torch.ones(a.nworld, dtype=torch.bool, device=dev))
        obs = env.observe()
        D = dict(seal=[], setd=[], d=[], lift=[], spd=[], rel=[], tilt=[])
        done_ct = 0
        step = 0
        while done_ct < a.episodes and step < 4000:
            step += 1
            with torch.no_grad():
                act = torch.tanh(ac.pi(obs))
            obs, r, done, info = env.step(act)
            if done.any():
                di = done.nonzero().squeeze(-1)
                done_ct += di.numel()
                sd = ~(info["timeout"][di] | info["off"][di])
                D["seal"].extend(info["ever_sealed"][di].cpu().tolist())
                D["setd"].extend(sd.cpu().tolist())
                D["d"].extend(info["final_d"][di].cpu().tolist())
                D["lift"].extend(info["max_lift"][di].cpu().tolist())
                D["spd"].extend(info["final_spd"][di].cpu().tolist())
                D["rel"].extend(info["release_h"][di].cpu().tolist())
                D["tilt"].extend(info["max_tilt"][di].cpu().tolist())
        n = len(D["seal"])
        seal = np.mean(D["seal"])
        setd = np.array(D["setd"])
        d = np.array(D["d"])
        lift = np.array(D["lift"])
        spd = np.array(D["spd"])
        rel = np.array(D["rel"])
        tilt = np.array(D["tilt"])
        at_rest = setd
        v1 = at_rest & (d < 0.035)
        v2 = v1 & (lift >= 0.30) & (spd < 0.08)
        v3 = v2 & (rel < 0.010) & (tilt < 0.44)
        dp = d[at_rest] if at_rest.any() else np.array([np.nan])
        row = dict(name=name.strip(), eps=n, seal=float(seal),
                   setd=float(setd.mean()), v1=float(v1.mean()),
                   v2=float(v2.mean()), v3=float(v3.mean()),
                   d_p50=float(np.nanpercentile(dp, 50)),
                   d_p90=float(np.nanpercentile(dp, 90)))
        rows.append(row)
        print(f"{name:24s} {n:4d} {seal:6.1%} {setd.mean():6.1%} "
              f"{v1.mean():6.1%} {v2.mean():6.1%} {v3.mean():6.1%} "
              f"{row['d_p50']*100:5.1f}c {row['d_p90']*100:5.1f}c")
    json.dump(rows, open(os.path.join(a.out, "bench.json"), "w"), indent=1)
    print(f"\nsaved {a.out}/bench.json  (current-physics env, deterministic, "
          f"DR off, pedestals on)")


if __name__ == "__main__":
    main()
