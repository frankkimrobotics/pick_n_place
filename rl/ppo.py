"""ppo :: batched PPO on PickEnv — on-policy comparator to sac.py.

Same obs/action space, same logging schema (plot with plot_training.py).
    python rl/ppo.py --nworld 4096 --steps 12000000 --mode pnp --dr
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import warp as wp                                          # noqa: E402
from sac import mlp, OBS_DIM, ACT_DIM                      # noqa: E402


def mlp_paper(inp, out, hidden=(256, 128, 64)):
    """Arafat et al. 2026 (QPAIN): MLP actor/critic [256,128,64], ELU, no normalisation."""
    seq, d = [], inp
    for h in hidden:
        seq += [nn.Linear(d, h), nn.ELU()]
        d = h
    seq += [nn.Linear(d, out)]
    return nn.Sequential(*seq)


class AC(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, arch="default"):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.arch = arch
        if arch == "paper":
            self.pi = mlp_paper(self.obs_dim, ACT_DIM)
            self.v = mlp_paper(self.obs_dim, 1)
        else:
            self.pi = mlp(self.obs_dim, ACT_DIM, ln=False)
            self.v = mlp(self.obs_dim, 1)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -0.5))

    def load_state_dict(self, sd, strict=True):
        """Obs-dim growth (37 -> 43 with the drive-lag obs, 2026-09-17):
        zero-pad first-layer weights of older checkpoints."""
        own = self.state_dict()
        sd = dict(sd)
        for k in list(sd.keys()):
            if k in own and own[k].shape != sd[k].shape:
                pad = torch.zeros_like(own[k])
                sl = tuple(slice(0, n) for n in sd[k].shape)
                pad[sl] = sd[k]
                sd[k] = pad
        return super().load_state_dict(sd, strict)

    def dist(self, obs):
        mu = self.pi(obs)
        return torch.distributions.Normal(mu, self.log_std.exp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworld", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=12_000_000)
    ap.add_argument("--rollout", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=16384)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.003)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mode", default="pnp", choices=["full", "attach", "pnp", "place", "carry", "mix"])
    ap.add_argument("--dr", action="store_true")
    ap.add_argument("--init", default=None)
    ap.add_argument("--release_mask", action="store_true")
    ap.add_argument("--mask_h", type=float, default=0.05)
    ap.add_argument("--tilt_pen", type=float, default=None)
    ap.add_argument("--target_max", type=float, default=0.30)
    ap.add_argument("--lift_req", type=float, default=0.0)
    ap.add_argument("--speed_bonus", type=float, default=0.0)
    ap.add_argument("--out", default=os.path.expanduser("~/pnp_rl/ppo1"))
    ap.add_argument("--scene", default=os.path.join(HERE, "scenes", "box_med.xml"))
    ap.add_argument("--drive", default="real", choices=["real", "ideal"],
                    help="real = measured Pro 630 velocity-drive model (2026-09-17); ideal = legacy stiff PD")
    ap.add_argument("--dq_max", type=float, default=2.0, help="per-decision joint delta clamp (deg)")
    ap.add_argument("--obs_lag", type=int, default=-1, help="append q_target-q to obs (-1: auto = drive==real)")
    ap.add_argument("--hover", type=float, nargs=2, default=[0.02, 0.04], help="attach/pnp start height range above the grasp point (m)")
    ap.add_argument("--init_std", type=float, default=None, help="initial policy log-std (default -0.5)")
    ap.add_argument("--smooth_w", type=float, default=None, help="weight of the delta-jerk cost (default W['smooth']=-0.01; 0 for the proven pnp economy)")
    ap.add_argument("--transport_w", type=float, default=None, help="weight of the carry-toward-target potential (default W['transport']=6)")
    ap.add_argument("--descend_sigma", type=float, default=0.07, help="xy gate width (m) of the descend-to-surface potential; 0.15 keeps it alive when the object drifts off target")
    # ---- paper setup (Arafat et al., QPAIN 2026): env + PPO details ----
    ap.add_argument("--env", default="pick", choices=["pick", "paper"], help="paper = env_paper.PaperPickEnv (reach/lift/track staged dense reward, no release)")
    ap.add_argument("--arch", default="default", choices=["default", "paper"], help="paper = [256,128,64] ELU actor/critic")
    ap.add_argument("--kl_target", type=float, default=None, help="adaptive LR on KL (paper 0.01): lr/1.5 if kl>2*target, lr*1.5 if kl<target/2")
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    # ---- demonstration anchor (DAPG-style): keeps a DAgger-initialised policy near the teacher ----
    ap.add_argument("--bc_data", default=None, help="bc_curobo dataset.pt {X obs, Y actions}; adds bc_coef * MSE(pi(X), atanh(Y)) to the loss")
    ap.add_argument("--bc_coef", type=float, default=1.0, help="initial weight of the demo MSE term")
    ap.add_argument("--bc_coef_min", type=float, default=0.1, help="floor the weight decays to")
    ap.add_argument("--bc_decay_steps", type=float, default=6e6, help="linear decay bc_coef -> bc_coef_min over this many env steps")
    ap.add_argument("--bc_batch", type=int, default=4096)
    ap.add_argument("--critic_warmup", type=int, default=0, help="updates during which only the critic is trained (actor frozen)")
    ap.add_argument("--lr_max", type=float, default=1e-3, help="ceiling for the adaptive LR (it ran to 3.8e-3 while the policy idled)")
    ap.add_argument("--grasp_shaping", type=int, default=1, help="paper env: suction press term + seal bonus (embodiment adaptation); 0 = pure paper reward")
    ap.add_argument("--obs_ee", type=int, default=1, help="paper env: append tcp position, cup axis and grasp-point-relative vector (0 = paper's obs only)")
    ap.add_argument("--reach_target", default="grasp", choices=["grasp", "centre"], help="paper env: reach term to the grasp point (suction) or the object centre (paper)")
    ap.add_argument("--lift_dense", type=int, default=1, help="paper env: dense ramp to h_min instead of the paper's indicator (0 = paper)")
    ap.add_argument("--w_track_c", type=float, default=2.0, help="paper env: coarse goal-tracking weight (sigma 0.10 m)")
    ap.add_argument("--w_track_f", type=float, default=4.0, help="paper env: fine goal-tracking weight (sigma 0.02 m)")
    ap.add_argument("--w_reach", type=float, default=1.0, help="paper env: reach weight")
    ap.add_argument("--drive_ramp", type=float, default=0.0, help="dynamics curriculum: anneal the measured drive's dead-time/accel from near-ideal (0) to measured (1) over this fraction of --steps (0 = off)")
    ap.add_argument("--vf_coef", type=float, default=0.5, help="paper 1.0")
    ap.add_argument("--value_clip", action="store_true", help="clipped value loss (paper: enabled)")
    ap.add_argument("--reg_ramp", type=float, default=0.4, help="paper env: lambda(t) ramps 0->lambda_max over this fraction of --steps")
    ap.add_argument("--ep_len", type=int, default=100, help="paper env: episode length in decisions (paper: 5 s)")
    ap.add_argument("--start", default="home", choices=["home", "hover"], help="paper env start pose")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda:0"
    torch.manual_seed(0)
    wp.init()
    from env_warp import PickEnv
    if a.env == "paper":
        from env_paper import PaperPickEnv
        env = PaperPickEnv(nworld=a.nworld, device=dev, xml=a.scene, dr=a.dr, drive=a.drive, dq_max_deg=a.dq_max,
                           obs_lag=(None if a.obs_lag < 0 else bool(a.obs_lag)), target_max=a.target_max,
                           start=a.start, ep_len=a.ep_len, grasp_shaping=bool(a.grasp_shaping), obs_ee=bool(a.obs_ee), reach_target=a.reach_target, lift_dense=bool(a.lift_dense),
                           w_track_c=a.w_track_c, w_track_f=a.w_track_f, w_reach=a.w_reach)
    else:
        env = PickEnv(nworld=a.nworld, device=dev, xml=a.scene, mode=a.mode, dr=a.dr, target_max=a.target_max, lift_req=a.lift_req, speed_bonus=a.speed_bonus, release_mask=a.release_mask, mask_h=a.mask_h, tilt_pen_w=a.tilt_pen,
                      drive=a.drive, dq_max_deg=a.dq_max, obs_lag=(None if a.obs_lag < 0 else bool(a.obs_lag)),
                      hover_range=tuple(a.hover), smooth_w=a.smooth_w, transport_w=a.transport_w, descend_sigma=a.descend_sigma)
    ac = AC(obs_dim=env.observe().shape[-1], arch=a.arch).to(dev)
    if a.init_std is not None:
        with torch.no_grad():
            ac.log_std.fill_(float(a.init_std))
    print(f"[ppo] obs_dim {ac.obs_dim} drive={a.drive} dq_max={a.dq_max} deg hover={a.hover} init_std={a.init_std}", flush=True)
    if a.init:
        ck = torch.load(a.init, map_location=dev, weights_only=False)
        try:
            if "ac" in ck:                      # PPO-format checkpoint
                sd = ck["ac"]
                own = ac.state_dict()
                for k in list(sd.keys()):           # obs-dim growth: zero-pad
                    if k in own and own[k].shape != sd[k].shape:
                        pad = torch.zeros_like(own[k])
                        sl = tuple(slice(0, s) for s in sd[k].shape)
                        pad[sl] = sd[k]
                        sd[k] = pad
                ac.load_state_dict(sd)
                print("[ppo] warm-started full AC (ppo ckpt, padded)", flush=True)
            elif "actor" in ck:                 # SAC-format: pi only
                ac.pi.load_state_dict({k[4:]: v for k, v in ck["actor"].items()
                                       if k.startswith("net.")}, strict=False)
                print("[ppo] warm-started pi (sac ckpt, partial)", flush=True)
        except Exception as e:
            print("[ppo] warm-start skipped:", e, flush=True)
    opt = torch.optim.Adam(ac.parameters(), lr=a.lr)
    bcX = bcY = None
    if a.bc_data:
        dsb = torch.load(a.bc_data, map_location=dev, weights_only=False)
        bcX, bcY = dsb["X"].to(dev), torch.atanh(dsb["Y"].to(dev).clamp(-0.97, 0.97))
        if bcX.shape[1] != ac.obs_dim:
            raise SystemExit(f"[ppo] bc_data obs {bcX.shape[1]} != env obs {ac.obs_dim}")
        print(f"[ppo] demo anchor: {bcX.shape[0]} steps, coef {a.bc_coef} -> {a.bc_coef_min} over {a.bc_decay_steps:.0f} steps", flush=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)

    N, T = a.nworld, a.rollout
    obs_b = torch.zeros(T, N, ac.obs_dim, device=dev)
    act_b = torch.zeros(T, N, ACT_DIM, device=dev)
    logp_b = torch.zeros(T, N, device=dev)
    rew_b = torch.zeros(T, N, device=dev)
    done_b = torch.zeros(T, N, device=dev)
    val_b = torch.zeros(T + 1, N, device=dev)

    obs = env.observe()
    step, n_up, t0 = 0, 0, time.time()
    best_succ = -1.0
    ep = dict(n=0, ret=0.0, len=0.0, placed=0, sealed=0,
              comp=np.zeros(len(env.RKEYS)))
    while step < a.steps:
        with torch.no_grad():
            for t in range(T):
                dist = ac.dist(obs)
                raw = dist.sample()
                actn = torch.tanh(raw)
                obs_b[t] = obs
                act_b[t] = raw
                logp_b[t] = dist.log_prob(raw).sum(-1)
                val_b[t] = ac.v(obs).squeeze(-1)
                obs, r, done, info = env.step(actn)
                rew_b[t] = r
                done_b[t] = done.float()
                if done.any():
                    di = torch.nonzero(done).squeeze(-1)
                    ep["n"] += di.numel()
                    ep["ret"] += float(info["ep_comp"][di].sum(-1).sum())
                    ep["len"] += float(info["ep_len"][di].float().sum())
                    ep["placed"] += int(info["placed"][di].sum())
                    pm = info["wmode"][di] == 0
                    ep["n_p"] = ep.get("n_p", 0) + int(pm.sum())
                    ep["placed_p"] = ep.get("placed_p", 0) + int(info["placed"][di][pm].sum())
                    ep["sealed"] += int(info["ever_sealed"][di].sum())
                    ep["comp"] += info["ep_comp"][di].sum(0).cpu().numpy()
            val_b[T] = ac.v(obs).squeeze(-1)
            adv = torch.zeros(T, N, device=dev)
            gae = torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                delta = rew_b[t] + a.gamma * (1 - done_b[t]) * val_b[t + 1] - val_b[t]
                gae = delta + a.gamma * a.lam * (1 - done_b[t]) * gae
                adv[t] = gae
            ret = adv + val_b[:T]
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        step += N * T
        if a.drive_ramp > 0 and a.drive == "real":
            env.drive_scale = min(1.0, step / max(1.0, a.drive_ramp * a.steps))
            env.apply_drive_scale()
        if a.env == "paper":          # lambda(t): regularisation curriculum (paper sec. III-C-2)
            env.reg_lambda = env.paper["lambda_max"] * min(1.0, step / max(1.0, a.reg_ramp * a.steps))

        fo = obs_b.reshape(-1, ac.obs_dim)
        fa = act_b.reshape(-1, ACT_DIM)
        fl = logp_b.reshape(-1)
        fadv = adv.reshape(-1)
        fret = ret.reshape(-1)
        idx_all = torch.randperm(fo.shape[0], device=dev)
        pl = vl = el = bl = 0.0
        nb = 0
        fv_old = val_b[:T].reshape(-1)
        kl_sum = 0.0
        for _ in range(a.epochs):
            for k in range(0, fo.shape[0], a.minibatch):
                mb = idx_all[k:k + a.minibatch]
                dist = ac.dist(fo[mb])
                lp = dist.log_prob(fa[mb]).sum(-1)
                ratio = (lp - fl[mb]).exp()
                l1 = ratio * fadv[mb]
                l2 = ratio.clamp(1 - a.clip, 1 + a.clip) * fadv[mb]
                lpi = -torch.min(l1, l2).mean()
                v_pred = ac.v(fo[mb]).squeeze(-1)
                if a.value_clip:
                    v_clip = fv_old[mb] + (v_pred - fv_old[mb]).clamp(-a.clip, a.clip)
                    lv = 0.5 * torch.max((v_pred - fret[mb]).pow(2), (v_clip - fret[mb]).pow(2)).mean()
                else:
                    lv = 0.5 * (v_pred - fret[mb]).pow(2).mean()
                lent = -dist.entropy().sum(-1).mean()
                if n_up < a.critic_warmup:
                    loss = a.vf_coef * lv
                else:
                    loss = lpi + a.vf_coef * lv + a.ent * lent
                if bcX is not None and n_up >= a.critic_warmup:
                    frac = min(1.0, step / max(1.0, a.bc_decay_steps))
                    bc_c = a.bc_coef + (a.bc_coef_min - a.bc_coef) * frac
                    jb = torch.randint(0, bcX.shape[0], (a.bc_batch,), device=dev)
                    lbc = (ac.pi(bcX[jb]) - bcY[jb]).pow(2).mean()
                    loss = loss + bc_c * lbc
                    bl += float(lbc)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), a.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    kl_sum += float((fl[mb] - lp).mean())
                pl += float(lpi); vl += float(lv); el += float(-lent); nb += 1
        if a.kl_target is not None and nb > 0:      # adaptive LR (rsl_rl-style schedule)
            kl = kl_sum / nb
            for g in opt.param_groups:
                if kl > 2.0 * a.kl_target:
                    g["lr"] = max(1e-5, g["lr"] / 1.5)
                elif kl < 0.5 * a.kl_target and kl > 0.0:
                    g["lr"] = min(a.lr_max, g["lr"] * 1.5)
        n_up += 1
        if n_up % 5 == 0:
            n_ep = max(1, ep["n"])
            rec = dict(step=step, updates=n_up, critic=vl / nb, actor=pl / nb,
                       alpha=0.0, entropy=el / nb, q_mean=float(val_b.mean()),
                       ep_ret=ep["ret"] / n_ep, ep_len=ep["len"] / n_ep,
                       success=ep["placed"] / n_ep, seal_rate=ep["sealed"] / n_ep,
                       succ_pnp=ep.get("placed_p", 0) / max(1, ep.get("n_p", 0)),
                       sps=step / (time.time() - t0), bc_mse=(bl / nb if bcX is not None else None),
                       comp={k: ep["comp"][i] / n_ep
                             for i, k in enumerate(env.RKEYS)},
                       lr=opt.param_groups[0]["lr"], reg_lambda=getattr(env, "reg_lambda", None),
                       drive_scale=getattr(env, "drive_scale", None))
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"[ppo] {step:>10,} | succ {rec['success']:.2%} pnp {rec['succ_pnp']:.2%} "
                  f"seal {rec['seal_rate']:.2%} ret {rec['ep_ret']:.2f} "
                  f"sps {rec['sps']:,.0f}", flush=True)
            ep = dict(n=0, ret=0.0, len=0.0, placed=0, sealed=0,
                      comp=np.zeros(len(env.RKEYS)))
            torch.save(dict(ac=ac.state_dict(), step=step),
                       os.path.join(a.out, "ac.pt"))
            if rec["success"] > best_succ:          # keep the peak (runs decay after it)
                best_succ = rec["success"]
                torch.save(dict(ac=ac.state_dict(), step=step, success=best_succ),
                           os.path.join(a.out, "best.pt"))
    torch.save(dict(ac=ac.state_dict(), step=step),
               os.path.join(a.out, "final.pt"))
    print("[ppo] done", flush=True)


if __name__ == "__main__":
    main()
