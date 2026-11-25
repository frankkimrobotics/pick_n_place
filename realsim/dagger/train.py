#!/usr/bin/env python3
"""realsim.dagger.train :: DAgger fine-tune of the 920M DiT on demos + expert-relabelled twin states.

Aggregated set = the original mega demos (scenes 980-999 and 90-99 held out) + labels.npz of every
collection run (realsim/dagger/label.py). Same CFM loss and train-time noise as the original run.
By default only the observation encoder (proj + head, 1.6 M weights) is trained and the result is
written as a weight patch for `eval_policy.py --patch`; `--net_prefix cemb,blocks.11` unfreezes
parts of the denoiser too.

  ~/miniconda3/envs/mj4/bin/python realsim/dagger/train.py --name d1 --gpu 1 \
      --runs ~/pnp_rollouts_realsim/dagger/c0_base ~/pnp_rollouts_realsim/dagger/c0_E3 \
      --val ~/pnp_rollouts_realsim/dagger/v0_dev
"""
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "policy")]
sys.path.append(os.path.expanduser("~/Desktop/lingbot-vision"))
os.environ["PNP_DS"] = os.path.expanduser("~/pnp_dataset_mega")
os.environ["PNP_EXCLUDE"] = "90-99,990-999"

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CK = os.path.expanduser("~/pnp_policies/cfm_cradio_dit_w2048_mega_dit2048/ckpt_final_ema.pt")


def load_labels(dirs, name="labels.npz"):
    from spline_action import waypoints_to_ctrl
    out = {k: [] for k in ("tok", "prop", "goal", "act", "wp", "scene", "phase")}
    for d in dirs:
        z = np.load(os.path.join(os.path.expanduser(d), name), allow_pickle=True)
        if not len(z["k"]):
            continue
        out["tok"].append(z["tok"].astype(np.float32)); out["prop"].append(z["prop"].astype(np.float32))
        out["goal"].append(z["goal"].astype(np.float32)); out["wp"].append(z["act"].astype(np.float32))
        out["act"].append(np.stack([waypoints_to_ctrl(w) for w in z["act"]]).astype(np.float32))
        out["scene"].append(z["scene"]); out["phase"].append(z["phase"])
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="collection run dirs with labels.npz")
    ap.add_argument("--val", nargs="*", default=[], help="held-out run dirs with labels.npz (offline metric only)")
    ap.add_argument("--init", default="", help="weight patch to start from (default: the unpatched ckpt)")
    ap.add_argument("--net_prefix", default="", help="comma list of denoiser parameter prefixes to unfreeze")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--frac", type=float, default=0.25, help="share of each batch drawn from the DAgger labels")
    ap.add_argument("--balance", type=int, default=0, help="1 = draw half of the DAgger share from pre-seal labels")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--demo_ok", type=int, default=0, help="1 = demo frames only from pick attempts that succeeded")
    ap.add_argument("--seed", type=int, default=-1, help="torch / sampling seed (-1 = unseeded)")
    ap.add_argument("--labels", default="labels.npz", help="labels file name inside each run dir")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast in the denoiser (Ampere+ only)")
    ap.add_argument("--out", default="~/pnp_rollouts_realsim/dagger/patches")
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    if a.seed >= 0:
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
    import real_rollout_zed as RZ
    RZ.DEV = dev
    import train_policy_zed as T
    from spline_action import ctrl_to_waypoints
    from realsim.wsp import apply_patch
    _tl = torch.load
    torch.load = lambda *x, **kw: _tl(*x, **{**kw, "map_location": "cpu"})
    pol = RZ.ZedPolicy(CK)
    torch.load = _tl
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    hp = {k: ck[k] for k in ("lat_shift", "goal_noise", "traj_noise", "goal_dropout", "tok_noise", "cam_dropout")}
    del ck
    T.ACTION_MODE, T.LAT_SHIFT, T.GOAL_SE3, T.RANGER_AUX = "bspline", int(hp["lat_shift"]), True, False
    T.GOAL_NOISE, T.TRAJ_NOISE, T.GOAL_DROPOUT = hp["goal_noise"], hp["traj_noise"], hp["goal_dropout"]
    T.VAL_SCENES = set(range(980, 990))
    if a.init:
        apply_patch(pol, os.path.expanduser(a.init))
    obs_enc, net, lo, hi = pol.obs_enc, pol.net, pol.lo, pol.hi
    norm = lambda x: ((x - lo) / (hi - lo)) * 2 - 1
    for p in list(net.parameters()) + list(obs_enc.parameters()):
        p.requires_grad_(False)
    train = {}
    for n, p in obs_enc.named_parameters():
        if n.startswith(("proj.", "head.")):
            train["obs_enc." + n] = p
    for pre in [s for s in a.net_prefix.split(",") if s]:
        for n, p in net.named_parameters():
            if n.startswith(pre):
                train["net." + n] = p
    for p in train.values():
        p.requires_grad_(True)
    w0 = {k: p.detach().clone() for k, p in train.items()}
    print(f"[dagger] training {len(train)} tensors, {sum(p.numel() for p in train.values()) / 1e6:.2f} M weights", flush=True)

    G = load_labels(a.runs, a.labels)
    V = load_labels(a.val) if a.val else None
    Gt = {k: torch.from_numpy(G[k]).to(dev) for k in ("tok", "prop", "goal", "act")}
    print(f"[dagger] labels {len(G['act'])} states from {len(set(G['scene']))} scenes; phases "
          f"{ {p: int((G['phase'] == p).sum()) for p in np.unique(G['phase'])} }"
          + (f"; val {len(V['act'])}" if V else ""), flush=True)
    i_pre = np.where(np.isin(G["phase"], ["approach", "press"]))[0]
    i_post = np.where(~np.isin(G["phase"], ["approach", "press"]))[0]
    tr = T.UmiDataset("cradio", "train", use_goal=True)
    va = T.UmiDataset("cradio", "val", use_goal=True)
    if a.demo_ok:                                                    # drop frames of failed attempts (press, no seal, back up)
        okf = {}
        for sid in {s for s, _ in tr.samples}:
            meta = json.load(open(os.path.join(os.environ["PNP_DS"], f"scene_{sid:03d}", "meta.json")))
            m_ = np.zeros(meta.get("frames", 0) + 1, bool)
            for p in meta.get("picks", []):
                if p.get("success"):
                    m_[max(0, p["frame_start"]):(p["frame_end"] if p["frame_end"] > 0 else len(m_))] = True
            okf[sid] = m_
        n0 = len(tr.samples)
        tr.samples = [(s, t_) for s, t_ in tr.samples if t_ < len(okf[s]) and okf[s][t_]]
        print(f"[dagger] demo_ok: kept {len(tr.samples)} of {n0} demo samples", flush=True)
    nb_g = int(round(a.batch * a.frac))
    dl = torch.utils.data.DataLoader(tr, batch_size=a.batch - nb_g, shuffle=True, num_workers=4, drop_last=True,
                                     persistent_workers=True)
    rs = np.random.default_rng(0)
    vi = rs.choice(len(va), 400, replace=False)
    vb = [va[int(i)] for i in vi]
    Dv = dict(tok=torch.stack([b[0] for b in vb]).float().to(dev), prop=torch.stack([b[1] for b in vb]).to(dev),
              goal=torch.stack([b[2] for b in vb]).to(dev), act=torch.stack([b[-1] for b in vb]).numpy())
    print(f"[dagger] demos: train {len(tr)} samples, val {len(va)} (scenes 980-989)", flush=True)

    @torch.no_grad()
    def sample(tok, prop, goal):
        """the deployed sampler (frozen noise, 16 Euler steps), batched -> waypoints [B,16,10]"""
        obs_enc.eval()
        out = []
        for i in range(0, len(tok), 256):
            cond = obs_enc(tok[i:i + 256], prop[i:i + 256], goal[i:i + 256])
            x = pol.noise0.expand(len(cond), -1, -1).clone()
            for j in range(16):
                x = x + net(x, torch.full((len(cond),), j / 16 * 50, device=dev), cond) / 16
            out.append(((x.transpose(1, 2) + 1) / 2 * (hi - lo) + lo).cpu().numpy())
        return np.stack([ctrl_to_waypoints(c) for c in np.concatenate(out)])

    def report(tag, ref=None):
        r = {}
        pd = sample(Dv["tok"], Dv["prop"], Dv["goal"])
        gt = np.stack([ctrl_to_waypoints(c) for c in Dv["act"]])
        r["demo_pos_mm"] = 1e3 * float(np.linalg.norm(pd[..., :3] - gt[..., :3], axis=-1).mean())
        r["demo_suc_err"] = float(((pd[..., 9] > .5) != (gt[..., 9] > .5)).mean())
        if ref is not None:
            r["demo_drift_mm"] = 1e3 * float(np.linalg.norm(pd[..., :3] - ref[..., :3], axis=-1).mean())
        for nm, S in (("fit", G), ("val", V)):
            if S is None:
                continue
            n = min(len(S["wp"]), 1500)
            ii = np.random.default_rng(1).choice(len(S["wp"]), n, replace=False)
            pg = sample(*[torch.from_numpy(S[k][ii]).to(dev) for k in ("tok", "prop", "goal")])
            wp = S["wp"][ii]
            r[f"{nm}_pos_mm"] = 1e3 * float(np.linalg.norm(pg[..., :3] - wp[..., :3], axis=-1).mean())
            r[f"{nm}_suc_err"] = float(((pg[..., 9] > .5) != (wp[..., 9] > .5)).mean())
            on = wp[:, :8, 9].min(1) > 0.5                              # expert keeps suction for the next 8 frames
            r[f"{nm}_keep"] = float((pg[on][:, :8, 9].min(1) > 0.5).mean()) if on.any() else float("nan")
        print(f"[dagger] {tag}: " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()), flush=True)
        return r, pd

    hist = {}
    hist["before"], ref = report("before")
    opt = torch.optim.AdamW(list(train.values()), lr=a.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    step, t0, run_l = 0, time.time(), []
    while step < a.steps:
        for tok, prop, goal_, gr_, act in dl:
            if step >= a.steps:
                break
            if a.balance and len(i_pre) and len(i_post):               # half pre-seal (approach / press), half post-seal
                gi_np = np.concatenate([rs.choice(i_pre, nb_g // 2), rs.choice(i_post, nb_g - nb_g // 2)])
            else:
                gi_np = rs.integers(0, len(G["act"]), nb_g)
            gi = torch.from_numpy(gi_np).to(dev)
            g_goal = Gt["goal"][gi].clone()
            g_goal[:, :3] += torch.randn_like(g_goal[:, :3]) * hp["goal_noise"]
            tok = torch.cat([tok.float().to(dev), Gt["tok"][gi]])
            prop = torch.cat([prop.to(dev), Gt["prop"][gi]])
            goal_ = torch.cat([goal_.to(dev), g_goal])
            act = torch.cat([act.to(dev), Gt["act"][gi]])
            tok = tok + torch.randn_like(tok) * (hp["tok_noise"] * tok.std())
            B = tok.shape[0]
            drop = torch.rand(B, device=dev) < hp["cam_dropout"]
            cam = torch.randint(0, 2, (B,), device=dev)
            mask = torch.ones_like(tok)
            for b in torch.nonzero(drop).flatten():
                mask[b, 2 * cam[b]:2 * cam[b] + 2] = 0.0
            tok = tok * mask
            obs_enc.train()
            x1 = norm(act).transpose(1, 2)
            cond = obs_enc(tok, prop, goal_)
            t = torch.rand(B, device=dev)
            x0 = torch.randn_like(x1)
            xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
                v = net(xt, t * 50, cond)
            per = ((v.float() - (x1 - x0)) ** 2).mean((1, 2))
            loss = per.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(train.values()), 1.0)
            opt.step()
            sched.step()
            run_l.append((float(per[:B - nb_g].mean()), float(per[B - nb_g:].mean())))
            step += 1
            if step % 250 == 0 or step == a.steps:
                l = np.mean(run_l[-250:], 0)
                print(f"[dagger] step {step}/{a.steps} loss demo {l[0]:.4f} dagger {l[1]:.4f} ({time.time() - t0:.0f} s)", flush=True)
            if step % 1000 == 0 and step < a.steps:
                hist[f"step{step}"], _ = report(f"step {step}", ref)
    hist["after"], _ = report("after", ref)
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    ops = [dict(key=k, kind="set", value=p.detach().cpu().clone(), index=None) for k, p in train.items()]
    rel = float(torch.sqrt(sum(((p.detach() - w0[k]) ** 2).sum() for k, p in train.items())
                           / sum((w0[k] ** 2).sum() for k in train)))
    torch.save(dict(ops=ops, note=f"dagger {a.name}: {len(G['act'])} labels, {a.steps} steps, |dW|/|W| {rel:.3f}"),
               os.path.join(out, a.name + ".pt"))
    json.dump(dict(args=vars(a), hist=hist, n_labels=int(len(G["act"])), rel_change=rel),
              open(os.path.join(out, a.name + ".json"), "w"), indent=1)
    print(f"[dagger] saved {out}/{a.name}.pt  |dW|/|W| {rel:.3f}", flush=True)


if __name__ == "__main__":
    main()
