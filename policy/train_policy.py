"""train_policy :: DP / CFM / ACT on the pnp dataset, UMI-paper settings,
frozen vision encoders (features precomputed by extract_features.py).

UMI interface (Table A1 + released config):
  obs  = wrist RGB features x2 frames (img_obs_horizon 2)
       + proprioception x2: EE pose relative to CURRENT pose (pos3 + rot6d)
         and suction state  (low_dim_obs_horizon 2)
  act  = 16-step relative EE trajectory wrt pose at obs time:
         pos3 + rot6d + suction1 = 10D  (action_horizon 16, rotation_6d)
  freq = 10 Hz. Min-max normalization to [-1, 1].

Heads:
  dp   ConditionalUnet1D + DDIM (50 train timesteps, 16 inference), lr 3e-4,
       EMA (power 0.75)  -- UMI's diffusion policy
  cfm  same UNet backbone, conditional flow matching (v-target = x1 - x0,
       t ~ U[0,1]), 16 Euler steps at inference
  act  CVAE transformer (z=32, dim 512, 8 heads, 4 enc / 7 dec layers,
       L1 + 10*KL, lr 1e-4) with chunk = 16, cross-attending patch tokens

Run:  python train_policy.py --model dp --enc clip --gpu 0
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DS = os.path.expanduser("~/pnp_dataset")
OUT = os.path.expanduser("~/pnp_policies")
TA = 16                 # action horizon
TO = 2                  # obs horizon (images + proprio)
ADIM = 10               # pos3 + rot6d + suction
PDIM = 10               # proprio per frame: pos3 + rot6d + suction
VAL_SCENES = set(range(90, 100))


# ----------------------------------------------------------------- rotations
def quat_to_R(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))


def rot6d(R):
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


# ----------------------------------------------------------------- dataset
class UmiDataset(torch.utils.data.Dataset):
    def __init__(self, enc, split):
        self.samples = []
        self.tokens = {}
        self.rel = {}
        for sdir in sorted(glob.glob(os.path.join(DS, "scene_*"))):
            sid = int(sdir[-3:])
            if (sid in VAL_SCENES) != (split == "val"):
                continue
            tk = np.load(os.path.join(sdir, f"{enc}_tokens.npy"), mmap_mode="r")
            e = np.load(os.path.join(sdir, "eef.npz"))
            pos, quat, suc = e["pos"], e["quat"], e["suction"].astype(np.float32)
            T = min(len(tk), len(pos))
            R = quat_to_R(quat[:T])
            self.tokens[sid] = tk            # mmap: shared page cache, no RAM copy
            # precompute absolute pose mats for relative transforms
            self.rel[sid] = (pos[:T].astype(np.float32), R.astype(np.float32), suc[:T])
            for t in range(T - TA):
                self.samples.append((sid, t))
        self.dim = self.tokens[next(iter(self.tokens))].shape[-1]

    def __len__(self):
        return len(self.samples)

    def _rel_pose(self, pos, R, suc, t_ref, t):
        dp = R[t_ref].T @ (pos[t] - pos[t_ref])
        dR = R[t_ref].T @ R[t]
        return np.concatenate([dp, rot6d(dR), [suc[t]]]).astype(np.float32)

    def __getitem__(self, i):
        sid, t = self.samples[i]
        pos, R, suc = self.rel[sid]
        t0 = max(t - 1, 0)
        obs_tok = torch.from_numpy(np.stack([self.tokens[sid][t0],
                                             self.tokens[sid][t]]).copy())
        prop = np.stack([self._rel_pose(pos, R, suc, t, t0),
                         self._rel_pose(pos, R, suc, t, t)])                # [2,10]
        act = np.stack([self._rel_pose(pos, R, suc, t, t + 1 + k)
                        for k in range(TA)])                                # [16,10]
        return obs_tok, torch.from_numpy(prop), torch.from_numpy(act)


def fit_normalizer(ds):
    acts = np.stack([ds[i][2].numpy() for i in
                     np.random.default_rng(0).choice(len(ds), min(5000, len(ds)),
                                                     replace=False)])
    lo = acts.reshape(-1, ADIM).min(0) - 1e-4
    hi = acts.reshape(-1, ADIM).max(0) + 1e-4
    return torch.from_numpy(lo).float(), torch.from_numpy(hi).float()


# ----------------------------------------------------------------- UNet1D
class SinEmb(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, t):
        half = self.d // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t.float()[:, None] * f[None]
        return torch.cat([a.sin(), a.cos()], -1)


class ResBlock1D(nn.Module):
    def __init__(self, ci, co, cond):
        super().__init__()
        self.c1 = nn.Conv1d(ci, co, 3, padding=1)
        self.c2 = nn.Conv1d(co, co, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, co)
        self.norm2 = nn.GroupNorm(8, co)
        self.film = nn.Linear(cond, co * 2)
        self.skip = nn.Conv1d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x, c):
        h = F.mish(self.norm1(self.c1(x)))
        g, b = self.film(c).chunk(2, -1)
        h = h * (1 + g[..., None]) + b[..., None]
        h = F.mish(self.norm2(self.c2(h)))
        return h + self.skip(x)


class CondUnet1D(nn.Module):
    """ConditionalUnet1D over [B, ADIM, TA] with FiLM conditioning."""
    def __init__(self, cond_dim, dims=(256, 512, 1024)):
        super().__init__()
        self.temb = nn.Sequential(SinEmb(128), nn.Linear(128, 256), nn.Mish(),
                                  nn.Linear(256, 256))
        cd = 256 + cond_dim
        self.d1 = ResBlock1D(ADIM, dims[0], cd)
        self.d2 = ResBlock1D(dims[0], dims[1], cd)
        self.d3 = ResBlock1D(dims[1], dims[2], cd)
        self.pool = nn.AvgPool1d(2)
        self.mid = ResBlock1D(dims[2], dims[2], cd)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.u2 = ResBlock1D(dims[2] + dims[1], dims[1], cd)
        self.u1 = ResBlock1D(dims[1] + dims[0], dims[0], cd)
        self.out = nn.Conv1d(dims[0], ADIM, 1)

    def forward(self, x, t, cond):
        c = torch.cat([self.temb(t), cond], -1)
        h1 = self.d1(x, c)
        h2 = self.d2(self.pool(h1), c)
        h3 = self.d3(self.pool(h2), c)
        m = self.mid(h3, c)
        u = self.u2(torch.cat([self.up(m), h2], 1), c)
        u = self.u1(torch.cat([self.up(u), h1], 1), c)
        return self.out(u)


class ObsEncoder(nn.Module):
    """pooled frozen tokens (mean) x2 frames + proprio -> cond vector."""
    def __init__(self, feat_dim, out=512):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(feat_dim, 512), nn.Mish(),
                                  nn.Linear(512, 256))
        self.head = nn.Sequential(nn.Linear(256 * TO + PDIM * TO, out), nn.Mish(),
                                  nn.Linear(out, out))

    def forward(self, tok, prop):
        f = self.proj(tok.float().mean(2))          # [B,2,256]
        return self.head(torch.cat([f.flatten(1), prop.flatten(1)], -1))


# ----------------------------------------------------------------- ACT
class ACT(nn.Module):
    def __init__(self, feat_dim, d=512, z=32):
        super().__init__()
        self.tokp = nn.Linear(feat_dim, d)
        self.propp = nn.Linear(PDIM * TO, d)
        self.zdim = z
        enc_layer = nn.TransformerEncoderLayer(d, 8, 2048, 0.1, batch_first=True)
        self.cvae_enc = nn.TransformerEncoder(enc_layer, 4)
        self.act_in = nn.Linear(ADIM, d)
        self.z_head = nn.Linear(d, z * 2)
        self.zp = nn.Linear(z, d)
        self.mem_enc = nn.TransformerEncoder(enc_layer, 4)
        dec_layer = nn.TransformerDecoderLayer(d, 8, 2048, 0.1, batch_first=True)
        self.dec = nn.TransformerDecoder(dec_layer, 7)
        self.query = nn.Parameter(torch.randn(TA, d) * 0.02)
        self.out = nn.Linear(d, ADIM)
        self.cls = nn.Parameter(torch.randn(1, d) * 0.02)

    def forward(self, tok, prop, act=None):
        B = tok.shape[0]
        if act is not None:
            seq = torch.cat([self.cls.expand(B, 1, -1),
                             self.propp(prop.flatten(1))[:, None],
                             self.act_in(act)], 1)
            mu, logv = self.z_head(self.cvae_enc(seq)[:, 0]).chunk(2, -1)
            zs = mu + torch.randn_like(mu) * (0.5 * logv).exp()
        else:
            mu = logv = None
            zs = torch.zeros(B, self.zdim, device=tok.device)
        mem = torch.cat([self.tokp(tok.float().flatten(1, 2)),
                         self.propp(prop.flatten(1))[:, None],
                         self.zp(zs)[:, None]], 1)
        mem = self.mem_enc(mem)
        h = self.dec(self.query.expand(B, -1, -1), mem)
        return self.out(h), mu, logv


# ----------------------------------------------------------------- training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["dp", "cfm", "act"])
    ap.add_argument("--enc", required=True, choices=["clip", "dinov2", "lingbot"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--val_every", type=int, default=5_000)
    ap.add_argument("--ckpt_every", type=int, default=50_000)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    torch.manual_seed(0)
    run = f"{a.model}_{a.enc}"
    rdir = os.path.join(OUT, run)
    os.makedirs(rdir, exist_ok=True)

    tr = UmiDataset(a.enc, "train")
    va = UmiDataset(a.enc, "val")
    lo, hi = fit_normalizer(tr)
    lo_d, hi_d = lo.to(dev), hi.to(dev)
    norm = lambda x: ((x - lo_d) / (hi_d - lo_d)) * 2 - 1
    print(f"[{run}] train {len(tr)} val {len(va)} feat_dim {tr.dim}")
    dl = torch.utils.data.DataLoader(tr, batch_size=a.batch, shuffle=True,
                                     num_workers=2, drop_last=True,
                                     persistent_workers=True)
    dv = torch.utils.data.DataLoader(va, batch_size=a.batch, num_workers=2)

    if a.model in ("dp", "cfm"):
        obs_enc = ObsEncoder(tr.dim).to(dev)
        net = CondUnet1D(cond_dim=512).to(dev)
        params = list(obs_enc.parameters()) + list(net.parameters())
        opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-6)
        ema = {k: v.detach().clone() for k, v in
               list(net.state_dict().items()) + list(obs_enc.state_dict().items())}
        if a.model == "dp":
            from diffusers import DDPMScheduler, DDIMScheduler
            nsched = DDPMScheduler(num_train_timesteps=50,
                                   beta_schedule="squaredcos_cap_v2",
                                   prediction_type="epsilon")
    else:
        net = ACT(tr.dim).to(dev)
        obs_enc = None
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)

    steps_total = a.steps
    warm = 2000
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / warm) * 0.5 *
        (1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, steps_total - warm)))))

    scaler = torch.amp.GradScaler("cuda", enabled=a.amp)
    log = open(os.path.join(rdir, "log.jsonl"), "w")
    step = 0
    t_start = time.time()

    def save_ckpt(tag):
        ck = dict(model=a.model, enc=a.enc, lo=lo, hi=hi, step=step,
                  net=net.state_dict(),
                  obs_enc=obs_enc.state_dict() if obs_enc else None)
        if a.model in ("dp", "cfm"):
            ck["ema"] = ema
        torch.save(ck, os.path.join(rdir, f"ckpt_{tag}.pt"))

    def validate():
        with torch.no_grad():
            vl = []
            for tok, prop, act in dv:
                tok, prop = tok.to(dev), prop.to(dev)
                x1 = norm(act.to(dev)).transpose(1, 2)
                if a.model == "dp":
                    cond = obs_enc(tok, prop)
                    t = torch.randint(0, 50, (x1.shape[0],), device=dev)
                    eps = torch.randn_like(x1)
                    vl.append(float(F.mse_loss(net(nsched.add_noise(x1, eps, t),
                                                   t, cond), eps)))
                elif a.model == "cfm":
                    cond = obs_enc(tok, prop)
                    t = torch.rand(x1.shape[0], device=dev)
                    x0 = torch.randn_like(x1)
                    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
                    vl.append(float(F.mse_loss(net(xt, t * 50, cond), x1 - x0)))
                else:
                    pred, _, _ = net(tok, prop, act=None)
                    vl.append(float(F.l1_loss(pred, norm(act.to(dev)))))
        return float(np.mean(vl))

    ep = 0
    while step < a.steps:
        for tok, prop, act in dl:
            if step >= a.steps:
                break
            tok, prop = tok.to(dev), prop.to(dev)
            x1 = norm(act.to(dev)).transpose(1, 2)          # [B,ADIM,TA]
            if a.model == "dp":
                cond = obs_enc(tok, prop)
                t = torch.randint(0, 50, (x1.shape[0],), device=dev)
                eps = torch.randn_like(x1)
                xt = nsched.add_noise(x1, eps, t)
                loss = F.mse_loss(net(xt, t, cond), eps)
            elif a.model == "cfm":
                cond = obs_enc(tok, prop)
                t = torch.rand(x1.shape[0], device=dev)
                x0 = torch.randn_like(x1)
                xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
                loss = F.mse_loss(net(xt, t * 50, cond), x1 - x0)
            else:
                pred, mu, logv = net(tok, prop, act=norm(act.to(dev)))
                l1 = F.l1_loss(pred, norm(act.to(dev)))
                kl = -0.5 * (1 + logv - mu.pow(2) - logv.exp()).mean()
                loss = l1 + 10.0 * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net.parameters() if obs_enc is None else
                list(net.parameters()) + list(obs_enc.parameters()), 1.0)
            opt.step()
            sched.step()
            if a.model in ("dp", "cfm"):
                with torch.no_grad():
                    decay = min(0.9999, (1 + step) / (10 + step)) ** 0.75
                    cur = dict(list(net.state_dict().items())
                               + list(obs_enc.state_dict().items()))
                    for k in ema:
                        ema[k].mul_(decay).add_(cur[k].float(), alpha=1 - decay)
            if step % 200 == 0:
                log.write(json.dumps({"step": step, "loss": float(loss)}) + "\n")
                log.flush()
            if step > 0 and step % a.val_every == 0:
                v = validate()
                log.write(json.dumps({"step": step, "val_loss": v}) + "\n")
                log.flush()
                print(f"[{run}] step {step}/{a.steps} loss {float(loss):.4f} "
                      f"val {v:.4f} ({time.time()-t_start:.0f}s)", flush=True)
            if step > 0 and step % a.ckpt_every == 0:
                save_ckpt(f"{step//1000}k")
            step += 1
        ep += 1

    save_ckpt("final")
    print(f"[{run}] done: {step} steps, saved {rdir}/ckpt_final.pt")


if __name__ == "__main__":
    main()
