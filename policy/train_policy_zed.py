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

DS = os.path.expanduser(os.environ.get("PNP_DS", "~/pnp_dataset_zed"))
OUT = os.path.expanduser("~/pnp_policies")
TA = 16                 # action horizon
TO = 2                  # obs horizon (images + proprio)
ADIM = 10               # pos3 + rot6d + suction
PDIM = 10               # proprio per frame: pos3 + rot6d + suction
VAL_SCENES = set(range(90, 100))
ACTION_MODE = "waypoint"        # "waypoint" | "bspline" (clamped cubic control points)
LAT_SHIFT = 0                   # bspline latency matching: chunk window starts at t+1+LAT_SHIFT
GOAL_NOISE = 0.0                # m: goal-input noise (robustness to graspmap/extrinsic error)
GOAL_SE3 = False                # 9-D SE3 goal (pos3 + rot6d) instead of pos3
GOAL_DROPOUT = 0.0              # optional conditioning: zero the goal with this prob
RANGER_AUX = False              # load per-frame tip raycast labels (traj.npz rf)
TRAJ_NOISE = 0.0                # m: displaced-state contraction augmentation


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
    def __init__(self, enc, split, use_goal=False):
        self.samples = []
        self.tokens = {}
        self.rel = {}
        self.goal = {}
        self.goal_q = {}
        self.rf = {}
        self.grasp = {}
        self.use_goal = use_goal
        max_sid = int(os.environ.get("PNP_MAX_SID", "100000"))
        _ex = []
        for part in os.environ.get("PNP_EXCLUDE", "").split(","):
            if "-" in part:
                a, b = part.split("-"); _ex.append((int(a), int(b)))
        for sdir in sorted(glob.glob(os.path.join(DS, "scene_*"))):
            sid = int(sdir[-3:])
            if sid >= max_sid or any(a <= sid <= b for a, b in _ex):
                continue
            if (sid in VAL_SCENES) != (split == "val"):
                continue
            pf, pfz = (os.path.join(sdir, f"{enc}_pool.npy"),
                       os.path.join(sdir, f"{enc}_zed_pool.npy"))
            if os.path.exists(pf) and os.path.exists(pfz):
                tk, tkz = np.load(pf, mmap_mode="r"), np.load(pfz, mmap_mode="r")
            else:
                tk = np.load(os.path.join(sdir, f"{enc}_tokens.npy"), mmap_mode="r")
                tkz = np.load(os.path.join(sdir, f"{enc}_zed_tokens.npy"), mmap_mode="r")
            e = np.load(os.path.join(sdir, "eef.npz"))
            pos, quat, suc = e["pos"], e["quat"], e["suction"].astype(np.float32)
            T = min(len(tk), len(tkz), len(pos))
            R = quat_to_R(quat[:T])
            self.tokens[sid] = (tk, tkz)     # mmap: shared page cache, no RAM copy
            if self.use_goal:
                import json as _json
                meta = _json.load(open(os.path.join(sdir, "meta.json")))
                g = np.zeros((T, 3), np.float32)
                gr = np.zeros((T, 1), np.float32)          # graspability of the active goal
                gq = np.zeros((T, 4), np.float32); gq[:, 0] = 1.0
                pks = [p for p in meta.get("picks", []) if p.get("grasp_nominal")]
                if pks:
                    g[:] = pks[0]["grasp_nominal"]["pos"]
                    gq[:] = pks[0]["grasp_nominal"].get("quat", [0.0, 0.7071, 0.7071, 0.0])
                    gr[:] = float(bool(pks[0].get("success")))
                    for p in pks:
                        f0 = max(0, min(T, p["frame_start"]))
                        f1 = max(0, min(T, p["frame_end"] if p["frame_end"] > 0 else T))
                        sealable = float(bool(p.get("success")))
                        g[f0:f1] = p["grasp_nominal"]["pos"]; gr[f0:f1] = sealable
                        gq[f0:f1] = p["grasp_nominal"].get("quat", [0.0, 0.7071, 0.7071, 0.0])
                        if f1 < T:
                            g[f1:] = p["grasp_nominal"]["pos"]; gr[f1:] = sealable
                            gq[f1:] = p["grasp_nominal"].get("quat", [0.0, 0.7071, 0.7071, 0.0])
                self.goal[sid] = g
                self.goal_q[sid] = gq
                self.grasp[sid] = gr
            if RANGER_AUX:
                tf = os.path.join(sdir, "traj.npz")
                self.rf[sid] = (np.load(tf)["rf"].astype(np.float32)
                                if os.path.exists(tf) else np.zeros(T, np.float32))
            # precompute absolute pose mats for relative transforms
            self.rel[sid] = (pos[:T].astype(np.float32), R.astype(np.float32), suc[:T])
            for t in range(T - TA - LAT_SHIFT):
                self.samples.append((sid, t))
        self.dim = self.tokens[next(iter(self.tokens))][0].shape[-1]

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
        tk, tkz = self.tokens[sid]
        obs_tok = torch.from_numpy(np.stack([tk[t0], tk[t], tkz[t0], tkz[t]]).copy())
        prop = np.stack([self._rel_pose(pos, R, suc, t, t0),
                         self._rel_pose(pos, R, suc, t, t)])                # [2,10]
        act = np.stack([self._rel_pose(pos, R, suc, t, t + 1 + LAT_SHIFT + k)
                        for k in range(TA)])                                # [16,10]
        delta = None
        if TRAJ_NOISE > 0:
            # displaced start: proprio/goal say the arm is delta off; labels converge back
            rng = np.random.default_rng((sid * 100003 + t) * 7 + 1)
            delta = rng.normal(0, TRAJ_NOISE, 3).astype(np.float32)
            rho_k = (0.5 ** np.arange(1, TA + 1, dtype=np.float32))[:, None]
            act = act.copy(); act[:, :3] -= rho_k * delta[None, :]
            prop = prop.copy(); prop[0, :3] -= delta       # pose(t-1) rel displaced pose(t)
        if ACTION_MODE == "bspline":
            # same tensor shape, different meaning: 16 clamped-cubic control points
            from spline_action import waypoints_to_ctrl
            act = waypoints_to_ctrl(act).astype(np.float32)
        if self.use_goal:
            grel = (R[t].T @ (self.goal[sid][t] - pos[t])).astype(np.float32)
            if delta is not None:
                grel = grel - delta                        # goal rel the displaced pose
            if GOAL_NOISE > 0:
                rng2 = np.random.default_rng((sid * 100003 + t) * 7 + 2)
                grel = grel + rng2.normal(0, GOAL_NOISE, 3).astype(np.float32)
            if GOAL_SE3:
                Rg = quat_to_R(self.goal_q[sid][t])
                grel = np.concatenate([grel, rot6d(R[t].T @ Rg)]).astype(np.float32)
            if GOAL_DROPOUT > 0:
                rng3 = np.random.default_rng((sid * 100003 + t) * 7 + 3)
                if rng3.random() < GOAL_DROPOUT:
                    grel = np.zeros_like(grel)             # optional conditioning
            out = [obs_tok, torch.from_numpy(prop), torch.from_numpy(grel),
                   torch.from_numpy(self.grasp[sid][t])]
            if RANGER_AUX:
                rfseq = self.rf[sid][t + 1:t + 1 + TA]
                rfseq = np.pad(rfseq, (0, TA - len(rfseq)), constant_values=rfseq[-1] if len(rfseq) else 0)
                out.append(torch.from_numpy(np.clip(rfseq, 0, 0.5).astype(np.float32) / 0.5))
            out.append(torch.from_numpy(act))
            return tuple(out)
        return obs_tok, torch.from_numpy(prop), torch.from_numpy(act)


def fit_normalizer(ds):
    acts = np.stack([ds[i][-1].numpy() for i in     # act is ALWAYS last (goal mode adds fields)
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



class DiTBlock1D(nn.Module):
    """AdaLN-Zero self-attention block; cond folded into the timestep embedding."""
    def __init__(self, d, heads=8):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.sa = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))
        self.ada = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, temb):
        s1, b1, g1, s2, b2, g2 = self.ada(temb)[:, None].chunk(6, -1)
        h = self.n1(x) * (1 + s1) + b1
        x = x + g1 * self.sa(h, h, h, need_weights=False)[0]
        h = self.n2(x) * (1 + s2) + b2
        return x + g2 * self.mlp(h)


class DiT1D(nn.Module):
    """Transformer denoiser over the TA action tokens (AdaLN-Zero).
    Same interface as CondUnet1D: forward(x [B,ADIM,TA], t, cond [B,cond_dim])."""
    def __init__(self, cond_dim=512, d=384, depth=6, heads=8):
        super().__init__()
        self.inp = nn.Linear(ADIM, d)
        self.pos = nn.Parameter(torch.randn(TA, d) * 0.02)
        self.temb = nn.Sequential(SinEmb(128), nn.Linear(128, d), nn.Mish(),
                                  nn.Linear(d, d))
        self.cemb = nn.Linear(cond_dim, d)
        self.blocks = nn.ModuleList([DiTBlock1D(d, heads) for _ in range(depth)])
        self.n_out = nn.LayerNorm(d, elementwise_affine=False)
        self.ada_out = nn.Linear(d, 2 * d)
        nn.init.zeros_(self.ada_out.weight)
        nn.init.zeros_(self.ada_out.bias)
        self.out = nn.Linear(d, ADIM)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, cond):
        h = self.inp(x.transpose(1, 2)) + self.pos[None]          # [B,TA,d]
        temb = self.temb(t) + self.cemb(cond)
        for blk in self.blocks:
            h = blk(h, temb)
        s, b = self.ada_out(temb)[:, None].chunk(2, -1)
        h = self.n_out(h) * (1 + s) + b
        return self.out(h).transpose(1, 2)                        # [B,ADIM,TA]


class MLPResBlock(nn.Module):
    def __init__(self, d, cond):
        super().__init__()
        self.l1 = nn.Linear(d, d)
        self.l2 = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.film = nn.Linear(cond, 2 * d)

    def forward(self, x, c):
        g, b = self.film(c).chunk(2, -1)
        h = F.mish(self.l1(self.norm(x))) * (1 + g) + b
        return x + self.l2(h)


class MLPDenoiser(nn.Module):
    """Flattened-chunk residual MLP denoiser (IBC/BESO-style baseline).
    Same interface: forward(x [B,ADIM,TA], t, cond)."""
    def __init__(self, cond_dim=512, d=1024, depth=6):
        super().__init__()
        self.inp = nn.Linear(ADIM * TA, d)
        self.temb = nn.Sequential(SinEmb(128), nn.Linear(128, 256), nn.Mish(),
                                  nn.Linear(256, 256))
        self.blocks = nn.ModuleList([MLPResBlock(d, 256 + cond_dim)
                                     for _ in range(depth)])
        self.out = nn.Linear(d, ADIM * TA)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, cond):
        B = x.shape[0]
        c = torch.cat([self.temb(t), cond], -1)
        h = self.inp(x.reshape(B, -1))
        for blk in self.blocks:
            h = blk(h, c)
        return self.out(h).reshape(B, ADIM, TA)


class ODEField(nn.Module):
    """ODE Policy (arXiv 2611.01729): learn the DYNAMICS of the action, not the action.
    An autonomous field a' = f(a, cond) over the 9-D pose part of the action state,
    integrated in physical chunk time (RK2) to produce the waypoint path; trained by
    matching the INTEGRATED path to the demo chunk, with the paper's contractive
    start-perturbation (targets decay back to the demo at rate rho) so off-path states
    flow back to the demonstration manifold -- the exact failure mode of our real
    hover-drift runs. Binary suction is a separate per-step classification head
    (the paper: discrete actions should not be ODE states)."""
    PDIM9 = 9

    def __init__(self, cond_dim=512, d=512, depth=4, H=8):
        super().__init__()
        self.H = H
        # +1: chunk phase k/H. An AUTONOMOUS field cannot represent motion reversals
        # (descend->press->lift visits the same pose with opposite velocities); our 1.6 s
        # chunks reverse, the paper's 0.2-0.4 s chunks don't. Phase input removes the
        # impossibility; H=8 (matching the executor's n_exec=8 replan cadence) halves
        # integration depth on top.
        self.inp = nn.Linear(self.PDIM9 + 1, d)
        self.films = nn.ModuleList([nn.Linear(cond_dim, 2 * d) for _ in range(depth)])
        self.layers = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.SiLU(),
                                                   nn.Linear(d, d)) for _ in range(depth)])
        self.out = nn.Linear(d, self.PDIM9)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.vmax = 0.5          # bound on the per-step increment (normalized units):
                                 # tanh-scaled field makes rollout divergence impossible
                                 # (unbounded field blew up at step 31.6k, loss 0.006->44)
        self.suc_head = nn.Sequential(nn.Linear(cond_dim, 256), nn.SiLU(), nn.Linear(256, H))

    def f(self, a, phase, cond):
        h = self.inp(torch.cat([a, phase.expand(a.shape[0], 1)], -1))
        for film, lay in zip(self.films, self.layers):
            g, b = film(cond).chunk(2, -1)
            h = h + lay(h) * (1 + g) + b
        return torch.tanh(self.out(h)) * self.vmax

    def rollout(self, a0, cond, H=None, sub=2):
        """integrate H chunk steps from a0 (RK2 midpoint, `sub` substeps/step) -> [B,H,9]"""
        H = H or self.H
        a = a0
        traj = []
        dt = 1.0 / sub
        for _k in range(H):
            ph = torch.tensor([(_k + 0.5) / H], device=a.device, dtype=a.dtype)
            for _s in range(sub):
                k1 = self.f(a, ph, cond)
                k2 = self.f(a + 0.5 * dt * k1, ph, cond)
                a = a + dt * k2
            a = a.clamp(-1.2, 1.2)   # excursion cap: in-domain targets live in [-1,1], so this
                                     # only binds under OOD conditioning (stops integration from
                                     # running to the bound: the 350-1000 mm real-frame endpoints)
            traj.append(a)
        return torch.stack(traj, 1)


class ObsEncoder(nn.Module):
    """pooled frozen tokens (mean) x2 frames + proprio [+ goal] -> cond vector."""
    def __init__(self, feat_dim, out=512, use_goal=False, goal_dim=3):
        super().__init__()
        self.use_goal = use_goal
        self.goal_dim = goal_dim
        self.proj = nn.Sequential(nn.Linear(feat_dim, 512), nn.Mish(),
                                  nn.Linear(512, 256))
        self.head = nn.Sequential(nn.Linear(256 * 2 * TO + PDIM * TO + (goal_dim if use_goal else 0), out),
                                  nn.Mish(), nn.Linear(out, out))
        self.grasp_head = nn.Sequential(nn.Linear(out, 128), nn.Mish(), nn.Linear(128, 1))
        # privileged-ranger distillation: predict the tip raycast distance over the chunk
        self.rf_head = nn.Sequential(nn.Linear(out, 128), nn.Mish(), nn.Linear(128, TA))

    def forward(self, tok, prop, goal=None):
        t = tok.float()
        if t.dim() == 4:
            t = t.mean(2)                            # pool patch tokens
        f = self.proj(t)                             # [B,NTOK,256]
        parts = [f.flatten(1), prop.flatten(1)]
        if self.use_goal:
            parts.append(goal)
        return self.head(torch.cat(parts, -1))

    def graspability(self, cond):
        """logit: can the commanded goal actually be sealed?"""
        return self.grasp_head(cond)


# ----------------------------------------------------------------- ACT
class ACT(nn.Module):
    def __init__(self, feat_dim, d=512, z=32, use_goal=False):
        super().__init__()
        self.use_goal = use_goal
        self.tokp = nn.Linear(feat_dim, d)
        self.propp = nn.Linear(PDIM * TO + (3 if use_goal else 0), d)
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

    def forward(self, tok, prop, act=None, goal=None):
        B = tok.shape[0]
        pv = prop.flatten(1)
        if self.use_goal:
            pv = torch.cat([pv, goal], -1)
        if act is not None:
            seq = torch.cat([self.cls.expand(B, 1, -1),
                             self.propp(pv)[:, None],
                             self.act_in(act)], 1)
            mu, logv = self.z_head(self.cvae_enc(seq)[:, 0]).chunk(2, -1)
            zs = mu + torch.randn_like(mu) * (0.5 * logv).exp()
        else:
            mu = logv = None
            zs = torch.zeros(B, self.zdim, device=tok.device)
        tokf = tok.float()
        if tokf.dim() == 3:
            tokf = tokf[:, :, None, :]
        mem = torch.cat([self.tokp(tokf.flatten(1, 2)),
                         self.propp(pv)[:, None],
                         self.zp(zs)[:, None]], 1)
        mem = self.mem_enc(mem)
        h = self.dec(self.query.expand(B, -1, -1), mem)
        return self.out(h), mu, logv


# ----------------------------------------------------------------- training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["dp", "cfm", "act", "ode"])
    ap.add_argument("--ode_sigma", type=float, default=0.15, help="contractive start-perturbation std (normalized units)")
    ap.add_argument("--ode_rho", type=float, default=0.5, help="perturbation decay rate toward the demo path")
    ap.add_argument("--arch", default="unet", choices=["unet", "dit", "mlp"],
                    help="denoiser architecture for dp/cfm (ACT ignores this)")
    ap.add_argument("--tag", default="", help="suffix for the run dir")
    ap.add_argument("--goal", action="store_true", help="condition on the active pick target (3-D rel grasp pos)")
    ap.add_argument("--graspability", action="store_true", help="auxiliary head: is the commanded goal sealable?")
    ap.add_argument("--action", default="waypoint", choices=["waypoint", "bspline"],
                    help="bspline: predict B-spline control points (ABPolicy-style splicing)")
    ap.add_argument("--goal_se3", action="store_true",
                    help="goal = SE3 grasp pose rel the cup (pos3+rot6d=9D) instead of pos3")
    ap.add_argument("--goal_dropout", type=float, default=0.0,
                    help="prob of zeroing the goal input per sample (makes goal conditioning optional)")
    ap.add_argument("--ranger_aux", type=float, default=0.0,
                    help="weight of the auxiliary head predicting the privileged tip raycast distance "
                         "over the chunk (distills height-above-surface into the features)")
    ap.add_argument("--depth", type=int, default=0, help="override denoiser depth (dit)")
    ap.add_argument("--goal_noise", type=float, default=0.0,
                    help="m: noise the goal INPUT (labels unchanged) -> vision corrects goal error")
    ap.add_argument("--traj_noise", type=float, default=0.0,
                    help="m: displaced-state augmentation: shift proprio+goal by -delta and bend the "
                         "label waypoints to converge back (act_k -= delta*rho^(k+1), rho=0.5)")
    ap.add_argument("--lat_shift", type=int, default=0,
                    help="bspline latency matching: fit the chunk to t+1+L..t+16+L (L decisions of "
                         "inference latency, ~1 = 100 ms at 10 Hz) so the curve covers the window "
                         "that actually executes; deploy with ab_refit(c0=handover state)")
    ap.add_argument("--dr", action="store_true", help="train-time noise: proprio 0.005, goal 0.010 m, tokens 1%%")
    ap.add_argument("--width", type=int, default=0,
                    help="override denoiser width (mlp: d, default 1024)")
    ap.add_argument("--enc", required=True, choices=["clip", "dinov2", "lingbot", "cradio"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--val_every", type=int, default=5_000)
    ap.add_argument("--ckpt_every", type=int, default=50_000)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    global ACTION_MODE, LAT_SHIFT, GOAL_NOISE, TRAJ_NOISE, GOAL_SE3, GOAL_DROPOUT, RANGER_AUX
    ACTION_MODE = a.action
    LAT_SHIFT = a.lat_shift
    GOAL_NOISE = a.goal_noise
    TRAJ_NOISE = a.traj_noise
    GOAL_SE3 = a.goal_se3
    GOAL_DROPOUT = a.goal_dropout
    RANGER_AUX = a.ranger_aux > 0
    dev = f"cuda:{a.gpu}"
    torch.manual_seed(0)
    run = f"{a.model}_{a.enc}" + ("" if a.arch == "unet" else f"_{a.arch}") + (f"_w{a.width}" if a.width else "") + (f"_{a.tag}" if a.tag else "")
    rdir = os.path.join(OUT, run)
    os.makedirs(rdir, exist_ok=True)

    tr = UmiDataset(a.enc, "train", use_goal=a.goal)
    va = UmiDataset(a.enc, "val", use_goal=a.goal)
    lo, hi = fit_normalizer(tr)
    lo_d, hi_d = lo.to(dev), hi.to(dev)
    norm = lambda x: ((x - lo_d) / (hi_d - lo_d)) * 2 - 1
    # ODE policy: integration starts at the identity action (dp=0, rot6d(I)) in normalized space
    _id10 = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0, 0.0], device=dev)
    a0_id = (norm(_id10)[None, :9]).float()                 # [1,9]
    print(f"[{run}] train {len(tr)} val {len(va)} feat_dim {tr.dim}")
    dl = torch.utils.data.DataLoader(tr, batch_size=a.batch, shuffle=True,
                                     num_workers=2, drop_last=True,
                                     persistent_workers=True)
    dv = torch.utils.data.DataLoader(va, batch_size=a.batch, num_workers=2)

    if a.model in ("dp", "cfm", "ode"):
        obs_enc = ObsEncoder(tr.dim, use_goal=a.goal, goal_dim=(9 if a.goal_se3 else 3)).to(dev)
        kw = {"d": a.width} if (a.width and a.arch == "mlp") else {}
        if a.arch == "dit":
            if a.width: kw["d"] = a.width
            if a.depth: kw["depth"] = a.depth
        if a.model == "ode":
            net = ODEField(cond_dim=512).to(dev)
        else:
            net = {"unet": CondUnet1D, "dit": DiT1D,
                   "mlp": MLPDenoiser}[a.arch](cond_dim=512, **kw).to(dev)
        params = list(obs_enc.parameters()) + list(net.parameters())
        opt = torch.optim.AdamW(params, lr=(1e-4 if a.model == "ode" else 3e-4),
                                weight_decay=1e-6)
        ema = {k: v.detach().clone() for k, v in
               list(net.state_dict().items()) + list(obs_enc.state_dict().items())}
        if a.model == "dp":
            from diffusers import DDPMScheduler, DDIMScheduler
            nsched = DDPMScheduler(num_train_timesteps=50,
                                   beta_schedule="squaredcos_cap_v2",
                                   prediction_type="epsilon")
    else:
        net = ACT(tr.dim, use_goal=a.goal).to(dev)
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
        ck = dict(model=a.model, enc=a.enc, arch=a.arch, goal=a.goal, graspability=a.graspability,
                  action=a.action, lo=lo, hi=hi, step=step,
                  ode_H=(net.H if a.model == "ode" else None), lat_shift=a.lat_shift,
                  goal_noise=a.goal_noise, traj_noise=a.traj_noise,
                  goal_se3=a.goal_se3, goal_dropout=a.goal_dropout, ranger_aux=a.ranger_aux,
                  depth=a.depth,
                  net=net.state_dict(),
                  obs_enc=obs_enc.state_dict() if obs_enc else None)
        if a.model in ("dp", "cfm", "ode"):
            ck["ema"] = ema
        torch.save(ck, os.path.join(rdir, f"ckpt_{tag}.pt"))

    def validate():
        with torch.no_grad():
            vl = []
            for batch in dv:
                if a.goal:
                    if RANGER_AUX: tok, prop, goal_, gr_, rf_, act = batch
                    else: tok, prop, goal_, gr_, act = batch
                    goal_ = goal_.to(dev)
                else: tok, prop, act = batch; goal_ = None
                tok, prop = tok.to(dev), prop.to(dev)
                x1 = norm(act.to(dev)).transpose(1, 2)
                if a.model == "dp":
                    cond = obs_enc(tok, prop, goal_)
                    t = torch.randint(0, 50, (x1.shape[0],), device=dev)
                    eps = torch.randn_like(x1)
                    vl.append(float(F.mse_loss(net(nsched.add_noise(x1, eps, t),
                                                   t, cond), eps)))
                elif a.model == "cfm":
                    cond = obs_enc(tok, prop, goal_)
                    t = torch.rand(x1.shape[0], device=dev)
                    x0 = torch.randn_like(x1)
                    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
                    vl.append(float(F.mse_loss(net(xt, t * 50, cond), x1 - x0)))
                elif a.model == "ode":
                    cond = obs_enc(tok, prop, goal_)
                    xr = norm(act.to(dev))
                    traj = net.rollout(a0_id.expand(xr.shape[0], -1), cond)
                    vl.append(float(F.mse_loss(traj, xr[:, :net.H, :9])))
                else:
                    pred, _, _ = net(tok, prop, act=None, goal=goal_)
                    vl.append(float(F.l1_loss(pred, norm(act.to(dev)))))
        return float(np.mean(vl))

    ep = 0
    while step < a.steps:
        for batch in dl:
            if step >= a.steps:
                break
            if a.goal:
                if RANGER_AUX: tok, prop, goal_, gr_, rf_, act = batch; rf_ = rf_.to(dev)
                else: tok, prop, goal_, gr_, act = batch; rf_ = None
                goal_ = goal_.to(dev); gr_ = gr_.to(dev)
            else: tok, prop, act = batch; goal_ = gr_ = rf_ = None
            tok, prop = tok.to(dev), prop.to(dev)
            if a.dr:
                prop = prop + torch.randn_like(prop) * 0.005
                tok = tok + torch.randn_like(tok.float()) * (0.01 * tok.float().std())
                if goal_ is not None:
                    goal_ = goal_ + torch.randn_like(goal_) * 0.010
            x1 = norm(act.to(dev)).transpose(1, 2)          # [B,ADIM,TA]
            if a.model == "dp":
                cond = obs_enc(tok, prop, goal_)
                t = torch.randint(0, 50, (x1.shape[0],), device=dev)
                eps = torch.randn_like(x1)
                xt = nsched.add_noise(x1, eps, t)
                loss = F.mse_loss(net(xt, t, cond), eps)
            elif a.model == "cfm":
                cond = obs_enc(tok, prop, goal_)
                t = torch.rand(x1.shape[0], device=dev)
                x0 = torch.randn_like(x1)
                xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
                loss = F.mse_loss(net(xt, t * 50, cond), x1 - x0)
            elif a.model == "ode":
                cond = obs_enc(tok, prop, goal_)
                xr = norm(act.to(dev))                      # [B,TA,10]
                tgt = xr[:, :net.H, :9]
                B = tgt.shape[0]
                # contractive perturbation: start off-path, supervise decay back to the demo
                delta = torch.randn(B, 9, device=dev) * a.ode_sigma
                decay = (a.ode_rho ** torch.arange(1, net.H + 1, device=dev,
                                                   dtype=torch.float32))[None, :, None]
                traj = net.rollout(a0_id.expand(B, -1) + delta, cond)
                # MSE learns the field fast and well (run-2 val 0.0038@5k ~= 15 mm) but
                # destabilizes once the field sharpens (~30k); Huber-0.1 starved the
                # zero-init out layer (40 mm max excursion = the zero-field basin).
                # So: MSE + bounded field + early stop before the instability zone.
                l_path = F.mse_loss(traj, tgt + decay * delta[:, None, :])
                suc_t = (act.to(dev)[:, :net.H, 9] > 0.5).float()
                l_suc = F.binary_cross_entropy_with_logits(net.suc_head(cond), suc_t)
                loss = l_path + 0.1 * l_suc
            else:
                pred, mu, logv = net(tok, prop, act=norm(act.to(dev)), goal=goal_)
                l1 = F.l1_loss(pred, norm(act.to(dev)))
                kl = -0.5 * (1 + logv - mu.pow(2) - logv.exp()).mean()
                loss = l1 + 10.0 * kl
            if a.graspability and gr_ is not None and obs_enc is not None:
                g_logit = obs_enc.graspability(obs_enc(tok, prop, goal_))
                loss = loss + 0.1 * F.binary_cross_entropy_with_logits(g_logit, gr_)
            if a.ranger_aux > 0 and rf_ is not None and obs_enc is not None:
                rf_pred = obs_enc.rf_head(obs_enc(tok, prop, goal_))
                loss = loss + a.ranger_aux * F.mse_loss(rf_pred, rf_)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net.parameters() if obs_enc is None else
                list(net.parameters()) + list(obs_enc.parameters()),
                0.5 if a.model == "ode" else 1.0)
            opt.step()
            sched.step()
            if a.model in ("dp", "cfm", "ode"):
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
