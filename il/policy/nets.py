"""Shared nets for the IL policies: observation encoder + conditional 1-D U-Net backbone.

Obs = { wrist_rgb, fixed_rgb : (B,To,3,H,W) ; proprio : (B,To,P=joints6+suction1) } → cond vector.
The Conditional1DUNet predicts a per-timestep vector over the action horizon (B,Tp,A), conditioned on
the obs cond + a scalar diffusion/flow time τ∈[0,1]. It is the shared backbone for the diffusion,
flow-matching, and drift heads (they differ only in training target + sampling).
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F


def sinusoidal_emb(t, dim):
    """t: (B,) in [0,1] → (B, dim) sinusoidal features."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / max(half - 1, 1))
    a = t.float()[:, None] * freqs[None] * 1000.0
    emb = torch.cat([torch.sin(a), torch.cos(a)], dim=-1)
    if emb.shape[-1] < dim:  # odd dim pad
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class SimpleImageEncoder(nn.Module):
    """Lightweight from-scratch CNN (no pretrained download). Swap for torchvision ResNet18 later."""
    def __init__(self, out=256, in_ch=3):
        super().__init__()
        chs = [32, 64, 128, 256]; layers = []; c = in_ch
        for oc in chs:
            layers += [nn.Conv2d(c, oc, 3, 2, 1), nn.GroupNorm(8, oc), nn.SiLU()]; c = oc
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(c, out)

    def forward(self, x):                       # x: (N,3,H,W)
        h = self.conv(x).mean(dim=(2, 3))       # global avg pool
        return self.head(h)


class ObsEncoder(nn.Module):
    """Per-camera CNN + proprio MLP, concatenated over the obs history To → cond vector."""
    def __init__(self, cams=("wrist_rgb", "fixed_rgb"), proprio_dim=7, To=2, img_feat=256, cond_dim=512):
        super().__init__()
        self.cams = list(cams); self.To = To; self.cond_dim = cond_dim
        self.img_enc = nn.ModuleDict({c: SimpleImageEncoder(img_feat) for c in self.cams})
        self.prop = nn.Sequential(nn.Linear(proprio_dim, 128), nn.SiLU(), nn.Linear(128, 128))
        feat = img_feat * len(self.cams) + 128
        self.merge = nn.Sequential(nn.Linear(feat * To, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))

    def forward(self, obs):
        per_t = []
        for t in range(self.To):
            parts = [self.img_enc[c](obs[c][:, t]) for c in self.cams]
            parts.append(self.prop(obs["proprio"][:, t]))
            per_t.append(torch.cat(parts, -1))
        return self.merge(torch.cat(per_t, -1))


class FiLM(nn.Module):
    def __init__(self, cond_dim, ch):
        super().__init__(); self.f = nn.Linear(cond_dim, ch * 2)
    def forward(self, x, g):                     # x:(B,C,T) g:(B,cond)
        s, b = self.f(g).chunk(2, -1)
        return x * (1 + s[..., None]) + b[..., None]


class Block1D(nn.Module):
    def __init__(self, ic, oc, cond_dim):
        super().__init__()
        self.c = nn.Conv1d(ic, oc, 3, padding=1); self.n = nn.GroupNorm(8, oc)
        self.film = FiLM(cond_dim, oc); self.act = nn.SiLU()
    def forward(self, x, g):
        return self.act(self.film(self.n(self.c(x)), g))


class Conditional1DUNet(nn.Module):
    """U-Net over the action-horizon axis (action dims = channels). Shared by all heads."""
    def __init__(self, action_dim, cond_dim, chs=(128, 256, 512)):
        super().__init__()
        self.tdim = cond_dim
        self.temb = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        gd = cond_dim * 2                        # global cond = [obs_cond ; time_emb]
        self.inp = nn.Conv1d(action_dim, chs[0], 1)
        self.down = nn.ModuleList(); self.pool = nn.ModuleList(); c = chs[0]
        for oc in chs:
            self.down.append(Block1D(c, oc, gd)); self.pool.append(nn.Conv1d(oc, oc, 3, 2, 1)); c = oc
        self.mid = Block1D(c, c, gd)
        self.up = nn.ModuleList(); self.ups = nn.ModuleList()
        for oc in reversed(chs):
            self.ups.append(nn.ConvTranspose1d(c, oc, 4, 2, 1)); self.up.append(Block1D(oc * 2, oc, gd)); c = oc
        self.out = nn.Conv1d(chs[0], action_dim, 1)

    def forward(self, x, tau, cond):             # x:(B,Tp,A) tau:(B,) cond:(B,cond_dim)
        g = torch.cat([cond, self.temb(sinusoidal_emb(tau, self.tdim))], -1)
        h = self.inp(x.transpose(1, 2)); skips = []
        for blk, pl in zip(self.down, self.pool):
            h = blk(h, g); skips.append(h); h = pl(h)
        h = self.mid(h, g)
        for ups, blk, sk in zip(self.ups, self.up, reversed(skips)):
            h = ups(h)
            if h.shape[-1] != sk.shape[-1]:
                h = F.interpolate(h, size=sk.shape[-1])
            h = blk(torch.cat([h, sk], 1), g)
        return self.out(h).transpose(1, 2)       # (B,Tp,A)
