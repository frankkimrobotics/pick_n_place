"""Policy = ObsEncoder + a generative action head, with the 4 selectable presets.

Action = joint targets q(6) + suction(1) = 7-d over a horizon Tp (predict Tp, execute Ta).
Action representation:
  * absolute : head models the joint targets directly            (diffusion / cfm / drift presets)
  * relative : head models Δjoints w.r.t. the current q (UMI)     (umi preset); suction stays absolute.
Suction (dim 6) is modelled as a continuous value and thresholded at execution.

Presets (the four options the task asked for):
  diffusion → DDPM/DDIM head, absolute joints
  umi       → same diffusion head, Δjoint (relative) representation + wrist-centric obs
  cfm       → conditional flow-matching head
  drift     → stochastic-interpolant drift-field head
"""
import torch, torch.nn as nn
from nets import ObsEncoder, Conditional1DUNet
from heads import HEADS

NJ = 6  # joint dims; index 6 = suction


class Policy(nn.Module):
    def __init__(self, head_name="diffusion", action_dim=7, proprio_dim=7, Tp=16, To=2,
                 cams=("wrist_rgb", "fixed_rgb"), cond_dim=512, action_rep="absolute", head_kw=None):
        super().__init__()
        self.enc = ObsEncoder(cams, proprio_dim, To, cond_dim=cond_dim)
        net = Conditional1DUNet(action_dim, cond_dim)
        self.head = HEADS[head_name](net, **(head_kw or {}))
        self.Tp, self.A, self.To, self.rep = Tp, action_dim, To, action_rep
        # normalization (set from dataset stats via set_norm)
        self.register_buffer("a_mean", torch.zeros(action_dim))
        self.register_buffer("a_std", torch.ones(action_dim))

    def set_norm(self, mean, std):
        self.a_mean.copy_(torch.as_tensor(mean)); self.a_std.copy_(torch.as_tensor(std).clamp_min(1e-4))

    def _raw_target(self, action, obs):
        """Pre-normalization target: absolute→as-is; relative→Δjoints vs current q (suction absolute)."""
        tgt = action.clone()
        if self.rep == "relative":
            base = obs["proprio"][:, -1, :NJ]                 # current joints
            tgt[..., :NJ] = action[..., :NJ] - base[:, None, :]
        return tgt

    def _to_target(self, action, obs):
        return (self._raw_target(action, obs) - self.a_mean) / self.a_std

    @torch.no_grad()
    def fit_norm(self, loader, n_batches=30, dev="cpu"):
        """Set normalization from the ACTUAL (rep-aware) target statistics — critical for the
        relative (umi) preset, whose Δjoint targets have a different scale than absolute joints."""
        ts = []
        for i, (obs, act) in enumerate(loader):
            obs = {k: v.to(dev) for k, v in obs.items()}
            ts.append(self._raw_target(act.to(dev), obs).reshape(-1, self.A).cpu())
            if i + 1 >= n_batches:
                break
        T = torch.cat(ts, 0)
        self.set_norm(T.mean(0), T.std(0))
        return self

    def _from_pred(self, a, obs):
        a = a * self.a_std + self.a_mean
        if self.rep == "relative":
            base = obs["proprio"][:, -1, :NJ]
            a[..., :NJ] = a[..., :NJ] + base[:, None, :]
        return a

    def loss(self, obs, action):
        """obs: dict of (B,To,...); action: (B,Tp,7) absolute joint targets + suction."""
        cond = self.enc(obs)
        return self.head.loss(self._to_target(action, obs), cond)

    @torch.no_grad()
    def predict(self, obs):
        """→ (B,Tp,7): joint targets + continuous suction (threshold >0.5 at execution)."""
        cond = self.enc(obs)
        a = self.head.sample(cond, (self.Tp, self.A))
        return self._from_pred(a, obs)


PRESETS = {
    "diffusion": dict(head_name="diffusion", action_rep="absolute"),
    "umi":       dict(head_name="diffusion", action_rep="relative"),
    "cfm":       dict(head_name="flow",      action_rep="absolute"),
    "drift":     dict(head_name="drift",     action_rep="absolute"),
}


def build_policy(name, **kw):
    if name not in PRESETS:
        raise ValueError(f"unknown policy '{name}'; choose from {list(PRESETS)}")
    cfg = dict(PRESETS[name]); cfg.update(kw)
    return Policy(**cfg)
