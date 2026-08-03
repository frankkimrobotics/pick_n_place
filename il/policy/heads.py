"""Three generative action heads over the shared Conditional1DUNet backbone. Each exposes
`loss(x0, cond)` (x0 = ground-truth normalized action chunk) and `sample(cond, shape)`.

- DiffusionHead : DDPM training, DDIM sampling            (Diffusion Policy / UMI preset)
- FlowHead      : conditional flow matching (OT interpolant, ODE sampling)   (cfm preset)
- DriftHead     : stochastic-interpolant drift field b(x,t)=E[ẋ_t|x_t], ODE sampling  (drift preset)
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F


class DiffusionHead(nn.Module):
    def __init__(self, net, n_steps=100, sample_steps=16):
        super().__init__(); self.net = net; self.T = n_steps; self.S = sample_steps
        betas = torch.linspace(1e-4, 0.02, n_steps)
        self.register_buffer("acp", torch.cumprod(1.0 - betas, 0))

    def loss(self, x0, cond):
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        acp = self.acp[t][:, None, None]; z = torch.randn_like(x0)
        xt = acp.sqrt() * x0 + (1 - acp).sqrt() * z
        pred = self.net(xt, t.float() / self.T, cond)
        return F.mse_loss(pred, z)

    @torch.no_grad()
    def sample(self, cond, shape):
        B = cond.shape[0]; x = torch.randn(B, *shape, device=cond.device)
        steps = torch.linspace(self.T - 1, 0, self.S).long()
        for i, t in enumerate(steps):
            tt = torch.full((B,), float(int(t)) / self.T, device=cond.device)
            acp = self.acp[int(t)]; z = self.net(x, tt, cond)
            x0 = (x - (1 - acp).sqrt() * z) / acp.sqrt()
            if i < len(steps) - 1:
                acp2 = self.acp[int(steps[i + 1])]
                x = acp2.sqrt() * x0 + (1 - acp2).sqrt() * z         # DDIM, eta=0
            else:
                x = x0
        return x


class FlowHead(nn.Module):
    """Conditional flow matching: x_t=(1-t)x0+t x1, target velocity v=x1-x0; sample by ODE."""
    def __init__(self, net, sample_steps=10):
        super().__init__(); self.net = net; self.S = sample_steps

    def loss(self, x1, cond):
        B = x1.shape[0]; t = torch.rand(B, device=x1.device)
        x0 = torch.randn_like(x1); tt = t[:, None, None]
        xt = (1 - tt) * x0 + tt * x1
        return F.mse_loss(self.net(xt, t, cond), x1 - x0)

    @torch.no_grad()
    def sample(self, cond, shape):
        B = cond.shape[0]; x = torch.randn(B, *shape, device=cond.device); dt = 1.0 / self.S
        for i in range(self.S):
            t = torch.full((B,), i * dt, device=cond.device)
            x = x + dt * self.net(x, t, cond)
        return x


class DriftHead(nn.Module):
    """Stochastic-interpolant drift field (Yilun Du / Albergo-Vanden-Eijnden style).
    x_t = (1-t)x0 + t x1 + γ(t) z, with x0,z ~ N(0,I), x1=data, γ(t)=c·sin(πt) (bounded, γ(0)=γ(1)=0).
    Learns the drift b(x,t)=E[ẋ_t | x_t]; the MSE target is the realized ẋ_t=(x1-x0)+γ'(t)z. Sample by
    integrating the probability-flow ODE dx/dt=b from base noise to data. The γz term is what makes it a
    'drift field' distinct from plain flow matching (γ≡0)."""
    def __init__(self, net, sample_steps=16, gamma=0.3):
        super().__init__(); self.net = net; self.S = sample_steps; self.c = gamma

    def loss(self, x1, cond):
        B = x1.shape[0]; t = torch.rand(B, device=x1.device)
        x0 = torch.randn_like(x1); z = torch.randn_like(x1); tt = t[:, None, None]
        g = self.c * torch.sin(math.pi * tt); gd = self.c * math.pi * torch.cos(math.pi * tt)
        xt = (1 - tt) * x0 + tt * x1 + g * z
        target = (x1 - x0) + gd * z
        return F.mse_loss(self.net(xt, t, cond), target)

    @torch.no_grad()
    def sample(self, cond, shape):
        B = cond.shape[0]; x = torch.randn(B, *shape, device=cond.device); dt = 1.0 / self.S
        for i in range(self.S):
            t = torch.full((B,), i * dt, device=cond.device)
            x = x + dt * self.net(x, t, cond)
        return x


HEADS = {"diffusion": DiffusionHead, "flow": FlowHead, "drift": DriftHead}
