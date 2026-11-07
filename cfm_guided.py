#!/usr/bin/env python3
"""cfm_guided :: faster + obstacle-steerable CFM sampling.

SPEED   capture the whole Euler integration in a CUDA graph (3.5x: 53 -> 15 ms on an
        A5000, bit-identical maths — the sampler is launch-overhead bound at batch 1).
STEER   classifier-style guidance on an ESDF. At every Euler step the velocity gets a
        repulsive term from the signed-distance field, so the SAMPLED chunk bends around
        obstacles instead of being filtered afterwards:

            x <- x + (v(x,t,cond) - lambda * dCost/dx) / n_steps
            Cost = sum_i softplus(margin - esdf(p_i)) over the 16 waypoints

ESDF is passed as a voxel grid (origin, voxel, grid[D,H,W]) so this stays env-agnostic:
nvblox (curobo2 env, Mapper.update_esdf + query_differentiable_layer) writes the grid,
the policy env only does a trilinear lookup, which is differentiable in plain torch.
"""
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------- ESDF
class VoxelESDF:
    """Trilinear, differentiable signed-distance lookup over a dense grid."""

    def __init__(self, grid, origin, voxel):
        self.g = grid[None, None]                       # [1,1,D,H,W] metres
        self.origin = torch.as_tensor(origin, dtype=torch.float32, device=grid.device)
        self.voxel = float(voxel)
        self.shape = torch.tensor(grid.shape[::-1], dtype=torch.float32,
                                  device=grid.device)   # (W,H,D)

    def __call__(self, p):
        """p [N,3] base-frame points -> signed distance [N] (outside grid = +big)."""
        idx = (p - self.origin) / self.voxel            # voxel coords (x,y,z)
        n = 2.0 * idx / (self.shape - 1) - 1.0          # -> [-1,1]
        samp = n[None, None, None]                      # [1,1,1,N,3]
        d = F.grid_sample(self.g, samp, mode="bilinear",
                          padding_mode="border", align_corners=True)
        d = d.reshape(-1)
        inside = ((n > -1) & (n < 1)).all(-1)
        return torch.where(inside, d, torch.full_like(d, 10.0))


# ----------------------------------------------------------------- guided sampler
def sample_guided(net, cond, p_t, R_t, lo, hi, esdf=None, margin=0.03,
                  lam=0.5, n_steps=16, ADIM=10, TA=16, generator=None):
    """Euler-integrate the flow; if an ESDF is given, steer the chunk clear of obstacles.

    p_t/R_t: current TCP pose (actions are relative to it).  Returns [TA, ADIM] unnormalised.
    """
    dev = cond.device
    x = torch.randn(1, ADIM, TA, device=dev, generator=generator)
    for k in range(n_steps):
        t = torch.full((1,), k / n_steps, device=dev)
        with torch.no_grad():
            v = net(x, t * 50, cond)
        if esdf is not None and lam > 0:
            xg = x.detach().requires_grad_(True)
            act = (xg.transpose(1, 2)[0] + 1) / 2 * (hi - lo) + lo   # unnormalise
            pts = p_t[None] + act[:, :3] @ R_t.T                      # waypoints -> base
            d = esdf(pts)
            cost = F.softplus((margin - d) * 50.0).sum() / 50.0       # smooth hinge
            (grad,) = torch.autograd.grad(cost, xg)
            v = v - lam * grad
        x = x + v / n_steps
    return (x.transpose(1, 2)[0] + 1) / 2 * (hi - lo) + lo


# ----------------------------------------------------------------- CUDA-graph speedup
class GraphedSampler:
    """CUDA-graph capture of the unguided sampler (static shapes, fixed step count)."""

    def __init__(self, net, cond_shape, n_steps=16, ADIM=10, TA=16, device="cuda:0"):
        self.n = n_steps
        self.x = torch.randn(1, ADIM, TA, device=device)
        self.cond = torch.zeros(cond_shape, device=device)
        self.ts = [torch.full((1,), k / n_steps, device=device) * 50 for k in range(n_steps)]
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                y = self.x.clone()
                for k in range(n_steps):
                    y = y + net(y, self.ts[k], self.cond) / n_steps
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            y = self.x.clone()
            for k in range(n_steps):
                y = y + net(y, self.ts[k], self.cond) / n_steps
            self.out = y

    def __call__(self, cond, generator=None):
        self.cond.copy_(cond)
        self.x.normal_(generator=generator)
        self.graph.replay()
        return self.out.clone()
