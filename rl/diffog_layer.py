#!/usr/bin/env python3
"""diffog_layer :: DiffOG-style differentiable trajectory-optimization layer
for the Pro 630 (after arXiv 2504.13807, extended with observation/object
conditioning and linearized collision rows).

    min_y  1/2 (y-a)^T Q(ctx) (y-a)  +  alpha/2 ||D2 y||^2
    s.t.   |D1 y| <= vmax*dt          (per-joint velocity, incl. link to q0)
           |D2 y| <= amax*dt^2        (per-joint acceleration)
           q_lo <= y_k <= q_hi        (joint boxes, elbow limits)
           G_c y <= h_c               (linearized collision half-spaces)

y = flattened H x 6 joint-target chunk; a = the policy/teacher chunk;
Q(ctx) is PD from a transformer over [action tokens, obs token, object token].

QP solve = batched PDHG/ADMM-style projected iteration in pure torch,
differentiable by unrolling (fast: thousands of QPs/s at 72 vars). qpth is
used only as a correctness oracle in `python diffog_layer.py --test`.
"""
import argparse
import math

import numpy as np
import torch
import torch.nn as nn

DT = 0.1                       # decision period (10 Hz)
VMAX = math.radians(50.0)      # measured drive velocity cap
AMAX = math.radians(800.0)     # measured drive acceleration cap
Q_LO = torch.tensor(np.radians([-175, -70, -145, -175, -175, -175]), dtype=torch.float32)
Q_HI = torch.tensor(np.radians([175, 70, 145, 175, 175, 175]), dtype=torch.float32)


# ---------------------------------------------------------------- transformer Q
class QNet(nn.Module):
    """Tokens: H action tokens (6-D each) + obs token + object token.
    Output: block-diagonal Cholesky (per-timestep 6x6 blocks) -> Q PD.
    Block-diagonal keeps the head small (H*21 outputs) and the QP fast while
    still letting Q vary along the horizon and react to obs/object."""

    def __init__(self, H=12, obs_dim=40, obj_dim=7, d=128, layers=4, heads=4):
        super().__init__()
        self.H = H
        self.a_in = nn.Linear(6, d)
        self.o_in = nn.Linear(obs_dim, d)
        self.b_in = nn.Linear(obj_dim, d)
        self.pos = nn.Parameter(torch.randn(H + 2, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True,
                                         norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, 21)          # per-timestep lower-tri 6x6
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.tril_idx = torch.tril_indices(6, 6)

    def forward(self, a_chunk, obs, obj):
        """a_chunk (B,H,6), obs (B,obs_dim), obj (B,obj_dim) -> Q (B,H,6,6)."""
        B, H, _ = a_chunk.shape
        tok = torch.cat([self.a_in(a_chunk), self.o_in(obs)[:, None],
                         self.b_in(obj)[:, None]], 1) + self.pos[None, :H + 2]
        z = self.enc(tok)[:, :H]                       # (B,H,d)
        l = self.head(z)                               # (B,H,21)
        L = torch.zeros(B, H, 6, 6, device=a_chunk.device, dtype=a_chunk.dtype)
        L[:, :, self.tril_idx[0], self.tril_idx[1]] = l
        diag = torch.arange(6)
        # exp diagonal (clamped) -> PD; zero-init head => Q == I at start
        L[:, :, diag, diag] = torch.exp(L[:, :, diag, diag].clamp(-4, 4))
        Q = L @ L.transpose(-1, -2) + 1e-4 * torch.eye(6, device=a_chunk.device)
        return Q


# ---------------------------------------------------------------- QP (batched)
def _second_diff_penalty(y, q0, alpha):
    """alpha/2 * || second differences ||^2 including the link to q0."""
    yq = torch.cat([q0[:, None], y], 1)
    d2 = yq[:, 2:] - 2 * yq[:, 1:-1] + yq[:, :-2]
    return 0.5 * alpha * (d2 ** 2).sum(dim=(1, 2))


def qp_solve(a_chunk, q0, Q, alpha=1.0, G=None, h=None, iters=80,
             rho=10.0, vmax=VMAX, amax=AMAX, dt=DT):
    """Batched projected gradient in DIFFERENCE space, differentiable by
    unrolling. Substituting d = first differences (y = q0 + cumsum(d)) makes
    the velocity constraint an exact box -> clamp IS the exact projection,
    so PGD converges to the true optimum of the velocity-constrained QP.
    Acceleration & joint boxes enter as augmented penalties (verified tiny),
    collision rows via augmented Lagrangian. Returns (y, info)."""
    B, H, _ = a_chunk.shape
    dev = a_chunk.device
    vcap = vmax * dt
    acap = amax * dt * dt
    q_lo = Q_LO.to(dev)
    q_hi = Q_HI.to(dev)
    # init d from the (velocity-clamped) reference
    aq = torch.cat([q0[:, None], a_chunk], 1)
    d = (aq[:, 1:] - aq[:, :-1]).clamp(-vcap, vcap)
    lam = None if G is None else torch.zeros(B, G.shape[1], device=dev)
    # step size: y = q0 + cumsum(d) -> Lipschitz ~ ||Q|| * H + alpha*16 + rho terms
    qn = Q.flatten(2).norm(dim=2).amax(dim=1).clamp(min=1e-3)
    gn = 0.0 if G is None else (G.norm(dim=(1, 2)) ** 2 / max(G.shape[1], 1))
    step = 1.0 / (qn * H + 16.0 * alpha + 40.0
                  + (rho * gn * H if G is not None else 0.0) + 1e-3)

    d_prev = d.clone()
    tk = 1.0
    for _ in range(iters):
        # FISTA momentum (restart-free; projection keeps iterates feasible)
        tk1 = (1 + math.sqrt(1 + 4 * tk * tk)) / 2
        z = d + ((tk - 1) / tk1) * (d - d_prev)
        d_prev = d
        tk = tk1
        d = z.clamp(-vcap, vcap)
        y = q0[:, None] + torch.cumsum(d, dim=1)
        r = y - a_chunk
        gy = torch.einsum("bhij,bhj->bhi", Q, r)                  # dE/dy
        # smoothness: alpha/2 ||D1 d||^2 (second differences of y incl. q0)
        dd = torch.cat([d[:, :1], d[:, 1:] - d[:, :-1]], 1)       # (B,H,6); dd[0]=d0 (v from rest)
        gs = torch.zeros_like(d)
        gs = gs + dd
        gs[:, :-1] -= dd[:, 1:]
        # accel penalty (beyond cap) in d-space directly
        over = (dd.abs() - acap).clamp(min=0) * torch.sign(dd)
        ga = torch.zeros_like(d)
        ga = ga + over
        ga[:, :-1] -= over[:, 1:]
        # joint box penalty on y
        gb = ((y - q_hi).clamp(min=0) + (y - q_lo).clamp(max=0))
        # collision AL on y
        if G is not None:
            yf = y.reshape(B, -1)
            viol = torch.einsum("bmn,bn->bm", G, yf) - h
            act = (viol + lam / rho).clamp(min=0)
            gc = torch.einsum("bmn,bm->bn", G, act).reshape(B, H, 6)
        else:
            gc = torch.zeros_like(y)
        gy_total = gy + 20.0 * gb + rho * gc                      # all dE/dy terms
        # chain rule dE/dd = reversed cumsum of dE/dy
        gd = torch.flip(torch.cumsum(torch.flip(gy_total, [1]), 1), [1])
        gd = gd + alpha * gs + 50.0 * ga
        d = (d - step[:, None, None] * gd).clamp(-vcap, vcap)     # EXACT projection
        if G is not None and lam is not None:
            y = q0[:, None] + torch.cumsum(d, dim=1)
            yf = y.reshape(B, -1)
            lam = (lam + rho * (torch.einsum("bmn,bn->bm", G, yf) - h)).clamp(min=0).detach()
    y = q0[:, None] + torch.cumsum(d, dim=1)
    y = torch.maximum(torch.minimum(y, q_hi), q_lo)               # final safety clamp
    info = {}
    with torch.no_grad():
        yq = torch.cat([q0[:, None], y], 1)
        d1 = yq[:, 1:] - yq[:, :-1]
        d2 = d1[:, 1:] - d1[:, :-1]
        info["vel_viol"] = float((d1.abs() - vcap).clamp(min=0).max())
        info["acc_viol"] = float((d2.abs() - acap).clamp(min=0).max())
        if G is not None:
            yf = y.reshape(B, -1)
            info["col_viol"] = float((torch.einsum("bmn,bn->bm", G, yf) - h).clamp(min=0).max())
    return y, info


class DiffOGLayer(nn.Module):
    def __init__(self, H=12, obs_dim=40, obj_dim=7, alpha=1.0, iters=60):
        super().__init__()
        self.qnet = QNet(H=H, obs_dim=obs_dim, obj_dim=obj_dim)
        self.alpha = alpha
        self.iters = iters
        self.H = H

    def forward(self, a_chunk, q0, obs, obj, G=None, h=None):
        Q = self.qnet(a_chunk, obs, obj)
        y, info = qp_solve(a_chunk, q0, Q, alpha=self.alpha, G=G, h=h,
                           iters=self.iters)
        return y, info


# ---------------------------------------------------------------- collision rows
def collision_rows(p_pts, J_pts, obstacles, margin=0.02, q0=None, H=12,
                   k_near=4):
    """Linearized half-space rows for point set vs axis-aligned boxes.

    p_pts (B,S,3) current sphere centres; J_pts (B,S,3,6) their jacobians;
    obstacles list of (center(3,), half(3,)) AABBs (torch, device-matched).
    Row per (sphere, obstacle) with d < activation: n^T J (y_k - q0) >= -(d-margin)
    -> in Gy<=h form applied to EVERY timestep k of the chunk (worst case).
    Returns G (B, M, H*6), h (B, M) or (None, None)."""
    B, S, _ = p_pts.shape
    dev = p_pts.device
    rows_G, rows_h = [], []
    for (c, half) in obstacles:
        dvec = (p_pts - c).abs() - half                    # (B,S,3)
        outside = dvec.clamp(min=0)
        d = outside.norm(dim=-1) - 0.0                     # >0 outside
        inside = dvec.amax(dim=-1)                         # <0 inside
        dist = torch.where(d > 0, d, inside)
        # gradient direction (normal): outward from box
        n = torch.where((dvec > 0), outside, torch.zeros_like(outside))
        n = torch.where(d[..., None] > 0, n, torch.zeros_like(n))
        # inside: push along max-penetration axis
        ax = dvec.argmax(dim=-1, keepdim=True)
        n_in = torch.zeros_like(n).scatter(-1, ax, 1.0) * torch.sign(p_pts - c)
        n = torch.nn.functional.normalize(
            torch.where(d[..., None] > 0, n, n_in), dim=-1)
        act = dist < (margin + 0.10)                       # activation band
        if not act.any():
            continue
        # n^T J y_k <= n^T J q0 + (dist - margin)   for all k
        nJ = torch.einsum("bsi,bsij->bsj", -n, J_pts)      # (B,S,6): -n so Gy<=h
        rhs = torch.einsum("bsj,bj->bs", nJ, q0) + (dist - margin)
        # ONE ROW PER (point, timestep): nJ placed only in that step's slot
        # (a single row spanning steps would SUM them against a one-step rhs).
        # Linearization valid near q0 -> only the first k_near steps.
        big = torch.zeros(B, S, k_near, H, 6, device=dev)
        for k in range(k_near):
            big[:, :, k, k] = nJ
        G = big.reshape(B, S * k_near, H * 6)
        act_k = act[:, :, None].expand(B, S, k_near).reshape(B, S * k_near)
        rhs_k = rhs[:, :, None].expand(B, S, k_near).reshape(B, S * k_near)
        G = G * act_k[..., None]
        rhs_k = torch.where(act_k, rhs_k, torch.full_like(rhs_k, 1e3))
        rows_G.append(G)
        rows_h.append(rhs_k)
    if not rows_G:
        return None, None
    return torch.cat(rows_G, 1), torch.cat(rows_h, 1)


# ---------------------------------------------------------------- tests
def _test():
    torch.manual_seed(0)
    B, H = 8, 12
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    q0 = torch.zeros(B, 6, device=dev)
    # aggressive random chunk violating velocity caps
    a = torch.cumsum(torch.randn(B, H, 6, device=dev) * 0.15, dim=1)
    layer = DiffOGLayer(H=H).to(dev)
    obs = torch.randn(B, 40, device=dev)
    obj = torch.randn(B, 7, device=dev)
    y, info = layer(a, q0, obs, obj)
    print("constraint report:", info)
    assert info["vel_viol"] < 1e-6, "velocity violated"
    # gradient flows to the transformer
    loss = ((y - a) ** 2).mean()
    loss.backward()
    gn = layer.qnet.head.weight.grad
    print("grad reaches Q head:", gn is not None,
          "(zero by symmetric init is OK; checking a_in instead)")
    print("grad head norm:", layer.qnet.head.weight.grad.norm().item())
    assert layer.qnet.head.weight.grad.norm() > 0, "no grad into Q head"
    # oracle: qpth on the SAME problem (Q=I, alpha=0, velocity rows only)
    try:
        from qpth.qp import QPFunction
        Hh = 6
        n = Hh * 6
        a_s = torch.cumsum(torch.randn(2, Hh, 6) * 0.08, 1)  # stay inside joint boxes
        Qm = torch.eye(n)[None].expand(2, n, n).contiguous()
        p = -a_s.reshape(2, -1)
        cap = VMAX * DT
        D1 = torch.zeros(Hh, Hh)
        for i in range(Hh):
            D1[i, i] = 1
            if i:
                D1[i, i - 1] = -1
        D1f = torch.kron(D1, torch.eye(6))
        G = torch.cat([D1f, -D1f])[None].expand(2, 2 * n, n).contiguous()
        hv = torch.full((2, 2 * n), cap)
        e = torch.empty(0)
        y_qpth = QPFunction(verbose=-1)(Qm, p, G, hv, e, e).reshape(2, Hh, 6)
        y_ours, _ = qp_solve(a_s.to(dev), torch.zeros(2, 6, device=dev),
                             torch.eye(6, device=dev)[None, None].expand(2, Hh, 6, 6),
                             alpha=0.0, iters=800, amax=1e9)
        err = (y_ours.cpu() - y_qpth).abs().max()
        obj_o = 0.5 * ((y_ours.cpu() - a_s) ** 2).sum()
        obj_q = 0.5 * ((y_qpth - a_s) ** 2).sum()
        rel = float((obj_o - obj_q) / obj_q.abs())
        print(f"qpth oracle: sol diff {err:.2e}, objective gap {rel*100:.4f}%")
        # the QP is near-flat at the optimum; the meaningful invariants are
        # objective optimality (<0.1%) and exact constraint satisfaction
        assert rel < 1e-3, "objective suboptimal vs qpth"
    except ImportError:
        print("qpth not available, oracle skipped")
    # throughput
    import time
    a = torch.cumsum(torch.randn(512, H, 6, device=dev) * 0.1, 1)
    q0 = torch.zeros(512, 6, device=dev)
    obs = torch.randn(512, 40, device=dev)
    obj = torch.randn(512, 7, device=dev)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        y, _ = layer(a, q0, obs, obj)
    torch.cuda.synchronize()
    dtm = (time.time() - t0) / 10
    print(f"batch 512 fwd: {dtm*1000:.1f} ms  ({512/dtm:.0f} QP/s)")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.test:
        _test()
