#!/usr/bin/env python3
"""qplan.proposals :: the N-chunk proposal set (plan section "Proposals").

Everything the critic is ever asked to score comes from here, so the collector uses the SAME
generator to inject open-loop proposal chunks into the replay buffer (`collect.py --explore`).
That is what keeps Q on-distribution: the paper's own stated limit is proposal support.

Families for N = 32 (scaled proportionally for N = 8 / 64):
  16 gaussian   cand[0] = pi's chunk itself (so pi is always in the support and the planner can
                never do worse than pi up to critic error); cand[1..] = chunk-correlated Gaussian
                noise on the six JOINT channels, sigma_shared = 0.15 (one draw per chunk) +
                sigma_step = 0.05 (per decision).  The suction channel is left at pi's value --
                a Gaussian on a logit whose SIGN is the command flips the cup at random; the
                structured family flips it deliberately instead.
  12 structured `ladder=False` (the plan as written): joint channels of the first four gaussian
                candidates scaled by 0.7 / 1.2 / 1.4.
                `ladder=True` (DEFAULT since the M1 iteration-2 diagnosis): a PURE scale ladder
                0.7 / 0.85 / 1.15 / 1.3 / 1.5 / 1.7 of pi's own chunk, plus 0.7 / 1.2 / 1.4 of
                two gaussian candidates.  Scaling a NOISY candidate mixes the speed axis with
                the noise axis, and `diag.py` showed the result: the x1.4 FAMILY scored much
                worse on Q_time than pi (-5.30 vs -5.00) while the pure ladder's optimum sat at
                1.15 -- i.e. the planner had no clean "same trajectory, faster" candidate to
                pick and t_goal never moved.  Either way the last two of the twelve flip the
                suction bit.
   4 safety     hold (zero joint deltas, pi's suction), 50 % slow, 25 % slow, hold with suction
                forced ON (stop without dropping a sealed object).
"""
import torch

from common import ACT_DIM, H, OBS_DIM  # noqa: E402


def family_sizes(n_cand):
    n_g = max(1, round(n_cand * 16 / 32))
    n_s = max(0, round(n_cand * 12 / 32))
    n_safe = n_cand - n_g - n_s
    if n_safe < 1:                       # keep at least one safety chunk
        n_safe, n_s = 1, n_cand - n_g - 1
    return n_g, n_s, n_safe


SAFE_SPECS = [(0.0, None), (0.5, None), (0.25, None), (0.0, +1.0),
              (0.75, None), (0.1, None), (0.5, +1.0), (0.0, -1.0)]
LADDER = [0.7, 0.85, 1.15, 1.3, 1.5, 1.7]      # pure scales of pi's chunk (ladder=True)
# `fine=True` replaces the GAUSSIAN family with these extra scales, so the whole proposal set is a
# deterministic function of pi's chunk.  Needed for the exported graph (no RNG inside it) and,
# measured, better than a FIXED noise table: one frozen perturbation direction re-applied at every
# decision integrates into a real trajectory bias (98.2 % -> 93.5 / 95.5 %), while fresh noise
# averages out.  See README "Speed head".
FINE = [0.5, 0.6, 0.8, 0.9, 1.05, 1.1, 1.25, 1.4, 0.95, 1.6, 0.75, 1.35, 0.65, 1.45, 0.55]


def noise_table(n_cand=32, sigma_shared=0.15, sigma_step=0.05, gen=None, device="cuda:0"):
    """A FIXED (n_g-1, H, 6) perturbation table for the gaussian family.

    Passed to `propose(noise=...)` it replaces the per-decision random draw, which is what makes
    the exported TensorRT graph deterministic (it has no RNG and no state): the same table is
    baked in as a constant.  Measured in the twin, fixing the table costs nothing -- the gaussian
    family's job is to give the softmax a few nearby directions, not to be a fresh sample."""
    n_g = family_sizes(n_cand)[0]
    if n_g < 2:
        return torch.zeros(0, H, 6, device=device)
    rn = (lambda sh: torch.randn(sh, device=device, generator=gen)) if gen is not None \
        else (lambda sh: torch.randn(sh, device=device))
    return rn((n_g - 1, 1, 6)) * sigma_shared + rn((n_g - 1, H, 6)) * sigma_step


def hash_tables(n_cand=32, obs_dim=OBS_DIM, gen=None, device="cuda:0", gain=5.0):
    """Fixed projections for `hash_noise` -- the gaussian family's RNG, replaced by a
    deterministic function of the observation.

    Why this exists: the exported graph cannot draw random numbers, but the gaussian family is
    NOT decoration.  Measured (README "Speed head"), freezing it kills the planner -- a fixed
    noise table costs 2.7-4.8 pp of success, replacing it with more ladder scales costs 2.7 pp,
    and collapsing it onto pi costs 1.7 pp AND the whole time gain.  Re-drawing it every decision
    is a 7-sample random search around pi, and the search is the point.  `sin(W z + b)` of the
    normalised observation reproduces the search: it is a pure function of o_t (so the graph stays
    deterministic and `f(obs)`), while `gain` is set so that one decision's change in o_t moves
    the phase by radians, which is what decorrelates consecutive draws."""
    n_g = family_sizes(n_cand)[0]
    k = max(0, n_g - 1)
    rn = (lambda sh: torch.randn(sh, device=device, generator=gen)) if gen is not None \
        else (lambda sh: torch.randn(sh, device=device))
    ru = (lambda sh: torch.rand(sh, device=device, generator=gen)) if gen is not None \
        else (lambda sh: torch.rand(sh, device=device))
    sc = gain / (obs_dim ** 0.5)
    return dict(k=k, Ws=rn((k * 6, obs_dim)) * sc, bs=ru((k * 6,)) * 6.2831853,
                Wt=rn((k * H * 6, obs_dim)) * sc, bt=ru((k * H * 6,)) * 6.2831853)


def hash_noise(z, tab, sigma_shared=0.15, sigma_step=0.05):
    """z (N, obs_dim) normalised observation -> (N, n_g-1, H, 6) perturbations.

    sqrt(2) * sin(uniform phase) has unit variance, so the marginal spread matches the
    `sigma_shared` / `sigma_step` of the random version it replaces."""
    N, k = z.shape[0], tab["k"]
    sh = torch.sin(z @ tab["Ws"].t() + tab["bs"]).view(N, k, 1, 6) * (sigma_shared * 1.4142136)
    st = torch.sin(z @ tab["Wt"].t() + tab["bt"]).view(N, k, H, 6) * (sigma_step * 1.4142136)
    return sh + st


def propose(a_pi, n_cand=32, sigma_shared=0.15, sigma_step=0.05, gen=None, ladder=True,
            noise=None, fine=False):
    """a_pi (N, 7) -> candidate chunks (N, C, H, 7), clamped to [-1, 1].

    cand[:, 0] is exactly repeat(a_pi, H) (the pi chunk).
    `noise` (n_g-1, H, 6) or (N, n_g-1, H, 6): use this FIXED table instead of drawing.
    """
    N, dev = a_pi.shape[0], a_pi.device
    n_g, n_s, n_safe = family_sizes(n_cand)
    base = a_pi[:, None, None, :].expand(N, 1, H, ACT_DIM)

    def rn(shape):
        return torch.randn(shape, device=dev, generator=gen) if gen is not None \
            else torch.randn(shape, device=dev)

    # ---- gaussian family -------------------------------------------------
    g = base.expand(N, n_g, H, ACT_DIM).clone()
    if n_g > 1 and fine:
        sc = torch.tensor([FINE[k % len(FINE)] for k in range(n_g - 1)],
                          device=dev, dtype=g.dtype)
        g[:, 1:, :, :6] = g[:, 1:, :, :6] * sc[None, :, None, None]
    elif n_g > 1:
        eps = (noise if noise.dim() == 4 else noise[None]) if noise is not None else \
            (rn((N, n_g - 1, 1, 6)) * sigma_shared + rn((N, n_g - 1, H, 6)) * sigma_step)
        g[:, 1:, :, :6] = g[:, 1:, :, :6] + eps
    # ---- structured family ----------------------------------------------
    outs = [g]
    if n_s > 0:
        rep = []
        if ladder:
            for sc in LADDER:                             # PURE scales of pi's chunk
                c = base.clone()
                c[..., :6] = c[..., :6] * sc
                rep.append(c)
            n_noisy = max(1, (n_s - len(LADDER)) // 3)
            src = g[:, 1:1 + n_noisy]
        else:
            src = g[:, :max(1, n_s // 3)]                 # "the 16's first four"
        for sc in (0.7, 1.2, 1.4):
            c = src.clone()
            c[..., :6] = c[..., :6] * sc
            rep.append(c)
        s = torch.cat(rep, dim=1)
        while s.shape[1] < n_s:                           # N = 64: widen with more noisy scales
            src2 = g[:, 1:1 + max(1, n_s // 6)]
            rep2 = []
            for sc in (0.85, 1.1, 1.3):
                c = src2.clone()
                c[..., :6] = c[..., :6] * sc
                rep2.append(c)
            s = torch.cat([s] + rep2, dim=1)
        s = s[:, :n_s].clone()
        n_flip = min(2, s.shape[1])
        if n_flip:
            s[:, -n_flip:, :, 6] = -s[:, -n_flip:, :, 6]
        outs.append(s)
    # ---- safety family ---------------------------------------------------
    safe = []
    for k in range(n_safe):
        sc, suck = SAFE_SPECS[k % len(SAFE_SPECS)]
        c = base.clone()
        c[..., :6] = c[..., :6] * sc
        if suck is not None:
            c[..., 6] = suck
        safe.append(c)
    if safe:
        outs.append(torch.cat(safe, dim=1))
    return torch.cat(outs, dim=1).clamp(-1.0, 1.0)


def sample_one(a_pi, n_cand=32, gen=None, **kw):
    """One uniformly drawn candidate per world (the collector's exploration chunk)."""
    c = propose(a_pi, n_cand=n_cand, gen=gen, **kw)
    N, C = c.shape[0], c.shape[1]
    j = torch.randint(0, C, (N,), device=c.device, generator=gen) if gen is not None \
        else torch.randint(0, C, (N,), device=c.device)
    return c[torch.arange(N, device=c.device), j], j
