#!/usr/bin/env python3
"""qplan.common :: shared constants, env/policy builders and the chunk bookkeeping.

The chunk convention follows the plan: H = 5 decisions (0.5 s at CTRL_HZ = 10) = the control
polygon the Pi's B-spline reference is built from, so a chunk is exactly one "plan" the deployed
streaming controller can swallow without re-planning.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.dirname(HERE)
ROOT = os.path.dirname(RL)
for p in (RL, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

H = 5                       # decisions per action chunk
ACT_DIM = 7                 # 6 joint deltas + suction logit
OBS_DIM = 40                # q6 qd6 p_obj3 goal3 a_prev7 tcp3 axis3 rel3 lag6
CHUNK_DIM = H * ACT_DIM     # 35
CTRL_HZ = 10.0

PI_PATH = os.path.join(RL, "weights", "resid3_fast_best.pt")
SCENE = os.path.join(RL, "scenes", "box_med.xml")
DATA_ROOT = os.path.expanduser("~/pnp_rl/qplan")

# Reward the critic regresses (NOT the env's dense PPO reward).
GAMMA = 0.99
R_SUCC_W = 1.0              # +1 for `placed` at the terminal decision
R_DIST_W = 0.02             # -0.02 per cm of final object-goal distance (graded)
R_TIME_W = 0.1              # -0.1 per decision until the object is lifted AND at the goal

# HL-Gauss support.  succ: terminal reward in [-0.6, 1]; discounted it never leaves [-0.7, 1.05].
# time: -0.1 * sum_{k<150} gamma^k = -7.77 at worst, 0 once at the goal.
V_RANGE = dict(succ=(-1.0, 1.2), time=(-8.0, 0.5))
# ep_len 180 (the scripted place phase needs the headroom): -0.1 * sum_{k<180} 0.99^k = -8.33
V_RANGE_PLACE = dict(succ=(-1.0, 1.2), time=(-9.0, 0.5))
N_BINS = 51
HLG_SIGMA_BINS = 0.75       # Gaussian kernel sigma in units of the bin width

ENV_KW = dict(ep_len=150, grasp_shaping=True, obs_ee=True, reach_target="grasp", lift_dense=True,
              w_reach=0.5, w_track_c=4, w_track_f=8)


def make_env(nworld, device="cuda:0", seed=0, dr=True, obj_err=True, dq_max_deg=3.0,
             scene=SCENE, ep_len=150, **kw):
    """The twin exactly as `eval_residual` builds it, plus the tracker-error DR."""
    from env_paper import PaperPickEnv
    env = PaperPickEnv(nworld=nworld, device=device, xml=scene, dr=dr, drive="real", seed=seed,
                       obs_obj_err=bool(obj_err), **dict(ENV_KW, ep_len=ep_len), **kw)
    env.dq_max = float(np.radians(dq_max_deg))
    env.auto_reset = False
    return env


def load_pi(device="cuda:0", path=PI_PATH, obs_dim=OBS_DIM, arch="paper"):
    """The frozen deterministic policy (a fused 3-level residual stack)."""
    from ppo import build_frozen_policy
    return build_frozen_policy(path, obs_dim, arch=arch, device=device)


def seed_env(env, seed, device="cuda:0"):
    """Paired evaluation: identical object/goal/DR draws AND identical observation-noise draws
    for every policy, independent of how much torch randomness the policy itself consumes."""
    env.rng = np.random.default_rng(seed)
    env.noise_gen = torch.Generator(device=device)
    env.noise_gen.manual_seed(int(seed) + 12345)
    env.auto_reset = False
    env.reset(torch.ones(env.nworld, dtype=torch.bool, device=device))
    return env


def pi_chunk(pol, obs):
    """The CHEAP approximation of pi's chunk used everywhere as the proposal centre:
    evaluate pi once at o_t and repeat that action H times, (N, H, 7).

    The exact chunk would need pi unrolled through the simulator (a model), which the deployed
    controller does not have.  The approximation is what the planner can actually build, so the
    critic is trained on data that CONTAINS such open-loop chunks (collect.py --explore) rather
    than on closed-loop pi chunks alone.
    """
    a = pol(obs)
    return a[:, None, :].expand(-1, H, -1).contiguous()


def flat(c):
    return c.reshape(c.shape[0], -1) if c.dim() == 3 else c


def unflat(c):
    return c.reshape(c.shape[0], H, ACT_DIM) if c.dim() == 2 else c


def chunks_from_traj(act, t, hor=H):
    """act (E, T, A) -> (E, hor, A) starting at decision t, padded by repeating the last action."""
    E, T, A = act.shape
    idx = torch.clamp(torch.arange(t, t + hor, device=act.device), max=T - 1)
    return act[:, idx]
