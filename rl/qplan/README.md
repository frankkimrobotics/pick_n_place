# qplan — Q-Planning on the Pro 630 twin (M0–M2)

Implements *Beyond Imitation: Self-Improving Robot Policies via Off-Policy Q-Planning*
(Giridhar et al., arXiv 2608.21204, <https://q-planning.github.io/>) on this repo's
mujoco_warp Pro 630 twin, following `policy/PLAN_QPLANNING.md`.

**The policy is frozen.** `rl/weights/resid3_fast_best.pt` (the fused 3-level residual stack,
40-D observation, `dq_max` 3 °/decision) is never trained here. Only a small critic
`Q_phi(obs, chunk)` is trained, and at run time it re-ranks N chunk proposals built around pi's
own chunk. Nothing in this directory ever talks to the robot.

```
common.py      env/policy builders, chunk bookkeeping, reward + HL-Gauss constants
proposals.py   the N-chunk proposal set (shared by the planner AND the collector)
collect.py     M0: roll pi / arbitrary proposals / the planner -> episode shards
critic.py      M1: Q_phi, HL-Gauss heads, replay buffer, H-step bootstrap + EMA target
planner.py     M1: the planner + the paired deterministic ablation
diag.py        per-family / scale-ladder diagnostic of what Q believes about a candidate
iterate.py     M2: deploy -> append -> retrain -> eval, K times, + the curve plot
```

## Design (and where it deviates from the plan)

**Chunk** H = 5 decisions (0.5 s at 10 Hz) = the control polygon the Pi's B-spline reference is
built from. A chunk is 5 × 7 = 35 numbers, flattened.

**Reward the critic regresses** (not the env's dense PPO reward):

| head | per-decision reward | support |
|---|---|---|
| `succ` | 0, except at the last decision: `placed - 0.02 · (final distance in cm)` | [-1.0, 1.2] |
| `time` | `-0.1` until the object is lifted AND within `succ_tol` of the goal | [-8.0, 0.5] |

γ = 0.99, H-step bootstrap `y = Σ_{i<H} γ^i r_{t+i} + (1-done) γ^H Q̄(o_{t+H}, c_{t+H})`,
EMA target τ = 0.005, Adam 3e-4, batch 4096. Both heads are HL-Gauss: 51 uniform bins, the
scalar target projected with a Gaussian kernel of σ = 0.75 bin widths (truncated-normal CDF
differences, renormalised), cross-entropy loss, `Q = Σ_b v_b softmax_b`. A unit test
(`γ`-decomposition and projection) is in the session notes: the Bellman split is exact to 5e-7
and HL-Gauss is unbiased to 1.2e-4.

**Bootstrap action** `c_{t+H}` is the chunk the BEHAVIOUR policy actually executed at `o_{t+H}`
(`--target_chunk exec`, default), not `repeat(pi(o_{t+H}), H)`. On the M0 data the behaviour
policy IS pi, so this is an exact SARSA backup; during the online iterations it is the planner,
which turns the loop into policy iteration (this is the intended direction of improvement).
`--target_chunk pi` implements the plan's literal version.

**Storage is by trajectory, not by transition.** A 150-decision episode holds 150 *overlapping*
chunked transitions; storing them flat would repeat each observation H+1 times. One episode is
151×40 + 150×7 + 150 values in fp16 = 14.6 kB, so a 4096-world batch is 59 MB on disk and
carries 614 400 chunked transitions. `critic.QBuffer` slices the chunks out on the GPU.

**The chunk of pi** is approximated as `repeat(pi(o_t), H)` — the exact chunk would need pi
unrolled through a model, which the deployed controller does not have. Because that
approximation is what the planner can actually build, the collector injects the *same* open-loop
chunks into the buffer (`--mode explore`), so the critic is trained where it is queried.

**Proposals** (N = 32; scaled proportionally for N = 8/16/64):

* 16 *gaussian* — `cand[0]` is pi's chunk itself (pi is always in the support); the rest add
  chunk-correlated noise on the six joint channels, σ_shared = 0.15 + σ_step = 0.05. The suction
  channel is left alone (a Gaussian on a logit whose *sign* is the command flips the cup at
  random); the structured family flips it deliberately.
* 12 *structured* — **deviation from the plan**: a PURE scale ladder
  `0.7 / 0.85 / 1.15 / 1.3 / 1.5 / 1.7 × pi's chunk`, plus `0.7 / 1.2 / 1.4 ×` two gaussian
  candidates, with the suction bit flipped on the last two. The plan scales *the 16's first
  four*, i.e. noisy candidates; §Results shows that mixing the speed axis with the noise axis
  is exactly why the first planner could not move `t_goal` at all.
* 4 *safety* — hold (zero joint deltas), 50 % slow, 25 % slow, hold with suction forced ON.

**Planner**, at every decision: score all N in one batched critic call, drop candidates with
`Q_succ < max − 0.1·|max|` (the plan's "0.9 × max", written so it still means "within 10 % of
the best" when the best value is negative), `w = softmax(Q_time / λ)` over the survivors,
`chunk = Σ_i w_i c_i`, execute that chunk's FIRST action and re-plan (receding horizon — the
deployed 10 Hz loop is unchanged). `--open_loop` executes all H actions instead.
Planner overhead at 1024 worlds is under 2 s per 150-decision episode (the env step dominates).

**New DR factor — tracker error** (`env_paper.PaperPickEnv(obs_obj_err=True)`, default OFF so
nothing else in the repo changes): a per-episode ±15 mm xy bias, ±5 mm per-step jitter and a
−6…+2 cm top error added to the OBSERVED object centre and grasp-relative vector only. The
physics, the seal test and the success metric all keep the true pose. See §Tracker-error DR for
why it is **not** part of the gated protocol.

## Usage

```bash
PY=~/miniconda3/envs/mjwarp/bin/python
cd ~/Desktop/2026/pick_and_place

# M0 — collect (GPU 1).  ~3 250 transitions/s => 200 k in 62 s
CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/collect.py --nworld 4096 --batches 2 --mode pi      --tag m0  --seed 0
CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/collect.py --nworld 4096 --batches 2 --mode explore --p_explore 0.15 --tag m0 --seed 5
CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/collect.py --nworld 4096 --batches 2 --mode scale --scale_lo 0.6 --scale_hi 1.6 --p_explore 0.08 --tag m0  --seed 30
CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/collect.py --nworld 4096 --batches 2 --mode scale --scale_lo 0.8 --scale_hi 1.5 --seg 15 --p_explore 0.08 --tag m0b --seed 40

# M1 — critic + paired ablation
CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/critic.py  --data ~/pnp_rl/qplan/data --steps 20000 --batch 4096 --out ~/pnp_rl/qplan/q0b
CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/planner.py --eval --q ~/pnp_rl/qplan/q0b/q.pt --nworld 1024 --which full  --out ~/pnp_rl/qplan/m1_ablation_it3.json
CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/planner.py --eval --q ~/pnp_rl/qplan/q0b/q.pt --nworld 1024 --which sweep --out ~/pnp_rl/qplan/m1_sweep.json
CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/diag.py    --q ~/pnp_rl/qplan/q0b/q.pt --nworld 512

# how many TD steps should Q get?  (the answer that explains M2)
CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/steps_curve.py --data ~/pnp_rl/qplan/data \
    --steps 2500 5000 10000 20000 40000 --nworld 1024

# M2 — the self-improvement loop (v5: the only recipe that does not decay)
CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/iterate.py --iters 6 --q ~/pnp_rl/qplan/q0b/q.pt \
    --data ~/pnp_rl/qplan/data --q_steps 20000 --fresh --fixed_frac 0.35 --patience 3 \
    --vel_margin 0 --n_cand 16 --lam 0.1 --out ~/pnp_rl/qplan/v5
CUDA_VISIBLE_DEVICES=1 $PY rl/qplan/plot_final.py          # -> ~/pnp_rl/qplan/iterations.png

# tracker-error DR and the scripted place phase (both OFF by default in env_paper)
CUDA_VISIBLE_DEVICES=0 $PY rl/qplan/collect.py --nworld 4096 --batches 2 --mode pi --obj_err 1 \
    --tag oe --seed 20 --out ~/pnp_rl/qplan/data_objerr
CUDA_VISIBLE_DEVICES=0 $PY rl/qplan/planner.py --eval --q ~/pnp_rl/qplan/q0b/q.pt --nworld 1024 \
    --obj_err 1 --which best --out ~/pnp_rl/qplan/m3_objerr_on.json
CUDA_VISIBLE_DEVICES=0 $PY rl/qplan/collect.py --nworld 4096 --batches 2 --ep_len 180 \
    --place_phase 1 --mode pi --tag p0 --seed 0 --out ~/pnp_rl/qplan/data_place
CUDA_VISIBLE_DEVICES=0 $PY rl/qplan/steps_curve.py --data ~/pnp_rl/qplan/data_place \
    --steps 10000 20000 40000 --nworld 1024 --ep_len 180 --place_phase 1
```

<!--RESULTS-->

## Results

All evaluations are paired: the same env seed, the same object/goal/DR draws and the same
observation-noise stream for every policy (`common.seed_env` gives the env its own
`torch.Generator`, so how much randomness a planner consumes cannot desynchronise it).
1024 episodes, 150 decisions, DR on, observation noise on, `dq_max` 3 deg, unless stated.
**Run-to-run spread** of the identical pi rollout is about +-0.4 pp on this metric (96.5-97.5 %
over seven runs) -- GPU reduction order is not deterministic and the twin is chaotic at contact.
Differences below ~0.8 pp are not meaningful.

### M0 - collector throughput (RTX 2080 Ti, GPU 1)

| mode | transitions | wall | rate | 200 k in | success of the collected episodes |
|---|---|---|---|---|---|
| `pi` | 614 400 / batch | 189 s | **3 251/s** | **62 s** | 96.4 / 97.2 % |
| `explore` (p 0.15) | 614 400 | 260 s | 2 364/s | 85 s | 18.0 % |
| `scale` (0.6-1.6) | 614 400 | 231 s | 2 642/s | 76 s | 41.8 % |
| `pi`, tracker DR | 614 400 | 219 s | 2 803/s | 71 s | 17.4 % |
| `pi`, place phase (ep_len 180) | 737 280 | 268 s | 2 751/s | 73 s | 85.2 % |

M0 target (200 k in under 10 min) is met with ~10x margin. The offline buffer used for M1 is
8 batches = **32 768 episodes / 4 915 200 chunked transitions**, 472 MB on disk.

### M1 - critic calibration (held-out episodes, 20 k TD steps)

| buffer | head | predicted | realised | bias | MAE | corr |
|---|---|---|---|---|---|---|
| pi + explore (`q0`) | succ | +0.1775 | +0.1780 | **-0.0005** | 0.141 | 0.898 |
| | time | -3.1164 | -2.9597 | -0.157 | 0.786 | 0.892 |
| + scale (`q0b`, used everywhere below) | succ | +0.1310 | +0.1229 | +0.0081 | 0.180 | 0.835 |
| | time | -3.3911 | -3.1920 | -0.199 | 1.006 | 0.856 |
| place phase (`q_place`) | succ | +0.1874 | +0.1931 | -0.0057 | 0.131 | 0.849 |
| | time | -5.5897 | -5.3371 | -0.253 | 1.167 | 0.765 |

Interpretable version: predicted success at the first decision (Q_succ / gamma^149) is
**26.4 %** against a realised **49.5 %** on `q0b`. That gap is not miscalibration of the head --
it is irreducible: at t = 0 the chunk does not reveal whether the *behaviour* policy will
deviate later, and the buffer mixes a 97 %-success and an 18 %-success behaviour. The unbiased
per-transition numbers above are the ones that matter for ranking.

### M1 - ablation (the table the plan asks for)

Frozen pi = `resid3_fast_best`, 96.88 % / t_goal 7.77 s / peak qd p90 43.5 deg/s.

| policy | success | seal | t_seal | t_goal | final d med | peak abs qd med/p90/max |
|---|---|---|---|---|---|---|
| **pi alone** | 96.88 % | 97.27 % | 5.23 s | 7.77 s | 1.0 cm | 38.7 / 43.5 / 55.2 |
| pi, blanket x1.3 (control) | 88.28 % | 89.45 % | 5.45 s | 7.26 s | 1.1 cm | 48.3 / 53.2 / 117.5 |
| planner weighted N=32 lam=0.05 | 97.17 % | 97.75 % | 4.91 s | 7.01 s | 1.0 cm | 43.0 / 48.9 / 62.4 |
| best-of-N N=32 | 97.36 % | 97.75 % | 4.79 s | **6.88 s** | 0.9 cm | 44.7 / 50.5 / 60.0 |
| planner N=8 | 97.75 % | 98.63 % | 5.54 s | 8.24 s | 1.0 cm | 39.3 / 45.5 / 54.2 |
| planner N=64 | 97.07 % | 97.75 % | 5.04 s | 7.14 s | 0.9 cm | 42.6 / 48.6 / 60.2 |
| planner lam=0.02 | 97.07 % | 97.56 % | 4.86 s | 6.95 s | 1.0 cm | 43.8 / 49.7 / 60.5 |
| planner lam=0.2 | 96.68 % | 97.27 % | 5.19 s | 7.45 s | 1.0 cm | 40.1 / 46.3 / 56.0 |
| planner lam=0.01 | 97.27 % | 97.66 % | 4.80 s | 6.90 s | 1.0 cm | 44.6 / 50.2 / 59.0 |
| planner N=32, full chunk open-loop | 94.43 % | 96.00 % | 5.18 s | 7.50 s | 1.5 cm | 43.1 / 49.6 / 59.7 |

The blanket-x1.3 control reproduces the number in the plan (96.9 -> 88.3 %): moving faster
everywhere costs 8.6 pp, and the whole point of the critic is to move faster only where it is
safe. Executing the full chunk open-loop costs 2.5 pp and 0.5 s -- receding horizon is worth
keeping.

`N`, `lambda` and the success band are one Pareto front, not independent knobs:

| N | lambda | succ_frac | success | t_goal | peak qd p90 |
|---|---|---|---|---|---|
| 32 | 0.05 | 0.97 | 95.21 % | 7.46 s | 48.0 |
| 32 | 0.05 | 0.95 | 96.29 % | 7.27 s | 48.5 |
| 32 | 0.05 | 0.90 | 97.17 % | 7.01 s | 48.9 |
| 32 | 0.05 | 0.80 | 97.75 % | 6.78 s | 49.7 |
| **16** | **0.10** | **0.90** | **99.02 %** | **7.14 s** | 48.5 |
| 16 | 0.20 | 0.90 | 99.02 % | 7.43 s | 47.8 |
| 16 | 0.40 | 0.90 | 98.73 % | 7.60 s | 47.2 |
| 16 | 1.00 | 0.90 | 98.83 % | 7.75 s | 46.9 |

Tightening the success band *hurts* (0.97 -> 95.2 %): with only 2-3 survivors the softmax is
forced to trust differences in Q_succ that are below the critic's own error, while at 0.90 pi's
own chunk survives 72-75 % of the time and Q_time does the work. **N = 16 beats N = 32**
because at N = 16 the twelve structured slots are exactly the pure scale ladder -- half the
gaussian noise candidates disappear and the survivors are the interpretable speed axis.

**Best planner: N = 16, lambda = 0.1, succ_frac = 0.9.** Reproduced on a second seed
(seed 1: pi 97.17 % -> 99.02 %, +1.85 pp; seed 1 with lambda 0.05: 97.46 % -> 98.93 %,
+1.47 pp), so the success gain is stable at +1.5 to +1.9 pp.

### M1 gate

| criterion | required | measured (N=16, lam=0.1) | |
|---|---|---|---|
| success vs pi | >= +1.0 pp | **+1.85 pp** (99.02 vs 97.17) | PASS |
| t_goal vs pi | >= -0.30 s | **-0.59 s** (7.14 vs 7.73) | PASS |
| peak qd p90 | <= pi p90 + 2 = 45.6 | 48.5 | **FAIL** |

Three iterations were needed to get there, and each one was diagnosed rather than guessed
(`diag.py` prints per-family Q_succ / Q_time / survival / weight, split before and after the
seal):

1. **Iteration 1** (buffer = pi + open-loop proposals): +0.39 pp success and **0.00 s** on
   t_goal. `diag.py`: Q_time ranked pi's own chunk best in every family (nominal -5.00 vs
   x1.4 -5.30 pre-seal) -- the time head had become a copy of the success head, because in that
   buffer every deviation is a deviation that FAILS (the explore episodes place 18 % of the
   time), so "chunk deviates" and "episode is slow" are the same feature.
2. **Iteration 2** (+ `--mode scale`: a blanket joint scale in [0.6, 1.6] held for an episode or
   a 10-15 decision segment, 42 % success -- episodes that mostly still succeed but finish at
   measurably different times). The scale ladder now has a real interior optimum: Q_time peaks at
   **1.15 x pi pre-seal** and **1.30 x post-seal**, and post-seal Q_succ is monotone *increasing*
   in the scale up to 1.7 (arriving sooner is also safer). Success unchanged, t_goal still flat.
3. **Iteration 3** (proposals: pure scale ladder instead of scaling noisy candidates). t_goal
   **-0.76 s** immediately. The plan scales "the 16's first four", three of which are gaussian
   draws, so the speed axis and the noise axis were entangled and the planner had no
   "same trajectory, faster" candidate to pick.

The peak-speed criterion cannot be met at the same time as the time criterion, and this is
structural, not a tuning failure. Two independent knobs trace the same front:

| | success | t_goal | peak qd p90 |
|---|---|---|---|
| pi | 96.78 % | 7.77 s | 43.5 |
| executed-chunk peak capped at 1.05 x pi's | 96.88 % | 8.13 s | **43.9** |
| capped at 1.15 x | 97.46 % | 7.65 s | 46.2 |
| capped at 1.30 x | 97.75 % | 7.40 s | 47.7 |
| uncapped | 98.44 % | 7.23 s | 48.6 |

and the lambda sweep above ends at 46.9 deg/s even when the planner is *slower* than pi. The
floor exists because the executed chunk is a weighted MIXTURE that changes more between
decisions than pi's own smooth output, and the accel-capped drive with the spline lead turns
that into peak speed. Nothing in Q prices joint speed: the fix is a third HL-Gauss head on a
per-decision speed-excess reward (the analogue of `env_warp._speed_penalty` / `--w_speed`),
not more tuning. **A hard velocity MASK on candidates is the wrong shape of fix** -- because
candidates are clamped to [-1,1], where pi saturates the fast ladder entries are identical to pi
and survive, while where pi is slow they are the ones dropped, so the surviving set is biased
slow and t_goal got *worse* than pi (8.39 s). Capping the executed chunk after the mixing keeps
the same guarantee without the bias.

### M2 - the self-improvement loop

Three loops were run. None beats **iteration 0** (the offline critic, no online data at all).

| loop | recipe | best iter | success (vs its paired pi) | t_goal (vs pi) | peak qd p90 |
|---|---|---|---|---|---|
| **iteration 0** | offline critic, 20 k TD steps, no online data | - | **98.83 %** (+1.76 pp) | **7.20 s** (-0.58 s) | **48.1** |
| v1 | the plan as written: raw union of the buffer, 10 k warm-start steps / iteration | 2 | 98.73 % (+1.76 pp) | 7.76 s (-0.03 s) | 51.6 |
| v3 | >= 35 % fixed-pool batches, 2.5 k steps @ lr 1e-4, velocity cap 1.2, best-checkpoint + early stop at 2 declines | 4 (stopped at 6) | 96.97 % (+0.20 pp) | 8.35 s (+0.57 s) | 47.9 |
| v5 | Q retrained FROM SCRATCH for 20 k steps on the grown buffer each iteration, early stop at 3 declines | 3 (stopped at 6) | 98.73 % (+2.15 pp) | 7.93 s (-0.14 s) | 52.2 |

v5's iteration sequence is 97.66 / 98.54 / **98.73** / 98.05 / 98.05 / 98.14 % -- it climbs back
toward iteration 0 and then flattens, which is what "the online episodes carry no new
information" looks like once the TD budget is held fixed.

v1 decays monotonically -- and it decays while the critic's calibration keeps *improving*:

| iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| success % | 98.44 | 98.73 | 98.14 | 97.17 | 97.56 | 96.88 | 96.58 | 96.97 | 95.61 | 96.58 |
| t_goal s | 7.59 | 7.76 | 7.98 | 8.33 | 8.26 | 8.45 | 8.79 | 8.73 | 8.99 | 8.84 |
| peak qd p90 | 51.3 | 51.6 | 51.8 | 51.8 | 52.1 | 51.8 | 52.0 | 52.2 | 52.2 | 51.9 |
| Q_succ bias | +.0060 | +.0085 | +.0048 | +.0032 | +.0025 | +.0038 | +.0032 | +.0018 | +.0023 | -.0001 |
| Q_succ corr | .851 | .859 | .868 | .870 | .874 | .878 | .879 | .881 | .884 | .884 |

**The cause is the TD budget, not the data.** Training the same critic on the same FIXED offline
buffer for different numbers of steps and evaluating the planner with each
(`steps_curve.py`, pi = 97.07 % / 7.78 s / 43.7):

| TD steps | success | t_goal | peak qd p90 | Q_succ corr |
|---|---|---|---|---|
| 2 500 | 26.76 % | 13.17 s | 50.4 | 0.810 |
| 5 000 | 90.04 % | 8.95 s | 51.0 | 0.822 |
| 10 000 | 95.12 % | 7.90 s | 49.9 | 0.827 |
| **20 000** | **98.83 %** | **7.20 s** | 48.1 | 0.835 |
| 40 000 | 97.56 % | 7.67 s | 45.6 | - |
| 60 000 | 95.31 % | 7.79 s | 44.5 | 0.835 |

Planning quality is an inverted U in the number of TD steps while calibration rises
monotonically, so a warm-started loop that adds 10 k steps per iteration walks straight off the
peak: iteration 1 of v1 is 30 k steps, iteration 10 is 120 k. As the critic converges it also
picks chunks closer and closer to pi (peak qd 50.4 -> 44.5, w(pi) 0.028 -> 0.194), which is the
same statement seen from the other side: **the useful signal for ranking chunks is an
early-training artefact of the value function, not its fixed point.**

Retraining from scratch with a 20 k budget removes the decay but does not produce improvement
either -- the online data is nearly redundant with the offline buffer. Training 20 k steps from
scratch on different pools:

| pool | success | vs pi |
|---|---|---|
| M0 only (32 768 episodes) | 98.83 % | +1.76 pp |
| M0 + tracker-error episodes | 98.34 % | +1.37 pp |
| M0 + four planner deployments (16 384 episodes) | 98.14 % | +1.56 pp |

**M2 gate: FAILED, and the loop plateaus at iteration 0.** success 99.0 % is at the gate,
t_goal 7.14 s against 6.5 s, peak qd 48.5 against 36 absolute / 45.6 relative. The most likely
cause, with the evidence above: pi's residual failures under full DR are *seal* failures
(seal 97.3 %, success 96.9 % -- essentially every non-placed episode is a non-sealed one), and a
seal that the drive DR makes impossible is not recoverable by re-weighting chunks of the same
policy. Q-planning can buy the 1.5-2 pp that better chunk timing is worth and ~8 % of cycle
time, and then it is out of headroom, so more deployments add episodes that carry no new
information. The paper's own stated limit -- proposal support -- is what binds: every proposal
is a scaled or jittered version of pi, so the reachable set of behaviours is a tube around pi.

### Tracker-error DR (the new factor)

`obs_obj_err` implements the plan's numbers exactly (+-15 mm xy bias, +-5 mm jitter, top
-6..+2 cm on the observed object only). It is **not** part of the gated protocol, because at
those magnitudes it is not a perturbation of the task, it is a different task (512 episodes):

| tracker DR | pi success | pi seal |
|---|---|---|
| off | 96.68 % | 97.27 % |
| xy bias + jitter only | 62.30 % | 65.82 % |
| top error only | 29.49 % | 50.59 % |
| full | 18.75 % | 35.16 % |
| half magnitudes | 51.95 % | 83.79 % |

The twin's seal test needs the cup tip within `SEAL_DIST` = 12 mm (DR'd 9.6-14.4 mm) of the
object-top centre, so a +-15 mm xy bias makes a seal *geometrically impossible* on a large share
of episodes -- the tracker DR is a different task, not a harder version of this one, which is
why the gated protocol runs with it off.

Two evaluations under it, and they disagree in an instructive way (1024 episodes, DR on):

| critic | pi | planner N=16 lam=0.1 | best-of-N N=16 | planner, exec peak <= 1.2 x pi |
|---|---|---|---|---|
| trained ON 16 384 tracker-DR episodes | 17.29 % | **10.16 %** | 9.28 % | - |
| the clean critic `q0b` (no tracker-DR data) | 16.70 % | **19.82 %** (+3.1 pp) | **22.66 %** (+6.0 pp) | 18.46 % |

Training the critic on tracker-DR data makes it *worse than useless*: the bias is unobservable
from o_t, so the same observation carries wildly different outcomes and the head simply absorbs
the variance -- predicted success at t = 0 is **-40 %** against a realised 12.9 %, and the
planner it drives falls below pi. The clean critic, which scores the observed state as if it
were true, still buys +3.1 pp (weighted) and +6.0 pp (best-of-N, where aggressive selection now
pays because pi's approach is the thing failing). So Q-planning is not helpless under an
unobservable state error -- but the data that contains the error is poison for the critic, and
what this factor really needs is an observable cue (multi-frame tracker disagreement, force
feedback, a probing motion). Kept in the repo, off by default, with `--obj_err_p` so a fraction
of worlds can carry it.

### Place phase (`place_phase=True`)

The scripted end of the robot's cycle, in the twin: once `at_goal` has held 5 decisions the env
takes over, descends the tcp straight down at the CURRENT xy (damped least squares on a
finite-difference 6-D site Jacobian, 1.5 cm/decision, uniformly scaled to <= 1.2 deg on the
largest joint), releases when the object bottom is within 4 mm of the table or the object stops
descending, and settles 5 decisions. `placed` = released AND resting (|v| < 2 cm/s) within
3.5 cm of the goal **xy**; `t_placed` = the decision it happened. ep_len 180 (t_placed p90 is
128 decisions). Two implementation lessons, both measured:

* a **per-joint** clamp on the IK step rotates the Cartesian direction and the place-down drifts
  sideways: 61 % placed, 3.0 cm median error. Uniform scaling of the whole `dq` is required.
* the **orientation rows of the Jacobian matter**: with position only, the null space rotates the
  wrist during the descent and the object -- welded a cup radius plus half a box below the tcp --
  swings out. Position-only: 38 % placed, 4.4 cm. Full 6-D: 85.6 % placed, 1.9 cm.

Paired eval, 1024 episodes, ep_len 180, DR on (pi's own run-to-run spread on this metric is
**+-1.9 pp** -- 82.9 / 84.4 / 86.6 % over three identical runs -- because a 1.9 cm median error
against a 3.5 cm threshold puts many episodes on the boundary):

| policy | Q budget | placed | seal | t_placed (placed only) | place err med | peak qd med/p90 |
|---|---|---|---|---|---|---|
| pi alone (three identical runs) | - | 82.9 / 84.4 / 86.6 % | 97.3 % | 10.67-10.93 s | 1.8-1.9 cm | 36.9 / 42.4 |
| planner N=16 lam=0.1 | 20 k | 82.1 % (pi 86.6), 83.5 % (pi 82.9) | 97.6 % | 10.54 s | 2.0 cm | 41.4 / 48.4 |
| planner N=16 lam=0.1 | **40 k** | **86.2 %** (pi 82.9, **+3.3 pp**) | **99.1 %** | 10.75 s | 1.9 cm | 40.9 / 47.8 |
| planner N=16 lam=0.1 | 80 k | 83.2 % (pi 84.4, -1.2 pp) | 99.2 % | 11.14 s | 2.0 cm | 39.5 / 45.5 |
| best-of-N N=16 | 20 k | 64.2 % (pi 86.6) | 98.2 % | 11.61 s | 2.6 cm | 46.9 / 52.4 |
| planner, exec peak <= 1.2 x pi | 20 k | 81.3 % (pi 86.6) | 97.7 % | 10.62 s | 2.0 cm | 39.6 / 45.5 |

The same inverted-U in the TD budget appears (20 k under-trains this buffer, 40 k is the peak,
80 k is past it), and on the PLACEMENT metric the planner is within the noise of pi across the
whole sweep: -4.5, +0.6, +3.3, -1.2 pp against +-1.9 pp of run-to-run spread. What does move
consistently and well outside the noise is the **seal rate, 97.3 % -> 99.1-99.2 %** at 40 k and
80 k, i.e. the planner still buys what it bought before the place phase existed -- a better
approach -- and then gives it back at the release.

`t_placed` does not move at all (10.5-11.1 s against pi's 10.7-10.9 s). The scripted descent is
fixed-rate and starts only after the dwell, so the only time lever left is arriving earlier, and
the binding constraint has become the **lateral accuracy of the arrival**, which the planner's
proposals do not improve (place error 1.9-2.0 cm either way). Aggressive selection is now
actively harmful -- best-of-N drops 20 pp, because a hard argmax on Q_time at the moment of
arrival trades away exactly the precision the release needs. **Reading: with the real end of the
cycle in the loop, chunk re-ranking of this policy buys seals, not placements and not cycle
time; the next lever is lateral accuracy at the goal, which needs either a proposal family that
moves the arrival point or a place-phase the planner is allowed to shape.**

### What to do next

1. A **third HL-Gauss head** on a per-decision joint-speed-excess reward, and a speed term in the
   planner's filter. That is the only thing that can make the speed criterion and the time
   criterion compatible.
2. **Select the critic by the paired eval, not by TD loss.** `steps_curve.py` should be part of
   every critic build; the fixed point of TD is not the best ranker.
3. **Widen the proposal set beyond a tube around pi** (the plan's own "later: CFM/DP proposals").
   Everything measured here says the remaining headroom is not reachable from scaled pi chunks.
4. The tracker-error DR needs an **observable** error cue before any critic can help with it.
